#!/usr/bin/env python3
"""
Step 9 (multi-face experiment) -- choose the images and the target faces.   CPU only.

Inputs : the two FairFace runs (pad 0.15 = label run, pad 0.25 = stability check) and dev_set.csv
Outputs: <out_dir>/multi_faces.csv  (one row per TARGET face)   and   multi_summary.txt

Rules (fixed before any model result exists)
  * Target face : usable (crop side >= 64 px, >= 80% of the padded crop inside the image, face box
                  not touching the border, box >= 48 px) AND its race group (Caucasian-like / Asian /
                  Black) has the same label under both crops AND mean group probability >= group_thr.
  * Image       : at least 2 target faces, and not one of the dev-set images (the dev set was used to
                  choose the prompt).  The images used by the core set ARE allowed (different experiment).
  * Group crop  : the union of the target boxes plus a margin of view_margin x its longer side on every
                  side, clipped to the image, padded to a square and shown to the model at 336 x 336.
                  A target whose face would be smaller than min_face_px in that 336 view is dropped
                  (and the crop is recomputed); an image left with < 2 targets is dropped.  This is a
                  rule about face SIZE only; no labels or model outputs are used.
  * mixed image : the target faces show at least 2 different emotion labels (only then does pointing
                  at the right face matter).  Mixed images are put FIRST in the chunk order, so if time
                  runs out the most informative images are already finished.
  * chunks      : images (not faces) are shuffled with a fixed seed within the mixed / not-mixed blocks and
                  cut into chunks of images_per_chunk images.

Usage
  python 09_build_multiface_set.py --ff15 fairface_all_1.csv --ff25 fairface_all.csv --dev_set dev_set.csv --out_dir multiface
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("build_sets", os.path.join(HERE, "03_build_sets.py"))
bs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bs)          # re-use usable_mask / group_probs so the rules are identical

EMO = bs.EMO
KEY = bs.KEY
VIEW = 336


def region_for(boxes, W, H, margin):
    """boxes: array (k,4) of left, top, right, bottom -> clipped crop region and 336-view scale."""
    ux0, uy0 = boxes[:, 0].min(), boxes[:, 1].min()
    ux1, uy1 = boxes[:, 2].max(), boxes[:, 3].max()
    m = margin * max(ux1 - ux0, uy1 - uy0)
    x0, y0 = max(0, int(round(ux0 - m))), max(0, int(round(uy0 - m)))
    x1, y1 = min(W, int(round(ux1 + m))), min(H, int(round(uy1 + m)))
    side = max(x1 - x0, y1 - y0)
    return x0, y0, x1, y1, VIEW / float(side)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ff15", required=True)
    ap.add_argument("--ff25", required=True)
    ap.add_argument("--dev_set", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pad", type=float, default=0.15)
    ap.add_argument("--group_thr", type=float, default=0.5)
    ap.add_argument("--view_margin", type=float, default=0.15)
    ap.add_argument("--min_face_px", type=float, default=32.0)
    ap.add_argument("--images_per_chunk", type=int, default=40)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    b = pd.read_csv(args.ff15)
    a = pd.read_csv(args.ff25)
    need_a = (KEY + ["prediction_status", "gender_prediction", "age_group_prediction"]
              + [f"race_prob_{r}" for r in bs.RACE] + [f"gender_prob_{g}" for g in bs.GENDER])
    m = b.merge(a[need_a], on=KEY, suffixes=("", "_a"), validate="1:1")
    m = m[(m.prediction_status == "ok") & (m.prediction_status_a == "ok")].copy()

    m["usable"], m["box_min_px"] = bs.usable_mask(m, args.pad)
    Gb, Ga = bs.group_probs(m, ""), bs.group_probs(m, "_a")
    m["race_group"] = Gb.idxmax(axis=1)
    m["race_group_stable"] = m.race_group == Ga.idxmax(axis=1)
    m["race_group_prob"] = [(Ga.at[i, g] + Gb.at[i, g]) / 2 for i, g in zip(m.index, m.race_group)]
    m["gender15"] = m.gender_prediction
    m["gender_stable"] = m.gender_prediction == m.gender_prediction_a
    m["age_band15"] = m.age_group_prediction.map(bs.AGE_BAND)
    m["age_band_stable"] = m.age_band15 == m.age_group_prediction_a.map(bs.AGE_BAND)

    dev_images = set(pd.read_csv(args.dev_set).image_name)
    t = m[m.usable & m.race_group_stable & m.race_group.isin(bs.CORE_GROUPS)
          & (m.race_group_prob >= args.group_thr) & (~m.image_name.isin(dev_images))].copy()
    n_start = len(t)

    # ---- per image: drop too-small targets, recompute the crop, keep images with >= 2 targets ----
    kept_rows, stats = [], {"images_seen": 0, "images_kept": 0, "faces_dropped_small": 0}
    for name, g in t.groupby("image_name", sort=True):
        if len(g) < 2:
            continue
        stats["images_seen"] += 1
        W, H = int(g.image_width.iloc[0]), int(g.image_height.iloc[0])
        cur = g
        while True:
            if len(cur) < 2:
                break
            bx = cur[["face_box_left", "face_box_top", "face_box_right", "face_box_bottom"]].to_numpy(float)
            x0, y0, x1, y1, sc = region_for(bx, W, H, args.view_margin)
            px = np.minimum(bx[:, 2] - bx[:, 0], bx[:, 3] - bx[:, 1]) * sc
            ok = px >= args.min_face_px
            if ok.all():
                break
            stats["faces_dropped_small"] += int((~ok).sum())
            cur = cur[ok]
        if len(cur) < 2:
            continue
        cur = cur.copy()
        cur["reg_x0"], cur["reg_y0"], cur["reg_x1"], cur["reg_y1"] = x0, y0, x1, y1
        cur["view_scale"] = sc
        cur["face_px_view"] = px
        cur["n_targets"] = len(cur)
        cur["n_emotions"] = cur.expression_name.nunique()
        stats["images_kept"] += 1
        kept_rows.append(cur)
    if not kept_rows:
        sys.exit("No multi-face images left with these rules.")
    mf = pd.concat(kept_rows, ignore_index=True)
    mf["mixed"] = mf.n_emotions >= 2

    # ---- image ids, mixed-first chunk order ----
    rng = np.random.default_rng(args.seed)
    imgs = mf.groupby("image_name").mixed.first()
    mixed_imgs = sorted(imgs[imgs].index)
    other_imgs = sorted(imgs[~imgs].index)
    rng.shuffle(mixed_imgs)
    rng.shuffle(other_imgs)
    order = mixed_imgs + other_imgs
    rank = {nm: i for i, nm in enumerate(order)}
    mf["image_id"] = mf.image_name.map(rank)
    mf["image_key"] = mf.image_id.map(lambda i: f"m{i:05d}.jpg")
    mf["chunk_id"] = mf.image_id // args.images_per_chunk
    mf = mf.sort_values(["image_id", "face_id_in_image"]).reset_index(drop=True)

    keep = ["image_id", "image_key", "chunk_id", "mixed", "n_targets", "n_emotions", "row_id",
            "image_name", "face_id_in_image", "face_box_top", "face_box_left", "face_box_right",
            "face_box_bottom", "face_box_confidence", "expression_label", "expression_name",
            "image_width", "image_height", "reg_x0", "reg_y0", "reg_x1", "reg_y1", "view_scale",
            "face_px_view", "box_min_px", "race_group", "race_group_prob", "gender15",
            "gender_stable", "age_band15", "age_band_stable"]
    mf[keep].to_csv(os.path.join(args.out_dir, "multi_faces.csv"), index=False)

    # ---- summary ----
    ni = mf.image_id.nunique()
    L = [f"seed={args.seed} group_thr={args.group_thr} view_margin={args.view_margin} "
         f"min_face_px={args.min_face_px} images_per_chunk={args.images_per_chunk}",
         f"target faces before the size rule: {n_start} | dropped for being < {args.min_face_px:.0f}px in the view: "
         f"{stats['faces_dropped_small']}",
         f"IMAGES: {ni} (mixed-emotion {len(mixed_imgs)}, other {len(other_imgs)}) | TARGET FACES: {len(mf)} "
         f"(in mixed images: {int(mf.mixed.sum())})",
         f"chunks: {mf.chunk_id.nunique()} (the first {int(np.ceil(len(mixed_imgs) / args.images_per_chunk))} "
         f"are mixed-emotion images)",
         "targets per image: " + str(mf.groupby("image_id").size().value_counts().sort_index().to_dict()),
         "face size in the 336 view (px) quantiles 10/25/50/75/90: "
         + str(mf.face_px_view.quantile([.1, .25, .5, .75, .9]).round(0).tolist()),
         "emotion mix (all targets): " + str(mf.expression_name.value_counts().to_dict()),
         "emotion mix (mixed images): " + str(mf[mf.mixed].expression_name.value_counts().to_dict()),
         "\ngroup x emotion (all target faces):",
         mf.pivot_table(index="race_group", columns="expression_name", values="row_id", aggfunc="count")
         .reindex(columns=[e for e in EMO]).fillna(0).astype(int).to_string(),
         "\ngroup x emotion (mixed images only):",
         mf[mf.mixed].pivot_table(index="race_group", columns="expression_name", values="row_id", aggfunc="count")
         .reindex(columns=[e for e in EMO]).fillna(0).astype(int).to_string(),
         "\nexpected jobs (4 conditions + 2 extra positional prompts on mixed images): "
         f"{4 * len(mf) + 2 * int(mf.mixed.sum())}"]
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(args.out_dir, "multi_summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
