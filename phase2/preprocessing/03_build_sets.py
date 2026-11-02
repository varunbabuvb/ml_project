#!/usr/bin/env python3
"""
Step 3 -- build the DEVELOPMENT set and the CORE (evaluation) set.   CPU only, runs anywhere.

Inputs : the two FairFace runs (pad 0.25 = fairface_all.csv, pad 0.15 = fairface_all_1.csv)
Outputs: <out_dir>/dev_set.csv, core_set.csv, sets_summary.txt

Rules (all decided BEFORE looking at any model result)
  * usable face  : crop side >= 64 px, >= 80% of the padded crop inside the image,
                   face box not touching the image border, face box >= 48 px.
  * race group   : FairFace probabilities are summed into Caucasian-like (White + Middle
                   Eastern), Asian (East Asian + Southeast Asian), Black, Indian, Latino.
                   The label is the pad-0.15 prediction; a face is "stable" only if the
                   pad-0.25 run gives the same merged group.  Indian/Latino are not used.
  * dev set      : dev_per_class faces per emotion (random, any group). Used ONLY for
                   choosing the prompt / model / calibration and for debugging.
  * core set     : per_cell faces for every (race group x emotion) cell, drawn from stable
                   faces whose average group probability >= group_thr, from images that do
                   NOT appear in the dev set (so no image is ever in both).
  * chunks       : the core set is shuffled (fixed seed) and cut into chunk files.

Usage
  python 03_build_sets.py --ff15 fairface_all_1.csv --ff25 fairface_all.csv --out_dir sets
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

KEY = ["image_name", "face_id_in_image"]
EMO = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
RACE = ["White", "Black", "Latino_Hispanic", "East Asian",
        "Southeast Asian", "Indian", "Middle Eastern"]
GENDER = ["Male", "Female"]
GROUP_OF = {"White": "Caucasian-like", "Middle Eastern": "Caucasian-like",
            "Black": "Black", "East Asian": "Asian", "Southeast Asian": "Asian",
            "Indian": "Indian", "Latino_Hispanic": "Latino"}
CORE_GROUPS = ["Caucasian-like", "Asian", "Black"]
AGE_BAND = {"0-2": "0-19", "3-9": "0-19", "10-19": "0-19", "20-29": "20-29",
            "30-39": "30-39", "40-49": "40-59", "50-59": "40-59",
            "60-69": "60+", "70+": "60+"}


def group_probs(df, suffix=""):
    P = df[[f"race_prob_{r}{suffix}" for r in RACE]].to_numpy()
    out = {}
    for g in sorted(set(GROUP_OF.values())):
        idx = [i for i, r in enumerate(RACE) if GROUP_OF[r] == g]
        out[g] = P[:, idx].sum(1)
    return pd.DataFrame(out, index=df.index)


def usable_mask(df, pad):
    w = df.face_box_right - df.face_box_left
    h = df.face_box_bottom - df.face_box_top
    px, py = (pad * w).round(), (pad * h).round()
    x0, y0 = df.face_box_left - px, df.face_box_top - py
    x1, y1 = df.face_box_right + px, df.face_box_bottom + py
    full = (x1 - x0) * (y1 - y0)
    cw = (np.minimum(df.image_width, x1) - np.maximum(0, x0)).clip(lower=0)
    ch = (np.minimum(df.image_height, y1) - np.maximum(0, y0)).clip(lower=0)
    ret = (cw * ch) / full
    small = df[["crop_width", "crop_height"]].min(axis=1) < 64
    boxmin = np.minimum(w, h)
    ok = (~small) & (ret >= 0.8) & (~df.bbox_touches_image_edge.astype(bool)) & (boxmin >= 48)
    return ok, boxmin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ff15", required=True, help="FairFace run with pad 0.15 (the label run)")
    ap.add_argument("--ff25", required=True, help="FairFace run with pad 0.25 (stability check)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pad", type=float, default=0.15, help="padding used in the --ff15 run")
    ap.add_argument("--per_cell", type=int, default=75)
    ap.add_argument("--group_thr", type=float, default=0.5)
    ap.add_argument("--dev_per_class", type=int, default=40)
    ap.add_argument("--chunk_size", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    b = pd.read_csv(args.ff15)
    a = pd.read_csv(args.ff25)
    need_a = (KEY + ["prediction_status", "race_prediction", "gender_prediction", "age_group_prediction"]
              + [f"race_prob_{r}" for r in RACE] + [f"gender_prob_{g}" for g in GENDER])
    m = b.merge(a[need_a], on=KEY, suffixes=("", "_a"), validate="1:1")
    if len(m) != len(b):
        sys.exit(f"ERROR: the two runs do not cover the same faces ({len(b)} vs {len(m)} matched).")
    m = m[(m.prediction_status == "ok") & (m.prediction_status_a == "ok")].copy()

    # ---- usable, race group, stability ----
    m["usable"], m["box_min_px"] = usable_mask(m, args.pad)
    Gb, Ga = group_probs(m, ""), group_probs(m, "_a")
    m["race_group"] = Gb.idxmax(axis=1)
    m["race_group_pad25"] = Ga.idxmax(axis=1)
    m["race_group_stable"] = m.race_group == m.race_group_pad25
    lab = m.race_group.to_numpy()
    m["race_group_prob"] = [(Ga.at[i, g] + Gb.at[i, g]) / 2 for i, g in zip(m.index, lab)]
    m["race7_pad15"] = m.race_prediction
    m["race7_pad25"] = m.race_prediction_a

    # gender / age
    m["gender15"] = m.gender_prediction
    m["gender_stable"] = m.gender_prediction == m.gender_prediction_a
    gp = [m.at[i, f"gender_prob_{g}"] + m.at[i, f"gender_prob_{g}_a"]
          for i, g in zip(m.index, m.gender_prediction)]
    m["gender_prob"] = np.array(gp) / 2
    m["age_band15"] = m.age_group_prediction.map(AGE_BAND)
    m["age_band_stable"] = m.age_band15 == m.age_group_prediction_a.map(AGE_BAND)

    # ---- dev set (random, any group) ----
    pool = m[m.usable]
    dev = pd.concat([pool[pool.expression_name == e].sample(
        n=min(args.dev_per_class, (pool.expression_name == e).sum()), random_state=args.seed)
        for e in EMO])
    dev_images = set(dev.image_name)

    # ---- core set (image-disjoint from dev) ----
    cand = m[m.usable & m.race_group_stable & m.race_group.isin(CORE_GROUPS)
             & (m.race_group_prob >= args.group_thr) & (~m.image_name.isin(dev_images))]
    parts, cells = [], []
    for g in CORE_GROUPS:
        for e in EMO:
            p = cand[(cand.race_group == g) & (cand.expression_name == e)]
            n = min(args.per_cell, len(p))
            parts.append(p.sample(n=n, random_state=args.seed))
            cells.append((g, e, len(p), n))
    core = pd.concat(parts).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    core["chunk_id"] = core.index // args.chunk_size

    # ---- sanity checks ----
    assert not (set(core.image_name) & dev_images), "dev and core share an image"
    assert not core.duplicated(KEY).any() and not dev.duplicated(KEY).any()

    keep = (["row_id", "image_name", "face_id_in_image", "face_box_top", "face_box_left",
             "face_box_right", "face_box_bottom", "face_box_confidence", "expression_label",
             "expression_name", "image_width", "image_height", "box_min_px",
             "race_group", "race_group_stable", "race_group_prob", "race7_pad15",
             "gender15", "gender_stable", "gender_prob", "age_band15", "age_band_stable"])
    keep = [c for c in keep if c in m.columns]
    dev[keep].to_csv(os.path.join(args.out_dir, "dev_set.csv"), index=False)
    core[keep + ["chunk_id"]].to_csv(os.path.join(args.out_dir, "core_set.csv"), index=False)

    # ---- summary ----
    L = []
    L.append(f"seed={args.seed} per_cell={args.per_cell} group_thr={args.group_thr} "
             f"dev_per_class={args.dev_per_class} chunk_size={args.chunk_size}")
    L.append(f"faces in both runs: {len(m)} | usable: {int(m.usable.sum())}")
    L.append(f"DEV  : {len(dev)} faces, {dev.image_name.nunique()} images; per class: "
             f"{dev.expression_name.value_counts().reindex(EMO).to_dict()}")
    L.append(f"CORE : {len(core)} faces, {core.image_name.nunique()} images, "
             f"{core.chunk_id.nunique()} chunks of <= {args.chunk_size}")
    L.append("dev/core share no image: OK")
    L.append("\nCORE cells (available -> taken):")
    tab = pd.DataFrame(cells, columns=["group", "emotion", "available", "taken"])
    L.append(tab.pivot(index="group", columns="emotion", values="taken")[EMO].to_string())
    short = tab[tab.taken < args.per_cell]
    if len(short):
        L.append("\nCells smaller than per_cell (everything available was taken):")
        L.append(short.to_string(index=False))
    L.append("\nCORE gender (stable only): " +
             str(core[core.gender_stable].gender15.value_counts().to_dict()))
    L.append("CORE age band (stable only): " +
             str(core[core.age_band_stable].age_band15.value_counts().to_dict()))
    L.append("CORE median face box (px): " +
             str(core.groupby("race_group").box_min_px.median().round(0).to_dict()))
    L.append("CORE multi-face images share: %.3f" %
             (core.groupby("image_name").image_name.transform("size") > 1).mean())
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(args.out_dir, "sets_summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
