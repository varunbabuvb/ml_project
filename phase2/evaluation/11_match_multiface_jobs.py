#!/usr/bin/env python3
"""
Step 11 (multi-face experiment) -- link each target face to a detected, numbered face and write the jobs.
CPU only; run after stage 1-2 and sov_stage3_5_overlay_scaled.py have processed the group images.

  python 11_match_multiface_jobs.py --targets ctx_multi/targets_view.csv --overlay_root ctx_multi/overlay \
        --name multi --extras mixed

MATCHING  one-to-one: all (target, detection) pairs with IoU >= min_iou are sorted by IoU and assigned
          greedily, so two neighbouring targets can never claim the same detection.  Targets without a
          detection get no jobs (reported by group: the detector's misses are a result in themselves).

JOBS      identical text in every condition for the same reference type (prompt variant A, no hint):
   plain, box          positional  'What is the emotion of the {second} face from the left? ...'
                       (N = the face's left-to-right number among the detected faces)
   box_number, sov     numbered    'What is the emotion of the person labeled "k"? ...'
   extras (box_number, sov with the POSITIONAL prompt) for mixed-emotion images (--extras mixed, default),
                       all images (--extras all) or none (--extras none).  Comparing numbered vs positional
                       on the SAME overlay isolates the effect of using the number.
ORDER     jobs are written chunk by chunk (mixed-emotion images first), so a partly finished run
          is always a set of complete chunks.  Split work across GPUs with --chunk_mod 2 --chunk_rem 0|1.

Outputs (in --out_dir, default <overlay_root>/prompts)
   jobs_<name>.jsonl, matches_<name>.csv, jobs_<name>_meta.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
       "tenth", "eleventh", "twelfth"]
CONDS = ("plain", "box", "box_number", "sov")
NUMBERED = ("box_number", "sov")
POSITIONAL_BASE = ("plain", "box")


def ordinal(n):
    return ORD[n - 1] if 1 <= n <= len(ORD) else f"{n}th"


def build_prompt(qtype, sov_id):
    opts = ", ".join(EMOTIONS)
    if qtype == "numbered":
        who = f'the person labeled "{sov_id}"'
    elif qtype == "positional":
        who = f"the {ordinal(sov_id)} face from the left"
    else:
        raise ValueError(qtype)
    return f"What is the emotion of {who}? Choose exactly one of: {opts}. Answer with the emotion word only."


def iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def assign_one_to_one(targets, faces, min_iou):
    """targets: list of boxes; faces: list of boxes. Returns {target_idx: (face_idx, iou)}."""
    pairs = []
    for i, t in enumerate(targets):
        for j, f in enumerate(faces):
            v = iou(t, f)
            if v >= min_iou:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    used_t, used_f, out = set(), set(), {}
    for v, i, j in pairs:
        if i in used_t or j in used_f:
            continue
        used_t.add(i)
        used_f.add(j)
        out[i] = (j, v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True, help="targets_view.csv from step 10")
    ap.add_argument("--overlay_root", required=True)
    ap.add_argument("--name", default="multi")
    ap.add_argument("--variant", default="A")
    ap.add_argument("--conditions", default=",".join(CONDS))
    ap.add_argument("--extras", default="mixed", choices=["mixed", "all", "none"])
    ap.add_argument("--min_iou", type=float, default=0.3)
    ap.add_argument("--chunk_mod", type=int, default=1)
    ap.add_argument("--chunk_rem", type=int, default=0)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    root = Path(args.overlay_root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "prompts"
    out_dir.mkdir(parents=True, exist_ok=True)
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    bad = [c for c in conds if c not in CONDS]
    if bad:
        sys.exit(f"unknown conditions: {bad}")

    tv = pd.read_csv(args.targets).sort_values(["chunk_id", "image_id", "face_id_in_image"])
    rows, jobs = [], []
    for key, g in tv.groupby("image_key", sort=False):
        g = g.sort_values("face_id_in_image")
        fp = root / "final" / f"{key}.json"
        base = {"image_key": key}
        if not fp.exists():
            for r in g.itertuples():
                rows.append({**base, "row_id": int(r.row_id), "matched": False, "reason": "no_final_json"})
            continue
        with open(fp, "r", encoding="utf-8") as f:
            fin = json.load(f)
        faces = fin["faces"]
        tboxes = [[r.tx1, r.ty1, r.tx2, r.ty2] for r in g.itertuples()]
        fboxes = [f_["box"] for f_ in faces]
        asg = assign_one_to_one(tboxes, fboxes, args.min_iou)
        expected = [int(f_["sov_id"]) for f_ in faces]
        for i, r in enumerate(g.itertuples()):
            rec = {**base, "row_id": int(r.row_id), "n_final": int(fin["num_final"]),
                   "n_targets": int(r.n_targets), "matched": False, "sov_id": None, "iou": 0.0, "reason": ""}
            if i not in asg:
                rec["reason"] = "no_detection" if not faces else "no_unique_match"
                rows.append(rec)
                continue
            j, v = asg[i]
            face = faces[j]
            sid, tag = int(face["sov_id"]), int(face.get("tag_number", face["sov_id"]))
            rec.update(matched=True, sov_id=sid, iou=round(v, 4))
            rows.append(rec)
            if int(r.chunk_id) % args.chunk_mod != args.chunk_rem:
                continue
            plan = []
            for c in conds:
                if c in POSITIONAL_BASE:
                    plan.append((c, "positional", sid))
                else:
                    plan.append((c, "numbered", tag))
            if args.extras == "all" or (args.extras == "mixed" and bool(r.mixed)):
                plan += [(c, "positional", sid) for c in NUMBERED if c in conds]
            for c, qt, num in plan:
                rel = f"overlays/{key}/{c}.png"
                if not (root / rel).exists():
                    rec["reason"] = f"overlay_missing:{c}"
                    continue
                jobs.append({
                    "job_id": f"{key}|{c}|{args.variant}|{qt}|{int(r.row_id)}",
                    "row_id": int(r.row_id), "image_key": key, "image": key, "condition": c,
                    "image_relpath": rel, "prompt_set": args.variant, "query_type": qt,
                    "face_id": sid, "prompt": build_prompt(qt, num),
                    "n_faces_final": int(fin["num_final"]), "expected_face_ids": expected,
                    "target_sov_id": sid, "n_targets": int(r.n_targets), "mixed": bool(r.mixed),
                    "chunk_id": int(r.chunk_id), "image_id": int(r.image_id),
                    "label": str(r.expression_name)})

    mt = pd.DataFrame(rows)
    full = tv.merge(mt.drop(columns=["image_key"], errors="ignore"), on="row_id", how="left")
    ids = [j["job_id"] for j in jobs]
    if len(ids) != len(set(ids)):
        sys.exit("ERROR: duplicate job ids")
    jp = out_dir / f"jobs_{args.name}.jsonl"
    with open(jp, "w", encoding="utf-8") as f:
        for j in jobs:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    full.to_csv(out_dir / f"matches_{args.name}.csv", index=False)
    meta = {"name": args.name, "variant": args.variant, "conditions": conds, "extras": args.extras,
            "min_iou": args.min_iou, "chunk_mod": args.chunk_mod, "chunk_rem": args.chunk_rem,
            "n_targets": len(tv), "n_matched": int(mt.matched.sum()), "n_jobs": len(jobs)}
    with open(out_dir / f"jobs_{args.name}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"target faces: {len(tv)} | matched: {int(mt.matched.sum())} ({mt.matched.mean() * 100:.1f}%) "
          f"| jobs written: {len(jobs)} (chunk_mod={args.chunk_mod}, chunk_rem={args.chunk_rem})")
    print("not matched, by reason:", mt[~mt.matched].reason.value_counts().to_dict())
    print("IoU of matched faces, quantiles 10/50/90:", mt[mt.matched].iou.quantile([.1, .5, .9]).round(2).tolist())
    mm = mt[mt.matched]
    print("images where the detector found a different number of faces than we have targets: "
          f"{int((mm.n_final != mm.n_targets).sum())} faces of {len(mm)}")
    for col in ("race_group", "gender15", "mixed"):
        if col in full.columns:
            print(f"\nmatch rate by {col}:")
            print((full.groupby(col).matched.mean() * 100).round(1).to_string())
    full["size_tier"] = pd.qcut(full.face_px_view, 3, labels=["small", "medium", "large"])
    print("\nmatch rate by face size in the view:")
    print((full.groupby("size_tier", observed=True).matched.mean() * 100).round(1).to_string())
    print("\njobs file:", jp)


if __name__ == "__main__":
    main()
