#!/usr/bin/env python3
"""
Step 5 -- link each target face to a detected, numbered face and write the job list.
Runs on CPU, anywhere, AFTER your existing stage 1-2 and stage 3-5 scripts have processed
the context images from step 4.

Order of operations
  1. python 04_make_context_images.py ...                      (laptop)
  2. python sov_stage1_2_detect.py --input <ctx>/images --output <ctx>/det
  3. python sov_stage3_5_overlay.py --input <ctx>/images --detections <ctx>/det/detections \
                                    --output <ctx>/overlay
  4. python 05_match_and_jobs.py --targets <ctx>/targets.csv --overlay_root <ctx>/overlay \
                                 --name dev --variants A,B --conditions plain,sov

What it does
  * MATCHING: for every target face, take the detected face whose box overlaps the dataset's
    face box the most (IoU).  If the best IoU is below --min_iou the face is "not detected";
    it gets no jobs, and is reported (by group) because detector recall is itself a result.
  * JOBS (one job = one image x one condition x one question), all with identical text in
    every condition:
        center    "What is the emotion of the person in the center of the image? ..."
                  for every condition in --conditions.
        numbered  'What is the emotion of the person labeled "k"? ...'  only for the numbered
                  conditions (box_number, sov), only when --numbered is given and only for
                  images with more than one numbered face.
  * PROMPT VARIANTS:  A = no hint     B = adds: Use "Neutral" when the face shows no clear emotion.
    Use A,B for the development set; pass ONLY the chosen one for the core set.

Outputs (in --out_dir, default <overlay_root>/prompts)
    jobs_<name>.jsonl      the jobs (same fields as your stage 6, plus row_id and labels)
    matches_<name>.csv     one row per target face: matched?, sov_id, IoU, #faces
    jobs_<name>_meta.json  settings and counts
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
NEUTRAL_HINT = ' Use "Neutral" when the face shows no clear emotion.'
CONDITIONS_ALL = ("plain", "box", "box_number", "sov")
NUMBERED = ("box_number", "sov")


def build_prompt(variant, qtype, sov_id=None):
    if variant not in ("A", "B"):
        raise ValueError("variant must be A or B")
    hint = NEUTRAL_HINT if variant == "B" else ""
    opts = ", ".join(EMOTIONS)
    if qtype == "center":
        who = "the person in the center of the image"
    elif qtype == "numbered":
        who = f'the person labeled "{sov_id}"'
    else:
        raise ValueError(qtype)
    return (f"What is the emotion of {who}? Choose exactly one of: {opts}.{hint} "
            "Answer with the emotion word only.")


def iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match_target(target_box, faces):
    """Return (best_face or None, best_iou, second_best_iou)."""
    scored = sorted(((iou(target_box, f["box"]), f) for f in faces), key=lambda x: -x[0])
    if not scored:
        return None, 0.0, 0.0
    second = scored[1][0] if len(scored) > 1 else 0.0
    return scored[0][1], scored[0][0], second


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True, help="targets.csv from step 04")
    ap.add_argument("--overlay_root", required=True, help="OUTPUT folder of stage 3-5")
    ap.add_argument("--name", required=True, help="dev or core (used in file names)")
    ap.add_argument("--variants", default="A", help="A, B or A,B")
    ap.add_argument("--conditions", default=",".join(CONDITIONS_ALL))
    ap.add_argument("--numbered", action="store_true",
                    help="also add numbered questions for box_number / sov on multi-face images")
    ap.add_argument("--min_iou", type=float, default=0.3)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    root = Path(args.overlay_root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "prompts"
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    bad = [c for c in conds if c not in CONDITIONS_ALL]
    if bad:
        sys.exit(f"unknown conditions: {bad}")

    tg = pd.read_csv(args.targets)
    rows, jobs = [], []
    for _, r in tg.iterrows():
        key = r["key"]
        fp = root / "final" / f"{key}.json"
        rec = {"row_id": int(r.row_id), "key": key, "matched": False, "sov_id": None,
               "iou": 0.0, "iou_second": 0.0, "n_final": None, "n_detected": None,
               "reason": ""}
        if not fp.exists():
            rec["reason"] = "no_final_json (detection/overlay stage did not finish for this image)"
            rows.append(rec)
            continue
        with open(fp, "r", encoding="utf-8") as f:
            fin = json.load(f)
        faces = fin["faces"]
        rec["n_final"], rec["n_detected"] = int(fin["num_final"]), int(fin["num_detected"])
        best, b_iou, s_iou = match_target([r.tx1, r.ty1, r.tx2, r.ty2], faces)
        rec["iou"], rec["iou_second"] = round(b_iou, 4), round(s_iou, 4)
        if best is None or b_iou < args.min_iou:
            rec["reason"] = "not_detected" if not faces else f"best_iou_below_{args.min_iou}"
            rows.append(rec)
            continue
        rec["matched"], rec["sov_id"] = True, int(best["sov_id"])
        rows.append(rec)

        expected = [int(f["sov_id"]) for f in faces]
        for variant in variants:
            for cond in conds:
                rel = f"overlays/{key}/{cond}.png"
                if not (root / rel).exists():
                    rec["reason"] = f"overlay_missing:{cond}"
                    continue
                qts = [("center", None)]
                if args.numbered and cond in NUMBERED and len(faces) > 1:
                    qts.append(("numbered", int(best["sov_id"])))
                for qt, fid in qts:
                    jobs.append({
                        "job_id": f"{key}|{cond}|{variant}|{qt}" + (f"|{fid}" if fid else ""),
                        "row_id": int(r.row_id), "image_key": key, "image": key,
                        "condition": cond, "image_relpath": rel,
                        "prompt_set": variant, "query_type": qt, "face_id": fid,
                        "prompt": build_prompt(variant, qt, fid),
                        "n_faces_final": len(faces), "expected_face_ids": expected,
                        "target_sov_id": int(best["sov_id"]),
                        "label": str(r.expression_name),
                    })

    mt = pd.DataFrame(rows)
    full = tg.merge(mt, on=["row_id", "key"], how="left")
    ids = [j["job_id"] for j in jobs]
    if len(ids) != len(set(ids)):
        sys.exit("ERROR: duplicate job ids")

    jp = out_dir / f"jobs_{args.name}.jsonl"
    with open(jp, "w", encoding="utf-8") as f:
        for j in jobs:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    full.to_csv(out_dir / f"matches_{args.name}.csv", index=False)
    meta = {"name": args.name, "variants": variants, "conditions": conds,
            "numbered": args.numbered, "min_iou": args.min_iou, "n_targets": len(tg),
            "n_matched": int(mt.matched.sum()), "n_jobs": len(jobs)}
    with open(out_dir / f"jobs_{args.name}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"targets: {len(tg)} | matched: {int(mt.matched.sum())} "
          f"({mt.matched.mean() * 100:.1f}%) | jobs: {len(jobs)}")
    print("not matched, by reason:", mt[~mt.matched].reason.value_counts().to_dict())
    print("IoU of matched faces, quantiles 10/50/90:",
          mt[mt.matched].iou.quantile([.1, .5, .9]).round(2).tolist())
    amb = mt[mt.matched & (mt.iou_second > 0.7 * mt.iou)]
    print(f"ambiguous matches (2nd best IoU > 70% of best): {len(amb)}")
    for col in ("race_group", "gender15", "expression_name"):
        if col in full.columns:
            print(f"\nmatch rate by {col}:")
            print((full.groupby(col).matched.mean() * 100).round(1).to_string())
    print("\njobs file:", jp)


if __name__ == "__main__":
    main()
