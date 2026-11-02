#!/usr/bin/env python3
"""
Step 12 -- tag-reading probe (replaces the old probe in 06_score_labels.py, whose numbers were misleading).

Question: can the model READ the number written on a face's tag?
Why the old probe failed: with left-to-right numbering the numbers equal the positions, so a model that
just lists "1, 2, 3" looks like it reads them; and "invented" numbers were not penalised in the hit rate.

This probe uses overlays rendered with SHUFFLED tag numbers (sov_id still counts left to right, but the
tags show a random permutation of 1..n).  For each image with >= --min_faces faces we ask, for the
box_number and sov conditions:
      "What number is written on the tag of the leftmost face? Answer with the number only."
      "What number is written on the tag of the rightmost face? Answer with the number only."
and we only ask a question when the correct answer differs from the position it would have without
shuffling (leftmost != 1, rightmost != n), so counting cannot give the answer.
Scoring: exact match.  Reported next to it: the chance level (mean of 1/n), the share of answers that are
a number which appears on some tag, and the share that equal the "unshuffled" position guess.

Needs a GPU (Colab / Kaggle) and 06_score_labels.py in the same folder.

  # 1) render shuffled overlays from the SAME detections as the multi-face run
  !python sov_stage3_5_overlay_scaled.py --input /content/ctx_multi/images --detections /content/ctx_multi/det/detections \
        --output /content/ctx_multi/overlay_shuffled --shuffle_numbers 7
  # 2) run the probe  (about 150 images x 2 conditions x 2 questions = about 600 short jobs)
  !python 12_tag_probe.py --overlay_root /content/ctx_multi/overlay_shuffled --model llava15 \
        --out_dir /content/drive/MyDrive/sov/probe --n_images 150
"""
import argparse
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("score_labels", os.path.join(HERE, "06_score_labels.py"))
sl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sl)

CONDS = ("box_number", "sov")


def build_jobs(root, n_images, min_faces):
    jobs, n_used = [], 0
    for fp in sorted((root / "final").glob("*.json")):
        if fp.name.endswith(".prev.json"):
            continue
        with open(fp, "r", encoding="utf-8") as f:
            rec = json.load(f)
        faces = sorted(rec["faces"], key=lambda x: x["sov_id"])
        n = len(faces)
        if n < min_faces:
            continue
        key = fp.name[:-len(".json")]
        shown = sorted(int(f["tag_number"]) for f in faces)
        qs = []
        if faces[0]["tag_number"] != 1:
            qs.append(("leftmost", int(faces[0]["tag_number"])))
        if faces[-1]["tag_number"] != n:
            qs.append(("rightmost", int(faces[-1]["tag_number"])))
        if not qs:
            continue
        n_used += 1
        for cond in CONDS:
            rel = f"overlays/{key}/{cond}.png"
            if not (root / rel).exists():
                continue
            for which, expected in qs:
                jobs.append({"job_id": f"{key}|{cond}|{which}", "image_key": key, "condition": cond,
                             "image_relpath": rel, "which": which, "expected": expected, "n_faces": n,
                             "shown_numbers": shown,
                             "prompt": f"What number is written on the tag of the {which} face? "
                                       "Answer with the number only."})
        if n_used >= n_images:
            break
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay_root", required=True, help="output folder of the SHUFFLED scaled overlay run")
    ap.add_argument("--model", default="llava15", choices=["llava15", "qwen25vl", "fake"])
    ap.add_argument("--out_dir", default="probe")
    ap.add_argument("--n_images", type=int, default=150)
    ap.add_argument("--min_faces", type=int, default=3)
    ap.add_argument("--no_4bit", action="store_true")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--qwen_side", type=int, default=336)
    ap.add_argument("--summary_only", action="store_true", help="just re-print the summary of an earlier run")
    args = ap.parse_args()

    root, out_dir = Path(args.overlay_root), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / f"probe_tags_{args.model}.jsonl"

    if not args.summary_only:
        jobs = build_jobs(root, args.n_images, args.min_faces)
        if not jobs:
            sys.exit("No probe jobs: need shuffled overlays with at least --min_faces faces "
                     "(run sov_stage3_5_overlay_scaled.py with --shuffle_numbers).")
        sl.ensure_trailing_newline(res_path)
        done = {r["job_id"] for r in sl.read_jsonl(res_path) if r.get("error") is None}
        todo = [j for j in jobs if j["job_id"] not in done]
        print(f"{len(jobs)} probe jobs, {len(done)} already done, {len(todo)} to run | model={args.model}")
        if todo:
            be = sl.build_backend(args)
            t0 = time.time()
            cache = {"rel": None, "img": None}
            for i, j in enumerate(todo, 1):
                try:
                    if cache["rel"] != j["image_relpath"]:
                        cache.update(rel=j["image_relpath"], img=sl.load_image(root / j["image_relpath"]))
                    txt = be.generate_text(cache["img"], j["prompt"], 6)
                    m = re.search(r"\d+", txt)
                    pred = int(m.group(0)) if m else None
                    sl.append_jsonl(res_path, {**{k: j[k] for k in ("job_id", "image_key", "condition", "which",
                                                                      "expected", "n_faces", "shown_numbers")},
                                               "output": txt, "pred": pred, "error": None, **be.info()})
                except Exception as e:
                    print("FAILED", j["job_id"], e)
                    sl.append_jsonl(res_path, {"job_id": j["job_id"], "error": repr(e)})
                if i % 50 == 0 or i == len(todo):
                    print(f"[{i}/{len(todo)}] {(time.time() - t0) / i:.2f}s/job")

    rows = [r for r in sl.read_jsonl(res_path) if r.get("error") is None]
    if not rows:
        print("No results yet.")
        return
    print("\n=== TAG-READING PROBE ===")
    for cond in CONDS:
        for which in ("leftmost", "rightmost", "both"):
            sub = [r for r in rows if r["condition"] == cond and (which == "both" or r["which"] == which)]
            if not sub:
                continue
            n = len(sub)
            acc = sum(r["pred"] == r["expected"] for r in sub) / n
            chance = sum(1.0 / r["n_faces"] for r in sub) / n
            valid = sum(r["pred"] in r["shown_numbers"] for r in sub if r["pred"] is not None) / n
            pos = sum(r["pred"] == (1 if r["which"] == "leftmost" else r["n_faces"]) for r in sub) / n
            print(f"{cond:11s} {which:9s} n={n:4d}  exact-match {acc:.3f}  chance {chance:.3f}  "
                  f"answer is a shown number {valid:.3f}  answer = unshuffled position {pos:.3f}")
    print("\nReading: exact-match near chance means the model is NOT reading the tags; "
          "a high 'unshuffled position' share means it answers by position.")
    print("results:", res_path)


if __name__ == "__main__":
    main()
