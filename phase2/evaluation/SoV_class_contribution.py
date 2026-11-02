#!/usr/bin/env python3
"""Decompose the paired core-set SoV - Plain change by true emotion class.

Inputs are the successful Stage-6 result JSONL and core_set.csv for ONE model.
The script uses only faces with both Plain and SoV outputs, and clusters bootstrap
resampling by source image. It reports two additive decompositions:
  * balanced-accuracy contribution = (SoV recall_c - Plain recall_c) / K
  * accuracy contribution = prevalence_c * (SoV recall_c - Plain recall_c)
The first sums to the SoV - Plain balanced-accuracy difference; the second sums
to the ordinary accuracy difference. Contributions are descriptive, not causal.

Example:
python SoV_class_contribution.py --results results_llava15_jobs_core.jsonl \
  --core_set core_set.csv --model LLaVA --out_dir class_contribution_llava
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: skipped malformed JSON at line {line_no}: {e}")
    return pd.DataFrame(rows)


def norm_label(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    return next((c for c in CLASSES if c.lower() == s), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="Stage-6 core results JSONL for one model")
    ap.add_argument("--core_set", required=True, help="core_set.csv used for this run")
    ap.add_argument("--model", default="model", help="Model label for output metadata")
    ap.add_argument("--out_dir", default="class_contribution")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    core = pd.read_csv(args.core_set)
    required_core = {"row_id", "expression_name", "image_name"}
    if not required_core.issubset(core.columns):
        raise SystemExit(f"core_set missing columns: {sorted(required_core - set(core.columns))}")
    core = core[["row_id", "expression_name", "image_name"]].copy()
    core["truth"] = core.expression_name.map(norm_label)
    core = core[core.truth.isin(CLASSES)].drop_duplicates("row_id")

    res = read_jsonl(args.results)
    if res.empty:
        raise SystemExit("No result rows found in JSONL.")
    for col in ("row_id", "condition", "pred"):
        if col not in res.columns:
            raise SystemExit(f"Results JSONL missing required field: {col}")
    # Use the main core prompt/run only; these fields are present in Stage-6 records.
    if "prompt_set" in res.columns:
        res = res[res.prompt_set.astype(str).str.upper().eq("A")]
    if "query_type" in res.columns:
        res = res[res.query_type.eq("center")]
    # Keep successful jobs. Error rows are counted separately for diagnostics.
    n_error = int(res.error.notna().sum()) if "error" in res.columns else 0
    if "error" in res.columns:
        res = res[res.error.isna()].copy()
    res["pred"] = res.pred.map(norm_label)
    res = res[res.condition.isin(["plain", "sov"]) & res.pred.isin(CLASSES)].copy()
    res = res.sort_values("created_utc" if "created_utc" in res.columns else "row_id")
    res = res.drop_duplicates(["row_id", "condition"], keep="last")

    wide = res.pivot(index="row_id", columns="condition", values="pred").reset_index()
    if not {"plain", "sov"}.issubset(wide.columns):
        raise SystemExit("Need successful Plain and SoV outputs in the supplied JSONL.")
    paired = wide.dropna(subset=["plain", "sov"]).merge(core, on="row_id", how="inner")
    if paired.empty:
        raise SystemExit("No paired Plain/SoV predictions matched core_set.csv.")
    paired["image_name"] = paired.image_name.fillna(paired.row_id.astype(str))

    # Each bootstrap draw samples source images with replacement, retaining all faces
    # from each selected image. Class-specific recall is computed within each draw.
    rng = np.random.default_rng(args.seed)
    images = paired.image_name.unique().tolist()
    grouped = {im: paired[paired.image_name == im] for im in images}
    boot_values = {c: [] for c in CLASSES}
    boot_ba = []
    for _ in range(args.boot):
        sampled = rng.choice(images, size=len(images), replace=True)
        b = pd.concat([grouped[im] for im in sampled], ignore_index=True)
        recalls = {}
        for c in CLASSES:
            sub = b[b.truth == c]
            if len(sub):
                recalls[c] = ((sub.sov == c).mean() - (sub.plain == c).mean())
            else:
                recalls[c] = np.nan
            boot_values[c].append(recalls[c])
        vals = [recalls[c] for c in CLASSES]
        boot_ba.append(np.nanmean(vals) if np.isfinite(vals).any() else np.nan)

    rows = []
    n_total = len(paired)
    for c in CLASSES:
        sub = paired[paired.truth == c]
        n = len(sub)
        if not n:
            continue
        r_plain = (sub.plain == c).mean()
        r_sov = (sub.sov == c).mean()
        delta = r_sov - r_plain
        prevalence = n / n_total
        bc = delta / len(CLASSES)
        ac = prevalence * delta
        arr = np.asarray(boot_values[c], dtype=float)
        arr = arr[np.isfinite(arr)]
        lo, hi = np.percentile(arr, [2.5, 97.5]) if len(arr) else (np.nan, np.nan)
        rows.append({
            "model": args.model, "emotion": c, "n_faces": n,
            "class_prevalence": prevalence,
            "plain_recall": r_plain, "sov_recall": r_sov,
            "delta_recall_pp": 100 * delta,
            "balanced_accuracy_contribution_pp": 100 * bc,
            "balanced_accuracy_contribution_ci_low_pp": 100 * lo / len(CLASSES),
            "balanced_accuracy_contribution_ci_high_pp": 100 * hi / len(CLASSES),
            "ordinary_accuracy_contribution_pp": 100 * ac,
        })
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(out / "class_contributions.csv", index=False)

    # Direct paired aggregate deltas on exactly the same matched sample.
    ba_plain = np.mean([(paired[paired.truth == c].plain == c).mean() for c in CLASSES if (paired.truth == c).any()])
    ba_sov = np.mean([(paired[paired.truth == c].sov == c).mean() for c in CLASSES if (paired.truth == c).any()])
    acc_plain = (paired.plain == paired.truth).mean()
    acc_sov = (paired.sov == paired.truth).mean()
    summary = pd.DataFrame([{
        "model": args.model, "paired_faces": len(paired), "source_images": len(images),
        "failed_or_error_rows_in_input": n_error,
        "plain_balanced_accuracy": ba_plain, "sov_balanced_accuracy": ba_sov,
        "delta_balanced_accuracy_pp": 100 * (ba_sov - ba_plain),
        "sum_class_ba_contributions_pp": detail.balanced_accuracy_contribution_pp.sum(),
        "plain_accuracy": acc_plain, "sov_accuracy": acc_sov,
        "delta_accuracy_pp": 100 * (acc_sov - acc_plain),
        "sum_class_accuracy_contributions_pp": detail.ordinary_accuracy_contribution_pp.sum(),
        "bootstrap_resamples": args.boot, "bootstrap_seed": args.seed,
    }])
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote: {out / 'class_contributions.csv'}")
    print(f"Wrote: {out / 'summary.csv'}")
    print("Interpretation: BA contributions sum to the paired balanced-accuracy change; "
          "ordinary-accuracy contributions sum to the paired accuracy change.")


if __name__ == "__main__":
    main()
