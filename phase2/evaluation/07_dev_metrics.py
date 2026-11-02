#!/usr/bin/env python3
"""
Step 7 -- score the DEVELOPMENT-set results: choose the prompt variant, run the model gate,
and fit the calibration vectors.   CPU only.

Input : one or more results_<model>_jobs_dev.jsonl files from step 6 (any mix of models),
        optionally probe_<model>.jsonl files (the tag-reading probe).
Output: dev_metrics.csv, calibration.json, and a printed report.

What it reports, per (model, prompt variant, condition)
  n, accuracy, macro-F1 (the 7 classes), recall per class, share of predictions per class,
  mean `mass` (how much probability the model puts on the 7 emotion words), and the same
  numbers after CALIBRATION.
Calibration (fixed procedure, applied identically in every condition): the dev set has the
same number of faces per emotion, so on average the model should predict every class equally
often. We divide the class scores by the model's average prediction (per model / variant /
condition).  The calibrated dev scores are optimistic (fit and scored on the same faces);
use them to see whether a bias such as "never says Neutral" is fixable, not as results.

GATE (thresholds fixed in advance; change them BEFORE you look, not after)
  1. tag reading   >= --min_tags   (default 0.80)  [needs the probe files]
  2. macro-F1 of the plain condition with the better variant >= --min_f1 (default 0.25;
     chance for 7 classes is about 0.14)
  3. the scored label equals the generated answer on >= --min_agree (default 0.95)
     [needs results made with --also_generate]

Usage
  python 07_dev_metrics.py --results res/results_llava15_jobs_dev.jsonl res/results_qwen25vl_jobs_dev.jsonl \
        --probes res/probe_llava15.jsonl res/probe_qwen25vl.jsonl --out_dir dev_report
"""
import argparse
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
SYNONYMS = {"anger": "Angry", "angry": "Angry", "disgusted": "Disgust", "disgust": "Disgust",
            "fear": "Fear", "fearful": "Fear", "scared": "Fear", "afraid": "Fear",
            "happy": "Happy", "happiness": "Happy", "joy": "Happy", "sad": "Sad",
            "sadness": "Sad", "surprise": "Surprise", "surprised": "Surprise",
            "neutral": "Neutral"}


def normalize_label(text):
    m = re.search(r"[A-Za-z]+", text or "")
    return SYNONYMS.get(m.group(0).lower()) if m else None


def read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def metrics(truth, pred):
    """truth, pred: arrays of label strings. Returns dict."""
    truth, pred = np.asarray(truth), np.asarray(pred)
    f1s, recalls = [], {}
    for c in EMOTIONS:
        tp = np.sum((truth == c) & (pred == c))
        fp = np.sum((truth != c) & (pred == c))
        fn = np.sum((truth == c) & (pred != c))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else float("nan")
        recalls[c] = r
        if tp + fn:                       # only classes that exist in the truth
            f1s.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    return {"n": len(truth), "acc": float(np.mean(truth == pred)),
            "macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
            "recall": recalls,
            "pred_share": {c: float(np.mean(pred == c)) for c in EMOTIONS}}


def softmax_matrix(lps):
    x = np.asarray(lps, dtype=float)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--probes", nargs="*", default=[])
    ap.add_argument("--out_dir", default="dev_report")
    ap.add_argument("--min_tags", type=float, default=0.80)
    ap.add_argument("--min_f1", type=float, default=0.25)
    ap.add_argument("--min_agree", type=float, default=0.95)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    recs = []
    for p in args.results:
        recs += [r for r in read_jsonl(p) if r.get("error") is None]
    df = pd.DataFrame(recs)
    df = df[df.query_type == "center"].copy()
    df["truth"] = df.label.str.capitalize()
    df["variant"] = df.prompt_set

    rows, calib = [], {}
    for (be, var, cond), g in df.groupby(["backend", "variant", "condition"]):
        lps = np.array([[r[e] for e in EMOTIONS] for r in g.label_logprobs])
        raw_pred = np.array(EMOTIONS)[lps.argmax(1)]
        m = metrics(g.truth.values, raw_pred)
        P = softmax_matrix(lps)
        log_pbar = np.log(P.mean(axis=0))
        cal_pred = np.array(EMOTIONS)[(np.log(P) - log_pbar).argmax(1)]
        mc = metrics(g.truth.values, cal_pred)
        calib[f"{be}|{var}|{cond}"] = dict(zip(EMOTIONS, map(float, log_pbar)))
        row = {"backend": be, "variant": var, "condition": cond, "n": m["n"],
               "acc": m["acc"], "macro_f1": m["macro_f1"],
               "neutral_recall": m["recall"]["Neutral"], "neutral_share": m["pred_share"]["Neutral"],
               "mean_mass": float(g["mass"].mean()),
               "cal_acc": mc["acc"], "cal_macro_f1": mc["macro_f1"],
               "cal_neutral_recall": mc["recall"]["Neutral"]}
        for c in EMOTIONS:
            row[f"recall_{c}"] = m["recall"][c]
            row[f"share_{c}"] = m["pred_share"][c]
        if "generated" in g and g.generated.notna().any():
            gen = g[g.generated.notna()]
            gl = gen.generated.map(normalize_label)
            row["gen_parseable"] = float(gl.notna().mean())
            row["agree_scored_vs_generated"] = float((gl == gen.pred).mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.out_dir, "dev_metrics.csv"), index=False)
    with open(os.path.join(args.out_dir, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2)

    pd.set_option("display.width", 200)
    cols = ["backend", "variant", "condition", "n", "acc", "macro_f1", "neutral_recall",
            "neutral_share", "mean_mass", "cal_macro_f1", "cal_neutral_recall"]
    print("=== DEV METRICS (raw unless the column starts with cal_) ===")
    print(out[cols].round(3).to_string(index=False))
    if "agree_scored_vs_generated" in out:
        print("\n=== scored label vs generated answer ===")
        print(out[["backend", "variant", "condition", "gen_parseable",
                   "agree_scored_vs_generated"]].round(3).to_string(index=False))

    # ---- gate, per backend ----
    probe_rate = {}
    for p in args.probes:
        pr = pd.DataFrame(read_jsonl(p))
        if len(pr):
            probe_rate[pr.backend.iloc[0]] = pr.read_correctly.sum() / max(1, pr.expected.sum())

    print("\n=== GATE ===")
    for be, g in out.groupby("backend"):
        pv = g.pivot_table(index="variant", columns="condition", values="macro_f1")
        use = [c for c in ("plain", "sov") if c in pv.columns]
        best = pv[use].mean(axis=1).idxmax()
        plain_f1 = float(pv.loc[best, "plain"]) if "plain" in pv.columns else float("nan")
        print(f"\n{be}:  better prompt variant (mean macro-F1 over {use}): {best}")
        print(f"  [{'PASS' if plain_f1 >= args.min_f1 else 'FAIL'}] plain macro-F1 = {plain_f1:.3f} "
              f"(needs >= {args.min_f1})")
        if be in probe_rate:
            ok = probe_rate[be] >= args.min_tags
            print(f"  [{'PASS' if ok else 'FAIL'}] tag reading = {probe_rate[be]:.2f} "
                  f"(needs >= {args.min_tags})")
        else:
            print("  [----] tag reading: no probe file given")
        if "agree_scored_vs_generated" in g:
            ag = float(g[g.variant == best].agree_scored_vs_generated.mean())
            print(f"  [{'PASS' if ag >= args.min_agree else 'FAIL'}] scored vs generated agreement = "
                  f"{ag:.3f} (needs >= {args.min_agree})")
        else:
            print("  [----] agreement: run step 6 with --also_generate")
        nr = g[g.variant == best]
        print(f"  neutral recall raw {nr.neutral_recall.mean():.2f} -> calibrated "
              f"{nr.cal_neutral_recall.mean():.2f}")
    print("\nreport files:", args.out_dir)


if __name__ == "__main__":
    main()
