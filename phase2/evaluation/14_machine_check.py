#!/usr/bin/env python3
"""
Machine check -- did your GPU and your teammate's GPU produce comparable results?   CPU only, 1 minute.

The multi-face images were split between the two machines by chunk parity (images were shuffled with a fixed
seed, mixed-emotion images first), so the two halves are two RANDOM samples of the same population and should
give the same accuracy per arm up to sampling noise.  A machine problem (different library version, different
numerics, a wrong prompt) would show up as a SYSTEMATIC difference in accuracy or in the predicted-class mix.

  py -3.13 14_machine_check.py --even results_llava15_jobs_multi_even.jsonl --odd results_llava15_jobs_multi_odd.jsonl --multi_faces multiface\\multi_faces.csv

For each arm (condition | prompt type) it prints, for the two halves: n, accuracy (all faces), accuracy on the expressive
emotions, the largest gap in the share of predictions per emotion class, seconds per job, and the difference in accuracy
(even - odd) with a 95% bootstrap interval (images resampled separately in each half; the halves share no image).

HOW TO READ IT
  * OK       : the accuracy interval contains 0 and the largest predicted-share gap is below 4 points.
  * LOOK     : the interval excludes 0, or the gap is >= 4 points.  One LOOK among ~6 arms can be chance (the
               interval is 95%); several LOOKs in the same direction mean a systematic machine difference.
  Rough noise levels: ~1400 faces per half -> accuracy differences up to about +-3.5 points are normal;
  on the expressive emotions (~150 faces per half) up to about +-10 points are normal.
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

EMO = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
EXPR = {"Angry", "Disgust", "Fear", "Sad", "Surprise"}


def read(path):
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


def per_image(df, arm, mf):
    d = df[df.arm == arm].merge(mf[["row_id", "image_id", "expression_name"]], on="row_id")
    d["truth"] = d.expression_name.str.capitalize()
    d["ok"] = (d.pred == d.truth).astype(float)
    d["expr"] = d.truth.isin(EXPR).astype(float)
    d["ok_expr"] = d.ok * d.expr
    g = d.groupby("image_id").agg(n=("ok", "size"), k=("ok", "sum"), ne=("expr", "sum"), ke=("ok_expr", "sum"))
    return d, g


def boot_diff(ga, gb, num, den, B, rng):
    out = []
    A, Bm = ga[[num, den]].to_numpy(float), gb[[num, den]].to_numpy(float)
    for _ in range(B):
        a = A[rng.integers(0, len(A), len(A))].sum(0)
        b = Bm[rng.integers(0, len(Bm), len(Bm))].sum(0)
        if a[1] > 0 and b[1] > 0:
            out.append(a[0] / a[1] - b[0] / b[1])
    return np.percentile(out, [2.5, 97.5]) if out else (np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--even", required=True)
    ap.add_argument("--odd", required=True)
    ap.add_argument("--multi_faces", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    mf = pd.read_csv(args.multi_faces)
    halves = {}
    for name, p in (("even", args.even), ("odd", args.odd)):
        r = pd.DataFrame(read(p))
        r = r[r.error.isna()].drop_duplicates("job_id", keep="last")
        r["arm"] = r.condition + "|" + r.query_type
        halves[name] = r
    shared = set(halves["even"].image_key) & set(halves["odd"].image_key)
    print(f"images only in one half: even {halves['even'].image_key.nunique()}, odd {halves['odd'].image_key.nunique()}; "
          f"shared images: {len(shared)} (must be 0)")
    for name, r in halves.items():
        faces = mf[mf.row_id.isin(r.row_id)]
        print(f"{name}: faces {faces.row_id.nunique()}, mixed-image share {faces.mixed.mean():.3f}, "
              f"emotion mix: " + str((faces.expression_name.value_counts(normalize=True).round(3)).to_dict()))
    rng = np.random.default_rng(args.seed)
    rows = []
    for arm in sorted(set(halves["even"].arm) & set(halves["odd"].arm)):
        de, ge = per_image(halves["even"], arm, mf)
        do, go = per_image(halves["odd"], arm, mf)
        acc_e, acc_o = ge.k.sum() / ge.n.sum(), go.k.sum() / go.n.sum()
        lo, hi = boot_diff(ge, go, "k", "n", args.boot, rng)
        ex_e = ge["ke"].sum() / max(ge["ne"].sum(), 1)
        ex_o = go["ke"].sum() / max(go["ne"].sum(), 1)
        elo, ehi = boot_diff(ge, go, "ke", "ne", args.boot, rng)
        share = lambda d: d.pred.value_counts(normalize=True).reindex(EMO).fillna(0)
        gap = float((share(de) - share(do)).abs().max())
        flag = "OK" if (lo <= 0 <= hi and gap < 0.04) else "LOOK"
        rows.append(dict(arm=arm, n_even=len(de), n_odd=len(do), acc_even=acc_e, acc_odd=acc_o,
                         diff=acc_e - acc_o, ci_lo=lo, ci_hi=hi, expr_even=ex_e, expr_odd=ex_o,
                         expr_diff=ex_e - ex_o, expr_ci_lo=elo, expr_ci_hi=ehi, max_share_gap=gap,
                         sec_even=de.seconds.mean(), sec_odd=do.seconds.mean(), flag=flag))
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n" + t.round(3).to_string(index=False))
    n_look = int((t.flag == "LOOK").sum())
    print(f"\n{len(t) - n_look} of {len(t)} arms OK, {n_look} LOOK.")
    if n_look >= 2 and np.sign(t[t.flag == 'LOOK']['diff']).abs().sum() and \
            (np.sign(t[t.flag == 'LOOK']['diff']).nunique() == 1):
        print("Several arms differ in the SAME direction: possible systematic machine difference.")
    t.to_csv("machine_check.csv", index=False)
    print("saved machine_check.csv")


if __name__ == "__main__":
    sys.exit(main())
