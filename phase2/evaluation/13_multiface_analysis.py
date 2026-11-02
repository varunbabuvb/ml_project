#!/usr/bin/env python3
"""
Step 13 -- analysis of the MULTI-FACE experiment.   CPU only; works on a partial run (uses the faces that have results).

  python 13_multiface_analysis.py --results results_llava15_jobs_multi_even.jsonl results_llava15_jobs_multi_odd.jsonl \
        --multi_faces multi_faces.csv --out_dir analysis_multi

ARMS (condition | prompt type)
  plain|positional  box|positional  box_number|numbered  sov|numbered            (all targets: the 4 designed conditions)
  box_number|positional  sov|positional                                          (extra runs, mixed-emotion images only)
  "positional" = 'the Nth face from the left' (the model has to count);  "numbered" = 'the person labeled "k"'.

METRICS (per subset of faces; every difference is PAIRED and has a 95% CLUSTER bootstrap over images, 2000 resamples)
  acc_all     pooled accuracy (the set is ~64% Happy, so this mostly measures Happy)
  acc_expr    accuracy on the "expressive" truths: Angry, Disgust, Fear, Sad, Surprise
  acc_happy   accuracy on Happy faces
  bal         balanced accuracy over the classes that have >= --min_class faces in the subset
  xref        MIS-REFERENCE rate: among faces whose image also contains a face with a DIFFERENT label, the
              share of answers that equal one of those other faces' labels (the model described the wrong face,
              or at least gave a neighbour's emotion).  Only defined on mixed-emotion images.
SUBSETS   all | mixed (the target faces in the image show >= 2 different emotions: here pointing at the right face
          matters) | not_mixed (all labels equal: a wrong reference still gives the right answer).
No calibration is applied here: the emotion mix is far from balanced, so "predict every class equally often" does
not hold.  Group results are limited to accuracy-type metrics and flagged when a cell has < --min_cell faces.
Outputs in --out_dir: tab_*.csv, fig_*.png, summary.md
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

EMO = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
EIDX = {e: i for i, e in enumerate(EMO)}
EXPR = [EIDX[e] for e in ("Angry", "Disgust", "Fear", "Sad", "Surprise")]
HAPPY = EIDX["Happy"]
BASE_ARMS = ["plain|positional", "box|positional", "box_number|numbered", "sov|numbered"]
EXTRA_ARMS = ["box_number|positional", "sov|positional"]
ARM_LABEL = {"plain|positional": "Plain (position)", "box|positional": "Box (position)",
             "box_number|numbered": "Box+Number (number)", "sov|numbered": "SoV (number)",
             "box_number|positional": "Box+Number (position)", "sov|positional": "SoV (position)"}
# (a, b, what it isolates)
CONTRASTS = [
    ("sov|numbered", "plain|positional", "SoV as designed vs Plain"),
    ("box_number|numbered", "plain|positional", "Box+Number as designed vs Plain"),
    ("box|positional", "plain|positional", "Boxes alone vs Plain (same prompt)"),
    ("box_number|numbered", "box|positional", "Numbers + numbered prompt vs boxes"),
    ("sov|numbered", "box_number|numbered", "Landmarks (same overlay type, same prompt)"),
    ("box_number|numbered", "box_number|positional", "Using the number vs counting (same image)"),
    ("sov|numbered", "sov|positional", "Using the number vs counting (SoV image)"),
    ("box_number|positional", "box|positional", "Number tags themselves (same prompt)"),
    ("sov|positional", "plain|positional", "SoV marks only (same prompt) vs Plain"),
]


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


def ci(arr):
    b = arr[1:]
    b = b[np.isfinite(b)]
    if not len(b):
        return np.nan, np.nan, np.nan
    lo, hi = np.percentile(b, [2.5, 97.5])
    p = 2 * min((b <= 0).mean(), (b >= 0).mean())
    return float(lo), float(hi), float(min(1.0, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--multi_faces", required=True)
    ap.add_argument("--out_dir", default="analysis_multi")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_class", type=int, default=30)
    ap.add_argument("--min_cell", type=int, default=30)
    ap.add_argument("--min_level", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    mf = pd.read_csv(args.multi_faces)
    rec = []
    for p in args.results:
        rec += read_jsonl(p)
    r = pd.DataFrame(rec)
    r = r[r.error.isna()].drop_duplicates("job_id", keep="last")
    r["arm"] = r.condition + "|" + r.query_type
    have_rows = set(r.row_id)
    d = mf[mf.row_id.isin(have_rows)].sort_values(["image_id", "face_id_in_image"]).reset_index(drop=True)
    n = len(d)
    pos = {rid: i for i, rid in enumerate(d.row_id)}
    truth = d.expression_name.str.capitalize().map(EIDX).to_numpy()
    arms = [a for a in BASE_ARMS + EXTRA_ARMS if (r.arm == a).any()]
    PRED, HAVE = {}, {}
    for a in arms:
        g = r[r.arm == a]
        p = np.full(n, -1)
        for rid, pr in zip(g.row_id, g.pred):
            if rid in pos:
                p[pos[rid]] = EIDX[pr]
        PRED[a], HAVE[a] = p, p >= 0
    nff = np.full(n, np.nan)
    for rid, k in zip(r.row_id, r.n_faces_final):
        if rid in pos:
            nff[pos[rid]] = k
    mixed = d.mixed.to_numpy().astype(bool)

    # ---- mis-reference indicator per arm (mixed images only) ----
    others = {}
    for img, g in mf.groupby("image_id"):
        labs = g.expression_name.str.capitalize().tolist()
        rids = g.row_id.tolist()
        for rid, lab in zip(rids, labs):
            others[rid] = {EIDX[x] for x in labs if x != lab}
    xelig = np.array([len(others.get(rid, ())) > 0 for rid in d.row_id])
    XREF = {a: np.array([(PRED[a][i] in others.get(rid, ())) if PRED[a][i] >= 0 else False
                         for i, rid in enumerate(d.row_id)], dtype=float) for a in arms}

    # ---- bootstrap weights (cluster = image) ----
    rng = np.random.default_rng(args.seed)
    u, inv = np.unique(d.image_id.values, return_inverse=True)
    counts = np.empty((args.boot, len(u)), dtype=np.float32)
    for b in range(args.boot):
        counts[b] = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))
    W = np.vstack([np.ones((1, n), dtype=np.float32), counts[:, inv]])
    ALL = np.ones(n, bool)

    def metric(name, arm, mask):
        m = mask & HAVE[arm]
        ok = (PRED[arm] == truth).astype(np.float32)
        if name == "xref":
            m = m & xelig
            w = W * m[None, :]
            with np.errstate(invalid="ignore", divide="ignore"):
                return (w * XREF[arm][None, :]).sum(1) / w.sum(1)
        if name == "acc_all":
            sel = m
        elif name == "acc_expr":
            sel = m & np.isin(truth, EXPR)
        elif name == "acc_happy":
            sel = m & (truth == HAPPY)
        elif name == "bal":
            cls = [c for c in range(7) if (m & (truth == c)).sum() >= args.min_class]
            if not cls:
                return np.full(W.shape[0], np.nan)
            rec_ = []
            for c in cls:
                w = W * (m & (truth == c))[None, :]
                with np.errstate(invalid="ignore", divide="ignore"):
                    rec_.append((w * ok[None, :]).sum(1) / w.sum(1))
            return np.nanmean(np.vstack(rec_), axis=0)
        else:
            raise ValueError(name)
        w = W * sel[None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            return (w * ok[None, :]).sum(1) / w.sum(1)

    def count(name, arm, mask):
        m = mask & HAVE[arm]
        if name == "xref":
            return int((m & xelig).sum())
        if name == "acc_expr":
            return int((m & np.isin(truth, EXPR)).sum())
        if name == "acc_happy":
            return int((m & (truth == HAPPY)).sum())
        return int(m.sum())

    subsets = {"all": ALL, "mixed": mixed, "not_mixed": ~mixed}
    metrics_all = ["acc_all", "acc_expr", "acc_happy", "bal", "xref"]

    # ===== TABLE 1: per arm =====
    rows = []
    for sname, smask in subsets.items():
        for a in arms:
            for mt in metrics_all:
                if mt == "xref" and sname != "mixed":
                    continue
                v = metric(mt, a, smask)
                lo, hi, _ = ci(v)
                rows.append(dict(subset=sname, arm=a, metric=mt, n=count(mt, a, smask), value=v[0], ci_lo=lo, ci_hi=hi))
    t1 = pd.DataFrame(rows)
    t1.to_csv(os.path.join(args.out_dir, "tab_arms.csv"), index=False)

    # ===== TABLE 2: paired contrasts =====
    rows = []
    for sname, smask in subsets.items():
        for a, b, what in CONTRASTS:
            if a not in HAVE or b not in HAVE:
                continue
            m = smask & HAVE[a] & HAVE[b]
            for mt in metrics_all:
                if mt == "xref" and sname != "mixed":
                    continue
                dv = metric(mt, a, m) - metric(mt, b, m)
                lo, hi, p = ci(dv)
                rows.append(dict(subset=sname, contrast=f"{a} - {b}", what=what, metric=mt,
                                 n=count(mt, a, m & HAVE[b]), delta=dv[0], ci_lo=lo, ci_hi=hi, p_boot=p,
                                 excludes_zero=bool(lo > 0 or hi < 0)))
    t2 = pd.DataFrame(rows)
    t2.to_csv(os.path.join(args.out_dir, "tab_contrasts.csv"), index=False)

    # ===== TABLE 3/4: group effects (accuracy-type metrics, same-prompt-agnostic key contrasts) =====
    tier = pd.qcut(d.face_px_view, 3, labels=["small", "medium", "large"]).astype(str).to_numpy()
    ref_clear = np.where(nff == d.n_targets.to_numpy(), "detections = targets", "extra/missing detections")
    gdefs = {
        "race_group": (d.race_group.to_numpy(), ["Caucasian-like", "Asian", "Black"], ALL),
        "gender": (d.gender15.to_numpy(), ["Male", "Female"], d.gender_stable.to_numpy().astype(bool)),
        "age_band": (d.age_band15.to_numpy(), ["0-19", "20-29", "30-39", "40-59", "60+"],
                     d.age_band_stable.to_numpy().astype(bool)),
        "face_size": (tier, ["small", "medium", "large"], ALL),
        "image_type": (np.where(mixed, "mixed", "not_mixed"), ["not_mixed", "mixed"], ALL),
        "reference_clear": (ref_clear, ["detections = targets", "extra/missing detections"], ALL),
        "n_targets": (np.where(d.n_targets.to_numpy() == 2, "2 faces", "3+ faces"), ["2 faces", "3+ faces"], ALL),
    }
    gcontrasts = [c for c in CONTRASTS[:5] if c[0] in HAVE and c[1] in HAVE]
    rows, rows_d = [], []
    for gname, (vals, levels, gm) in gdefs.items():
        sizes = {lv: int(((vals == lv) & gm).sum()) for lv in levels}
        ref = max(sizes, key=sizes.get)
        for mt in ("acc_all", "acc_happy", "acc_expr"):
            for a, b, what in gcontrasts:
                store = {}
                for lv in levels:
                    mk = (vals == lv) & gm & HAVE[a] & HAVE[b]
                    nn = count(mt, a, mk)
                    if mk.sum() == 0 or nn == 0:
                        continue
                    dv = metric(mt, a, mk) - metric(mt, b, mk)
                    lo, hi, p = ci(dv)
                    store[lv] = dv
                    rows.append(dict(group=gname, level=lv, metric=mt, contrast=f"{a} - {b}", n=nn,
                                     small_cell=nn < args.min_cell, delta=dv[0], ci_lo=lo, ci_hi=hi, p_boot=p))
                for lv, dv in store.items():
                    if lv == ref or ref not in store:
                        continue
                    dd = dv - store[ref]
                    lo, hi, p = ci(dd)
                    rows_d.append(dict(group=gname, level=lv, reference=ref, metric=mt,
                                       contrast=f"{a} - {b}", diff=dd[0], ci_lo=lo, ci_hi=hi, p_boot=p,
                                       excludes_zero=bool(lo > 0 or hi < 0)))
    t3 = pd.DataFrame(rows)
    t3.to_csv(os.path.join(args.out_dir, "tab_group_effects.csv"), index=False)
    t4 = pd.DataFrame(rows_d)
    t4.to_csv(os.path.join(args.out_dir, "tab_group_differences.csv"), index=False)

    # ===== TABLE 5: cell sizes (group x emotion) =====
    cells = d.pivot_table(index="race_group", columns="expression_name", values="row_id", aggfunc="count")
    cells = cells.reindex(columns=[e.lower() for e in EMO]).fillna(0).astype(int)
    cells.to_csv(os.path.join(args.out_dir, "tab_cell_sizes.csv"))
    small = [(g_, e, int(v)) for g_ in cells.index for e, v in cells.loc[g_].items() if v < args.min_cell]

    # ===== FIGURES =====
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        plt = None
    if plt is not None:
        col = {"plain|positional": "#6c757d", "box|positional": "#4c78a8", "box_number|numbered": "#f58518",
               "sov|numbered": "#e45756", "box_number|positional": "#f2b880", "sov|positional": "#f0a3a2"}
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
        for ax, (sname, mt, ttl, arm_list) in zip(axes, [
                ("all", "acc_all", "All faces: accuracy (4 designed conditions)", [a for a in arms if a in BASE_ARMS]),
                ("mixed", "acc_all", "Mixed-emotion images: accuracy (all arms)", arms),
                ("all", "acc_expr", "Expressive emotions only (4 designed conditions)", [a for a in arms if a in BASE_ARMS])]):
            sub = t1[(t1.subset == sname) & (t1.metric == mt)].set_index("arm").reindex(arm_list)
            sub = sub.dropna(subset=["value"])
            x = np.arange(len(sub))
            ax.bar(x, sub.value, color=[col[a] for a in sub.index], edgecolor="black", linewidth=0.5)
            ax.errorbar(x, sub.value, yerr=[sub.value - sub.ci_lo, sub.ci_hi - sub.value], fmt="none",
                        ecolor="black", capsize=3)
            ax.set_xticks(x)
            ax.set_xticklabels([ARM_LABEL[a].replace(" (", "\n(") for a in sub.index], fontsize=7, rotation=0)
            ax.set_title(ttl, fontsize=10)
            ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "fig_arms.png"), dpi=160)
        plt.close(fig)

        sub = t2[(t2.subset == "mixed") & (t2.metric == "acc_all")].reset_index(drop=True)
        if len(sub):
            y = np.arange(len(sub))[::-1]
            fig, ax = plt.subplots(figsize=(8, 0.42 * len(sub) + 1.4))
            ax.errorbar(sub.delta * 100, y, xerr=[(sub.delta - sub.ci_lo) * 100, (sub.ci_hi - sub.delta) * 100],
                        fmt="o", color="#e45756", ecolor="black", capsize=3)
            ax.axvline(0, color="gray", lw=1)
            ax.set_yticks(y)
            ax.set_yticklabels(sub.what, fontsize=8)
            ax.set_xlabel("difference in accuracy, points (95% CI), mixed-emotion images")
            fig.tight_layout()
            fig.savefig(os.path.join(args.out_dir, "fig_contrasts_mixed.png"), dpi=160)
            plt.close(fig)

    # ===== SUMMARY =====
    def f(r_):
        return f"{r_.delta * 100:+.1f} [{r_.ci_lo * 100:+.1f}, {r_.ci_hi * 100:+.1f}]"

    L = ["# Multi-face analysis summary (auto-generated)", "",
         f"Target faces with results: {n} of {len(mf)} | images: {len(u)} (mixed: {int(d[d.mixed].image_id.nunique())}) | "
         f"chunks present: {sorted(d.chunk_id.unique())[:6]}... ({d.chunk_id.nunique()} of {mf.chunk_id.nunique()}) | "
         f"bootstrap {args.boot} resamples of images",
         "", "## Accuracy per arm (95% CI)"]
    for sname, mt in (("all", "acc_all"), ("mixed", "acc_all"), ("all", "acc_expr"), ("mixed", "xref")):
        s = t1[(t1.subset == sname) & (t1.metric == mt)]
        L.append(f"- **{sname} / {mt}**: " + "; ".join(
            f"{ARM_LABEL[r_.arm]} {r_.value:.3f} [{r_.ci_lo:.3f}, {r_.ci_hi:.3f}] (n={r_.n})" for r_ in s.itertuples()))
    L += ["", "## Paired differences, points (95% CI)"]
    for sname, mt in (("all", "acc_all"), ("mixed", "acc_all"), ("mixed", "xref"), ("all", "acc_expr"), ("not_mixed", "acc_all")):
        s = t2[(t2.subset == sname) & (t2.metric == mt) & t2.delta.notna()]
        if len(s):
            L.append(f"- **{sname} / {mt}**")
            L += [f"    - {r_.what} ({r_.contrast}): {f(r_)}{'  *' if r_.excludes_zero else ''}" for r_ in s.itertuples()]
    L += ["", "## Does the SoV-vs-Plain effect differ between groups? (acc_all, 'SoV as designed vs Plain'; * = CI excludes 0)"]
    s = t4[(t4.metric == "acc_all") & (t4.contrast == "sov|numbered - plain|positional")]
    for r_ in s.itertuples():
        L.append(f"- {r_.group}: {r_.level} vs {r_.reference}: {r_.diff * 100:+.1f} [{r_.ci_lo * 100:+.1f}, {r_.ci_hi * 100:+.1f}]"
                 f"{'  *' if r_.excludes_zero else ''}")
    L += ["", "## Warnings",
          "- Emotions are very unbalanced (mostly Happy and Neutral; Neutral is never predicted). acc_all mostly measures Happy; "
          "use acc_expr and xref for the informative faces.",
          "- Group labels are FairFace ESTIMATES. Beyond Happy, the group cells are too small for group claims; "
          f"cells below {args.min_cell} faces: " + (", ".join(f"{g_}/{e}={v}" for g_, e, v in small) if small else "none"),
          "- Positional prompts make plain/box rely on the model counting faces; 'reference_clear' rows compare images where "
          "the number of detected faces equals the number of target faces with the others.",
          "- Many contrasts and subsets are reported; treat single borderline intervals as exploratory."]
    with open(os.path.join(args.out_dir, "summary.md"), "w", encoding="utf-8") as f_:
        f_.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("\nFiles written to", args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
