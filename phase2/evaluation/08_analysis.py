#!/usr/bin/env python3
"""
Step 8 -- analysis of the core-set results.   CPU only; runs on your laptop or in Colab.

Usage
  python 08_analysis.py --core_results results_llava15_jobs_core.jsonl --core_set core_set.csv \
        --dev_results results_llava15_jobs_dev.jsonl --out_dir analysis
  # optional: add the re-run with scaled overlays (any subset of conditions), label "scaled":
  python 08_analysis.py ... --extra results_llava15_jobs_core_scaled.jsonl:scaled

What it computes (all rules are fixed here, before looking at group results)
  * Metrics: accuracy, BALANCED ACCURACY (mean recall over the emotion classes; primary) and
    macro-F1 (overall only).  Class sets:  all7 | no_neutral (the model never says Neutral) |
    groups6 (all but Fear: Fear cells are too small for per-group results) | groups5 (also no Neutral).
    Overall results use all7 and no_neutral; per-group results use groups6 and groups5.
  * Three scoring variants: raw (argmax of the 7 label scores); self (each condition's scores divided
    by that condition's own average prediction over the core set; uses NO labels and is applied
    identically in every condition); dev (same, but the average comes from the dev set; only for the
    conditions that were run on the dev set).
  * Every difference is PAIRED (same faces, same context image) with a 95% CLUSTER BOOTSTRAP over
    images (2000 resamples), because faces from one photo are not independent.
  * Per-group effects: for each group level the paired difference between two conditions, and the
    DIFFERENCE BETWEEN GROUPS of that difference (relative to the largest level), each with its CI.
    Groups: race group (3), gender (stable labels only), age band (stable labels only), face-size
    tertile, single-face vs multi-face context crop.
  * A group x emotion cell is flagged when it has fewer than --min_cell faces.
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
CLASS_SETS = {"all7": EMO,
              "no_neutral": [e for e in EMO if e != "Neutral"],
              "groups6": [e for e in EMO if e != "Fear"],
              "groups5": [e for e in EMO if e not in ("Fear", "Neutral")]}
BASE = ["plain", "box", "box_number", "sov"]
PRETTY = {"plain": "Plain", "box": "Box", "box_number": "Box+Number", "sov": "SoV"}


# ------------------------------------------------------------------ loading
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


def load_results(path, label=None):
    df = pd.DataFrame(read_jsonl(path))
    df = df[df.error.isna()].copy()
    if label:
        df["condition"] = df.condition + "@" + label
    df = df.drop_duplicates(["job_id"], keep="last")
    return df


# ------------------------------------------------------------------ metrics
def metrics(pred, truth, W, mask, classes):
    """pred, truth: int arrays (n,). W: (B+1, n) weights, row 0 = ones.  Returns dict of arrays."""
    cls = np.array([EIDX[c] for c in classes])
    inc = np.isin(truth, cls)
    w = W * (mask & inc)[None, :]
    T = np.eye(7)[truth]
    P = np.eye(7)[np.clip(pred, 0, 6)]
    tc = w @ T[:, cls]
    tp = w @ (T * P)[:, cls]
    pc = w @ P[:, cls]
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(tc > 0, tp / tc, np.nan)
        prec = np.where(pc > 0, tp / pc, 0.0)
        f1 = np.where(tc > 0, np.where(prec + recall > 0, 2 * prec * recall / (prec + recall), 0.0), np.nan)
        acc = (w * (pred == truth)[None, :]).sum(1) / w.sum(1)
    return {"acc": acc, "bal": np.nanmean(recall, axis=1), "f1": np.nanmean(f1, axis=1),
            "recall": recall, "n": int((mask & inc).sum())}


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
    ap.add_argument("--core_results", required=True)
    ap.add_argument("--core_set", required=True)
    ap.add_argument("--dev_results", default=None)
    ap.add_argument("--extra", action="append", default=[], help="file.jsonl:label (repeatable)")
    ap.add_argument("--out_dir", default="analysis")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_cell", type=int, default=30)
    ap.add_argument("--min_level", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    core = pd.read_csv(args.core_set).sort_values("row_id").reset_index(drop=True)
    n = len(core)
    pos = {r: i for i, r in enumerate(core.row_id)}
    truth = core.expression_name.str.capitalize().map(EIDX).to_numpy()

    # ---------------- results -> arrays aligned with core ----------------
    frames = [load_results(args.core_results)]
    labels = []
    for e in args.extra:
        path, lab = e.rsplit(":", 1)
        frames.append(load_results(path, lab))
        labels.append(lab)
    allr = pd.concat(frames, ignore_index=True)
    cen = allr[allr.query_type == "center"]
    LP, HAVE = {}, {}
    for cond, g in cen.groupby("condition"):
        a = np.full((n, 7), np.nan)
        for r, d in zip(g.row_id, g.label_logprobs):
            if r in pos:
                a[pos[r]] = [d[e] for e in EMO]
        LP[cond] = a
        HAVE[cond] = ~np.isnan(a).any(1)
    conds = [c for c in BASE if c in LP] + [c for c in LP if c not in BASE]
    nfaces_final = np.full(n, np.nan)
    for r, k in zip(cen.row_id, cen.n_faces_final):
        if r in pos:
            nfaces_final[pos[r]] = k

    def softmax(x):
        x = x - x.max(1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(1, keepdims=True)

    # ---------------- scoring variants ----------------
    PRED, VARS = {}, {}
    for c in conds:
        L = np.nan_to_num(LP[c], nan=0.0)
        PRED[("raw", c)] = L.argmax(1)
        lpb = np.log(softmax(LP[c][HAVE[c]]).mean(0))
        PRED[("self", c)] = (L - lpb).argmax(1)
    if args.dev_results:
        dev = load_results(args.dev_results)
        dev = dev[(dev.query_type == "center")]
        if "prompt_set" in dev:
            dev = dev[dev.prompt_set == "A"]
        for c, g in dev.groupby("condition"):
            if c in LP:
                lps = np.array([[d[e] for e in EMO] for d in g.label_logprobs])
                lpb = np.log(softmax(lps).mean(0))
                PRED[("dev", c)] = (np.nan_to_num(LP[c], nan=0.0) - lpb).argmax(1)
    variants = ["raw", "self"] + (["dev"] if any(v == "dev" for v, _ in PRED) else [])

    # ---------------- bootstrap weights (cluster = image) ----------------
    rng = np.random.default_rng(args.seed)
    u, inv = np.unique(core.image_name.values, return_inverse=True)
    counts = np.empty((args.boot, len(u)), dtype=np.float32)
    for b in range(args.boot):
        counts[b] = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))
    W = np.vstack([np.ones((1, n), dtype=np.float32), counts[:, inv]])

    ALL = np.ones(n, bool)

    def M(variant, c, classes, mask):
        return metrics(PRED[(variant, c)], truth, W, mask & HAVE[c], classes)

    # ---------------- contrasts ----------------
    contrasts = [("sov", "plain"), ("box", "plain"), ("box_number", "plain"),
                 ("box_number", "box"), ("sov", "box_number")]
    for lab in labels:
        contrasts += [(f"box@{lab}", "plain"), (f"box_number@{lab}", "plain"),
                      (f"sov@{lab}", "plain"), (f"sov@{lab}", "sov"),
                      (f"sov@{lab}", f"box_number@{lab}")]
    contrasts = [(a, b) for a, b in contrasts if a in LP and b in LP]

    def delta(a, b, variant, classes, mask, metric="bal"):
        if (variant, a) not in PRED or (variant, b) not in PRED:
            return None
        m = mask & HAVE[a] & HAVE[b]
        return (M(variant, a, classes, m)[metric] - M(variant, b, classes, m)[metric]), int(m.sum())

    # ================= TABLE 1: overall per condition =================
    rows = []
    for variant in variants:
        for cs in ("all7", "no_neutral"):
            for c in conds:
                if (variant, c) not in PRED:
                    continue
                m = M(variant, c, CLASS_SETS[cs], ALL)
                for met in ("acc", "bal", "f1"):
                    lo, hi, _ = ci(m[met])
                    rows.append(dict(variant=variant, classes=cs, condition=c, metric=met,
                                     n=int((ALL & HAVE[c]).sum()), value=m[met][0], ci_lo=lo, ci_hi=hi))
    t1 = pd.DataFrame(rows)
    t1.to_csv(os.path.join(args.out_dir, "tab_overall.csv"), index=False)

    # ================= TABLE 2: paired overall differences =================
    rows = []
    for variant in variants:
        for cs in ("all7", "no_neutral"):
            for a, b in contrasts:
                for met in ("bal", "f1", "acc"):
                    r = delta(a, b, variant, CLASS_SETS[cs], ALL, met)
                    if r is None:
                        continue
                    d, nn = r
                    lo, hi, p = ci(d)
                    rows.append(dict(variant=variant, classes=cs, metric=met, contrast=f"{a} - {b}",
                                     n=nn, delta=d[0], ci_lo=lo, ci_hi=hi, p_boot=p,
                                     excludes_zero=bool(lo > 0 or hi < 0)))
    t2 = pd.DataFrame(rows)
    t2.to_csv(os.path.join(args.out_dir, "tab_paired_overall.csv"), index=False)

    # ================= TABLE 3/4: per-group effects and between-group differences =================
    tier = pd.qcut(core.box_min_px, 3, labels=["small", "medium", "large"]).astype(str)
    crop_type = np.where(nfaces_final == 1, "single-face", "multi-face")
    gdefs = {
        "overall": (np.array(["all faces"] * n), ["all faces"], ALL),
        "race_group": (core.race_group.values, ["Caucasian-like", "Asian", "Black"], ALL),
        "gender": (core.gender15.values, ["Male", "Female"], core.gender_stable.values.astype(bool)),
        "age_band": (core.age_band15.values, ["0-19", "20-29", "30-39", "40-59", "60+"],
                     core.age_band_stable.values.astype(bool)),
        "face_size": (tier.values, ["small", "medium", "large"], ALL),
        "crop_type": (crop_type, ["single-face", "multi-face"], ALL),
    }
    rows, rows_d = [], []
    for gname, (vals, levels, base_mask) in gdefs.items():
        sizes = {lv: int(((vals == lv) & base_mask).sum()) for lv in levels}
        ref = max(sizes, key=sizes.get)
        for variant in variants:
            for cs in ("groups6", "groups5"):
                for a, b in contrasts:
                    store = {}
                    for lv in levels:
                        mk = (vals == lv) & base_mask
                        r = delta(a, b, variant, CLASS_SETS[cs], mk)
                        if r is None or mk.sum() == 0:
                            continue
                        d, nn = r
                        lo, hi, p = ci(d)
                        store[lv] = d
                        rows.append(dict(group=gname, level=lv, variant=variant, classes=cs,
                                         contrast=f"{a} - {b}", n=nn, small_level=nn < args.min_level,
                                         delta=d[0], ci_lo=lo, ci_hi=hi, p_boot=p))
                    for lv in levels:
                        if lv == ref or lv not in store or ref not in store:
                            continue
                        dd = store[lv] - store[ref]
                        lo, hi, p = ci(dd)
                        rows_d.append(dict(group=gname, level=lv, reference=ref, variant=variant,
                                           classes=cs, contrast=f"{a} - {b}",
                                           delta_level=store[lv][0], delta_ref=store[ref][0],
                                           diff=dd[0], ci_lo=lo, ci_hi=hi, p_boot=p,
                                           excludes_zero=bool(lo > 0 or hi < 0)))
    t3 = pd.DataFrame(rows)
    t3.to_csv(os.path.join(args.out_dir, "tab_group_effects.csv"), index=False)
    t4 = pd.DataFrame(rows_d)
    t4.to_csv(os.path.join(args.out_dir, "tab_group_differences.csv"), index=False)

    # ================= TABLE 5: recall per emotion =================
    rows = []
    for c in conds:
        m = M("raw", c, EMO, ALL)
        for j, e in enumerate(EMO):
            lo, hi, _ = ci(m["recall"][:, j])
            rows.append(dict(condition=c, emotion=e, n_faces=int((truth == EIDX[e]).sum()),
                             recall=m["recall"][0, j], ci_lo=lo, ci_hi=hi,
                             pred_share=float((PRED[("raw", c)][HAVE[c]] == EIDX[e]).mean())))
    t5 = pd.DataFrame(rows)
    t5.to_csv(os.path.join(args.out_dir, "tab_emotion_recall.csv"), index=False)
    rows = []
    for a, b in contrasts:
        ma, mb = M("raw", a, EMO, ALL & HAVE[a] & HAVE[b]), M("raw", b, EMO, ALL & HAVE[a] & HAVE[b])
        for j, e in enumerate(EMO):
            d = ma["recall"][:, j] - mb["recall"][:, j]
            lo, hi, p = ci(d)
            rows.append(dict(contrast=f"{a} - {b}", emotion=e, delta_recall=d[0], ci_lo=lo, ci_hi=hi))
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "tab_emotion_delta.csv"), index=False)

    # ================= TABLE 6: cell sizes =================
    cells = core.pivot_table(index="race_group", columns="expression_name", values="row_id", aggfunc="count")
    cells = cells.reindex(columns=[e.lower() for e in EMO]).fillna(0).astype(int)
    cells.to_csv(os.path.join(args.out_dir, "tab_cell_sizes.csv"))
    small = [(g, e, int(v)) for g in cells.index for e, v in cells.loc[g].items() if v < args.min_cell]

    # ================= TABLE 7: numbered vs centre prompt (multi-face crops) =================
    num = allr[allr.query_type == "numbered"]
    rows = []
    for c in sorted(num.condition.unique()):
        g = num[num.condition == c]
        pn = np.full(n, -1)
        for r, p in zip(g.row_id, g.pred):
            if r in pos:
                pn[pos[r]] = EIDX[p]
        has = pn >= 0
        if not has.any() or ("raw", c) not in PRED:
            continue
        pc = PRED[("raw", c)]
        mm = metrics(pn, truth, W, has, EMO)["acc"]
        mc = metrics(pc, truth, W, has & HAVE[c], EMO)["acc"]
        d = mm - mc
        lo, hi, p = ci(d)
        row = dict(condition=c, n=int(has.sum()), acc_numbered_prompt=mm[0], acc_center_prompt=mc[0],
                   delta=d[0], ci_lo=lo, ci_hi=hi, p_boot=p)
        for ref in ("plain", "box"):
            if ("raw", ref) in PRED:
                row[f"acc_{ref}_center_same_faces"] = float(metrics(PRED[("raw", ref)], truth, W, has & HAVE[ref], EMO)["acc"][0])
        rows.append(row)
    t7 = pd.DataFrame(rows)
    t7.to_csv(os.path.join(args.out_dir, "tab_numbered_vs_center.csv"), index=False)

    # ================= FIGURES =================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        plt = None
    if plt is not None:
        col = {"plain": "#6c757d", "box": "#4c78a8", "box_number": "#f58518", "sov": "#e45756"}
        # fig 1: overall balanced accuracy
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, cs, ttl in zip(axes, ("all7", "no_neutral"), ("all 7 classes", "without Neutral")):
            for k, variant in enumerate(("raw", "self")):
                sub = t1[(t1.variant == variant) & (t1.classes == cs) & (t1.metric == "bal") & t1.condition.isin(BASE)]
                sub = sub.set_index("condition").reindex([c for c in BASE if c in set(sub.condition)])
                x = np.arange(len(sub)) + (k - 0.5) * 0.38
                ax.bar(x, sub.value, 0.36, color=[col[c] for c in sub.index],
                       alpha=1.0 if variant == "raw" else 0.55, edgecolor="black", linewidth=0.5,
                       label=("raw" if variant == "raw" else "self-calibrated"))
                ax.errorbar(x, sub.value, yerr=[sub.value - sub.ci_lo, sub.ci_hi - sub.value],
                            fmt="none", ecolor="black", capsize=3)
            ax.set_xticks(range(len(BASE)))
            ax.set_xticklabels([PRETTY[c] for c in BASE if c in LP])
            ax.set_title(f"Balanced accuracy, {ttl}")
            ax.set_ylim(0, 0.9)
        axes[0].set_ylabel("balanced accuracy (95% CI)")
        from matplotlib.patches import Patch
        axes[0].legend(handles=[Patch(facecolor="gray", edgecolor="black", label="raw (dark bars)"),
                                Patch(facecolor="gray", edgecolor="black", alpha=0.45, label="self-calibrated (light bars)")],
                       loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "fig_overall.png"), dpi=160)
        plt.close(fig)

        # fig 2: forest plot of SoV - Plain by group
        sub = t3[(t3.variant == "raw") & (t3.classes == "groups6") & (t3.contrast == "sov - plain")]
        if len(sub):
            labels_y = [f"{r.group}: {r.level} (n={r.n})" for r in sub.itertuples()]
            y = np.arange(len(sub))[::-1]
            fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(sub) + 1.5))
            ax.errorbar(sub.delta * 100, y, xerr=[(sub.delta - sub.ci_lo) * 100, (sub.ci_hi - sub.delta) * 100],
                        fmt="o", color="#e45756", ecolor="black", capsize=3)
            ax.axvline(0, color="gray", lw=1)
            ax.set_yticks(y)
            ax.set_yticklabels(labels_y, fontsize=8)
            ax.set_xlabel("SoV - Plain, balanced accuracy (points, 95% CI)")
            ax.set_title("Effect of SoV by group (6 classes, raw)")
            fig.tight_layout()
            fig.savefig(os.path.join(args.out_dir, "fig_group_forest.png"), dpi=160)
            plt.close(fig)

        # fig 3: recall by emotion
        fig, ax = plt.subplots(figsize=(9, 4))
        wbar = 0.8 / len(BASE)
        for k, c in enumerate([c for c in BASE if c in LP]):
            s = t5[t5.condition == c].set_index("emotion").reindex(EMO)
            ax.bar(np.arange(7) + (k - 1.5) * wbar, s.recall, wbar, color=col[c], label=PRETTY[c])
        ax.set_xticks(range(7))
        ax.set_xticklabels(EMO)
        ax.set_ylabel("recall")
        ax.set_title("Recall by emotion (raw)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "fig_recall.png"), dpi=160)
        plt.close(fig)

        # fig 4: predicted-class share
        fig, ax = plt.subplots(figsize=(9, 4))
        for k, c in enumerate([c for c in BASE if c in LP]):
            s = t5[t5.condition == c].set_index("emotion").reindex(EMO)
            ax.bar(np.arange(7) + (k - 1.5) * wbar, s.pred_share, wbar, color=col[c], label=PRETTY[c])
        ax.set_xticks(range(7))
        ax.set_xticklabels(EMO)
        ax.set_ylabel("share of predictions")
        ax.set_title("What the model predicts (raw)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "fig_pred_share.png"), dpi=160)
        plt.close(fig)

    # ================= SUMMARY =================
    def line_overall(variant, cs):
        s = t1[(t1.variant == variant) & (t1.classes == cs) & (t1.metric == "bal") & t1.condition.isin(BASE)]
        return ", ".join(f"{PRETTY[r.condition]} {r.value:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]" for r in s.itertuples())

    L = ["# Analysis summary (auto-generated)", "",
         f"Faces: {n} | images: {len(u)} | bootstrap: {args.boot} resamples of images | seed {args.seed}", "",
         "## Overall balanced accuracy (95% CI)",
         f"- raw, 7 classes: {line_overall('raw', 'all7')}",
         f"- raw, without Neutral: {line_overall('raw', 'no_neutral')}",
         f"- self-calibrated, 7 classes: {line_overall('self', 'all7')}", "",
         "## Paired differences, balanced accuracy, points (95% CI)"]
    for variant in variants:
        for cs in ("all7", "no_neutral"):
            s = t2[(t2.variant == variant) & (t2.classes == cs) & (t2.metric == "bal")]
            L.append(f"- **{variant}, {cs}**: " + "; ".join(
                f"{r.contrast} {r.delta * 100:+.1f} [{r.ci_lo * 100:+.1f}, {r.ci_hi * 100:+.1f}]" for r in s.itertuples()))
    L += ["", "## SoV - Plain by group (raw, 6 classes; points)"]
    s = t3[(t3.variant == "raw") & (t3.classes == "groups6") & (t3.contrast == "sov - plain")]
    for r in s.itertuples():
        L.append(f"- {r.group} / {r.level} (n={r.n}{', SMALL' if r.small_level else ''}): "
                 f"{r.delta * 100:+.1f} [{r.ci_lo * 100:+.1f}, {r.ci_hi * 100:+.1f}]")
    L += ["", "## Is the SoV - Plain effect different between groups? (raw, 6 classes; difference vs reference level)"]
    s = t4[(t4.variant == "raw") & (t4.classes == "groups6") & (t4.contrast == "sov - plain")]
    for r in s.itertuples():
        L.append(f"- {r.group}: {r.level} vs {r.reference}: {r.diff * 100:+.1f} [{r.ci_lo * 100:+.1f}, {r.ci_hi * 100:+.1f}]"
                 f" -> {'CI excludes 0' if r.excludes_zero else 'no clear difference'}")
    if len(t7):
        L += ["", "## Numbered vs centre prompt on multi-face crops (accuracy)"]
        for r in t7.itertuples():
            L.append(f"- {r.condition} (n={r.n}): numbered {r.acc_numbered_prompt:.3f} vs centre "
                     f"{r.acc_center_prompt:.3f}, diff {r.delta * 100:+.1f} [{r.ci_lo * 100:+.1f}, {r.ci_hi * 100:+.1f}]")
    L += ["", "## Warnings",
          "- Fear is excluded from per-group results (too few faces); the model predicts Neutral almost never, "
          "so Neutral recall is ~0 in every condition (see the no_neutral rows).",
          "- Group labels are FairFace ESTIMATES; gender/age use only faces whose label was stable across two crops.",
          f"- Group x emotion cells below {args.min_cell} faces: "
          + (", ".join(f"{g}/{e}={v}" for g, e, v in small) if small else "none")]
    with open(os.path.join(args.out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("\nFiles written to", args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
