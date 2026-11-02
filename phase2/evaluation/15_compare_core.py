import json, csv, os
import numpy as np

CLASSES = ["angry","disgust","fear","happy","sad","surprise","neutral"]
CIDX = {c: i for i, c in enumerate(CLASSES)}
CONDS = ["plain", "box", "box_number", "sov"]
B, SEED = 2000, 42
FILES = {
    "llava15":  ("results_llava15_jobs_core.jsonl",  "results_llava15_jobs_core_halfdots.jsonl"),
    "qwen25vl": ("results_qwen25vl_jobs_core.jsonl", "results_qwen25vl_jobs_core_halfdots.jsonl"),
}

def center(path):
    out = {}
    for l in open(path, encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("error") or r.get("query_type") != "center":
            continue
        out[(r["row_id"], r["condition"])] = r   # keep-last handles resume duplicates
    return out

data = {}
for m, (core, half) in FILES.items():
    rows = center(core)
    d = {c: {rid: r for (rid, cc), r in rows.items() if cc == c} for c in CONDS}
    d["half"] = {rid: r for (rid, cc), r in center(half).items() if cc == "sov"}
    data[m] = d
    print(m, {k: len(v) for k, v in d.items()})

ids = sorted(set.intersection(*[set(data[m][c]) for m in data for c in data[m]]))
print("faces common to all runs:", len(ids), "(expected 1473)")
ref = data["llava15"]["plain"]
bad = sum(data[m][c][i]["label"] != ref[i]["label"] for m in data for c in data[m] for i in ids)
print("label mismatches across runs:", bad, "(expected 0)")

img = [ref[i]["image_key"] for i in ids]
uimg = sorted(set(img)); jx = {u: k for k, u in enumerate(uimg)}
J, K = len(uimg), len(CLASSES)
lab = [CIDX[ref[i]["label"].lower()] for i in ids]
T = np.zeros((J, K))
for i, k in zip(ids, lab):
    T[jx[ref[i]["image_key"]], k] += 1

def corr(m, c):
    Cm = np.zeros((J, K))
    for i, k in zip(ids, lab):
        if data[m][c][i]["pred"].strip().lower() == CLASSES[k]:
            Cm[jx[ref[i]["image_key"]], k] += 1
    return Cm

Cs = {(m, c): corr(m, c) for m in data for c in data[m]}

def ba(Cm, w):
    num, den = w @ Cm, w @ T
    rec = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return rec.mean(axis=1)

rng = np.random.default_rng(SEED)
W = rng.multinomial(J, np.ones(J) / J, size=B).astype(float)
ones = np.ones((1, J))
pt = {k: ba(v, ones)[0] for k, v in Cs.items()}
bs = {k: ba(v, W) for k, v in Cs.items()}

def ci(x):
    return np.percentile(x, [2.5, 97.5])

CONTR = [("SoV - Plain", "sov", "plain"), ("Box - Plain", "box", "plain"),
         ("Box+Num - Plain", "box_number", "plain"), ("SoV - Box+Num", "sov", "box_number"),
         ("Half-dot SoV - SoV", "half", "sov"), ("Half-dot SoV - Plain", "half", "plain")]

out = []
print("\n=== raw balanced accuracy, 7 classes ===")
for m in data:
    print(m, {c: round(pt[(m, c)], 3) for c in CONDS + ["half"]})

print("\n=== paired contrasts, points (95% CI, bootstrap over images) ===")
for name, a, b in CONTR:
    row = [name]
    est = {}
    for m in data:
        d_pt = (pt[(m, a)] - pt[(m, b)]) * 100
        lo, hi = ci((bs[(m, a)] - bs[(m, b)]) * 100)
        est[m] = (bs[(m, a)] - bs[(m, b)]) * 100
        print(f"{name:22s} {m:9s} {d_pt:+6.1f} [{lo:+.1f}, {hi:+.1f}]")
        row += [round(d_pt, 2), round(lo, 2), round(hi, 2)]
    dd = est["qwen25vl"] - est["llava15"]
    d_pt = ((pt[("qwen25vl", a)] - pt[("qwen25vl", b)]) - (pt[("llava15", a)] - pt[("llava15", b)])) * 100
    lo, hi = ci(dd)
    flag = "  *" if (lo > 0 or hi < 0) else ""
    print(f"{name:22s} Qwen-LLaVA {d_pt:+6.1f} [{lo:+.1f}, {hi:+.1f}]{flag}")
    row += [round(d_pt, 2), round(lo, 2), round(hi, 2)]
    out.append(row)

print("\n=== per-class recall ===")
print("model     cond        " + "  ".join(f"{c[:8]:>8s}" for c in CLASSES))
for m in data:
    for c in CONDS + ["half"]:
        rec = Cs[(m, c)].sum(0) / T.sum(0)
        print(f"{m:9s} {c:11s} " + "  ".join(f"{x:8.2f}" for x in rec))

os.makedirs("analysis_compare", exist_ok=True)
with open("analysis_compare/core_compare.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["contrast", "llava_pt", "llava_lo", "llava_hi", "qwen_pt", "qwen_lo", "qwen_hi", "diff_pt", "diff_lo", "diff_hi"])
    w.writerows(out)
print("\nwrote analysis_compare/core_compare.csv")
