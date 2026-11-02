import json, os
import numpy as np
import pandas as pd

CLASSES = ["angry","disgust","fear","happy","sad","surprise","neutral"]
B, SEED = 2000, 42
MODELS = {
    "llava15":  ["results_llava15_jobs_multi_even.jsonl",  "results_llava15_jobs_multi_odd.jsonl"],
    "qwen25vl": ["results_qwen25vl_jobs_multi_even.jsonl", "results_qwen25vl_jobs_multi_odd.jsonl"],
}

mf = pd.read_csv("multi_faces.csv")
mf["label"] = mf["expression_name"].astype(str).str.lower().str.strip()
F = mf["row_id"].to_numpy()
lab = mf["label"].to_numpy()
mixed = mf["mixed"].astype(bool).to_numpy()
uimg, jx = np.unique(mf["image_key"].to_numpy(), return_inverse=True)
J = len(uimg)
print("targets:", len(mf), "images:", J, "mixed faces:", int(mixed.sum()))

preds, qts = {}, set()
for m, files in MODELS.items():
    d = {}
    for f in files:
        for l in open(f, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            if r.get("error"):
                continue
            d[(r["row_id"], r["condition"], r["query_type"])] = r["pred"].strip().lower()
            qts.add(r["query_type"])
    preds[m] = d
print("query types seen:", sorted(qts))
NUM = [q for q in qts if q != "positional"]
assert len(NUM) == 1, f"expected one non-positional query type, got {NUM}"
NUM = NUM[0]

ARMS = {
    "plain":   ("plain", "positional"),
    "box":     ("box", "positional"),
    "bn_num":  ("box_number", NUM),
    "sov_num": ("sov", NUM),
    "bn_pos":  ("box_number", "positional"),
    "sov_pos": ("sov", "positional"),
}
MAIN = ["plain", "box", "bn_num", "sov_num"]

def arm_pred(m, arm):
    cond, qt = ARMS[arm]
    return np.array([preds[m].get((r, cond, qt)) for r in F], dtype=object)

P = {(m, a): arm_pred(m, a) for m in MODELS for a in ARMS}
OK = {k: np.array([x is not None for x in v]) for k, v in P.items()}
COR = {k: np.array([(x == l) if x is not None else False for x, l in zip(v, lab)], dtype=float) for k, v in P.items()}
for m in MODELS:
    print(m, {a: int(OK[(m, a)].sum()) for a in ARMS})

SUB = {"all": np.ones(len(mf), bool), "expr": ~np.isin(lab, ["happy", "neutral"]), "mixed": mixed}
rng = np.random.default_rng(SEED)
W = rng.multinomial(J, np.ones(J) / J, size=B).astype(float)

def acc_bs(corr, mask):
    C = np.bincount(jx, weights=corr * mask, minlength=J)
    N = np.bincount(jx, weights=mask.astype(float), minlength=J)
    den = W @ N
    bs = np.divide(W @ C, den, out=np.full(B, np.nan), where=den > 0)
    return C.sum() / N.sum(), bs

def contrast(a, b, subset):
    mask = SUB[subset].copy()
    for m in MODELS:
        mask &= OK[(m, a)] & OK[(m, b)]
    res = {}
    for m in MODELS:
        pa, ba = acc_bs(COR[(m, a)], mask)
        pb, bb = acc_bs(COR[(m, b)], mask)
        res[m] = ((pa - pb) * 100, (ba - bb) * 100)
    dpt = res["qwen25vl"][0] - res["llava15"][0]
    dbs = res["qwen25vl"][1] - res["llava15"][1]
    return int(mask.sum()), res, dpt, dbs

def fmt(pt, bs):
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return f"{pt:+6.1f} [{lo:+.1f}, {hi:+.1f}]", lo, hi

CONTRASTS = [
    ("SoV as designed - Plain",            "sov_num", "plain",   "all"),
    ("Box+Num as designed - Plain",        "bn_num",  "plain",   "all"),
    ("Landmarks (SoV - Box+Num, numbered)", "sov_num", "bn_num",  "all"),
    ("SoV as designed - Plain",            "sov_num", "plain",   "expr"),
    ("Landmarks (SoV - Box+Num, numbered)", "sov_num", "bn_num",  "expr"),
    ("Number tags alone (BoxNum - Box, positional)", "bn_pos", "box", "mixed"),
    ("Number vs counting, SoV image",      "sov_num", "sov_pos", "mixed"),
    ("Number vs counting, BoxNum image",   "bn_num",  "bn_pos",  "mixed"),
]

print("\n=== paired contrasts, accuracy points (95% CI, bootstrap over images) ===")
rows = []
for name, a, b, sub in CONTRASTS:
    n, res, dpt, dbs = contrast(a, b, sub)
    row = [name, sub, n]
    print(f"\n{name} [{sub}, n={n}]")
    for m in MODELS:
        s, lo, hi = fmt(*res[m])
        print(f"   {m:9s} {s}")
        row += [round(res[m][0], 2), round(lo, 2), round(hi, 2)]
    s, lo, hi = fmt(dpt, dbs)
    print(f"   Qwen-LLaVA {s}" + ("  *" if lo > 0 or hi < 0 else ""))
    row += [round(dpt, 2), round(lo, 2), round(hi, 2)]
    rows.append(row)

core = np.ones(len(mf), bool)
for m in MODELS:
    for a in MAIN:
        core &= OK[(m, a)]
print("\nfaces with all 4 main arms in both models:", int(core.sum()))

print("\n=== pooled accuracy per arm (faces above) ===")
for m in MODELS:
    for sub in ["all", "expr"]:
        mk = core & SUB[sub]
        print(f"{m:9s} {sub:5s} n={int(mk.sum()):5d} " + "  ".join(f"{a}={COR[(m, a)][mk].mean():.3f}" for a in MAIN))

print("\n=== recall per true emotion (n per class in header) ===")
print("n:", {c: int(((lab == c) & core).sum()) for c in CLASSES})
print("model     arm       " + " ".join(f"{c[:8]:>8s}" for c in CLASSES))
for m in MODELS:
    for a in MAIN:
        rec = []
        for c in CLASSES:
            mk = core & (lab == c)
            rec.append(np.mean(P[(m, a)][mk] == c) if mk.sum() else float("nan"))
        print(f"{m:9s} {a:9s} " + " ".join(f"{x:8.2f}" for x in rec))

print("\n=== what Happy faces are predicted as (share) ===")
for m in MODELS:
    for a in MAIN:
        mk = core & (lab == "happy")
        p = P[(m, a)][mk]
        h, ne = np.mean(p == "happy"), np.mean(p == "neutral")
        print(f"{m:9s} {a:9s} happy={h:.2f} neutral={ne:.2f} other={1 - h - ne:.2f}")

os.makedirs("analysis_compare", exist_ok=True)
pd.DataFrame(rows, columns=["contrast", "subset", "n", "llava_pt", "llava_lo", "llava_hi",
                            "qwen_pt", "qwen_lo", "qwen_hi", "diff_pt", "diff_lo", "diff_hi"]
             ).to_csv("analysis_compare/multi_compare.csv", index=False)
print("\nwrote analysis_compare/multi_compare.csv")
