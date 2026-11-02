import json, collections
for model in ["llava15", "qwen25vl"]:
    rows = [json.loads(l) for l in open(f"results_{model}_jobs_core.jsonl") if l.strip()]
    for cond in ["plain", "box", "box_number", "sov"]:
        r = [x for x in rows if x["condition"] == cond and x["query_type"] == "center" and not x.get("error")]
        n = len(r)
        neu = [x for x in r if x["label"].lower() == "neutral"]
        rec = sum(x["pred"].lower() == "neutral" for x in neu) / max(1, len(neu))
        pred = collections.Counter(x["pred"] for x in r)
        print(model, cond, n, "neutral recall %.2f" % rec, {k: round(v / n, 2) for k, v in pred.most_common()})
