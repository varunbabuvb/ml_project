"""
SoV pipeline - Stage 6: prompt building
=======================================

Turns the output of stage 3-5 into a list of "jobs" for the model stage. One job =
(image, condition, question). No model is loaded here, so this runs anywhere (CPU).

What the SoV paper says about prompts (Sec. 3.1, 3.3, Fig. 2, 3, 5, 10-15, App. A.3)
------------------------------------------------------------------------------------
* Eq. 2: the VLLM gets SoV(I) as the image and the SAME text prompt Q. Only the image
  changes between conditions.
* Plain text prompt  : a general question about the group (Fig. 5 left, Fig. 2).
* Combined text-vision prompt: questions about a specific face by its number
  (Fig. 5 right, Fig. 15: 'what is the emotion of the person labeled "2"').
* Fig. 3 shows two questions per image: "How many visible faces ...?" and
  "What is the emotion for each face?".
* Appendix A.3 also shows a 4-step chain-of-thought question list. NOT used here
  (needs multi-turn chat); can be added later.
* The paper does NOT give the exact prompts used for its benchmark numbers, nor how
  free-form answers were scored. So the "paper" set below uses the paper's example
  wordings, and the "constrained" set is OUR addition that makes answers parseable.

Prompt sets
-----------
  paper        paper's own wording, free-form answers. Text identical in all conditions.
  constrained  same questions + the 7 allowed emotion words + a fixed answer format.
               Text is still identical in all four conditions (so only the image
               differs), and works for single-face images too (e.g. RAF-DB).

Query types
-----------
  count      "How many visible faces are there in the image?"
  all_faces  one answer listing every face's emotion
  one_face   one question per numbered face (ONLY for box_number / sov: the plain and
             box images have no labels to refer to). Off by default; enable it with
             query_types=("count","all_faces","one_face"). Useful for weaker models that
             struggle to list many faces at once; costs one model call per face.

Outputs (under --output, default <overlay_root>/prompts)
--------------------------------------------------------
  jobs_<set>.jsonl           one JSON object per job
  jobs_<set>_meta.json       settings, counts, provenance
  prompts_preview_<set>.txt  human-readable example of every distinct prompt (for the report)
Existing files are never overwritten silently: the old one is kept as *.prev-<time>.

Usage
-----
    python sov_stage6_prompts.py --overlay_root ./out_overlay
Notebook:
    from sov_stage6_prompts import run
    run("/content/output_overlay")
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]  # paper's order
CONDITIONS = ("plain", "box", "box_number", "sov")
NUMBERED_CONDITIONS = ("box_number", "sov")
QUERY_TYPES = ("count", "all_faces", "one_face")
PROMPT_SETS = ("paper", "constrained")
NEUTRAL_HINT = ' Use "Neutral" when the face shows no clear emotion.'


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# The prompts themselves
# ---------------------------------------------------------------------------
def build_prompt(prompt_set, query_type, face_id=None, labels=EMOTIONS, neutral_hint=False):
    if prompt_set not in PROMPT_SETS:
        raise ValueError(f"prompt_set must be one of {PROMPT_SETS}, got {prompt_set!r}")
    if query_type not in QUERY_TYPES:
        raise ValueError(f"query_type must be one of {QUERY_TYPES}, got {query_type!r}")
    if query_type == "one_face" and face_id is None:
        raise ValueError("one_face prompts need a face_id")
    opts = ", ".join(labels)
    hint = NEUTRAL_HINT if neutral_hint else ""

    if prompt_set == "paper":
        if query_type == "count":
            return "How many visible faces are there in the image?"
        if query_type == "all_faces":
            return "What is the emotion for each face?"
        return f'What is the emotion of the person labeled "{face_id}"?'

    # constrained
    if query_type == "count":
        return ("How many visible faces are there in the image? "
                "Answer with a single number only.")
    if query_type == "all_faces":
        return (f"What is the emotion of each visible face? Choose from: {opts}.{hint} "
                "If faces are labeled with numbers in the image, use those numbers; "
                "otherwise number the faces yourself from left to right. "
                'Answer with one line per face in the format "Face <number>: <emotion>" '
                "and nothing else.")
    return (f'What is the emotion of the person labeled "{face_id}"? '
            f"Choose exactly one of: {opts}.{hint} Answer with the emotion word only.")


# ---------------------------------------------------------------------------
# Job manifest
# ---------------------------------------------------------------------------
def build_jobs(overlay_root, prompt_set, query_types, labels, neutral_hint):
    overlay_root = Path(overlay_root)
    final_dir = overlay_root / "final"
    if not final_dir.is_dir():
        raise FileNotFoundError(f"No 'final' folder in {overlay_root}. Run stage 3-5 first "
                                "and pass its OUTPUT folder as overlay_root.")
    jobs, problems, eps_seen = [], [], set()
    files = [p for p in sorted(final_dir.glob("*.json")) if not p.name.endswith(".prev.json")]
    if not files:
        raise FileNotFoundError(f"No final/*.json files in {final_dir}.")

    for p in files:
        key = p.name[:-len(".json")]
        try:
            with open(p, "r", encoding="utf-8") as f:
                rec = json.load(f)
            face_ids = [int(f["sov_id"]) for f in rec["faces"]]
            eps_seen.add(rec.get("params", {}).get("eps"))
        except Exception as e:
            problems.append(f"{key}: unreadable final json ({e})")
            continue
        for cond in CONDITIONS:
            rel = f"overlays/{key}/{cond}.png"
            if not (overlay_root / rel).exists():
                problems.append(f"{key}/{cond}: overlay image missing ({rel}); jobs skipped")
                continue
            for qt in query_types:
                if qt == "one_face":
                    if cond not in NUMBERED_CONDITIONS:
                        continue  # no labels in plain/box images to refer to
                    targets = face_ids
                else:
                    targets = [None]
                for fid in targets:
                    jobs.append({
                        "job_id": f"{key}|{cond}|{prompt_set}|{qt}" + (f"|{fid}" if fid else ""),
                        "image_key": key,
                        "image": rec["image"],
                        "condition": cond,
                        "image_relpath": rel,
                        "prompt_set": prompt_set,
                        "query_type": qt,
                        "face_id": fid,
                        "prompt": build_prompt(prompt_set, qt, fid, labels, neutral_hint),
                        "n_faces_final": rec["num_final"],
                        "expected_face_ids": face_ids,
                    })
    ids = [j["job_id"] for j in jobs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate job_ids; image keys are not unique.")
    return jobs, problems, sorted(e for e in eps_seen if e is not None)


def _atomic_write_text(path, text):
    path = Path(path)
    if path.exists():  # never silently overwrite
        stamp = time.strftime("%Y%m%d-%H%M%S")
        os.replace(path, path.with_name(f"{path.stem}.prev-{stamp}{path.suffix}"))
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def run(overlay_root, output_dir=None, prompt_sets=PROMPT_SETS,
        query_types=("count", "all_faces"), neutral_hint=False, labels=EMOTIONS):
    overlay_root = Path(overlay_root)
    output_dir = Path(output_dir) if output_dir else overlay_root / "prompts"
    output_dir.mkdir(parents=True, exist_ok=True)
    query_types = tuple(query_types)
    bad = [q for q in query_types if q not in QUERY_TYPES]
    if bad:
        raise ValueError(f"Unknown query types {bad}; choose from {QUERY_TYPES}")

    results = {}
    for ps in prompt_sets:
        jobs, problems, eps = build_jobs(overlay_root, ps, query_types, labels, neutral_hint)

        _atomic_write_text(output_dir / f"jobs_{ps}.jsonl",
                           "".join(json.dumps(j, ensure_ascii=False) + "\n" for j in jobs))

        counts = {}
        for j in jobs:
            counts.setdefault(j["condition"], {}).setdefault(j["query_type"], 0)
            counts[j["condition"]][j["query_type"]] += 1
        meta = {"created_utc": utc_now(), "prompt_set": ps, "query_types": list(query_types),
                "labels": list(labels), "neutral_hint": neutral_hint,
                "num_jobs": len(jobs), "jobs_per_condition_and_query": counts,
                "stage3_5_eps_values": eps, "problems": problems,
                "overlay_root_at_creation": str(overlay_root)}
        _atomic_write_text(output_dir / f"jobs_{ps}_meta.json",
                           json.dumps(meta, indent=2, ensure_ascii=False))

        lines = [f"PROMPT SET: {ps}   (neutral_hint={neutral_hint})", ""]
        for qt in query_types:
            ex = 1 if qt == "one_face" else None
            lines += [f"[{qt}]", build_prompt(ps, qt, ex, labels, neutral_hint), ""]
        _atomic_write_text(output_dir / f"prompts_preview_{ps}.txt", "\n".join(lines))

        print(f"[{ps}] {len(jobs)} jobs -> {output_dir / f'jobs_{ps}.jsonl'}")
        for c in CONDITIONS:
            print(f"   {c:<11} {counts.get(c, {})}")
        for pr in problems:
            print("   WARNING:", pr)
        results[ps] = jobs
    print("\nPreview of the prompts:\n")
    for ps in prompt_sets:
        print((output_dir / f"prompts_preview_{ps}.txt").read_text(encoding="utf-8"))
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="SoV stage 6: build prompts and job manifest.")
    ap.add_argument("--overlay_root", required=True, help="OUTPUT folder of stage 3-5")
    ap.add_argument("--output", default=None, help="default: <overlay_root>/prompts")
    ap.add_argument("--prompt_sets", default="paper,constrained")
    ap.add_argument("--query_types", default="count,all_faces",
                    help="comma list from: count, all_faces, one_face")
    ap.add_argument("--neutral_hint", action="store_true",
                    help='add: Use "Neutral" when the face shows no clear emotion.')
    args, _ = ap.parse_known_args(argv)
    run(args.overlay_root, args.output, args.prompt_sets.split(","),
        args.query_types.split(","), args.neutral_hint)


if __name__ == "__main__":
    main()
