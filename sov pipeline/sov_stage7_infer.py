"""
SoV pipeline - Stage 7: run the model on the jobs
=================================================

Reads jobs_<prompt_set>.jsonl (from stage 6), runs the model on every job and appends
one JSON line per finished job to results.jsonl. The model sits behind a tiny "backend"
interface, so Qwen2.5-VL (or anything else) can be added later without touching the rest.

Safety
------
* Every finished job is appended and fsync'ed immediately -> a crash / Colab disconnect
  loses at most the job that was running. Re-running skips finished jobs (resume).
* A failing job (error, out-of-memory) is logged to errors.jsonl and skipped, never
  written to results.jsonl, so it is retried on the next run.
* A half-written last line (crash mid-write) is tolerated.
* Results never overwrite anything; one results file per (model, prompt set).
* PUT THE OUTPUT FOLDER ON GOOGLE DRIVE, otherwise a runtime reset deletes the results.

Important LLaVA-1.5 detail: the Hugging Face preprocessing resizes the shortest side to
336 and then CENTER-CROPS to 336x336. On a wide image (e.g. 1024x576) that silently cuts
away the left and right sides, so faces there would never be seen. We therefore pad the
image to a square (grey) first, exactly like the original LLaVA-1.5 does. This keeps every
face visible but shrinks the image to 336 px overall (a 1024 px image -> factor 0.33).

Modes
-----
  dry_run=True   no model; checks the jobs file, every image path, padding. CPU only.
  probe=True     asks the model to read the face-number tags in box_number / sov images and
                 reports how many it can read (tells you if the overlay survives the resize).
  (default)      the full run.

Usage (Colab, GPU runtime, fresh session)
-----------------------------------------
    !pip install -q -U transformers accelerate bitsandbytes
    from sov_stage7_infer import run
    run("/content/new_output_folder", dry_run=True)          # 1) pre-flight, no GPU needed
    run("/content/new_output_folder", probe=True)            # 2) can it read the tags?
    run("/content/new_output_folder")                        # 3) full run
"""

import os
import re
import sys
import json
import time
import argparse
import logging
import traceback
import platform
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("USE_TF", "0")                 # keep TensorFlow out of this stage
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from PIL import Image, ImageOps

log = logging.getLogger("sov7")
PAD_FILL = (122, 116, 104)   # CLIP mean colour, what LLaVA-1.5 pads with


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def pkg_version(name):
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return None


def _ensure_trailing_newline(path):
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb+") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")  # repair a half-written last line


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path):
    out = []
    if not Path(path).exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass  # tolerate a corrupt/half line
    return out


def load_image(path):
    with Image.open(path) as im:
        im.load()
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")


def pad_to_square(img, fill=PAD_FILL):
    w, h = img.size
    if w == h:
        return img
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), fill)
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    return canvas


def pick_max_new_tokens(job, override=None):
    if override:
        return int(override)
    q = job["query_type"]
    if q in ("count", "one_face"):
        return 24
    if job["prompt_set"] == "constrained":
        return min(640, 32 + 12 * int(job["n_faces_final"]))
    return 512  # free-form 'paper' answers can be long


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class Backend:
    name = "base"
    model_id = None

    def generate(self, image, prompt, max_new_tokens):
        """Return {'text': str, 'n_input_tokens': int, 'n_new_tokens': int}."""
        raise NotImplementedError

    def info(self):
        return {"backend": self.name, "model_id": self.model_id}


class LlavaBackend(Backend):
    """LLaVA-1.5-7B (Hugging Face port), 4-bit, greedy decoding."""
    name = "llava15"
    model_id = "llava-hf/llava-1.5-7b-hf"

    def __init__(self, load_4bit=True):
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU found. In Colab: Runtime > Change runtime type > T4 GPU, "
                               "then restart the session.")
        self.torch = torch
        kwargs = dict(torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16)
        log.info("Loading %s (4-bit=%s). The first time this downloads ~14 GB; be patient.",
                 self.model_id, load_4bit)
        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(self.model_id, **kwargs)
        self.model.eval()
        self.load_4bit = load_4bit
        log.info("Model loaded in %.0fs", time.time() - t0)

    def info(self):
        return {"backend": self.name, "model_id": self.model_id, "load_4bit": self.load_4bit,
                "decoding": "greedy"}

    def generate(self, image, prompt, max_new_tokens):
        torch = self.torch
        conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        try:
            text = self.processor.apply_chat_template(conv, add_generation_prompt=True)
        except Exception:
            text = f"USER: <image>\n{prompt} ASSISTANT:"
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        n_in = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new = out[0][n_in:]
        return {"text": self.processor.decode(new, skip_special_tokens=True).strip(),
                "n_input_tokens": n_in, "n_new_tokens": int(new.shape[0])}


class FakeBackend(Backend):
    """For testing the plumbing without a GPU. Answers with dummy text."""
    name = "fake"
    model_id = "fake"

    def generate(self, image, prompt, max_new_tokens):
        if "How many" in prompt:
            return {"text": "3", "n_input_tokens": 10, "n_new_tokens": 1}
        return {"text": "Face 1: Happy\nFace 2: Neutral", "n_input_tokens": 50, "n_new_tokens": 12}


BACKENDS = {"llava15": LlavaBackend, "fake": FakeBackend}
# Qwen2.5-VL will be added here as another Backend subclass in the next step.


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def _setup_logging(log_dir):
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(Path(log_dir) / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)


def _preview(s, n=90):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n] + "..."


def _gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        return None


def run(overlay_root, prompt_set="constrained", backend="llava15", output_dir=None,
        jobs_file=None, limit=None, conditions=None, query_types=None,
        pad_square=True, max_new_tokens=None, dry_run=False, probe=False):
    overlay_root = Path(overlay_root)
    jobs_path = Path(jobs_file) if jobs_file else overlay_root / "prompts" / f"jobs_{prompt_set}.jsonl"
    if not jobs_path.exists():
        raise FileNotFoundError(f"Jobs file not found: {jobs_path} (run stage 6 first)")
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; choose from {list(BACKENDS)}")
    out_dir = Path(output_dir) if output_dir else overlay_root / "results" / backend
    out_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(out_dir)

    jobs = read_jsonl(jobs_path)
    if conditions:
        jobs = [j for j in jobs if j["condition"] in conditions]
    if query_types:
        jobs = [j for j in jobs if j["query_type"] in query_types]
    if limit:
        jobs = jobs[:limit]
    log.info("Loaded %d job(s) from %s", len(jobs), jobs_path)
    if not jobs:
        log.error("No jobs to run (check filters).")
        return

    # --- pre-flight: every image must exist and open, before we load a big model ---
    missing, sizes = [], {}
    for rel in sorted({j["image_relpath"] for j in jobs}):
        p = overlay_root / rel
        if not p.exists():
            missing.append(rel)
            continue
        try:
            with Image.open(p) as im:
                sizes[rel] = im.size
        except Exception as e:
            missing.append(f"{rel} (unreadable: {e})")
    if missing:
        raise FileNotFoundError("Missing/unreadable images:\n  " + "\n  ".join(missing))
    log.info("Pre-flight OK: %d distinct images found.", len(sizes))

    if dry_run:
        for j in jobs[:3]:
            log.info("example job %s\n    prompt: %s\n    max_new_tokens=%d, image=%s",
                     j["job_id"], _preview(j["prompt"], 150), pick_max_new_tokens(j, max_new_tokens),
                     sizes[j["image_relpath"]])
        s = max(max(v) for v in sizes.values())
        log.info("Dry run finished. Largest image side %d px -> after square padding and the "
                 "336 px resize the scale factor is about %.2f.", s, 336.0 / s)
        return

    results_path = out_dir / f"results_{prompt_set}.jsonl"
    errors_path = out_dir / "errors.jsonl"
    probe_path = out_dir / "probe_results.jsonl"
    append_jsonl(out_dir / "run_history.jsonl", {
        "started_utc": utc_now(), "overlay_root": str(overlay_root), "prompt_set": prompt_set,
        "mode": "probe" if probe else "full", "pad_square": pad_square, "limit": limit,
        "python": sys.version.split()[0], "platform": platform.platform(), "gpu": _gpu_name(),
        "versions": {p: pkg_version(p) for p in
                     ("torch", "transformers", "accelerate", "bitsandbytes", "pillow")}})

    be = BACKENDS[backend]()
    img_cache = {"rel": None, "img": None}

    def get_image(rel):
        if img_cache["rel"] != rel:
            im = load_image(overlay_root / rel)
            img_cache.update(rel=rel, img=(pad_to_square(im) if pad_square else im),
                             orig=im.size)
        return img_cache["img"], img_cache["orig"]

    if probe:
        return _run_probe(be, jobs, get_image, probe_path, errors_path, max_new_tokens)

    _ensure_trailing_newline(results_path)
    done = {r["job_id"] for r in read_jsonl(results_path) if r.get("error") is None}
    todo = [j for j in jobs if j["job_id"] not in done]
    log.info("%d already done, %d to run.", len(jobs) - len(todo), len(todo))

    n_ok = n_err = 0
    t_start = time.time()
    try:
        for i, j in enumerate(todo, 1):
            t0 = time.time()
            try:
                img, orig = get_image(j["image_relpath"])
                mnt = pick_max_new_tokens(j, max_new_tokens)
                g = be.generate(img, j["prompt"], mnt)
                rec = {
                    "job_id": j["job_id"], "image_key": j["image_key"], "condition": j["condition"],
                    "prompt_set": j["prompt_set"], "query_type": j["query_type"],
                    "face_id": j["face_id"], "n_faces_final": j["n_faces_final"],
                    "prompt": j["prompt"], "output": g["text"],
                    "n_input_tokens": g["n_input_tokens"], "n_new_tokens": g["n_new_tokens"],
                    "max_new_tokens": mnt, "hit_token_limit": g["n_new_tokens"] >= mnt,
                    "image_size": list(orig), "padded_to_square": pad_square,
                    "seconds": round(time.time() - t0, 2), "created_utc": utc_now(),
                    "error": None, **be.info()}
                append_jsonl(results_path, rec)  # saved immediately
                n_ok += 1
                log.info("[%d/%d] %s -> %s (%.1fs%s)", i, len(todo), j["job_id"].split("|", 1)[1],
                         _preview(g["text"]), time.time() - t0,
                         ", HIT TOKEN LIMIT" if rec["hit_token_limit"] else "")
            except Exception as e:
                n_err += 1
                log.error("[%d/%d] FAILED %s: %s", i, len(todo), j["job_id"], e)
                append_jsonl(errors_path, {"time_utc": utc_now(), "job_id": j["job_id"],
                                           "error": repr(e), "traceback": traceback.format_exc()})
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()  # helps after an out-of-memory error
                except Exception:
                    pass
    except KeyboardInterrupt:
        log.warning("Interrupted. Finished jobs are saved; re-run to resume.")
    finally:
        log.info("Done. ok=%d failed=%d in %.0fs | results: %s",
                 n_ok, n_err, time.time() - t_start, results_path)


def _run_probe(be, jobs, get_image, probe_path, errors_path, max_new_tokens):
    """Can the model read the face-number tags? One question per numbered image."""
    prompt = "List every number written on the small colored tags in this image, separated by commas."
    seen, rows = set(), []
    for j in jobs:
        if j["condition"] not in ("box_number", "sov"):
            continue
        k = (j["image_key"], j["condition"])
        if k in seen:
            continue
        seen.add(k)
        try:
            img, _ = get_image(j["image_relpath"])
            g = be.generate(img, prompt, max_new_tokens or 120)
            said = {int(x) for x in re.findall(r"\d+", g["text"])}
            exp = set(j["expected_face_ids"])
            hit = len(said & exp)
            row = {"image_key": j["image_key"], "condition": j["condition"],
                   "expected": len(exp), "read_correctly": hit,
                   "invented": len(said - exp), "output": g["text"]}
            append_jsonl(probe_path, row)
            rows.append(row)
            log.info("probe %-45s %-10s read %d/%d tags, %d invented",
                     j["image_key"][:45], j["condition"], hit, len(exp), len(said - exp))
        except Exception as e:
            log.error("probe failed for %s: %s", k, e)
            append_jsonl(errors_path, {"time_utc": utc_now(), "probe": list(k),
                                       "error": repr(e), "traceback": traceback.format_exc()})
    log.info("Probe finished. Results: %s", probe_path)


def main(argv=None):
    ap = argparse.ArgumentParser(description="SoV stage 7: run the model on the jobs.")
    ap.add_argument("--overlay_root", required=True, help="OUTPUT folder of stage 3-5")
    ap.add_argument("--prompt_set", default="constrained", choices=["paper", "constrained"])
    ap.add_argument("--backend", default="llava15", choices=list(BACKENDS))
    ap.add_argument("--output", default=None, help="default: <overlay_root>/results/<backend>")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--conditions", default=None, help="comma list, e.g. plain,sov")
    ap.add_argument("--query_types", default=None, help="comma list, e.g. count")
    ap.add_argument("--no_pad", action="store_true", help="do NOT pad to square (not advised)")
    ap.add_argument("--max_new_tokens", type=int, default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--probe", action="store_true")
    a, _ = ap.parse_known_args(argv)
    run(a.overlay_root, a.prompt_set, a.backend, a.output, None, a.limit,
        a.conditions.split(",") if a.conditions else None,
        a.query_types.split(",") if a.query_types else None,
        not a.no_pad, a.max_new_tokens, a.dry_run, a.probe)


if __name__ == "__main__":
    main()