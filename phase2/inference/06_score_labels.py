#!/usr/bin/env python3
"""
Step 6 -- run the model on the jobs from step 5 and record, for every job, the model's score
for each of the 7 emotion words (no text parsing needed).   Needs a GPU (Colab / Kaggle).

How the scoring works
  The prompt ends with "Answer with the emotion word only." We run ONE forward pass and read
  the model's next-token log-probabilities for the first token of each emotion word
  (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral).  The predicted label is the highest
  of the seven.  `mass` = probability the model puts on those seven tokens in total (low mass
  means the model wanted to say something else).  With --also_generate the model also writes
  a short normal answer so that step 7 can check the two agree.

Same file for Colab and Kaggle.  Install once per fresh session:
    !pip install -q -U transformers accelerate bitsandbytes
Examples
    # 0) look at how the label words are tokenised (cheap, loads only the tokenizer)
    !python 06_score_labels.py --model llava15 --check_tokens
    # 1) smoke test on 20 jobs
    !python 06_score_labels.py --overlay_root /content/ctx_dev/overlay --jobs /content/ctx_dev/overlay/prompts/jobs_dev.jsonl \
         --model llava15 --out_dir /content/drive/MyDrive/sov_results --limit 20 --also_generate
    # 2) tag-reading probe (can the model read the numbers?)
    !python 06_score_labels.py ... --probe --probe_n 100
    # 3) full run: same command without --limit (re-run it after a disconnect: it resumes)
Models:  llava15 (llava-hf/llava-1.5-7b-hf)   qwen25vl (Qwen/Qwen2.5-VL-7B-Instruct)   fake (no GPU)
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time
import traceback
import zlib
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("USE_TF", "0")                 # keep TensorFlow out of this stage
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from PIL import Image, ImageOps

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
PAD_FILL = (122, 116, 104)
SYNONYMS = {"anger": "Angry", "angry": "Angry", "disgusted": "Disgust", "disgust": "Disgust",
            "fear": "Fear", "fearful": "Fear", "scared": "Fear", "afraid": "Fear",
            "happy": "Happy", "happiness": "Happy", "joy": "Happy", "sad": "Sad",
            "sadness": "Sad", "surprise": "Surprise", "surprised": "Surprise",
            "neutral": "Neutral"}


# ------------------------------------------------------------------ helpers
def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass  # half-written last line
    return out


def ensure_trailing_newline(path):
    p = Path(path)
    if p.exists() and p.stat().st_size > 0:
        with open(p, "rb+") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")


def load_image(path):
    with Image.open(path) as im:
        im.load()
        return ImageOps.exif_transpose(im).convert("RGB")


def pad_to_square(img):
    w, h = img.size
    if w == h:
        return img
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), PAD_FILL)
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    return canvas


def normalize_label(text):
    m = re.search(r"[A-Za-z]+", text or "")
    return SYNONYMS.get(m.group(0).lower()) if m else None


def logsumexp(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def candidate_token_ids(tok, labels):
    """First-token ids for each label (with / without leading space, capitalised / lower).
    Raises if two labels share a token, so a bad tokenisation can never go unnoticed."""
    cands = []
    for lab in labels:
        ids = set()
        for variant in (lab, " " + lab, lab.lower(), " " + lab.lower()):
            enc = tok(variant, add_special_tokens=False).input_ids
            if enc and tok.decode(enc[:1]).strip():
                ids.add(int(enc[0]))
        cands.append(sorted(ids))
    seen = {}
    for lab, ids in zip(labels, cands):
        for i in ids:
            if i in seen:
                raise ValueError(f"Token {i} is a first token of both {seen[i]} and {lab}.")
            seen[i] = lab
    return cands


# ------------------------------------------------------------------ backends
class BaseScorer:
    name = "base"
    model_id = None

    def info(self):
        return {"backend": self.name, "model_id": self.model_id}

    def score(self, image, prompt, n_generate=0):
        raise NotImplementedError

    def generate_text(self, image, prompt, max_new_tokens):
        raise NotImplementedError


class _HFScorer(BaseScorer):
    """Shared scoring code for Hugging Face vision-language models."""
    torch = None
    tok = None
    cand_ids = None

    def _prepare(self, image, prompt):
        raise NotImplementedError

    def _setup_tokens(self):
        self.cand_ids = candidate_token_ids(self.tok, EMOTIONS)

    def score(self, image, prompt, n_generate=0):
        torch = self.torch
        inputs = self._prepare(image, prompt)
        n_in = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            out = self.model(**inputs)
            logits = out.logits[0, -1].float()
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError(
                "Non-finite logits (fp16 overflow?). For Qwen try --dtype bfloat16 or float32.")
        lp = torch.log_softmax(logits, dim=-1)
        label_lp = [float(torch.logsumexp(lp[ids], dim=0)) for ids in
                    [torch.tensor(c, device=lp.device) for c in self.cand_ids]]
        res = {"label_logprobs": dict(zip(EMOTIONS, label_lp)),
               "mass": float(math.exp(logsumexp(label_lp))), "n_input_tokens": n_in,
               "text": None}
        if n_generate:
            with torch.inference_mode():
                g = self.model.generate(**inputs, max_new_tokens=n_generate, do_sample=False)
            res["text"] = self.tok.decode(g[0][n_in:], skip_special_tokens=True).strip()
        return res

    def generate_text(self, image, prompt, max_new_tokens):
        torch = self.torch
        inputs = self._prepare(image, prompt)
        n_in = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            g = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tok.decode(g[0][n_in:], skip_special_tokens=True).strip()


class LlavaScorer(_HFScorer):
    name = "llava15"
    model_id = "llava-hf/llava-1.5-7b-hf"

    def __init__(self, load_4bit=True, tokenizer_only=False):
        import torch
        from transformers import AutoProcessor
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.tok = self.processor.tokenizer
        self._setup_tokens()
        self.load_4bit = load_4bit
        if tokenizer_only:
            return
        from transformers import LlavaForConditionalGeneration
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU. Colab: Runtime > Change runtime type > T4 GPU. "
                               "Kaggle: Settings > Accelerator > GPU.")
        kw = dict(torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16)
        self.model = LlavaForConditionalGeneration.from_pretrained(self.model_id, **kw).eval()

    def info(self):
        return {"backend": self.name, "model_id": self.model_id, "load_4bit": self.load_4bit}

    def _prepare(self, image, prompt):
        conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        try:
            text = self.processor.apply_chat_template(conv, add_generation_prompt=True)
        except Exception:
            text = f"USER: <image>\n{prompt} ASSISTANT:"
        inputs = self.processor(images=pad_to_square(image), text=text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(self.torch.float16)
        return inputs


class QwenScorer(_HFScorer):
    name = "qwen25vl"
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

    def __init__(self, load_4bit=True, dtype="float16", side=336, tokenizer_only=False):
        import torch
        from transformers import AutoProcessor
        self.torch = torch
        self.side = side
        self.dtype = dtype
        self.load_4bit = load_4bit
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.tok = self.processor.tokenizer
        self._setup_tokens()
        if tokenizer_only:
            return
        from transformers import Qwen2_5_VLForConditionalGeneration
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU found.")
        dt = {"float16": torch.float16, "bfloat16": torch.bfloat16,
              "float32": torch.float32}[dtype]
        kw = dict(torch_dtype=dt, device_map="auto", attn_implementation="sdpa")
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dt,
                llm_int8_skip_modules=["visual", "lm_head"])
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_id, **kw).eval()

    def info(self):
        return {"backend": self.name, "model_id": self.model_id, "load_4bit": self.load_4bit,
                "dtype": self.dtype, "input_side": self.side}

    def _prepare(self, image, prompt):
        s = int(round(self.side / 28.0)) * 28      # Qwen works in 28-px steps
        if image.size != (s, s):
            image = image.resize((s, s), Image.LANCZOS)
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
        return {k: v.to(self.model.device) for k, v in inputs.items()}


class FakeScorer(BaseScorer):
    """No GPU: random but repeatable scores. Only for testing the plumbing."""
    name = "fake"
    model_id = "fake"

    def score(self, image, prompt, n_generate=0):
        rnd = random.Random(zlib.crc32((prompt + str(image.size)).encode()))
        lp = [math.log(rnd.random() + 1e-3) - 2 for _ in EMOTIONS]
        best = EMOTIONS[max(range(7), key=lambda i: lp[i])]
        return {"label_logprobs": dict(zip(EMOTIONS, lp)), "mass": float(math.exp(logsumexp(lp))),
                "n_input_tokens": 100, "text": best if n_generate else None}

    def generate_text(self, image, prompt, max_new_tokens):
        return "Numbers: 1, 2"


def build_backend(args, tokenizer_only=False):
    if args.model == "llava15":
        return LlavaScorer(load_4bit=not args.no_4bit, tokenizer_only=tokenizer_only)
    if args.model == "qwen25vl":
        return QwenScorer(load_4bit=not args.no_4bit, dtype=args.dtype, side=args.qwen_side,
                          tokenizer_only=tokenizer_only)
    if args.model == "fake":
        return FakeScorer()
    raise ValueError(args.model)


# ------------------------------------------------------------------ main
def run_probe(be, jobs, root, out_path, n_max):
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
        if len(seen) > n_max:
            break
        img = load_image(root / j["image_relpath"])
        txt = be.generate_text(img, prompt, 40)
        said = {int(x) for x in re.findall(r"\d+", txt)}
        exp = set(j["expected_face_ids"])
        row = {"image_key": j["image_key"], "condition": j["condition"], "expected": len(exp),
               "read_correctly": len(said & exp), "invented": len(said - exp), "output": txt,
               **be.info()}
        append_jsonl(out_path, row)
        rows.append(row)
    tot = sum(r["expected"] for r in rows)
    ok = sum(r["read_correctly"] for r in rows)
    print(f"probe: read {ok}/{tot} tags correctly ({(ok / tot * 100) if tot else 0:.1f}%), "
          f"invented {sum(r['invented'] for r in rows)} | file: {out_path}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay_root")
    ap.add_argument("--jobs")
    ap.add_argument("--model", default="llava15", choices=["llava15", "qwen25vl", "fake"])
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--conditions", default=None, help="comma list, e.g. plain,sov")
    ap.add_argument("--variants", default=None, help="comma list, e.g. A,B")
    ap.add_argument("--query_types", default=None, help="comma list: center,numbered")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--also_generate", action="store_true",
                    help="also generate a short answer (for the agreement check)")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--probe_n", type=int, default=100)
    ap.add_argument("--check_tokens", action="store_true")
    ap.add_argument("--no_4bit", action="store_true")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--qwen_side", type=int, default=336)
    args, _ = ap.parse_known_args(argv)

    if args.check_tokens:
        be = build_backend(args, tokenizer_only=True)
        for lab, ids in zip(EMOTIONS, be.cand_ids):
            print(f"{lab:9s} ids={ids} tokens={be.tok.convert_ids_to_tokens(ids)}")
        print("OK: no label shares a first token with another label.")
        return
    if not args.overlay_root or not args.jobs:
        ap.error("--overlay_root and --jobs are required")

    root, jobs_path = Path(args.overlay_root), Path(args.jobs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = read_jsonl(jobs_path)
    if args.conditions:
        jobs = [j for j in jobs if j["condition"] in args.conditions.split(",")]
    if args.variants:
        jobs = [j for j in jobs if j["prompt_set"] in args.variants.split(",")]
    if args.query_types:
        jobs = [j for j in jobs if j["query_type"] in args.query_types.split(",")]
    if args.limit:
        jobs = jobs[:args.limit]
    if not jobs:
        sys.exit("No jobs after filtering.")
    missing = sorted({j["image_relpath"] for j in jobs if not (root / j["image_relpath"]).exists()})
    if missing:
        sys.exit("Missing images, e.g. " + ", ".join(missing[:3]))
    print(f"{len(jobs)} job(s) | model={args.model} | out_dir={out_dir}")

    be = build_backend(args)
    if args.probe:
        run_probe(be, jobs, root, out_dir / f"probe_{args.model}.jsonl", args.probe_n)
        return

    res_path = out_dir / f"results_{args.model}_{jobs_path.stem}.jsonl"
    err_path = out_dir / "errors.jsonl"
    ensure_trailing_newline(res_path)
    done = {r["job_id"] for r in read_jsonl(res_path) if r.get("error") is None}
    todo = [j for j in jobs if j["job_id"] not in done]
    print(f"{len(done)} already done, {len(todo)} to run.")

    cache = {"rel": None, "img": None}
    n_ok = n_err = 0
    secs = []
    t_start = time.time()
    try:
        for i, j in enumerate(todo, 1):
            t0 = time.time()
            try:
                if cache["rel"] != j["image_relpath"]:
                    cache.update(rel=j["image_relpath"], img=load_image(root / j["image_relpath"]))
                g = be.score(cache["img"], j["prompt"], 6 if args.also_generate else 0)
                lp = g["label_logprobs"]
                pred = max(lp, key=lp.get)
                rec = {"job_id": j["job_id"], "row_id": j["row_id"], "image_key": j["image_key"],
                       "condition": j["condition"], "prompt_set": j["prompt_set"],
                       "query_type": j["query_type"], "face_id": j["face_id"],
                       "n_faces_final": j["n_faces_final"], "label": j["label"],
                       "pred": pred, "label_logprobs": lp, "mass": round(g["mass"], 5),
                       "generated": g["text"], "n_input_tokens": g["n_input_tokens"],
                       "seconds": round(time.time() - t0, 3), "created_utc": utc_now(),
                       "error": None, **be.info()}
                append_jsonl(res_path, rec)
                n_ok += 1
                secs.append(rec["seconds"])
                if i % 25 == 0 or i == len(todo):
                    avg = sum(secs[-50:]) / len(secs[-50:])
                    print(f"[{i}/{len(todo)}] {avg:.2f}s/job (last 50) | "
                          f"~{(len(todo) - i) * avg / 3600:.2f} h left")
            except Exception as e:
                n_err += 1
                print(f"FAILED {j['job_id']}: {e}")
                append_jsonl(err_path, {"time_utc": utc_now(), "job_id": j["job_id"],
                                        "error": repr(e), "traceback": traceback.format_exc()})
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
    except KeyboardInterrupt:
        print("Interrupted. Finished jobs are saved; re-run the same command to resume.")
    finally:
        if secs:
            w = secs[3:] if len(secs) > 10 else secs   # skip warm-up
            avg = sum(w) / len(w)
            print(f"\nDone: ok={n_ok} failed={n_err} in {time.time() - t_start:.0f}s | "
                  f"mean {avg:.2f}s/job (after warm-up)")
            for n in (1000, 6000):
                print(f"   projection: {n} jobs ~ {n * avg / 3600:.1f} h")
        print("results:", res_path)


if __name__ == "__main__":
    main()
