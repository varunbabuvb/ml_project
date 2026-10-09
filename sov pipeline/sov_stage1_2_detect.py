"""
SoV pipeline - Stage 1 (image loading) + Stage 2 (face detection)
==================================================================

What this does
--------------
1. Loads every image in --input (EXIF-rotation fixed, converted to RGB).
2. Detects faces with RetinaFace (same detector as the SoV paper) and gets
   5 landmarks per face (eyes, nose, mouth corners).
3. Saves, for every image:
     <output>/detections/<image>.json   boxes, scores, landmarks, metadata
     <output>/viz/<image>.png           raw boxes + landmarks drawn for eyeballing
4. Writes <output>/summary.csv with face counts per image.

Crash / loss safety
-------------------
* Every JSON/PNG is written atomically (temp file -> os.replace), so a crash
  can never leave a half-written result.
* A result is written the moment an image finishes. Re-running the script
  skips images already done (same settings), so you can resume after a crash.
* One bad image never stops the run; the error + traceback go to
  <output>/logs/errors.jsonl and that image is retried on the next run.
* Old results are never overwritten silently: if you re-run with --force or
  different settings, the previous JSON is kept as <image>.prev.json.
* Every run appends its settings + library versions to run_history.jsonl.
* Original images are never modified. Their SHA-256 is stored so we can tell
  later if a file changed.

Usage (terminal)
----------------
    python sov_stage1_2_detect.py --check_env
    python sov_stage1_2_detect.py --input ./examples --output ./out --limit 1
    python sov_stage1_2_detect.py --input ./examples --output ./out

Usage (Colab / notebook)
------------------------
    from sov_stage1_2_detect import run, env_check
    env_check("retinaface")
    run("/content/examples", "/content/drive/MyDrive/sov_out")
"""

import os
import sys
import csv
import json
import time
import math
import hashlib
import argparse
import logging
import traceback
import platform
from datetime import datetime, timezone
from pathlib import Path
import importlib.metadata as md

# ---------------------------------------------------------------------------
# Environment setup. This MUST happen before TensorFlow is imported anywhere.
# ---------------------------------------------------------------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")           # quieter TF logs
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")   # don't grab all GPU memory
try:
    import tf_keras  # noqa: F401  (needed by retina-face on Keras-3 TensorFlow)
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
except Exception:
    pass

import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFont

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SCHEMA_VERSION = 1
log = logging.getLogger("sov")


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def pkg_version(name):
    try:
        return md.version(name)
    except Exception:
        return None


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def finite_or_none(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def write_json_atomic(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_png_atomic(img, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    img.save(tmp, format="PNG")
    os.replace(tmp, path)


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Stage 1: image listing + loading
# ---------------------------------------------------------------------------
def list_images(input_dir, output_dir=None):
    input_dir = Path(input_dir)
    out = []
    for p in sorted(input_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        rel_parts = p.relative_to(input_dir).parts
        if any(part.startswith(".") for part in rel_parts):
            continue
        if output_dir is not None and Path(output_dir).resolve() in p.resolve().parents:
            continue  # never treat our own outputs as inputs
        out.append(p)
    return out


def image_key(path, input_dir):
    """Unique, filesystem-safe name for an image (keeps subfolders distinct)."""
    rel = Path(path).relative_to(input_dir).as_posix()
    return rel.replace("/", "__")


def load_image(path):
    """Returns (PIL RGB image, info dict). Fixes EXIF rotation (phone photos!)."""
    with Image.open(path) as im:
        im.load()
        orient = None
        try:
            orient = im.getexif().get(0x0112)
        except Exception:
            pass
        rotated = orient not in (None, 1)
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")  # handles RGBA / palette / grayscale / CMYK
    return im, {"exif_orientation_applied": bool(rotated)}


# ---------------------------------------------------------------------------
# Landmark canonicalisation (backend independent)
# We name landmarks by IMAGE side (smaller x = "img_left"), because different
# libraries use different left/right conventions (subject's vs viewer's).
# ---------------------------------------------------------------------------
def _order_pair(pair):
    if pair is None or len(pair) != 2 or any(p is None for p in pair):
        return None
    a, b = sorted(pair, key=lambda p: p[0])
    return a, b


def canonical_landmarks(eyes, nose, mouth):
    e, m = _order_pair(eyes), _order_pair(mouth)
    return {
        "eye_img_left": e[0] if e else None,
        "eye_img_right": e[1] if e else None,
        "nose": nose,
        "mouth_img_left": m[0] if m else None,
        "mouth_img_right": m[1] if m else None,
    }


# ---------------------------------------------------------------------------
# Stage 2: detector backends
# ---------------------------------------------------------------------------
class RetinaFaceBackend:
    """RetinaFace via the `retina-face` pip package (what the SoV paper uses)."""
    name = "retinaface"

    def __init__(self, threshold):
        from retinaface import RetinaFace  # lazy import, so errors are clear
        self.RF = RetinaFace
        self.threshold = threshold
        if hasattr(RetinaFace, "build_model"):
            RetinaFace.build_model()  # downloads weights on first use

    def detect(self, rgb):
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])  # library expects BGR arrays
        res = self.RF.detect_faces(bgr, threshold=self.threshold)
        faces = []
        if not isinstance(res, dict):
            return faces  # "no faces" (or unexpected format) -> treated as empty
        for _, v in res.items():
            if not isinstance(v, dict) or "facial_area" not in v:
                continue
            lm = v.get("landmarks") or {}

            def pt(k):
                p = lm.get(k)
                return [float(p[0]), float(p[1])] if p is not None else None

            faces.append({
                "score": finite_or_none(v.get("score")),
                "box": [float(b) for b in v["facial_area"]],
                "landmarks": canonical_landmarks(
                    [pt("right_eye"), pt("left_eye")], pt("nose"),
                    [pt("mouth_right"), pt("mouth_left")]),
            })
        return faces


class MTCNNBackend:
    """OPTIONAL fallback if RetinaFace will not install. Deviates from the paper."""
    name = "mtcnn"

    def __init__(self, threshold):
        import torch
        from facenet_pytorch import MTCNN
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.m = MTCNN(keep_all=True, device=dev)
        self.threshold = threshold

    def detect(self, rgb):
        boxes, probs, points = self.m.detect(Image.fromarray(rgb), landmarks=True)
        faces = []
        if boxes is None:
            return faces
        for b, p, pts in zip(boxes, probs, points):
            if p is None or p < self.threshold:
                continue
            pts = [[float(x), float(y)] for x, y in pts]  # eyeL, eyeR, nose, mouthL, mouthR
            faces.append({
                "score": finite_or_none(p),
                "box": [float(v) for v in b],
                "landmarks": canonical_landmarks([pts[0], pts[1]], pts[2], [pts[3], pts[4]]),
            })
        return faces


BACKENDS = {"retinaface": RetinaFaceBackend, "mtcnn": MTCNNBackend}


def build_backend(name, threshold):
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend {name!r}. Choose from {list(BACKENDS)}")
    return BACKENDS[name](threshold)


def env_check(backend="retinaface", threshold=0.9):
    """Quick install check: prints versions, builds the detector, runs it on a blank image."""
    print("python:", sys.version.split()[0], "| platform:", platform.platform())
    for p in ("numpy", "pillow", "tensorflow", "tf-keras", "keras", "retina-face",
              "torch", "facenet-pytorch"):
        print(f"  {p}: {pkg_version(p)}")
    t0 = time.time()
    det = build_backend(backend, threshold)
    print(f"detector '{backend}' built in {time.time() - t0:.1f}s")
    blank = np.full((256, 256, 3), 127, dtype=np.uint8)
    out = det.detect(blank)
    print(f"blank-image test OK, faces found: {len(out)} (expected 0)")
    return det


# ---------------------------------------------------------------------------
# Post-processing of raw detections
# ---------------------------------------------------------------------------
def _area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def scale_and_clean(faces, inv_scale, width, height):
    """Map coords back to original image size, clamp to image, drop degenerate boxes,
    sort by area (largest first), attach raw_id and a landmark sanity flag."""
    cleaned, dropped = [], 0
    for f in faces:
        x1, y1, x2, y2 = [v * inv_scale for v in f["box"]]
        x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
        y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
        if (x2 - x1) < 2 or (y2 - y1) < 2:
            dropped += 1
            continue
        lms = {}
        for k, p in f["landmarks"].items():
            lms[k] = None if p is None else [p[0] * inv_scale, p[1] * inv_scale]
        # sanity: are all landmarks within the box (+15% margin)?
        mx, my = 0.15 * (x2 - x1), 0.15 * (y2 - y1)
        lm_ok = all(
            p is None or (x1 - mx <= p[0] <= x2 + mx and y1 - my <= p[1] <= y2 + my)
            for p in lms.values())
        cleaned.append({"score": f["score"], "box": [x1, y1, x2, y2],
                        "landmarks": lms, "landmarks_inside_box": bool(lm_ok)})
    cleaned.sort(key=lambda f: (-_area(f["box"]), f["box"][0]))
    for i, f in enumerate(cleaned):
        f["raw_id"] = i
    return cleaned, dropped


# ---------------------------------------------------------------------------
# Visualisation (raw detections, for eyeballing only)
# ---------------------------------------------------------------------------
def _font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def draw_raw(img, faces):
    out = img.copy()
    d = ImageDraw.Draw(out)
    w, h = out.size
    t = max(2, int(round(min(w, h) / 250)))
    fs = max(12, int(min(w, h) / 40))
    font = _font(fs)
    for f in faces:
        x1, y1, x2, y2 = f["box"]
        d.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=t)
        sc = f["score"]
        label = f'{f["raw_id"]}:{sc:.2f}' if sc is not None else f'{f["raw_id"]}'
        d.text((x1 + t, max(0, y1 - fs - 2)), label, fill=(255, 255, 0), font=font)
        r = max(2, t)
        for p in f["landmarks"].values():
            if p:
                d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=(255, 0, 0))
    return out


# ---------------------------------------------------------------------------
# Main run
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


def _already_done(det_path, params):
    if not det_path.exists():
        return False
    try:
        with open(det_path, "r", encoding="utf-8") as f:
            rec = json.load(f)
        return rec.get("schema") == SCHEMA_VERSION and rec.get("params") == params
    except Exception:
        return False  # corrupt / unreadable -> redo


def write_summary(det_dir, out_csv):
    rows = []
    for p in sorted(Path(det_dir).glob("*.json")):
        if p.name.endswith(".prev.json"):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                r = json.load(f)
            scores = [f["score"] for f in r["faces"] if f["score"] is not None]
            rows.append({
                "image": r["image"], "width": r["width"], "height": r["height"],
                "num_faces": r["num_faces"],
                "min_score": round(min(scores), 4) if scores else "",
                "landmarks_outside_box": sum(1 for f in r["faces"] if not f["landmarks_inside_box"]),
                "dropped_degenerate": r.get("dropped_degenerate", 0),
            })
        except Exception as e:
            rows.append({"image": p.name, "num_faces": f"UNREADABLE: {e}"})
    cols = ["image", "width", "height", "num_faces", "min_score",
            "landmarks_outside_box", "dropped_degenerate"]
    tmp = Path(str(out_csv) + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out_csv)
    return rows


def run(input_dir, output_dir, backend="retinaface", threshold=0.9,
        max_side=1600, force=False, limit=None):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    det_dir, viz_dir, log_dir = output_dir / "detections", output_dir / "viz", output_dir / "logs"
    for d in (det_dir, viz_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)
    _setup_logging(log_dir)

    params = {"backend": backend, "threshold": threshold, "max_side": max_side}
    append_jsonl(output_dir / "run_history.jsonl", {
        "started_utc": utc_now(), "params": params, "force": force, "limit": limit,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "versions": {p: pkg_version(p) for p in
                     ("numpy", "pillow", "tensorflow", "tf-keras", "retina-face",
                      "torch", "facenet-pytorch")},
    })

    images = list_images(input_dir, output_dir)
    if not images:
        log.error("No images (%s) found under %s", sorted(IMG_EXTS), input_dir)
        return []
    if limit:
        images = images[:limit]
    log.info("Found %d image(s). Settings: %s", len(images), params)

    det = build_backend(backend, threshold)
    n_ok = n_skip = n_err = 0

    try:
        for i, path in enumerate(images, 1):
            key = image_key(path, input_dir)
            det_path = det_dir / f"{key}.json"
            if not force and _already_done(det_path, params):
                log.info("[%d/%d] skip (already done): %s", i, len(images), key)
                n_skip += 1
                continue
            t0 = time.time()
            try:
                img, info = load_image(path)
                w, h = img.size
                s = 1.0
                det_img = img
                if max(w, h) > max_side:  # speed/memory guard for huge images
                    s = max_side / float(max(w, h))
                    det_img = img.resize((max(1, round(w * s)), max(1, round(h * s))),
                                         Image.LANCZOS)
                raw = det.detect(np.array(det_img))
                faces, dropped = scale_and_clean(raw, 1.0 / s, w, h)

                rec = {
                    "schema": SCHEMA_VERSION, "image": path.relative_to(input_dir).as_posix(),
                    "sha256": sha256_file(path), "width": w, "height": h,
                    **info, "detector": det.name, "params": params,
                    "detection_scale": s, "num_faces": len(faces), "faces": faces,
                    "dropped_degenerate": dropped,
                    "seconds": round(time.time() - t0, 2), "created_utc": utc_now(),
                }
                if det_path.exists():  # never silently overwrite an older result
                    os.replace(det_path, det_dir / f"{key}.prev.json")
                write_json_atomic(det_path, rec)  # results first...
                try:                               # ...picture second (can't cost us results)
                    save_png_atomic(draw_raw(img, faces), viz_dir / f"{key}.png")
                except Exception:
                    log.warning("viz failed for %s:\n%s", key, traceback.format_exc())
                log.info("[%d/%d] %s: %d face(s) in %.1fs%s", i, len(images), key,
                         len(faces), time.time() - t0,
                         "" if s == 1.0 else f" (detected at scale {s:.2f})")
                n_ok += 1
            except Exception as e:
                n_err += 1
                log.error("[%d/%d] FAILED %s: %s", i, len(images), key, e)
                append_jsonl(log_dir / "errors.jsonl", {
                    "time_utc": utc_now(), "image": str(path), "error": repr(e),
                    "traceback": traceback.format_exc()})
    except KeyboardInterrupt:
        log.warning("Interrupted by user. Finished results are saved; re-run to resume.")
    finally:
        rows = write_summary(det_dir, output_dir / "summary.csv")
        log.info("Done. ok=%d skipped=%d failed=%d | summary: %s",
                 n_ok, n_skip, n_err, output_dir / "summary.csv")
        for r in rows:
            log.info("  %-40s faces=%s", r["image"], r["num_faces"])
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="SoV stage 1+2: load images, detect faces.")
    ap.add_argument("--input", help="folder with images")
    ap.add_argument("--output", help="folder for results (use Google Drive on Colab)")
    ap.add_argument("--backend", default="retinaface", choices=list(BACKENDS))
    ap.add_argument("--threshold", type=float, default=0.9, help="min detection confidence")
    ap.add_argument("--max_side", type=int, default=1600,
                    help="downscale images larger than this (px) for detection only")
    ap.add_argument("--force", action="store_true", help="redo images even if already done")
    ap.add_argument("--limit", type=int, default=None, help="only first N images (smoke test)")
    ap.add_argument("--check_env", action="store_true", help="test install and exit")
    args, _ = ap.parse_known_args(argv)  # tolerant of notebook-injected args

    if args.check_env:
        env_check(args.backend, args.threshold)
        return
    if not args.input or not args.output:
        ap.error("--input and --output are required (or use --check_env)")
    run(args.input, args.output, args.backend, args.threshold,
        args.max_side, args.force, args.limit)


if __name__ == "__main__":
    main()