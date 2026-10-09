"""
SoV pipeline - Stage 3 (overlap handling) + Stage 4 (numbering) + Stage 5 (overlays)
=====================================================================================

Reads the detections written by sov_stage1_2_detect.py and, for every image:

  3. Overlap handling (paper's Algorithm 1 / Eq. 4-7): sort faces by box area,
     largest first; keep a face unless it overlaps an already-kept (larger) face by
     more than EPS, where overlap = intersection_area / min(area_a, area_b).
  4. Numbering: kept faces get IDs 1..n from left to right (by box centre x).
     (The paper does not specify an order; this is our choice.)
  5. Renders the paper's four conditions:
        plain        original image, untouched
        box          boxes only                          (uniform colour)
        box_number   boxes + face numbers                (colour per face)
        sov          boxes + numbers + 5 landmarks       (full SoV)
     plus debug_dropped.png showing which faces were removed and why.

Outputs (under --output):
    final/<key>.json                kept faces (with sov_id), dropped faces + reasons
    overlays/<key>/plain|box|box_number|sov.png
    overlays/<key>/debug_dropped.png
    summary.csv, logs/run.log, logs/errors.jsonl, run_history.jsonl

Same safety rules as stage 1-2: atomic writes, resume by skipping finished images,
a failing image never stops the run, nothing is silently overwritten (.prev.json).
Originals are only read, never modified.

Usage
-----
    python sov_stage3_5_overlay.py --input ./images --detections ./out/detections \
           --output ./out_overlay --eps 0.4
Notebook:
    from sov_stage3_5_overlay import run
    run("/content/images", "/content/output_folder/detections", "/content/output_overlay")
"""

import os
import sys
import csv
import json
import time
import hashlib
import argparse
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps, ImageFont

SCHEMA_VERSION = 1
STYLE_VERSION = 1          # bump if the drawing style changes (forces re-render)
CONDITIONS = ("plain", "box", "box_number", "sov")
log = logging.getLogger("sov3")

# ---- drawing style --------------------------------------------------------------
BOX_ONLY_COLOR = (255, 165, 0)   # uniform orange for the "box" condition
PALETTE = [(255, 64, 64), (64, 160, 255), (60, 200, 90), (255, 200, 0), (200, 80, 255),
           (255, 128, 0), (0, 210, 210), (255, 90, 180), (150, 220, 0), (120, 120, 255),
           (0, 180, 120), (230, 230, 60)]
LM_COLORS = {"eye_img_left": (0, 255, 255), "eye_img_right": (0, 128, 255),
             "nose": (255, 255, 0), "mouth_img_left": (255, 0, 255),
             "mouth_img_right": (255, 128, 0)}


# ---------------------------------------------------------------------------
# Utilities (kept local so this file is standalone)
# ---------------------------------------------------------------------------
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


def load_image(path):
    """PIL RGB image with EXIF rotation applied (same rule as stage 1-2)."""
    with Image.open(path) as im:
        im.load()
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")


def _font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Stage 3: overlap handling
# ---------------------------------------------------------------------------
def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def overlap_ratio(a, b):
    """intersection area / area of the smaller box (paper Eq. 5)."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    m = min(area(a), area(b))
    return (iw * ih) / m if m > 0 else 0.0


def handle_overlaps(faces, eps):
    """Greedy, largest-first: a face is dropped if it overlaps a kept face by > eps."""
    ordered = sorted(faces, key=lambda f: (-area(f["box"]), f["box"][0]))
    kept, dropped = [], []
    for f in ordered:
        clash = None  # (raw_id of the kept face, overlap ratio)
        for k in kept:
            r = overlap_ratio(f["box"], k["box"])
            if r > eps and (clash is None or r > clash[1]):
                clash = (k["raw_id"], r)
        if clash is None:
            kept.append(f)
        else:
            dropped.append({"raw_id": f["raw_id"], "box": f["box"], "score": f["score"],
                            "overlaps_kept_raw_id": clash[0],
                            "overlap_ratio": round(clash[1], 4)})
    return kept, dropped


# ---------------------------------------------------------------------------
# Stage 4: numbering (left to right by box centre x; ties by centre y)
# ---------------------------------------------------------------------------
def number_faces(kept):
    def centre(f):
        b = f["box"]
        return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
    out = []
    for i, f in enumerate(sorted(kept, key=centre), 1):
        g = dict(f)
        g["sov_id"] = i
        out.append(g)
    return out


# ---------------------------------------------------------------------------
# Stage 5: rendering
# ---------------------------------------------------------------------------
def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _text_colour(rgb):
    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return (0, 0, 0) if lum > 140 else (255, 255, 255)


def _draw_tag(d, img_size, box, text, colour, thickness):
    x1, y1, x2, y2 = box
    fh = y2 - y1
    font = _font(int(_clamp(round(0.35 * fh), 12, 28)))
    l, t, r, b = d.textbbox((0, 0), text, font=font)
    tw, th, pad = r - l, b - t, 2
    tag_w, tag_h = tw + 2 * pad, th + 2 * pad
    tx = x1
    ty = y1 - tag_h - 1
    if ty < 0:                       # no room above -> put it inside the box corner
        ty = y1 + thickness
    tx = _clamp(tx, 0, max(0, img_size[0] - tag_w))
    d.rectangle([tx, ty, tx + tag_w, ty + tag_h], fill=colour)
    d.text((tx + pad - l, ty + pad - t), text, fill=_text_colour(colour), font=font)


def render(img, faces, condition):
    if condition == "plain":
        return img.copy()
    out = img.copy()
    d = ImageDraw.Draw(out)
    for f in faces:
        x1, y1, x2, y2 = f["box"]
        m = min(x2 - x1, y2 - y1)
        t = int(_clamp(round(0.05 * m), 2, 4))
        colour = BOX_ONLY_COLOR if condition == "box" else PALETTE[(f["sov_id"] - 1) % len(PALETTE)]
        d.rectangle([x1, y1, x2, y2], outline=colour, width=t)
        if condition == "sov":
            r = int(_clamp(round(0.045 * m), 2, 4))
            for name, p in f["landmarks"].items():
                if p:
                    c = LM_COLORS.get(name, (255, 0, 0))
                    d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r],
                              fill=c, outline=(0, 0, 0))
        if condition in ("box_number", "sov"):
            _draw_tag(d, out.size, f["box"], str(f["sov_id"]), colour, t)
    return out


def render_dropped(img, kept, dropped):
    """Debug picture: kept faces in green, dropped faces in red with the reason."""
    out = img.copy()
    d = ImageDraw.Draw(out)
    for f in kept:
        d.rectangle(f["box"], outline=(0, 255, 0), width=2)
    for x in dropped:
        d.rectangle(x["box"], outline=(255, 0, 0), width=3)
        font = _font(int(_clamp(round(0.4 * (x["box"][3] - x["box"][1])), 11, 28)))
        d.text((x["box"][0], max(0, x["box"][1] - 16)),
               f'drop {x["raw_id"]} (dup of {x["overlaps_kept_raw_id"]}, {x["overlap_ratio"]:.2f})',
               fill=(255, 60, 60), font=font)
    return out


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


def _all_outputs_exist(ov_dir):
    return all((ov_dir / f"{c}.png").exists() for c in CONDITIONS) and \
        (ov_dir / "debug_dropped.png").exists()


def _already_done(final_path, ov_dir, params):
    if not final_path.exists() or not _all_outputs_exist(ov_dir):
        return False
    try:
        with open(final_path, "r", encoding="utf-8") as f:
            rec = json.load(f)
        return rec.get("schema") == SCHEMA_VERSION and rec.get("params") == params
    except Exception:
        return False


def write_summary(final_dir, out_csv):
    rows = []
    for p in sorted(Path(final_dir).glob("*.json")):
        if p.name.endswith(".prev.json"):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                r = json.load(f)
            rows.append({"image": r["image"], "detected": r["num_detected"],
                         "dropped_overlap": len(r["dropped"]), "final_faces": r["num_final"],
                         "eps": r["params"]["eps"],
                         "image_changed_since_detection": r.get("image_sha256_mismatch", False)})
        except Exception as e:
            rows.append({"image": p.name, "detected": f"UNREADABLE: {e}"})
    cols = ["image", "detected", "dropped_overlap", "final_faces", "eps",
            "image_changed_since_detection"]
    tmp = Path(str(out_csv) + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out_csv)
    return rows


def run(input_dir, detections_dir, output_dir, eps=0.4, force=False, limit=None):
    input_dir, detections_dir, output_dir = Path(input_dir), Path(detections_dir), Path(output_dir)
    if not detections_dir.is_dir():
        raise FileNotFoundError(f"Detections folder not found: {detections_dir}")
    # forgiving: if given the stage 1-2 OUTPUT folder, use its 'detections' subfolder
    if not any(detections_dir.glob("*.json")) and (detections_dir / "detections").is_dir():
        detections_dir = detections_dir / "detections"
    final_dir, ov_root, log_dir = output_dir / "final", output_dir / "overlays", output_dir / "logs"
    for d in (final_dir, ov_root, log_dir):
        d.mkdir(parents=True, exist_ok=True)
    _setup_logging(log_dir)

    params = {"eps": eps, "style_version": STYLE_VERSION, "numbering": "left_to_right"}
    append_jsonl(output_dir / "run_history.jsonl", {
        "started_utc": utc_now(), "params": params, "force": force, "limit": limit,
        "python": sys.version.split()[0]})

    det_files = [p for p in sorted(detections_dir.glob("*.json")) if not p.name.endswith(".prev.json")]
    if not det_files:
        log.error("No detection JSON files in %s (run sov_stage1_2_detect.py first)", detections_dir)
        return []
    if limit:
        det_files = det_files[:limit]
    log.info("Found %d detection file(s). Settings: %s", len(det_files), params)

    n_ok = n_skip = n_err = 0
    try:
        for i, dp in enumerate(det_files, 1):
            key = dp.name[:-len(".json")]
            final_path, ov_dir = final_dir / f"{key}.json", ov_root / key
            if not force and _already_done(final_path, ov_dir, params):
                log.info("[%d/%d] skip (already done): %s", i, len(det_files), key)
                n_skip += 1
                continue
            t0 = time.time()
            try:
                with open(dp, "r", encoding="utf-8") as f:
                    det = json.load(f)
                img_path = input_dir / det["image"]
                if not img_path.exists():
                    raise FileNotFoundError(f"image not found: {img_path}")
                mismatch = sha256_file(img_path) != det.get("sha256")
                if mismatch:
                    log.warning("%s: image file changed since detection (sha256 differs)!", key)
                img = load_image(img_path)
                if list(img.size) != [det["width"], det["height"]]:
                    raise ValueError(f"image size {img.size} != detection size "
                                     f"{(det['width'], det['height'])}; re-run stage 1-2")

                kept, dropped = handle_overlaps(det["faces"], eps)
                faces = number_faces(kept)

                rec = {"schema": SCHEMA_VERSION, "image": det["image"], "params": params,
                       "image_sha256_mismatch": bool(mismatch),
                       "width": det["width"], "height": det["height"],
                       "num_detected": len(det["faces"]), "num_final": len(faces),
                       "faces": faces, "dropped": dropped, "created_utc": utc_now()}
                if final_path.exists():
                    os.replace(final_path, final_dir / f"{key}.prev.json")
                write_json_atomic(final_path, rec)   # data first...
                ov_dir.mkdir(parents=True, exist_ok=True)
                for c in CONDITIONS:                  # ...then pictures
                    save_png_atomic(render(img, faces, c), ov_dir / f"{c}.png")
                save_png_atomic(render_dropped(img, faces, dropped), ov_dir / "debug_dropped.png")
                log.info("[%d/%d] %s: %d detected -> %d final (%d dropped) in %.1fs",
                         i, len(det_files), key, len(det["faces"]), len(faces),
                         len(dropped), time.time() - t0)
                n_ok += 1
            except Exception as e:
                n_err += 1
                log.error("[%d/%d] FAILED %s: %s", i, len(det_files), key, e)
                append_jsonl(log_dir / "errors.jsonl", {
                    "time_utc": utc_now(), "detection_file": str(dp), "error": repr(e),
                    "traceback": traceback.format_exc()})
    except KeyboardInterrupt:
        log.warning("Interrupted. Finished results are saved; re-run to resume.")
    finally:
        rows = write_summary(final_dir, output_dir / "summary.csv")
        log.info("Done. ok=%d skipped=%d failed=%d | summary: %s",
                 n_ok, n_skip, n_err, output_dir / "summary.csv")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="SoV stage 3-5: overlaps, numbering, overlays.")
    ap.add_argument("--input", required=True, help="folder with the ORIGINAL images")
    ap.add_argument("--detections", required=True, help="the 'detections' folder from stage 1-2")
    ap.add_argument("--output", required=True, help="folder for results")
    ap.add_argument("--eps", type=float, default=0.4,
                    help="overlap threshold: drop a face if intersection/min-area > eps")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args, _ = ap.parse_known_args(argv)
    run(args.input, args.detections, args.output, args.eps, args.force, args.limit)


if __name__ == "__main__":
    main()