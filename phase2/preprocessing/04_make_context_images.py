#!/usr/bin/env python3
"""
Step 4 -- build the "context images" that go into the SoV pipeline.   Run on your LAPTOP
(it needs the original ExpW images). Run once for dev_set.csv and once for core_set.csv.

For every selected face:
  1. cut a square-ish region around the face: ctx x the larger side of the face box
     (default 2.75), clipped to the image,
  2. pad it to a square with the CLIP-mean grey that LLaVA-1.5 uses,
  3. resize it to exactly size x size (default 336, LLaVA-1.5's input size),
  4. remember where the target face is inside the new image (target box).
Because the image is already 336x336, the SoV boxes / numbers / landmarks that stage 3-5
draws later are drawn at the model's own resolution and survive the model's resize.
The same context image is used in ALL four conditions.

Outputs (in <out_dir>)
  images/<row_id>.jpg    the context images (feed these to sov_stage1_2_detect.py)
  targets.csv            one row per face: image key, target box in context-image
                         coordinates, size of the face in the model's view, + the set columns
  <set_name>_context_images.zip   images/ + targets.csv, ready to upload

Usage
  python 04_make_context_images.py --set_csv sets\\dev_set.csv  --img_dir "C:\\...\\origin" --out_dir ctx_dev
  python 04_make_context_images.py --set_csv sets\\core_set.csv --img_dir "C:\\...\\origin" --out_dir ctx_core
  (add --limit 20 for a quick test)
"""
import argparse
import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from PIL import Image

PAD_FILL = (122, 116, 104)  # CLIP mean colour, what LLaVA-1.5 pads with


def make_one_image(task):
    img_path, rows, images_dir, ctx, size, q = task
    try:
        im = Image.open(img_path).convert("RGB")
    except Exception:
        return [{"row_id": r["row_id"], "status": "image_open_failed"} for r in rows]
    W, H = im.size
    out = []
    for r in rows:
        l, t = r["face_box_left"], r["face_box_top"]
        rt, b = r["face_box_right"], r["face_box_bottom"]
        w, h = rt - l, b - t
        cx, cy = (l + rt) / 2.0, (t + b) / 2.0
        side = ctx * max(w, h)
        x0, y0, x1, y1 = cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2
        clipped = bool(x0 < 0 or y0 < 0 or x1 > W or y1 > H)
        rx0, ry0 = int(max(0, round(x0))), int(max(0, round(y0)))
        rx1, ry1 = int(min(W, round(x1))), int(min(H, round(y1)))
        if rx1 - rx0 < 2 or ry1 - ry0 < 2:
            out.append({"row_id": r["row_id"], "status": "empty_region"})
            continue
        crop = im.crop((rx0, ry0, rx1, ry1))
        cw, ch = crop.size
        S = max(cw, ch)
        canvas = Image.new("RGB", (S, S), PAD_FILL)
        px, py = (S - cw) // 2, (S - ch) // 2
        canvas.paste(crop, (px, py))
        scale = size / float(S)
        canvas = canvas.resize((size, size), Image.LANCZOS)
        # target face box in the new coordinates (clamped to the canvas)
        tb = [(l - rx0 + px) * scale, (t - ry0 + py) * scale,
              (rt - rx0 + px) * scale, (b - ry0 + py) * scale]
        tb = [min(max(v, 0.0), float(size)) for v in tb]
        key = f"{int(r['row_id']):06d}.jpg"
        canvas.save(os.path.join(images_dir, key), quality=q)
        out.append({"row_id": r["row_id"], "status": "ok", "key": key,
                    "tx1": tb[0], "ty1": tb[1], "tx2": tb[2], "ty2": tb[3],
                    "face_px_in_view": min(tb[2] - tb[0], tb[3] - tb[1]),
                    "ctx_scale": scale, "region_clipped_by_image": clipped,
                    "region_w": cw, "region_h": ch})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set_csv", required=True)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ctx", type=float, default=2.75)
    ap.add_argument("--size", type=int, default=336)
    ap.add_argument("--jpeg_quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    df = pd.read_csv(args.set_csv)
    if args.limit:
        df = df.head(args.limit)
    images_dir = os.path.join(args.out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    tasks = []
    for name, g in df.groupby("image_name", sort=False):
        tasks.append((os.path.join(args.img_dir, name), g.to_dict("records"),
                      images_dir, args.ctx, args.size, args.jpeg_quality))
    print(f"{len(df)} faces in {len(tasks)} images | ctx={args.ctx} size={args.size} "
          f"workers={args.workers}")

    res = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(make_one_image, tasks, chunksize=16), 1):
            res.extend(r)
            if i % 500 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} images ({time.time() - t0:.0f}s)")

    tg = df.merge(pd.DataFrame(res), on="row_id", how="left")
    bad = tg[tg.status != "ok"]
    tg = tg[tg.status == "ok"].copy()
    tg_path = os.path.join(args.out_dir, "targets.csv")
    tg.to_csv(tg_path, index=False)

    set_name = os.path.splitext(os.path.basename(args.set_csv))[0]
    zip_path = os.path.join(args.out_dir, f"{set_name}_context_images.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
        z.write(tg_path, "targets.csv")
        for k in tg.key:
            z.write(os.path.join(images_dir, k), f"images/{k}")

    print("\n=== SUMMARY ===")
    print("ok:", len(tg), "| failed:", len(bad), (bad.status.value_counts().to_dict() if len(bad) else ""))
    print("face size in the model's 336 view (px) quantiles 10/50/90:",
          tg.face_px_in_view.quantile([.1, .5, .9]).round(0).tolist())
    print("regions clipped by the image border: %.1f%%" % (tg.region_clipped_by_image.mean() * 100))
    print("targets:", tg_path)
    print("zip    :", zip_path, f"({os.path.getsize(zip_path) / 1e6:.0f} MB)")
    print(f"time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
