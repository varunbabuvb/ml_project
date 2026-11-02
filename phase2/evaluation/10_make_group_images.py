#!/usr/bin/env python3
"""
Step 10 (multi-face experiment) -- build the group-crop images.   Run on your LAPTOP
(it needs the original ExpW images).

For every image listed in multi_faces.csv (from step 9):
  1. cut the region reg_x0..reg_y1 chosen in step 9 (all target faces + margin),
  2. pad it to a square with the CLIP-mean grey,
  3. resize it to size x size (336),
  4. write the position of every target face in the new image.

Outputs (in --out_dir)
  images/<image_key>      one 336x336 image per original image  (feed this folder to stage 1-2)
  targets_view.csv        one row per target face: box in the 336 view + all columns of multi_faces.csv
  multi_group_images.zip  images/ + targets_view.csv, ready to upload

Usage
  python 10_make_group_images.py --multi_csv multiface\\multi_faces.csv --img_dir "C:\\...\\origin" --out_dir ctx_multi
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

PAD_FILL = (122, 116, 104)


def make_one(task):
    img_path, rows, images_dir, size, q = task
    r0 = rows[0]
    try:
        im = Image.open(img_path).convert("RGB")
    except Exception:
        return [{"row_id": r["row_id"], "status": "image_open_failed"} for r in rows]
    x0, y0, x1, y1 = int(r0["reg_x0"]), int(r0["reg_y0"]), int(r0["reg_x1"]), int(r0["reg_y1"])
    crop = im.crop((x0, y0, x1, y1))
    cw, ch = crop.size
    S = max(cw, ch)
    canvas = Image.new("RGB", (S, S), PAD_FILL)
    px, py = (S - cw) // 2, (S - ch) // 2
    canvas.paste(crop, (px, py))
    scale = size / float(S)
    canvas = canvas.resize((size, size), Image.LANCZOS)
    canvas.save(os.path.join(images_dir, r0["image_key"]), quality=q)
    out = []
    for r in rows:
        tb = [(r["face_box_left"] - x0 + px) * scale, (r["face_box_top"] - y0 + py) * scale,
              (r["face_box_right"] - x0 + px) * scale, (r["face_box_bottom"] - y0 + py) * scale]
        tb = [min(max(v, 0.0), float(size)) for v in tb]
        out.append({"row_id": r["row_id"], "status": "ok", "tx1": tb[0], "ty1": tb[1],
                    "tx2": tb[2], "ty2": tb[3],
                    "face_px_check": min(tb[2] - tb[0], tb[3] - tb[1])})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multi_csv", required=True)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--size", type=int, default=336)
    ap.add_argument("--jpeg_quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (testing)")
    args = ap.parse_args()

    t0 = time.time()
    df = pd.read_csv(args.multi_csv)
    if args.limit:
        keep = sorted(df.image_id.unique())[:args.limit]
        df = df[df.image_id.isin(keep)]
    images_dir = os.path.join(args.out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    tasks = [(os.path.join(args.img_dir, name), g.to_dict("records"), images_dir, args.size,
              args.jpeg_quality) for name, g in df.groupby("image_name", sort=False)]
    print(f"{len(df)} target faces in {len(tasks)} images | size={args.size} workers={args.workers}")
    res = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(make_one, tasks, chunksize=16), 1):
            res.extend(r)
            if i % 250 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} images ({time.time() - t0:.0f}s)")

    tv = df.merge(pd.DataFrame(res), on="row_id", how="left")
    bad = tv[tv.status != "ok"]
    tv = tv[tv.status == "ok"].copy()
    tv_path = os.path.join(args.out_dir, "targets_view.csv")
    tv.to_csv(tv_path, index=False)
    zip_path = os.path.join(args.out_dir, "multi_group_images.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
        z.write(tv_path, "targets_view.csv")
        for k in sorted(tv.image_key.unique()):
            z.write(os.path.join(images_dir, k), f"images/{k}")

    print("\n=== SUMMARY ===")
    print("ok faces:", len(tv), "| failed:", len(bad), (bad.status.value_counts().to_dict() if len(bad) else ""))
    print("images written:", tv.image_key.nunique())
    print("face size in the 336 view (px) quantiles 10/50/90:",
          tv.face_px_check.quantile([.1, .5, .9]).round(0).tolist())
    print("targets:", tv_path)
    print("zip    :", zip_path, f"({os.path.getsize(zip_path) / 1e6:.0f} MB)")
    print(f"time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
