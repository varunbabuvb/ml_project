#!/usr/bin/env python3
"""
Day 1, step 1 -- run on your LAPTOP, as a script from a terminal
(not inside Jupyter on Windows, because it uses multiprocessing).

What it does
  1. Reads ExpW label.lst, drops duplicate rows, keeps faces with
     min_conf < confidence <= max_conf (default 60 < conf <= 100).
  2. Crops every eligible face (with 25% padding, like FairFace's own
     preprocessing) and saves it as crops/<row_id>.jpg
  3. Writes manifest.csv (one row per face, with the quality flags)
  4. Zips crops + manifest into expw_crops.zip for upload to Drive / Kaggle.

Example (Windows; keep the quotes because of the spaces in the path):
  python 01_crop_faces.py ^
     --label_lst "C:\\Users\\VARUN\\exp w dataset\\label.lst" ^
     --img_dir   "C:\\Users\\VARUN\\exp w dataset\\origin" ^
     --out_dir   "C:\\Users\\VARUN\\exp w dataset\\full_crops"

Quick test on 200 faces first:  add  --limit 200
"""
import argparse
import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from PIL import Image

COLS = ["image_name", "face_id_in_image", "face_box_top", "face_box_left",
        "face_box_right", "face_box_bottom", "face_box_confidence",
        "expression_label"]
NAMES = {0: "angry", 1: "disgust", 2: "fear", 3: "happy",
         4: "sad", 5: "surprise", 6: "neutral"}
# Counts you reported earlier (conf > 60 and <= 100); used only as a sanity check.
EXPECTED = {"angry": 1477, "disgust": 1404, "fear": 375, "happy": 11164,
            "sad": 2919, "surprise": 2808, "neutral": 9399}


def crop_one_image(task):
    """Open one image once and crop every labelled face in it."""
    img_path, rows, crops_dir, pad, tiny_px, jpeg_q = task
    results = []
    try:
        im = Image.open(img_path).convert("RGB")
    except Exception:
        return [{"row_id": r["row_id"], "crop_status": "image_open_failed"}
                for r in rows]

    W, H = im.size
    for r in rows:
        left, top = r["face_box_left"], r["face_box_top"]
        right, bottom = r["face_box_right"], r["face_box_bottom"]
        w, h = right - left, bottom - top
        rec = {"row_id": r["row_id"], "image_width": W, "image_height": H}
        rec["bbox_touches_image_edge"] = bool(
            left <= 0 or top <= 0 or right >= W or bottom >= H)
        rec["bbox_outside_image"] = bool(
            w <= 0 or h <= 0 or right <= 0 or bottom <= 0
            or left >= W or top >= H)
        rec["tiny_bbox"] = bool(min(w, h) < tiny_px)

        if w <= 0 or h <= 0:
            rec.update(crop_status="invalid_box", crop_touches_image_edge=True,
                       crop_width=0, crop_height=0)
            results.append(rec)
            continue

        px, py = int(round(pad * w)), int(round(pad * h))
        x0, y0, x1, y1 = left - px, top - py, right + px, bottom + py
        # True when the padded crop had to be cut by the image border
        rec["crop_touches_image_edge"] = bool(x0 < 0 or y0 < 0 or x1 > W or y1 > H)
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)

        if cx1 - cx0 < 2 or cy1 - cy0 < 2:
            rec.update(crop_status="empty_crop", crop_width=0, crop_height=0)
            results.append(rec)
            continue

        crop = im.crop((cx0, cy0, cx1, cy1))
        crop.save(os.path.join(crops_dir, f"{r['row_id']:06d}.jpg"),
                  quality=jpeg_q)
        rec.update(crop_status="ok", crop_width=crop.width,
                   crop_height=crop.height)
        results.append(rec)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_lst", required=True)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_conf", type=float, default=60.0)
    ap.add_argument("--max_conf", type=float, default=100.0)
    ap.add_argument("--pad", type=float, default=0.25)
    ap.add_argument("--tiny_px", type=int, default=48)
    ap.add_argument("--jpeg_quality", type=int, default=95)
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0,
                    help="only process the first N eligible faces (testing)")
    args = ap.parse_args()

    t0 = time.time()
    crops_dir = os.path.join(args.out_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    # ---------- 1. read + clean the label file ----------
    df = pd.read_csv(args.label_lst, sep=r"\s+", names=COLS, header=None)
    n_raw = len(df)
    df = df.drop_duplicates()
    n_after_dup = len(df)
    df = df.drop_duplicates(["image_name", "face_id_in_image"])
    n_after_key = len(df)
    df = df[(df.face_box_confidence > args.min_conf)
            & (df.face_box_confidence <= args.max_conf)]
    df = df[df.expression_label.isin(NAMES.keys())]
    df = df.sort_values(["image_name", "face_id_in_image"]).reset_index(drop=True)
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=42)
        df = df.sort_values(["image_name", "face_id_in_image"]).reset_index(drop=True)
    df["row_id"] = range(len(df))
    df["expression_name"] = df.expression_label.map(NAMES)

    print(f"raw rows: {n_raw} | after exact-duplicate removal: {n_after_dup} "
          f"| after (image, face) dedup: {n_after_key}")
    print(f"eligible faces ({args.min_conf} < conf <= {args.max_conf}): {len(df)}")
    counts = df.expression_name.value_counts()
    if not args.limit:
        print("class counts vs your earlier numbers:")
        for k, v in EXPECTED.items():
            print(f"  {k:9s} {int(counts.get(k, 0)):6d}   (earlier: {v})")

    # ---------- 2. crop (one task per image, run in parallel) ----------
    tasks = []
    missing_images = 0
    for image_name, g in df.groupby("image_name", sort=False):
        path = os.path.join(args.img_dir, image_name)
        if not os.path.exists(path):
            missing_images += 1
        tasks.append((path, g.to_dict("records"), crops_dir, args.pad,
                      args.tiny_px, args.jpeg_quality))
    print(f"images to open: {len(tasks)} (missing on disk: {missing_images}) "
          f"| workers: {args.workers}")

    results = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(crop_one_image, tasks, chunksize=32):
            results.extend(res)
            done += 1
            if done % 2000 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} images done "
                      f"({time.time() - t0:.0f}s)")

    # ---------- 3. manifest ----------
    res_df = pd.DataFrame(results)
    manifest = df.merge(res_df, on="row_id", how="left")
    for c in ["bbox_touches_image_edge", "bbox_outside_image", "tiny_bbox",
              "crop_touches_image_edge"]:
        manifest[c] = manifest[c].fillna(True).astype(bool)
    manifest["crop_width"] = manifest["crop_width"].fillna(0).astype(int)
    manifest["crop_height"] = manifest["crop_height"].fillna(0).astype(int)
    manifest["crop_file"] = manifest.row_id.map(lambda i: f"crops/{i:06d}.jpg")
    manifest.loc[manifest.crop_status != "ok", "crop_file"] = ""
    small = manifest[["crop_width", "crop_height"]].min(axis=1) < 64
    manifest["flagged"] = (manifest.bbox_outside_image
                           | manifest.crop_touches_image_edge | small
                           | (manifest.crop_status != "ok"))
    man_path = os.path.join(args.out_dir, "manifest.csv")
    manifest.to_csv(man_path, index=False)

    # ---------- 4. zip for upload ----------
    zip_path = os.path.join(args.out_dir, "expw_crops.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
        z.write(man_path, "manifest.csv")
        for f in manifest.loc[manifest.crop_status == "ok", "crop_file"]:
            z.write(os.path.join(args.out_dir, f), f)

    # ---------- summary ----------
    print("\n=== SUMMARY ===")
    print("crop_status:", manifest.crop_status.value_counts().to_dict())
    print("flags: bbox_outside", int(manifest.bbox_outside_image.sum()),
          "| crop_touches_edge", int(manifest.crop_touches_image_edge.sum()),
          "| side<64px", int(small.sum()),
          "| tiny_bbox(<%dpx)" % args.tiny_px, int(manifest.tiny_bbox.sum()))
    print("flagged total:", int(manifest.flagged.sum()),
          "| clean:", int((~manifest.flagged).sum()))
    print("manifest:", man_path)
    print("zip     :", zip_path,
          f"({os.path.getsize(zip_path) / 1e6:.0f} MB)")
    print(f"total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
