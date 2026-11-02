#!/usr/bin/env python3
"""
Day 1, step 2 -- run FairFace (pretrained ResNet-34, 7-race model) on all crops.

The SAME file works in Google Colab and Kaggle; only the paths you pass differ.
Run it from a notebook cell with:   !python 02_fairface_all.py --crops_zip ... 

Colab  (after mounting Drive):
  !python 02_fairface_all.py \
      --crops_zip /content/drive/MyDrive/expw/expw_crops.zip \
      --model_pt  /content/drive/MyDrive/expw/res34_fair_align_multi_7_20190809.pt \
      --out_csv   /content/drive/MyDrive/expw/fairface_all.csv

Kaggle (zip and .pt added as private Datasets; output goes to /kaggle/working):
  !python 02_fairface_all.py \
      --crops_zip /kaggle/input/expw-crops/expw_crops.zip \
      --model_pt  /kaggle/input/fairface-model/res34_fair_align_multi_7_20190809.pt \
      --out_csv   /kaggle/working/fairface_all.csv

Quick test first: add  --limit 300
Optional cross-check against your pilot run: add  --pilot_csv <old fairface_predictions.csv>
"""
import argparse
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# Output order of the FairFace 18-way head: 7 race | 2 gender | 9 age
RACE = ["White", "Black", "Latino_Hispanic", "East Asian",
        "Southeast Asian", "Indian", "Middle Eastern"]
GENDER = ["Male", "Female"]
AGE = ["0-2", "3-9", "10-19", "20-29", "30-39",
       "40-49", "50-59", "60-69", "70+"]


class CropDataset(Dataset):
    def __init__(self, manifest, root, tfm):
        self.m, self.root, self.tfm = manifest.reset_index(drop=True), root, tfm

    def __len__(self):
        return len(self.m)

    def __getitem__(self, i):
        f = self.m.at[i, "crop_file"]
        try:
            im = Image.open(os.path.join(self.root, f)).convert("RGB")
            return self.tfm(im), True, i
        except Exception:
            return torch.zeros(3, 224, 224), False, i


def load_model(path, device):
    model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 18)
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)  # strict: fails loudly on a wrong file
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops_zip", required=True)
    ap.add_argument("--model_pt", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--workdir", default="/tmp/expw_full")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pilot_csv", default="")
    args = ap.parse_args()

    env = ("Kaggle" if os.path.exists("/kaggle/working")
           else "Colab" if os.path.exists("/content") else "other")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"env: {env} | device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    if device == "cpu":
        print("WARNING: no GPU found; this will be slow. Turn on a GPU runtime.")

    # ---- unzip crops to local disk (the zip itself may be read-only) ----
    t0 = time.time()
    marker = os.path.join(args.workdir, "manifest.csv")
    if not os.path.exists(marker):
        os.makedirs(args.workdir, exist_ok=True)
        with zipfile.ZipFile(args.crops_zip) as z:
            z.extractall(args.workdir)
    print(f"crops ready in {args.workdir} ({time.time() - t0:.0f}s)")

    manifest = pd.read_csv(marker)
    if args.limit:
        manifest = manifest.sample(n=min(args.limit, len(manifest)),
                                   random_state=42).reset_index(drop=True)
    manifest["crop_file"] = manifest["crop_file"].fillna("")
    print(f"manifest rows: {len(manifest)}")

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = CropDataset(manifest, args.workdir, tfm)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=(device == "cuda"))
    model = load_model(args.model_pt, device)
    print("FairFace checkpoint loaded (strict).")

    probs = np.zeros((len(ds), 18), dtype=np.float32)
    ok = np.zeros(len(ds), dtype=bool)
    t1 = time.time()
    with torch.inference_mode():
        for n, (x, good, idx) in enumerate(dl):
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=(device == "cuda")):
                out = model(x)
            out = out.float().cpu()
            p = torch.cat([out[:, :7].softmax(1), out[:, 7:9].softmax(1),
                           out[:, 9:18].softmax(1)], dim=1).numpy()
            probs[idx.numpy()] = p
            ok[idx.numpy()] = good.numpy()
            if n % 20 == 0:
                print(f"  batch {n + 1}/{len(dl)} ({time.time() - t1:.0f}s)")
    print(f"inference done in {time.time() - t1:.0f}s")

    # ---- assemble the output table ----
    res = manifest.copy()
    res["prediction_status"] = np.where(ok, "ok", "failed")
    pr, pg, pa = probs[:, :7], probs[:, 7:9], probs[:, 9:18]
    res["race_prediction"] = np.array(RACE)[pr.argmax(1)]
    res["race_confidence"] = pr.max(1)
    res["gender_prediction"] = np.array(GENDER)[pg.argmax(1)]
    res["gender_confidence"] = pg.max(1)
    res["age_group_prediction"] = np.array(AGE)[pa.argmax(1)]
    res["age_group_confidence"] = pa.max(1)
    for j, name in enumerate(RACE):
        res[f"race_prob_{name}"] = pr[:, j]
    for j, name in enumerate(GENDER):
        res[f"gender_prob_{name}"] = pg[:, j]
    for j, name in enumerate(AGE):
        res[f"age_prob_{name}"] = pa[:, j]
    bad = ~ok
    res.loc[bad, ["race_prediction", "gender_prediction",
                  "age_group_prediction"]] = np.nan
    res.to_csv(args.out_csv, index=False)

    # ---- summary ----
    print("\n=== FAIRFACE SUMMARY ===")
    print("status:", res.prediction_status.value_counts().to_dict())
    g = res[res.prediction_status == "ok"]
    print("\nrace:\n", g.race_prediction.value_counts().to_string())
    print("\ngender:\n", g.gender_prediction.value_counts().to_string())
    print("\nage:\n", g.age_group_prediction.value_counts().to_string())
    for c in ["race_confidence", "gender_confidence", "age_group_confidence"]:
        print(f"{c}: mean {g[c].mean():.3f} | <0.5: {(g[c] < 0.5).sum()}")
    if "flagged" in g:
        print("flagged crops:", int(g.flagged.sum()),
              "| clean:", int((~g.flagged).sum()))
    print("saved:", args.out_csv)

    # ---- optional: compare with the old pilot run ----
    if args.pilot_csv and os.path.exists(args.pilot_csv):
        pil = pd.read_csv(args.pilot_csv)
        keys = ["image_name", "face_id_in_image"]
        mrg = g.merge(pil[keys + ["race_prediction", "gender_prediction",
                                  "age_group_prediction"]],
                      on=keys, suffixes=("", "_pilot"))
        print(f"\npilot cross-check on {len(mrg)} shared faces:")
        for c in ["race_prediction", "gender_prediction",
                  "age_group_prediction"]:
            print(f"  {c} agreement: "
                  f"{(mrg[c] == mrg[c + '_pilot']).mean() * 100:.1f}%")
        print("  (a very low agreement, especially for gender, would point to "
              "a label-order or preprocessing mismatch)")


if __name__ == "__main__":
    sys.exit(main())
