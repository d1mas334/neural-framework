#!/usr/bin/env python3
"""
Download balanced real/fake subset from:
Rajarshi-Roy-research/Defactify_Image_Dataset

Expected columns:
- Image   (image)
- Label_A (int; 0=real, 1=fake)
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Defactify subset into real/fake folders")
    p.add_argument("--repo", default="Rajarshi-Roy-research/Defactify_Image_Dataset")
    p.add_argument("--split", default="train")
    p.add_argument("--out", type=Path, default=Path("data/binary_images"))
    p.add_argument("--max-per-class", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image-col", default="Image")
    p.add_argument("--label-col", default="Label_A")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    print(f"Loading dataset {args.repo} [{args.split}] ...")
    ds = load_dataset(args.repo, split=args.split)
    print(f"Rows available: {len(ds)}")

    missing = [c for c in (args.image_col, args.label_col) if c not in ds.column_names]
    if missing:
        raise SystemExit(f"Missing columns: {missing}. Available: {ds.column_names}")

    real_idx = []
    fake_idx = []
    for i in range(len(ds)):
        lbl = int(ds[i][args.label_col])
        if lbl == 0:
            real_idx.append(i)
        elif lbl == 1:
            fake_idx.append(i)

    if not real_idx or not fake_idx:
        raise SystemExit("Could not find both classes 0 and 1 in label column.")

    random.shuffle(real_idx)
    random.shuffle(fake_idx)
    n = min(len(real_idx), len(fake_idx), args.max_per_class)

    out_real = args.out / "real"
    out_fake = args.out / "fake"
    if out_real.exists():
        shutil.rmtree(out_real)
    if out_fake.exists():
        shutil.rmtree(out_fake)
    out_real.mkdir(parents=True, exist_ok=True)
    out_fake.mkdir(parents=True, exist_ok=True)

    for k in range(n):
        r = ds[real_idx[k]][args.image_col]
        f = ds[fake_idx[k]][args.image_col]
        r.save(out_real / f"{k + 1:06d}.png")
        f.save(out_fake / f"{k + 1:06d}.png")
        if (k + 1) % 200 == 0:
            print(f"Saved {k + 1}/{n} pairs...")

    print("Done.")
    print(f"Saved real: {n} -> {out_real}")
    print(f"Saved fake: {n} -> {out_fake}")


if __name__ == "__main__":
    main()

