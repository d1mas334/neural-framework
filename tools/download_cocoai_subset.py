#!/usr/bin/env python3
"""
Download a subset from Hugging Face dataset NasrinImp/COCO_AI and save as:
  data/binary_images/real
  data/binary_images/fake

Real images are taken from column: coco_image
Fake images are taken from a generated column (e.g. sdxl_image, sd3_image, sd35_image, sd21_image, dalle_image, midjourney_image)
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download COCO_AI subset into real/fake folders")
    p.add_argument("--repo", default="NasrinImp/COCO_AI", help="HF dataset repo")
    p.add_argument("--split", default="train", help="Dataset split")
    p.add_argument("--out", type=Path, default=Path("data/binary_images"), help="Output folder")
    p.add_argument("--generated-column", default="sdxl_image", help="Generated image column name")
    p.add_argument("--max-samples", type=int, default=3000, help="Number of paired samples")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    print(f"Loading dataset {args.repo} [{args.split}] ...")
    ds = load_dataset(args.repo, split=args.split)
    total = len(ds)
    print(f"Rows available: {total}")

    required_cols = {"coco_image", args.generated_column}
    missing = [c for c in required_cols if c not in ds.column_names]
    if missing:
        raise SystemExit(f"Missing columns in dataset: {missing}. Available: {ds.column_names}")

    n = min(args.max_samples, total)
    indices = list(range(total))
    random.shuffle(indices)
    indices = indices[:n]

    out_real = args.out / "real"
    out_fake = args.out / "fake"
    if out_real.exists():
        shutil.rmtree(out_real)
    if out_fake.exists():
        shutil.rmtree(out_fake)
    out_real.mkdir(parents=True, exist_ok=True)
    out_fake.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i, idx in enumerate(indices, start=1):
        row = ds[idx]
        real_img = row["coco_image"]
        fake_img = row[args.generated_column]
        if real_img is None or fake_img is None:
            continue

        real_path = out_real / f"{i:06d}.png"
        fake_path = out_fake / f"{i:06d}.png"

        real_img.save(real_path)
        fake_img.save(fake_path)
        saved += 1

        if i % 200 == 0:
            print(f"Saved {i}/{n} pairs...")

    print("Done.")
    print(f"Saved pairs: {saved}")
    print(f"Real dir: {out_real}")
    print(f"Fake dir: {out_fake}")


if __name__ == "__main__":
    main()

