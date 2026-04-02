#!/usr/bin/env python3
"""
Build a CSV for binary classification:
label, f1, f2, ..., fN

label: 0 = real, 1 = generated
features: flattened grayscale pixels in [0, 1]
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from PIL import Image


VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(root: Path) -> list[Path]:
    files: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXT:
            files.append(p)
    return files


def image_to_features(path: Path, size: int) -> list[float]:
    with Image.open(path) as img:
        img = img.convert("L").resize((size, size))
        pixels = list(img.getdata())
    return [px / 255.0 for px in pixels]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build generated-vs-real CSV for Lua framework")
    parser.add_argument("--real-dir", required=True, type=Path, help="Directory with real photos")
    parser.add_argument("--fake-dir", required=True, type=Path, help="Directory with generated photos")
    parser.add_argument("--out", required=True, type=Path, help="Output CSV path")
    parser.add_argument("--size", type=int, default=28, help="Image side after resize (default: 28)")
    parser.add_argument("--max-per-class", type=int, default=0, help="0 means use all")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    real_files = collect_images(args.real_dir)
    fake_files = collect_images(args.fake_dir)

    if not real_files:
        raise SystemExit(f"No images found in real dir: {args.real_dir}")
    if not fake_files:
        raise SystemExit(f"No images found in fake dir: {args.fake_dir}")

    random.shuffle(real_files)
    random.shuffle(fake_files)

    if args.max_per_class > 0:
        real_files = real_files[: args.max_per_class]
        fake_files = fake_files[: args.max_per_class]

    # Keep classes balanced
    n = min(len(real_files), len(fake_files))
    real_files = real_files[:n]
    fake_files = fake_files[:n]

    rows: list[list[float | int]] = []
    for p in real_files:
        rows.append([0, *image_to_features(p, args.size)])
    for p in fake_files:
        rows.append([1, *image_to_features(p, args.size)])

    random.shuffle(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    feature_count = args.size * args.size
    header = ["label"] + [f"p{i}" for i in range(1, feature_count + 1)]

    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print("Done.")
    print(f"Output: {args.out}")
    print(f"Rows: {len(rows)} (real={n}, generated={n})")
    print(f"Features per sample: {feature_count}")


if __name__ == "__main__":
    main()

