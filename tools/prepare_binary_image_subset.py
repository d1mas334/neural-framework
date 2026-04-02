#!/usr/bin/env python3
"""
Prepare a balanced binary image subset from a raw dataset tree.

Class inference by folder names:
- real class: folder contains one of ["real", "human", "photo", "natural"]
- fake class: folder contains one of ["fake", "ai", "generated", "synthetic"]
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REAL_TAGS = ("real", "human", "photo", "natural")
FAKE_TAGS = ("fake", "ai", "generated", "synthetic")


def collect_images(root: Path) -> list[Path]:
    items: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXT:
            items.append(p)
    return items


def infer_label(path: Path) -> int | None:
    s = str(path).lower()
    real = any(tag in s for tag in REAL_TAGS)
    fake = any(tag in s for tag in FAKE_TAGS)
    if real and not fake:
        return 0
    if fake and not real:
        return 1
    return None


def copy_subset(files: list[Path], out_dir: Path, max_count: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(len(files), max_count)
    for i, src in enumerate(files[:n], start=1):
        dst = out_dir / f"{i:06d}{src.suffix.lower()}"
        shutil.copy2(src, dst)
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare balanced real/fake subset from raw image dataset")
    parser.add_argument("--src", required=True, type=Path, help="Raw dataset root")
    parser.add_argument("--out", required=True, type=Path, help="Output root with real/ and fake/")
    parser.add_argument("--max-per-class", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    all_files = collect_images(args.src)
    if not all_files:
        raise SystemExit(f"No images found in: {args.src}")

    real_files: list[Path] = []
    fake_files: list[Path] = []
    unknown = 0

    for p in all_files:
        lbl = infer_label(p)
        if lbl == 0:
            real_files.append(p)
        elif lbl == 1:
            fake_files.append(p)
        else:
            unknown += 1

    if not real_files or not fake_files:
        raise SystemExit(
            "Could not infer both classes from folder names.\n"
            "Need path names containing tags like real/human and fake/ai/generated."
        )

    random.shuffle(real_files)
    random.shuffle(fake_files)

    n = min(len(real_files), len(fake_files), args.max_per_class)
    out_real = args.out / "real"
    out_fake = args.out / "fake"

    if out_real.exists():
        shutil.rmtree(out_real)
    if out_fake.exists():
        shutil.rmtree(out_fake)

    c_real = copy_subset(real_files, out_real, n)
    c_fake = copy_subset(fake_files, out_fake, n)

    print("Done.")
    print(f"Source files: {len(all_files)}")
    print(f"Unknown label files skipped: {unknown}")
    print(f"Saved real: {c_real} -> {out_real}")
    print(f"Saved fake: {c_fake} -> {out_fake}")


if __name__ == "__main__":
    main()

