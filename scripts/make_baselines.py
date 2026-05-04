"""Generate reference 'method' folders so the eval pipeline is runnable
before any real cloaking algorithm is implemented.

Two reference methods are produced:

  - ``noise``  : original image + low-amplitude Gaussian noise (sanity-check
                 that retrieval is still strong; this is NOT a cloak).
  - ``blur``   : Gaussian-blurred original (a weak quality-only baseline).

Each method writes flat files named exactly like the ``filename`` column of
``splits/protected.csv`` to ``outputs/methods/<name>/``.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from facial_cloaking.paths import PROJECT_ROOT


def _add_gaussian_noise(img: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    noise = rng.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    out = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(PROJECT_ROOT / "splits" / "protected.csv"))
    parser.add_argument("--out-root", default=str(PROJECT_ROOT / "outputs" / "methods"))
    parser.add_argument("--noise-sigma", type=float, default=4.0)
    parser.add_argument("--blur-sigma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    noise_dir = out_root / "noise"
    blur_dir = out_root / "blur"
    noise_dir.mkdir(parents=True, exist_ok=True)
    blur_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        src = PROJECT_ROOT / r["image_path"].strip()
        fname = r["filename"].strip()
        with Image.open(src) as img:
            img = img.convert("RGB")
            _add_gaussian_noise(img, args.noise_sigma, rng).save(noise_dir / fname, "JPEG", quality=95)
            img.filter(ImageFilter.GaussianBlur(radius=args.blur_sigma)).save(
                blur_dir / fname, "JPEG", quality=95
            )

    print(f"[baselines] wrote {len(rows)} images each to:")
    print(f"  - {noise_dir}")
    print(f"  - {blur_dir}")


if __name__ == "__main__":
    main()
