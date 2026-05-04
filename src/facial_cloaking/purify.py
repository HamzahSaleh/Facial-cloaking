"""Image-purification attacks used in the robustness evaluation.

Each function takes a PIL.Image and returns a PIL.Image so they can be chained
into the same encoding pipeline as the cloaked images themselves.
"""
from __future__ import annotations

from io import BytesIO
from typing import Callable

import numpy as np
from PIL import Image, ImageFilter

try:  # cv2 is in requirements.txt; bilateral needs it
    import cv2
except Exception:  # pragma: no cover - environment-dependent
    cv2 = None


Purification = Callable[[Image.Image], Image.Image]


def jpeg_compress(quality: int) -> Purification:
    """Re-encode the image through JPEG at the given quality (1-100)."""
    def _apply(img: Image.Image) -> Image.Image:
        buf = BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    _apply.__name__ = f"jpeg_q{int(quality)}"
    return _apply


def gaussian_blur(sigma: float) -> Purification:
    """Apply a Gaussian blur with the given sigma (PIL units)."""
    def _apply(img: Image.Image) -> Image.Image:
        return img.convert("RGB").filter(ImageFilter.GaussianBlur(radius=float(sigma)))
    _apply.__name__ = f"blur_s{sigma}"
    return _apply


def bilateral_filter(d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> Purification:
    """OpenCV bilateral filter (edge-preserving smoothing)."""
    def _apply(img: Image.Image) -> Image.Image:
        if cv2 is None:
            raise RuntimeError("opencv-python is required for bilateral_filter")
        arr = np.asarray(img.convert("RGB"))
        out = cv2.bilateralFilter(arr, d, sigma_color, sigma_space)
        return Image.fromarray(out)
    _apply.__name__ = f"bilateral_d{d}"
    return _apply


def bit_depth_reduction(bits: int) -> Purification:
    """Quantize each channel to `bits` bits (1-8)."""
    bits = int(bits)
    if not 1 <= bits <= 8:
        raise ValueError("bits must be in [1, 8]")
    levels = 2 ** bits

    def _apply(img: Image.Image) -> Image.Image:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        q = np.round(arr * (levels - 1)) / (levels - 1)
        out = np.clip(q * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(out)
    _apply.__name__ = f"bits{bits}"
    return _apply


def standard_robustness_grid() -> dict[str, Purification]:
    """The standard purification grid from `plan.md`."""
    grid: dict[str, Purification] = {
        "jpeg_q95": jpeg_compress(95),
        "jpeg_q75": jpeg_compress(75),
        "jpeg_q50": jpeg_compress(50),
        "blur_s1.0": gaussian_blur(1.0),
        "blur_s2.0": gaussian_blur(2.0),
        "bits6": bit_depth_reduction(6),
        "bits4": bit_depth_reduction(4),
    }
    if cv2 is not None:
        grid["bilateral"] = bilateral_filter()
    return grid


def parse_purification_spec(spec: str) -> tuple[str, Purification]:
    """Parse a CLI-style spec into (name, callable).

    Examples:
        "jpeg-75"     -> jpeg_compress(75)
        "blur-1.0"    -> gaussian_blur(1.0)
        "bits-4"      -> bit_depth_reduction(4)
        "bilateral"   -> bilateral_filter()
    """
    s = spec.strip().lower()
    if s == "bilateral":
        return s, bilateral_filter()
    if "-" not in s:
        raise ValueError(f"unknown purification spec: {spec!r}")
    kind, arg = s.split("-", 1)
    if kind == "jpeg":
        return f"jpeg_q{arg}", jpeg_compress(int(arg))
    if kind == "blur":
        return f"blur_s{arg}", gaussian_blur(float(arg))
    if kind == "bits":
        return f"bits{arg}", bit_depth_reduction(int(arg))
    raise ValueError(f"unknown purification spec: {spec!r}")
