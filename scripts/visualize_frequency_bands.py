"""Visualize low/mid/high frequency reconstructions using block DCT.

Usage example:
python scripts/visualize_frequency_bands.py \
  --input portrait-white-man-isolated.jpg \
  --output-dir outputs/frequency_debug
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


INDUSTRY_PROFILE = {
    "block_size": 8,
    "low_start": 0,
    "low_end": 0,
    "mid_start": 2,
    "mid_end": 12,
    "high_start": 13,
    "high_end": 63,
    "display_mode": "magnitude",
    "display_percentile": 99.5,
    "low_gain": 1.0,
    "mid_gain": 8.0,
    "high_gain": 12.0,
    "color_space": "ycbcr",
    "y_weight": 1.0,
    "cbcr_weight": 0.5,
}


def dct_basis(n: int = 8) -> np.ndarray:
    """Return an orthonormal DCT-II basis matrix of shape (n, n)."""
    c = np.zeros((n, n), dtype=np.float32)
    factor = np.pi / (2.0 * n)
    scale0 = np.sqrt(1.0 / n)
    scale = np.sqrt(2.0 / n)

    for k in range(n):
        alpha = scale0 if k == 0 else scale
        for i in range(n):
            c[k, i] = alpha * np.cos((2 * i + 1) * k * factor)
    return c


def rgb_to_ycbcr_bt601(x: np.ndarray) -> np.ndarray:
    """Convert RGB [0,1] image to YCbCr [0,1] using BT.601 full-range style."""
    m = np.array(
        [
            [0.2990, 0.5870, 0.1140],
            [-0.168736, -0.331264, 0.500000],
            [0.500000, -0.418688, -0.081312],
        ],
        dtype=np.float32,
    )
    ycbcr = np.tensordot(x, m.T, axes=1)
    ycbcr[..., 1:] += 0.5
    return ycbcr


def ycbcr_to_rgb_bt601(x: np.ndarray) -> np.ndarray:
    """Convert YCbCr [0,1] image to RGB [0,1] using BT.601 full-range style."""
    ycbcr = x.copy()
    ycbcr[..., 1:] -= 0.5
    m_inv = np.array(
        [
            [1.000000, 0.000000, 1.402000],
            [1.000000, -0.344136, -0.714136],
            [1.000000, 1.772000, 0.000000],
        ],
        dtype=np.float32,
    )
    rgb = np.tensordot(ycbcr, m_inv.T, axes=1)
    return np.clip(rgb, 0.0, 1.0)


def zigzag_indices(n: int = 8) -> list[tuple[int, int]]:
    """Return (row, col) indices in JPEG zig-zag order for n x n blocks."""
    order: list[tuple[int, int]] = []
    for s in range(2 * n - 1):
        if s % 2 == 0:
            r = min(s, n - 1)
            c = s - r
            while r >= 0 and c < n:
                order.append((r, c))
                r -= 1
                c += 1
        else:
            c = min(s, n - 1)
            r = s - c
            while c >= 0 and r < n:
                order.append((r, c))
                r += 1
                c -= 1
    return order


def mask_from_band_range(n: int, start_idx: int, end_idx: int) -> np.ndarray:
    """Build an n x n mask with ones for zig-zag indices [start_idx, end_idx]."""
    if start_idx < 0 or end_idx >= n * n or start_idx > end_idx:
        raise ValueError("Invalid band range.")

    m = np.zeros((n, n), dtype=np.float32)
    zz = zigzag_indices(n)
    for k in range(start_idx, end_idx + 1):
        r, c = zz[k]
        m[r, c] = 1.0
    return m


def band_energy_stats(
    x: np.ndarray,
    dct_c: np.ndarray,
    block: int,
    low_range: tuple[int, int],
    mid_range: tuple[int, int],
    high_range: tuple[int, int],
) -> dict[str, float]:
    """Compute energy share per frequency band from block-DCT coefficients."""
    coeff, _ = block_dct2d(x, dct_c, block=block)
    coeff_energy = coeff * coeff
    total = float(np.sum(coeff_energy))
    if total < 1e-12:
        total = 1e-12

    def _band_energy(band_range: tuple[int, int]) -> float:
        m = mask_from_band_range(block, band_range[0], band_range[1])
        return float(np.sum(coeff_energy * m[None, None, None, :, :]))

    low_e = _band_energy(low_range)
    mid_e = _band_energy(mid_range)
    high_e = _band_energy(high_range)
    covered_e = low_e + mid_e + high_e
    residual_e = max(0.0, total - covered_e)

    return {
        "total_energy": total,
        "low_energy": low_e,
        "mid_energy": mid_e,
        "high_energy": high_e,
        "residual_energy": residual_e,
        "low_pct": 100.0 * low_e / total,
        "mid_pct": 100.0 * mid_e / total,
        "high_pct": 100.0 * high_e / total,
        "residual_pct": 100.0 * residual_e / total,
    }


def pad_to_block_multiple(x: np.ndarray, block: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Reflect-pad HxWxC image to nearest multiple of block size."""
    h, w, _ = x.shape
    pad_h = (block - (h % block)) % block
    pad_w = (block - (w % block)) % block

    if pad_h == 0 and pad_w == 0:
        return x, (h, w)

    x_pad = np.pad(x, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    return x_pad, (h, w)


def block_dct2d(x: np.ndarray, c: np.ndarray, block: int = 8) -> tuple[np.ndarray, tuple[int, int]]:
    """Apply 2D DCT to each block. Returns coeffs and original (h, w)."""
    x_pad, orig_hw = pad_to_block_multiple(x, block)
    h, w, ch = x_pad.shape
    hb, wb = h // block, w // block

    blocks = x_pad.reshape(hb, block, wb, block, ch).transpose(0, 2, 4, 1, 3)
    coeff = np.einsum("ab,xycbd,de->xycae", c, blocks, c.T, optimize=True)
    return coeff, orig_hw


def block_idct2d(coeff: np.ndarray, c: np.ndarray, orig_hw: tuple[int, int], block: int = 8) -> np.ndarray:
    """Apply inverse 2D DCT to each block and crop to original size."""
    hb, wb, ch, _, _ = coeff.shape
    blocks = np.einsum("ab,xycbd,de->xycae", c.T, coeff, c, optimize=True)
    x_pad = blocks.transpose(0, 3, 1, 4, 2).reshape(hb * block, wb * block, ch)

    h, w = orig_hw
    return x_pad[:h, :w, :]


def reconstruct_band(
    img_float: np.ndarray,
    dct_c: np.ndarray,
    band_start: int,
    band_end: int,
    block: int = 8,
) -> np.ndarray:
    """Keep only selected zig-zag DCT bands and reconstruct image."""
    coeff, orig_hw = block_dct2d(img_float, dct_c, block=block)
    band_mask = mask_from_band_range(block, band_start, band_end)
    coeff_band = coeff * band_mask[None, None, None, :, :]
    recon = block_idct2d(coeff_band, dct_c, orig_hw, block=block)
    return recon


def to_pil_uint8(x: np.ndarray) -> Image.Image:
    """Convert float image in [0,1] to uint8 PIL image."""
    return Image.fromarray(np.clip(x * 255.0, 0, 255).astype(np.uint8))


def scale_signed_for_display(x: np.ndarray, gain: float = 1.0, percentile: float = 99.0) -> np.ndarray:
    """Scale signed band reconstructions so positive/negative structure is visible."""
    # Remove per-channel mean so oscillatory components are centered for display.
    y = x - x.mean(axis=(0, 1), keepdims=True)
    robust = np.percentile(np.abs(y), percentile)
    if robust < 1e-8:
        robust = 1e-8
    y = y / robust
    y = 0.5 + 0.5 * gain * y
    return np.clip(y, 0.0, 1.0)


def scale_magnitude_for_display(x: np.ndarray, gain: float = 1.0, percentile: float = 99.0) -> np.ndarray:
    """Scale absolute band energy so weak details are visible as intensity."""
    y = np.abs(x)
    robust = np.percentile(y, percentile)
    if robust < 1e-8:
        robust = 1e-8
    y = gain * (y / robust)
    return np.clip(y, 0.0, 1.0)


def draw_label(img: Image.Image, text: str) -> Image.Image:
    """Add a top-left text label for quick comparison."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, 260, 30), fill=(0, 0, 0))
    draw.text((8, 8), text, fill=(255, 255, 255))
    return out


def make_panel(images: list[Image.Image], cols: int = 2) -> Image.Image:
    """Build a simple tiled panel from equally sized images."""
    if not images:
        raise ValueError("No images provided for panel.")

    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    panel = Image.new("RGB", (cols * w, rows * h), color=(30, 30, 30))

    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        panel.paste(img, (c * w, r * h))
    return panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize frequency bands with 8x8 block DCT.")
    parser.add_argument("--input", required=True, type=Path, help="Input image path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "frequency_debug",
        help="Directory for generated images.",
    )
    parser.add_argument("--block-size", type=int, default=8, help="DCT block size (default: 8).")
    parser.add_argument(
        "--profile",
        choices=["industry", "custom"],
        default="industry",
        help="Preset profile. 'industry' applies JPEG/BT.601/Y-priority defaults.",
    )

    # Zig-zag index ranges.
    parser.add_argument("--low-start", type=int, default=0)
    parser.add_argument("--low-end", type=int, default=2)
    parser.add_argument("--mid-start", type=int, default=3)
    parser.add_argument("--mid-end", type=int, default=12)
    parser.add_argument("--high-start", type=int, default=13)
    parser.add_argument("--high-end", type=int, default=63)
    parser.add_argument(
        "--aggressive-bands",
        action="store_true",
        help="Use a more separated split: low[0-0], mid[6-15], high[16-63].",
    )
    parser.add_argument(
        "--display-percentile",
        type=float,
        default=99.0,
        help="Robust percentile used for display scaling (default: 99).",
    )
    parser.add_argument(
        "--low-gain",
        type=float,
        default=1.2,
        help="Display gain for low-band reconstruction.",
    )
    parser.add_argument(
        "--mid-gain",
        type=float,
        default=4.0,
        help="Display gain for mid-band reconstruction.",
    )
    parser.add_argument(
        "--high-gain",
        type=float,
        default=6.0,
        help="Display gain for high-band reconstruction.",
    )
    parser.add_argument(
        "--display-mode",
        choices=["signed", "magnitude"],
        default="magnitude",
        help="How to render band-limited outputs for visibility.",
    )
    parser.add_argument(
        "--color-space",
        choices=["rgb", "ycbcr"],
        default="ycbcr",
        help="Color space used before DCT processing.",
    )
    parser.add_argument(
        "--y-weight",
        type=float,
        default=1.0,
        help="Weight for Y channel when using YCbCr (visual emphasis only).",
    )
    parser.add_argument(
        "--cbcr-weight",
        type=float,
        default=0.5,
        help="Weight for Cb/Cr channels when using YCbCr (visual emphasis only).",
    )

    return parser.parse_args()


def apply_industry_profile(args: argparse.Namespace) -> argparse.Namespace:
    """Apply canonical defaults for reproducible frequency visualization."""
    if args.profile != "industry":
        return args

    for key, value in INDUSTRY_PROFILE.items():
        setattr(args, key, value)
    return args


def apply_aggressive_band_split(args: argparse.Namespace) -> argparse.Namespace:
    """Apply a more separated low/mid/high split for clearer visual differences."""
    if not args.aggressive_bands:
        return args

    args.low_start, args.low_end = 0, 0
    args.mid_start, args.mid_end = 6, 15
    args.high_start, args.high_end = 16, 63
    return args


def main() -> int:
    args = apply_aggressive_band_split(apply_industry_profile(parse_args()))

    if not args.input.exists():
        raise FileNotFoundError(f"Input image not found: {args.input}")

    if args.block_size != 8:
        raise ValueError("This script expects block_size=8 for JPEG-style zig-zag indexing.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(args.input).convert("RGB")
    x_rgb = np.asarray(img).astype(np.float32) / 255.0
    if args.color_space == "ycbcr":
        x = rgb_to_ycbcr_bt601(x_rgb)
        # Industry default: emphasize luminance contribution during visualization.
        x = x.copy()
        x[..., 0] *= args.y_weight
        x[..., 1] *= args.cbcr_weight
        x[..., 2] *= args.cbcr_weight
    else:
        x = x_rgb

    c = dct_basis(n=args.block_size)

    low = reconstruct_band(x, c, args.low_start, args.low_end, block=args.block_size)
    mid = reconstruct_band(x, c, args.mid_start, args.mid_end, block=args.block_size)
    high = reconstruct_band(x, c, args.high_start, args.high_end, block=args.block_size)

    if args.display_mode == "signed":
        low_disp = scale_signed_for_display(low, gain=args.low_gain, percentile=args.display_percentile)
        mid_disp = scale_signed_for_display(mid, gain=args.mid_gain, percentile=args.display_percentile)
        high_disp = scale_signed_for_display(high, gain=args.high_gain, percentile=args.display_percentile)
    else:
        low_disp = scale_magnitude_for_display(low, gain=args.low_gain, percentile=args.display_percentile)
        mid_disp = scale_magnitude_for_display(mid, gain=args.mid_gain, percentile=args.display_percentile)
        high_disp = scale_magnitude_for_display(high, gain=args.high_gain, percentile=args.display_percentile)

    if args.color_space == "ycbcr":
        low_disp = ycbcr_to_rgb_bt601(low_disp)
        mid_disp = ycbcr_to_rgb_bt601(mid_disp)
        high_disp = ycbcr_to_rgb_bt601(high_disp)

    original = draw_label(img, "Original")
    low_img = draw_label(
        to_pil_uint8(low_disp),
        f"Low [{args.low_start}-{args.low_end}] {args.display_mode} x{args.low_gain:g}",
    )
    mid_img = draw_label(
        to_pil_uint8(mid_disp),
        f"Mid [{args.mid_start}-{args.mid_end}] {args.display_mode} x{args.mid_gain:g}",
    )
    high_img = draw_label(
        to_pil_uint8(high_disp),
        f"High [{args.high_start}-{args.high_end}] {args.display_mode} x{args.high_gain:g}",
    )

    stem = args.input.stem
    original.save(args.output_dir / f"{stem}_original.png")
    low_img.save(args.output_dir / f"{stem}_low.png")
    mid_img.save(args.output_dir / f"{stem}_mid.png")
    high_img.save(args.output_dir / f"{stem}_high.png")

    panel = make_panel([original, low_img, mid_img, high_img], cols=2)
    panel.save(args.output_dir / f"{stem}_panel.png")

    stats = band_energy_stats(
        x,
        c,
        block=args.block_size,
        low_range=(args.low_start, args.low_end),
        mid_range=(args.mid_start, args.mid_end),
        high_range=(args.high_start, args.high_end),
    )
    stats_payload = {
        "profile": args.profile,
        "color_space": args.color_space,
        "weights": {"y": args.y_weight, "cbcr": args.cbcr_weight},
        "bands": {
            "low": [args.low_start, args.low_end],
            "mid": [args.mid_start, args.mid_end],
            "high": [args.high_start, args.high_end],
        },
        "energy": stats,
    }
    stats_path = args.output_dir / f"{stem}_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_payload, f, indent=2)

    print("Saved:")
    print(args.output_dir / f"{stem}_original.png")
    print(args.output_dir / f"{stem}_low.png")
    print(args.output_dir / f"{stem}_mid.png")
    print(args.output_dir / f"{stem}_high.png")
    print(args.output_dir / f"{stem}_panel.png")
    print(stats_path)
    print(f"Profile: {args.profile}")
    if args.aggressive_bands:
        print("Band preset: aggressive")
    print(f"Color space: {args.color_space} (Y={args.y_weight}, CbCr={args.cbcr_weight})")
    print(
        "Bands: "
        f"low[{args.low_start}-{args.low_end}] "
        f"mid[{args.mid_start}-{args.mid_end}] "
        f"high[{args.high_start}-{args.high_end}]"
    )
    print(
        "Energy share (%): "
        f"low={stats['low_pct']:.2f}, "
        f"mid={stats['mid_pct']:.2f}, "
        f"high={stats['high_pct']:.2f}, "
        f"residual={stats['residual_pct']:.2f}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
