from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


FREQUENCY_PROFILES = {
    "industry": {
        "low": (0, 2),
        "mid": (3, 12),
        "high": (13, 63),
    },
    "aggressive": {
        "low": (0, 0),
        "mid": (6, 15),
        "high": (16, 63),
    },
}


@dataclass(frozen=True)
class BlockGeometry:
    """Geometry metadata needed to reconstruct an image after block processing."""

    orig_h: int
    orig_w: int
    padded_h: int
    padded_w: int
    blocks_h: int
    blocks_w: int


def dct_basis_matrix(n: int = 8, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Return an orthonormal DCT-II basis matrix with shape (n, n)."""
    if n <= 0:
        raise ValueError("n must be a positive integer")

    i = torch.arange(n, device=device, dtype=dtype if dtype is not None else torch.float32)
    k = torch.arange(n, device=device, dtype=dtype if dtype is not None else torch.float32)

    alpha = torch.full((n,), (2.0 / n) ** 0.5, device=i.device, dtype=i.dtype)
    alpha[0] = (1.0 / n) ** 0.5

    theta = torch.pi * (2.0 * i[None, :] + 1.0) * k[:, None] / (2.0 * n)
    return alpha[:, None] * torch.cos(theta)


def zigzag_indices(n: int = 8) -> list[tuple[int, int]]:
    """Return (row, col) indices in JPEG zig-zag order for n x n coefficients."""
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


def band_mask(block_size: int = 8, start: int = 0, end: int = 63, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Build a mask for zig-zag coefficients in [start, end] inclusive."""
    max_idx = block_size * block_size - 1
    if start < 0 or end > max_idx or start > end:
        raise ValueError(f"Invalid band range [{start}, {end}] for block size {block_size}")

    mask = torch.zeros((block_size, block_size), device=device, dtype=dtype if dtype is not None else torch.float32)
    zz = zigzag_indices(block_size)
    for idx in range(start, end + 1):
        r, c = zz[idx]
        mask[r, c] = 1.0
    return mask


def mid_frequency_mask(block_size: int = 8, low: int = 3, high: int = 12, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Convenience wrapper for the project's default mid-frequency range."""
    return band_mask(block_size=block_size, start=low, end=high, device=device, dtype=dtype)


def profile_band_ranges(profile: str = "industry") -> dict[str, tuple[int, int]]:
    """Return low/mid/high ranges for a named profile."""
    if profile not in FREQUENCY_PROFILES:
        raise ValueError(f"Unknown frequency profile: {profile}")
    return FREQUENCY_PROFILES[profile]


def _pad_to_block_multiple(x: torch.Tensor, block_size: int) -> tuple[torch.Tensor, BlockGeometry]:
    if x.ndim != 4:
        raise ValueError("Expected input shape [B, C, H, W]")

    b, c, h, w = x.shape
    pad_h = (block_size - (h % block_size)) % block_size
    pad_w = (block_size - (w % block_size)) % block_size

    if pad_h == 0 and pad_w == 0:
        x_pad = x
    else:
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    hp, wp = x_pad.shape[-2:]
    return x_pad, BlockGeometry(
        orig_h=h,
        orig_w=w,
        padded_h=hp,
        padded_w=wp,
        blocks_h=hp // block_size,
        blocks_w=wp // block_size,
    )


def blockwise_dct2d(x: torch.Tensor, block_size: int = 8) -> tuple[torch.Tensor, BlockGeometry]:
    """Apply blockwise DCT to x, returning coeffs [B,C,Hb,Wb,b,b] and geometry."""
    x_pad, geo = _pad_to_block_multiple(x, block_size)
    b, c, _, _ = x_pad.shape

    basis = dct_basis_matrix(block_size, device=x.device, dtype=x.dtype)
    unfolded = F.unfold(x_pad, kernel_size=block_size, stride=block_size)
    blocks = unfolded.view(b, c, block_size, block_size, geo.blocks_h, geo.blocks_w)
    blocks = blocks.permute(0, 1, 4, 5, 2, 3).contiguous()

    tmp = torch.einsum("ij,bchwjk->bchwik", basis, blocks)
    coeff = torch.einsum("bchwik,kj->bchwij", tmp, basis.t())
    return coeff, geo


def blockwise_idct2d(coeff: torch.Tensor, geometry: BlockGeometry, block_size: int = 8) -> torch.Tensor:
    """Inverse blockwise DCT for coeffs [B,C,Hb,Wb,b,b], cropped to original H,W."""
    if coeff.ndim != 6:
        raise ValueError("Expected coeff shape [B, C, Hb, Wb, b, b]")

    basis = dct_basis_matrix(block_size, device=coeff.device, dtype=coeff.dtype)

    tmp = torch.einsum("ij,bchwjk->bchwik", basis.t(), coeff)
    blocks = torch.einsum("bchwik,kj->bchwij", tmp, basis)

    b, c, hb, wb, _, _ = blocks.shape
    folded_src = blocks.permute(0, 1, 4, 5, 2, 3).contiguous()
    folded_src = folded_src.view(b, c * block_size * block_size, hb * wb)

    x_pad = F.fold(
        folded_src,
        output_size=(geometry.padded_h, geometry.padded_w),
        kernel_size=block_size,
        stride=block_size,
    )
    return x_pad[:, :, : geometry.orig_h, : geometry.orig_w]


def rgb_to_ycbcr(x: torch.Tensor) -> torch.Tensor:
    """Convert [B,3,H,W] RGB in [0,1] to BT.601 YCbCr in [0,1]-like range."""
    if x.shape[1] != 3:
        raise ValueError("Expected 3-channel RGB tensor")

    m = torch.tensor(
        [
            [0.299000, 0.587000, 0.114000],
            [-0.168736, -0.331264, 0.500000],
            [0.500000, -0.418688, -0.081312],
        ],
        device=x.device,
        dtype=x.dtype,
    )
    ycbcr = torch.einsum("ij,bjhw->bihw", m, x)
    ycbcr[:, 1:, :, :] += 0.5
    return ycbcr


def ycbcr_to_rgb(x: torch.Tensor) -> torch.Tensor:
    """Convert [B,3,H,W] BT.601 YCbCr back to RGB and clamp to [0,1]."""
    if x.shape[1] != 3:
        raise ValueError("Expected 3-channel YCbCr tensor")

    x0 = x.clone()
    x0[:, 1:, :, :] -= 0.5
    m = torch.tensor(
        [
            [1.000000, 0.000000, 1.402000],
            [1.000000, -0.344136, -0.714136],
            [1.000000, 1.772000, 0.000000],
        ],
        device=x.device,
        dtype=x.dtype,
    )
    rgb = torch.einsum("ij,bjhw->bihw", m, x0)
    return torch.clamp(rgb, 0.0, 1.0)


def band_energy_stats(
    x: torch.Tensor,
    *,
    block_size: int = 8,
    low_range: tuple[int, int] = (0, 2),
    mid_range: tuple[int, int] = (3, 12),
    high_range: tuple[int, int] = (13, 63),
) -> dict[str, float]:
    """Return total and per-band DCT energy percentages for the input tensor."""
    coeff, _ = blockwise_dct2d(x, block_size=block_size)
    energy = coeff.pow(2)

    total = float(energy.sum().item())
    if total < 1e-12:
        total = 1e-12

    def _energy_for_range(band: tuple[int, int]) -> float:
        m = band_mask(block_size, band[0], band[1], device=x.device, dtype=x.dtype)
        m = m.view(1, 1, 1, 1, block_size, block_size)
        return float((energy * m).sum().item())

    low_e = _energy_for_range(low_range)
    mid_e = _energy_for_range(mid_range)
    high_e = _energy_for_range(high_range)
    covered = low_e + mid_e + high_e
    residual = max(0.0, total - covered)

    return {
        "total_energy": total,
        "low_energy": low_e,
        "mid_energy": mid_e,
        "high_energy": high_e,
        "residual_energy": residual,
        "low_pct": 100.0 * low_e / total,
        "mid_pct": 100.0 * mid_e / total,
        "high_pct": 100.0 * high_e / total,
        "residual_pct": 100.0 * residual / total,
    }
