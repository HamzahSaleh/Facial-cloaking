from __future__ import annotations

from dataclasses import dataclass

import torch

from .dct_utils import (
    band_energy_stats,
    band_mask,
    profile_band_ranges,
)


@dataclass
class CloakConfig:
    """Core config values used by the cloaking optimizer."""

    epsilon: float = 8.0 / 255.0
    n_steps: int = 300
    lr: float = 0.01

    lambda_embed: float = 1.0
    lambda_freq: float = 0.3
    lambda_quality: float = 2.0

    block_size: int = 8
    frequency_profile: str = "industry"


@dataclass(frozen=True)
class FrequencyBands:
    """Resolved low/mid/high zig-zag index ranges."""

    low: tuple[int, int]
    mid: tuple[int, int]
    high: tuple[int, int]


@dataclass(frozen=True)
class FrequencyMasks:
    """Per-band masks in [1,1,1,1,b,b] broadcast shape for block-DCT coeffs."""

    low: torch.Tensor
    mid: torch.Tensor
    high: torch.Tensor


def resolve_frequency_bands(config: CloakConfig) -> FrequencyBands:
    """Resolve profile name into concrete low/mid/high zig-zag ranges."""
    ranges = profile_band_ranges(config.frequency_profile)
    return FrequencyBands(
        low=ranges["low"],
        mid=ranges["mid"],
        high=ranges["high"],
    )


def build_frequency_masks(
    config: CloakConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> FrequencyMasks:
    """Build broadcast-ready masks for low/mid/high coefficient bands."""
    bands = resolve_frequency_bands(config)
    b = config.block_size

    def _mk(r: tuple[int, int]) -> torch.Tensor:
        m = band_mask(b, r[0], r[1], device=device, dtype=dtype)
        return m.view(1, 1, 1, 1, b, b)

    return FrequencyMasks(
        low=_mk(bands.low),
        mid=_mk(bands.mid),
        high=_mk(bands.high),
    )


def summarize_frequency_energy(
    x: torch.Tensor,
    config: CloakConfig,
) -> dict[str, float]:
    """Compute quantifiable energy share for the configured low/mid/high bands."""
    bands = resolve_frequency_bands(config)
    return band_energy_stats(
        x,
        block_size=config.block_size,
        low_range=bands.low,
        mid_range=bands.mid,
        high_range=bands.high,
    )
