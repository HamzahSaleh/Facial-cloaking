from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image

from .dct_utils import (
    band_energy_stats,
    band_mask,
    blockwise_dct2d,
    profile_band_ranges,
)


# CLIP ViT-B/32 normalization constants (open_clip / OpenAI CLIP).
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_CLIP_INPUT_SIZE = 224

# Bias used inside softplus for the NL-CM forward so that the identity
# initialization (W = I, delta = 0) reproduces the input image to within
# ~2% on [0, 1]. softplus(z + b) - b approaches z as b grows.
_NLCM_SOFTPLUS_BIAS = 4.0


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


# ---------------------------------------------------------------------------
# Tensor / PIL conversion
# ---------------------------------------------------------------------------

def _pil_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    """Convert a PIL RGB image to a [3, H, W] float tensor in [0, 1]."""
    import numpy as np
    arr = np.array(img.convert("RGB"), dtype=np.uint8)  # copy -> writable
    t = torch.from_numpy(arr).to(device=device, dtype=torch.float32) / 255.0
    return t.permute(2, 0, 1).contiguous()


def _tensor_to_pil(x: torch.Tensor) -> Image.Image:
    """Convert a [3, H, W] tensor in [0, 1] back to a PIL RGB image."""
    arr = (x.detach().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    arr = arr.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------------------
# Differentiable CLIP encoding
# ---------------------------------------------------------------------------

def _clip_normalize(x: torch.Tensor) -> torch.Tensor:
    """CLIP normalization on a [B, 3, H, W] tensor in [0, 1]. Differentiable."""
    mean = torch.tensor(_CLIP_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_CLIP_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def _encode_image_tensor(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Encode a [3, H, W] tensor in [0, 1] through CLIP with gradients flowing.

    Replicates open_clip's preprocess (resize to 224x224 bicubic + normalize)
    in pure torch so the optimizer can backprop into pixel space.
    """
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.shape[-2:] != (_CLIP_INPUT_SIZE, _CLIP_INPUT_SIZE):
        x = F.interpolate(
            x, size=(_CLIP_INPUT_SIZE, _CLIP_INPUT_SIZE),
            mode="bicubic", align_corners=False, antialias=True,
        )
    x = x.clamp(0.0, 1.0)
    features = model.encode_image(_clip_normalize(x))
    return F.normalize(features, dim=-1).squeeze(0)


# ---------------------------------------------------------------------------
# Differentiable SSIM (single-scale, windowed Gaussian)
# ---------------------------------------------------------------------------

def _gaussian_kernel(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    return g


def differentiable_ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Per-channel windowed-Gaussian SSIM, averaged. Inputs: [C, H, W] or [B, C, H, W]."""
    if x.ndim == 3:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
    c = x.shape[1]
    g1d = _gaussian_kernel(window_size, sigma, x.device, x.dtype)
    kernel = (g1d[:, None] * g1d[None, :]).expand(c, 1, window_size, window_size)
    pad = window_size // 2

    def _conv(t: torch.Tensor) -> torch.Tensor:
        return F.conv2d(t, kernel, padding=pad, groups=c)

    mu_x = _conv(x)
    mu_y = _conv(y)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sig_x2 = _conv(x * x) - mu_x2
    sig_y2 = _conv(y * y) - mu_y2
    sig_xy = _conv(x * y) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    num = (2 * mu_xy + c1) * (2 * sig_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sig_x2 + sig_y2 + c2)
    return (num / den).mean()


# ---------------------------------------------------------------------------
# Loss components
# ---------------------------------------------------------------------------

def _l_embed(e_cloak: torch.Tensor, e_id: torch.Tensor, e_null: torch.Tensor) -> torch.Tensor:
    """Embedding-collapse loss: pull cloaked away from identity, toward null.

    Returns cos(e_cloak, e_id) - cos(e_cloak, e_null). Lower (more negative)
    means a better cloak. All inputs assumed L2-normalized [D].
    """
    return torch.dot(e_cloak, e_id) - torch.dot(e_cloak, e_null)


def _l_quality(x_cloak: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Visual-fidelity loss: 1 - SSIM. Lower is better."""
    return 1.0 - differentiable_ssim(x_cloak, x)


def _l_freq(
    x_cloak: torch.Tensor,
    x: torch.Tensor,
    mid_mask: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Frequency-entanglement loss: -||M_mid * DCT(x_cloak - x)||_2.

    More negative means more perturbation energy concentrated in the mid band,
    i.e. structurally entangled with identity-bearing frequencies. Used by
    the freq-only and combined variants (not by NL-CM-only).
    """
    diff = (x_cloak - x).unsqueeze(0)
    coeff, _ = blockwise_dct2d(diff, block_size=block_size)
    masked = coeff * mid_mask
    return -torch.linalg.vector_norm(masked)


# ---------------------------------------------------------------------------
# Shared optimizer skeleton
# ---------------------------------------------------------------------------

def _project_linf(delta_image: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Clamp the image-space perturbation to the L_inf ball of radius epsilon."""
    return delta_image.clamp(-epsilon, epsilon)


def _project_w(W: torch.Tensor, max_frob: float = 0.1) -> torch.Tensor:
    """Project W so that ||W - I||_F <= max_frob. Returns a new tensor."""
    eye = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
    delta_w = W - eye
    norm = torch.linalg.matrix_norm(delta_w, ord="fro")
    if norm > max_frob:
        delta_w = delta_w * (max_frob / norm)
    return eye + delta_w


def _optimize(
    x: torch.Tensor,
    e_id: torch.Tensor,
    e_null: torch.Tensor,
    model: torch.nn.Module,
    *,
    forward_fn,         # (x, params) -> x_cloak in roughly [0, 1]
    init_params: dict,  # learnable parameters (will be cloned + grad-enabled)
    constrain_params,   # callable(params) -> None, applied after each opt step (no-grad)
    use_freq_loss: bool,
    config: CloakConfig,
) -> tuple[torch.Tensor, dict]:
    """Run the per-image cloaking optimization. Returns (x_cloak, metrics)."""
    device = x.device
    dtype = x.dtype

    if use_freq_loss:
        masks = build_frequency_masks(config, device=device, dtype=dtype)
        mid_mask = masks.mid
    else:
        mid_mask = None

    # Freeze CLIP so backward only computes activation gradients (to reach our
    # learnable params) and skips per-layer weight-grad computation.
    for p in model.parameters():
        p.requires_grad_(False)

    params = {k: v.clone().detach().requires_grad_(True) for k, v in init_params.items()}
    optimizer = torch.optim.Adam(list(params.values()), lr=config.lr)

    history = {"l_total": [], "l_embed": [], "l_quality": [], "l_freq": []}

    for step in range(config.n_steps):
        optimizer.zero_grad()
        x_cloak_raw = forward_fn(x, params)

        # Image-space L_inf projection (kept differentiable via residual form).
        delta_img = _project_linf(x_cloak_raw - x, config.epsilon)
        x_cloak = (x + delta_img).clamp(0.0, 1.0)

        e_cloak = _encode_image_tensor(model, x_cloak)

        l_embed = _l_embed(e_cloak, e_id, e_null)
        l_quality = _l_quality(x_cloak, x)
        if use_freq_loss:
            l_freq = _l_freq(x_cloak, x, mid_mask, config.block_size)
        else:
            l_freq = torch.zeros((), device=device, dtype=dtype)

        loss = (
            config.lambda_embed * l_embed
            + config.lambda_quality * l_quality
            + (config.lambda_freq * l_freq if use_freq_loss else 0.0)
        )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            constrain_params(params)

        history["l_total"].append(float(loss.item()))
        history["l_embed"].append(float(l_embed.item()))
        history["l_quality"].append(float(l_quality.item()))
        history["l_freq"].append(float(l_freq.item()) if use_freq_loss else 0.0)

    with torch.no_grad():
        x_cloak_raw = forward_fn(x, params)
        delta_img = _project_linf(x_cloak_raw - x, config.epsilon)
        x_cloak = (x + delta_img).clamp(0.0, 1.0)
        e_cloak = _encode_image_tensor(model, x_cloak)

    metrics = {
        "loss_history": history,
        "final_loss": history["l_total"][-1],
        "final_l_embed": history["l_embed"][-1],
        "final_l_quality": history["l_quality"][-1],
        "final_l_freq": history["l_freq"][-1],
        "final_ssim_proxy": float(1.0 - history["l_quality"][-1]),
        "cos_to_identity": float(torch.dot(e_cloak, e_id).item()),
        "cos_to_null": float(torch.dot(e_cloak, e_null).item()),
    }
    return x_cloak, metrics


# ---------------------------------------------------------------------------
# Non-linear channel mixing forward + cloaker
# ---------------------------------------------------------------------------

def _forward_nlcm(x: torch.Tensor, params: dict) -> torch.Tensor:
    """x_cloak = softplus(W @ x + delta + bias) - bias.

    With W = I and delta = 0, the bias term makes softplus near-linear over [0, 1]
    so the identity initialization reproduces x to within ~2%. The non-linearity
    is what blocks the additive `x + delta` inverse: there is no closed-form
    way to recover delta given W and the cloaked output.
    """
    C, H, Wd = x.shape
    W = params["W"]
    delta = params["delta"]
    x_flat = x.reshape(C, H * Wd)
    mixed = (W @ x_flat).reshape(C, H, Wd)
    pre = mixed + delta + _NLCM_SOFTPLUS_BIAS
    return F.softplus(pre) - _NLCM_SOFTPLUS_BIAS


def cloak_image_nlcm(
    img: Image.Image,
    e_identity: torch.Tensor,
    e_null: torch.Tensor,
    model: torch.nn.Module,
    *,
    config: CloakConfig | None = None,
    device: torch.device | str | None = None,
    w_max_frob: float = 0.1,
) -> tuple[Image.Image, dict]:
    """Non-linear channel mixing cloak. Loss = lambda_embed * L_embed + lambda_quality * L_quality.

    Learnable parameters: per-pixel delta in R^{3xHxW} (init 0) and a 3x3 channel
    mixing matrix W (init I). Constraints applied after each step:
      * ||delta||_inf <= epsilon
      * ||W - I||_F <= w_max_frob
    Final image is also clamped to [0, 1].

    L_freq is intentionally NOT included in this method so the ablation
    isolates the non-linear-mixing parameterization from the frequency
    regularizer used by the freq-only and combined variants.
    """
    config = config or CloakConfig()
    device = torch.device(device) if device is not None else next(model.parameters()).device
    x = _pil_to_tensor(img, device)
    e_id = e_identity.to(device).detach()
    e_n = e_null.to(device).detach()

    init_params = {
        "delta": torch.zeros_like(x),
        "W": torch.eye(3, device=device, dtype=x.dtype),
    }

    def constrain(p):
        p["delta"].clamp_(-config.epsilon, config.epsilon)
        p["W"].copy_(_project_w(p["W"], max_frob=w_max_frob))

    x_cloak, metrics = _optimize(
        x, e_id, e_n, model,
        forward_fn=_forward_nlcm,
        init_params=init_params,
        constrain_params=constrain,
        use_freq_loss=False,
        config=config,
    )
    metrics["method"] = "nlcm"
    return _tensor_to_pil(x_cloak), metrics
