from __future__ import annotations

from dataclasses import dataclass

import torch

from facial_cloaking.dct_utils import *
from facial_cloaking.embed import encode_image

@dataclass
class CloakConfig :
    epsilon: float = 8/255
    n_steps: int = 300
    lr: float = 0.01
    lambda_embed: float = 1.0
    lambda_freq: float = 0.3
    lambda_quality: float = 2.0
    freq_low: int = 3
    freq_high: int = 12

def channel_mix(
    W: torch.Tensor,
    x: torch.Tensor,
    delta_pixel: torch.Tensor
) -> torch.Tensor :
    
    mixed = (x + delta_pixel).permute(0, 2, 3, 1)
    mixed = mixed @ W.t()
    mixed = mixed.permute(0, 3, 1, 2)

    x_cloak = torch.nn.functional.softplus(mixed)
    x_cloak = torch.clamp(x_cloak, 0.0, 1.0)
    return x_cloak

