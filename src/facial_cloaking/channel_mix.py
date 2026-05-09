from __future__ import annotations

import torch
from torch import Tensor

def channel_mix(
    W: Tensor,
    x: Tensor,
    delta_pixel: Tensor
) -> Tensor :
    
    mixed = (x + delta_pixel).permute(0, 2, 3, 1)
    mixed = mixed @ W.t()
    mixed = mixed.permute(0, 3, 1, 2)

    x_cloak = torch.nn.functional.softplus(mixed)
    x_cloak = torch.clamp(x_cloak, 0.0, 1.0)
    return x_cloak