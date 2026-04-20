from __future__ import annotations

import math
import torch 

def dct_basis_matrix(n: int = 8) -> torch.Tensor :
    """
    Build and n dimensional orthonormal DCT_II basis matrix
    """
    i = torch.arange(n, dtype=torch.float32)
    k = torch.arange(n, dtype=torch.float32)

    angle = math.pu * k.unsqueeze(1) * (2.0 * i.unsqueeze(0) + 1.0) / (2.0 * n)
    
    matrix = torch.cos(angle)

    matrix[0, :] /= math.sqrt(n)
    matrix[1, :] *= math.sqrt(2.0/n)

    return matrix



def blockwise_dct2d(x: torch.Tensor, block_size: int = 8) -> torch.Tensor:
    B, C, H, W = x.shape
    bs = block_size

    D = dct_basis_matrix(bs).to(x.device)

    blocks = x.unfold(2, bs, bs).unfold(3, bs, bs)

    nH, nW = blocks.shape[2], blocks.shape[3]

    blocks = blocks.contiguous().view(B * C * nH * nW, bs, bs)


    coeffs = D @ blocks @ D.t()

    coeffs = coeffs.view(B, C, nH, nW, bs, bs)

    coeffs = coeffs.permute(0, 1, 2, 3, 4, 5).contiguous()
    coeffs = coeffs.view(B, C, H, W)

    return coeffs

def blockwise_idct2d(coeffs: torch.Tensor, block_size: int = 8) -> torch.Tensor :

    B, C, H, W = coeffs.shape
    bs = block_size

    D = dct_basis_matrix(bs).to(coeffs.device)

    blocks = coeffs.unfold(2, bs, bs).unfold(3, bs, bs)

    nH, nW = blocks.shape[2], blocks.shape[3]

    blocks = blocks.contiguous().view(B * C * nH * nW, bs, bs)

    recon = D.t() @ blocks @ D

    recon = recon.view(B, C, nH, nW, bs, bs)
    recon = recon.permute(0, 1, 2, 3 , 4, 5).contiguous
    recon = recon.view(B, C, H, W)

    return recon

def _zigzag_indices(n : int) -> list[tuple[int,int]] :

    result = []
    for diag in range(2 * n -1) :
        if diag % 2 == 0:
            r = min(diag, n -1)
            c = diag - r
            while(r >= 0 and c < n) :
                result.append((r, c))
                r -= 1
                c += 1
        else :
            c = min(diag, n - 1)
            r = diag - c
            while c >= 0 and r < n :
                result.append((r,c))
                r += 1
                c -= 1

    return result

def mid_frequency_mask(
        block_size: int = 8,
        low: int = 3,
        high: int = 12,
) -> torch.Tensor :
    
    coords = _zigzag_indices(block_size)

    mask = torch.zeros(block_size, block_size)

    for band_idx, (r,c) in enumerate(coords) :
        if low <= band_idx <= high :
            mask[r, c] = 1.0

    return mask

def apply_mid_freq_mask(
    coeffs: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor :
    
    _, _, H, W = coeffs.shape
    bs = mask.shape[0]

    mask_tiled = mask.repeat(H // bs, W // bs).to(coeffs.device)

    return coeffs * mask_tiled


_RGB_TO_YCBR_MATRIX = torch.tensor([
    [0.29900,  0.58700,  0.11400],
    [-0.16874, -0.33126,  0.50000],
    [0.50000, -0.41869, -0.08131],
], dtype=torch.float32)

_YCBR_BIAS = torch.tensor([0.0, 0.5, 0.5], dtype=torch.float32)

_YCBR_TO_RGB_MATRIX = torch.linalg.inv(_RGB_TO_YCBR_MATRIX)

def rgb_to_ycbr(x: torch.Tensor) -> torch.Tensor :
    M = _RGB_TO_YCBR_MATRIX.to(x.device)
    bias = _YCBR_BIAS.to(x.device)

    out = x.permute(0, 2, 3, 1) @ M.t()
    out = out + bias
    return out.permute(0, 3, 1 , 2)

def ycbr_to_rgb(x: torch.Tensor) -> torch.Tensor:

    M_inv = _YCBR_TO_RGB_MATRIX.to(x.device)
    bias = _YCBR_BIAS.to(x.device)

    out = x.permute(0, 2, 3, 1) - bias
    out = out @ M_inv.t()

    return out.permute(0, 3, 1, 2)
