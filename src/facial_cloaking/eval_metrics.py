"""Evaluation metrics for the three cloaking surfaces and visual quality.

Functions are pure and operate on already-encoded tensors / PIL images so the
pipeline orchestrator stays thin and we can unit-test without CLIP.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ---------------------------------------------------------------------------
# Identity surfaces (operate on L2-normalized embeddings)
# ---------------------------------------------------------------------------

def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    return x if x.dim() == 2 else x.unsqueeze(0)


def cosine_matrix(queries: torch.Tensor, gallery: torch.Tensor) -> torch.Tensor:
    """Cosine similarity matrix [Nq, Ng]; inputs assumed L2-normalized."""
    return _ensure_2d(queries) @ _ensure_2d(gallery).T


def rank1_accuracy(
    queries: torch.Tensor,
    query_labels: list[str],
    gallery: torch.Tensor,
    gallery_labels: list[str],
) -> float:
    """Top-1 retrieval accuracy: fraction of queries whose nearest gallery
    centroid is the true identity."""
    if not query_labels:
        return 0.0
    sims = cosine_matrix(queries, gallery)
    pred_idx = sims.argmax(dim=1).cpu().tolist()
    correct = sum(1 for i, p in enumerate(pred_idx) if gallery_labels[p] == query_labels[i])
    return correct / len(query_labels)


def mean_true_cosine(
    queries: torch.Tensor,
    query_labels: list[str],
    gallery: torch.Tensor,
    gallery_labels: list[str],
) -> float:
    """Mean cosine similarity of each query to its true-identity centroid."""
    if not query_labels:
        return 0.0
    label_to_idx = {lab: i for i, lab in enumerate(gallery_labels)}
    sims = cosine_matrix(queries, gallery).cpu()
    vals = [sims[i, label_to_idx[lab]].item() for i, lab in enumerate(query_labels) if lab in label_to_idx]
    return float(np.mean(vals)) if vals else 0.0


def qa_gap(
    queries: torch.Tensor,
    query_labels: list[str],
    gallery: torch.Tensor,
    gallery_labels: list[str],
) -> float:
    """Mean (cos-to-true-id  -  max cos-to-other-id).

    Positive => still recognizable as true identity (Q&A succeeds).
    Negative => closer to a wrong identity (Q&A fails).
    """
    if not query_labels:
        return 0.0
    label_to_idx = {lab: i for i, lab in enumerate(gallery_labels)}
    sims = cosine_matrix(queries, gallery).cpu()
    gaps: list[float] = []
    for i, lab in enumerate(query_labels):
        if lab not in label_to_idx:
            continue
        true_i = label_to_idx[lab]
        true_sim = sims[i, true_i].item()
        masked = sims[i].clone()
        masked[true_i] = float("-inf")
        max_other = masked.max().item()
        gaps.append(true_sim - max_other)
    return float(np.mean(gaps)) if gaps else 0.0


def null_attractor_gap(
    queries: torch.Tensor,
    query_labels: list[str],
    gallery: torch.Tensor,
    gallery_labels: list[str],
    e_null: torch.Tensor,
) -> float:
    """Mean (cos-to-true-id  -  cos-to-null).

    Negative => embeddings have been pulled toward the null attractor
    (collapse working as designed).
    """
    if not query_labels:
        return 0.0
    label_to_idx = {lab: i for i, lab in enumerate(gallery_labels)}
    null = torch.nn.functional.normalize(e_null.flatten().unsqueeze(0), dim=-1)
    sims_g = cosine_matrix(queries, gallery).cpu()
    sims_n = cosine_matrix(queries, null).cpu().squeeze(1)
    diffs: list[float] = []
    for i, lab in enumerate(query_labels):
        if lab not in label_to_idx:
            continue
        true_sim = sims_g[i, label_to_idx[lab]].item()
        diffs.append(true_sim - sims_n[i].item())
    return float(np.mean(diffs)) if diffs else 0.0


# ---------------------------------------------------------------------------
# Editing-resistance hook
# ---------------------------------------------------------------------------

def editing_resistance(
    edited_query: torch.Tensor,
    query_labels: list[str],
    gallery: torch.Tensor,
    gallery_labels: list[str],
) -> dict[str, float]:
    """Score embeddings produced by an external editing pipeline.

    Lower mean true-cos and lower rank-1 => editing failed to recover the
    original identity from the cloaked input (cloak resisted the edit).
    """
    return {
        "edited_rank1": rank1_accuracy(edited_query, query_labels, gallery, gallery_labels),
        "edited_mean_true_cos": mean_true_cosine(edited_query, query_labels, gallery, gallery_labels),
    }


# ---------------------------------------------------------------------------
# Visual quality
# ---------------------------------------------------------------------------

def _to_uint8(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def image_quality(original: Image.Image, candidate: Image.Image) -> dict[str, float]:
    """SSIM + PSNR between two same-size PIL images."""
    a = _to_uint8(original)
    b = _to_uint8(candidate)
    if a.shape != b.shape:
        b_resized = candidate.resize(original.size, Image.BILINEAR)
        b = _to_uint8(b_resized)
    ssim = structural_similarity(a, b, channel_axis=-1, data_range=255)
    psnr = peak_signal_noise_ratio(a, b, data_range=255)
    return {"ssim": float(ssim), "psnr": float(psnr)}


def aggregate_quality(pairs: list[tuple[Path, Path]]) -> dict[str, float]:
    """Mean SSIM/PSNR across a list of (original, candidate) path pairs."""
    ssims: list[float] = []
    psnrs: list[float] = []
    for orig_p, cand_p in pairs:
        with Image.open(orig_p) as o, Image.open(cand_p) as c:
            q = image_quality(o, c)
        ssims.append(q["ssim"])
        psnrs.append(q["psnr"])
    if not ssims:
        return {"ssim": float("nan"), "psnr": float("nan")}
    return {"ssim": float(np.mean(ssims)), "psnr": float(np.mean(psnrs))}
