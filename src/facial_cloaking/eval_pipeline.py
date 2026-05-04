"""Orchestrator for evaluating one or more cloaking methods.

A "method" is identified by name and is resolved to a list of
(identity, true_image_path, candidate_image_path) rows.

- The special method name ``"uncloaked"`` resolves candidate paths to the
  original image paths from the CSV (upper-bound retrieval reference).
- Any other method resolves candidates as ``<method_dir>/<filename>``
  using the ``filename`` column of the CSV.

The pipeline is method-agnostic: it never knows how images were produced.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image

from . import eval_metrics
from .data import load_image
from .embed import encode_pil_images
from .paths import PROJECT_ROOT
from .purify import Purification


@dataclass(frozen=True)
class MethodSpec:
    """Where to find the candidate (cloaked) image for each protected row."""

    name: str
    candidate_dir: Path | None  # None => use the original image_path (uncloaked baseline)

    @classmethod
    def uncloaked(cls) -> "MethodSpec":
        return cls(name="uncloaked", candidate_dir=None)

    @classmethod
    def folder(cls, name: str, directory: Path | str) -> "MethodSpec":
        return cls(name=name, candidate_dir=Path(directory))


@dataclass
class EvalRow:
    identity: str
    filename: str
    original_path: Path
    candidate_path: Path


@dataclass
class Gallery:
    """Identity-centroid gallery + null attractor (cached on disk)."""

    labels: list[str]
    embeddings: torch.Tensor   # [N, D], L2-normalized
    e_null: torch.Tensor       # [D],    L2-normalized

    def to_dict(self) -> dict:
        return {
            "labels": self.labels,
            "embeddings": self.embeddings.cpu(),
            "e_null": self.e_null.cpu(),
        }

    @classmethod
    def from_dict(cls, payload: dict, device: str = "cpu") -> "Gallery":
        return cls(
            labels=list(payload["labels"]),
            embeddings=payload["embeddings"].to(device),
            e_null=payload["e_null"].to(device),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_dict(), path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "Gallery":
        return cls.from_dict(torch.load(path, map_location=device), device=device)


# ---------------------------------------------------------------------------
# CSV row resolution
# ---------------------------------------------------------------------------

def _read_csv(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def resolve_method_rows(method: MethodSpec, csv_path: Path) -> list[EvalRow]:
    rows: list[EvalRow] = []
    for r in _read_csv(csv_path):
        identity = r["identity"].strip()
        filename = r["filename"].strip()
        orig = PROJECT_ROOT / r["image_path"].strip()
        if method.candidate_dir is None:
            cand = orig
        else:
            cand = method.candidate_dir / filename
        if not cand.exists():
            continue
        rows.append(EvalRow(identity=identity, filename=filename,
                            original_path=orig, candidate_path=cand))
    return rows


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _encode_images(model, preprocess, images: list[Image.Image], device: str,
                   batch_size: int = 32) -> torch.Tensor:
    if not images:
        return torch.empty(0)
    return encode_pil_images(model, preprocess, images, device, batch_size=batch_size)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class MethodScore:
    method: str
    n_images: int
    rank1: float
    mean_true_cos: float
    qa_gap: float
    null_gap: float
    ssim: float
    psnr: float
    purifications: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "n_images": self.n_images,
            "rank1": self.rank1,
            "mean_true_cos": self.mean_true_cos,
            "qa_gap": self.qa_gap,
            "null_gap": self.null_gap,
            "ssim": self.ssim,
            "psnr": self.psnr,
            "purifications": self.purifications,
        }


def _identity_metrics(
    queries: torch.Tensor,
    labels: list[str],
    gallery: Gallery,
) -> dict[str, float]:
    return {
        "rank1": eval_metrics.rank1_accuracy(queries, labels, gallery.embeddings, gallery.labels),
        "mean_true_cos": eval_metrics.mean_true_cosine(queries, labels, gallery.embeddings, gallery.labels),
        "qa_gap": eval_metrics.qa_gap(queries, labels, gallery.embeddings, gallery.labels),
        "null_gap": eval_metrics.null_attractor_gap(
            queries, labels, gallery.embeddings, gallery.labels, gallery.e_null
        ),
    }


def score_method(
    method: MethodSpec,
    *,
    csv_path: Path,
    gallery: Gallery,
    model,
    preprocess,
    device: str,
    purifications: dict[str, Purification] | None = None,
    batch_size: int = 32,
) -> MethodScore:
    """Score one method on the 3 surfaces + visual quality, plus optional
    robustness purifications."""
    rows = resolve_method_rows(method, csv_path)
    if not rows:
        raise FileNotFoundError(
            f"No candidate images found for method '{method.name}' "
            f"(dir={method.candidate_dir})"
        )

    # Load candidate images once; reuse for clean + each purification.
    cand_imgs = [load_image(r.candidate_path) for r in rows]
    labels = [r.identity for r in rows]

    queries = _encode_images(model, preprocess, cand_imgs, device, batch_size)
    base = _identity_metrics(queries, labels, gallery)

    # Visual quality vs. originals (skip when method is uncloaked).
    if method.candidate_dir is None:
        ssim = float("nan")
        psnr = float("nan")
    else:
        pairs = [(r.original_path, r.candidate_path) for r in rows]
        q = eval_metrics.aggregate_quality(pairs)
        ssim = q["ssim"]
        psnr = q["psnr"]

    purif_results: dict[str, dict[str, float]] = {}
    for pname, fn in (purifications or {}).items():
        purified = [fn(img) for img in cand_imgs]
        pq = _encode_images(model, preprocess, purified, device, batch_size)
        purif_results[pname] = _identity_metrics(pq, labels, gallery)

    return MethodScore(
        method=method.name,
        n_images=len(rows),
        rank1=base["rank1"],
        mean_true_cos=base["mean_true_cos"],
        qa_gap=base["qa_gap"],
        null_gap=base["null_gap"],
        ssim=ssim,
        psnr=psnr,
        purifications=purif_results,
    )


def compare_methods(
    methods: Iterable[MethodSpec],
    *,
    csv_path: Path,
    gallery: Gallery,
    model,
    preprocess,
    device: str,
    purifications: dict[str, Purification] | None = None,
    batch_size: int = 32,
) -> list[MethodScore]:
    return [
        score_method(
            m,
            csv_path=csv_path,
            gallery=gallery,
            model=model,
            preprocess=preprocess,
            device=device,
            purifications=purifications,
            batch_size=batch_size,
        )
        for m in methods
    ]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_markdown_table(scores: list[MethodScore]) -> str:
    header = (
        "| method | n | rank1 | mean_true_cos | qa_gap | null_gap | ssim | psnr |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for s in scores:
        lines.append(
            f"| {s.method} | {s.n_images} | "
            f"{s.rank1:.3f} | {s.mean_true_cos:.3f} | {s.qa_gap:+.3f} | "
            f"{s.null_gap:+.3f} | {s.ssim:.3f} | {s.psnr:.2f} |"
        )
    if any(s.purifications for s in scores):
        lines.append("")
        lines.append("### Robustness (rank1 / mean_true_cos under purification)")
        all_purifs = sorted({p for s in scores for p in s.purifications})
        head = "| method | " + " | ".join(all_purifs) + " |"
        sep = "|---|" + "|".join(["---:"] * len(all_purifs)) + "|"
        lines.append(head)
        lines.append(sep)
        for s in scores:
            cells = []
            for p in all_purifs:
                m = s.purifications.get(p)
                if m is None:
                    cells.append("—")
                else:
                    cells.append(f"{m['rank1']:.2f} / {m['mean_true_cos']:.2f}")
            lines.append(f"| {s.method} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
