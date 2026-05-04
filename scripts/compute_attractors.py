"""Precompute the identity-centroid gallery and the null attractor.

Run once:
    python scripts/compute_attractors.py \
        --csv splits/clean_test.csv \
        --output outputs/attractors.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# allow `python scripts/compute_attractors.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from facial_cloaking.embed import compute_identity_embeddings, load_clip_model
from facial_cloaking.eval_pipeline import Gallery
from facial_cloaking.paths import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(PROJECT_ROOT / "splits" / "clean_test.csv"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "attractors.pt"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"[attractors] loading CLIP on {args.device} ...")
    model, preprocess = load_clip_model(args.device)

    print(f"[attractors] encoding identities from {args.csv} ...")
    id_embs = compute_identity_embeddings(model, preprocess, Path(args.csv), args.device)

    labels = sorted(id_embs.keys())
    embeddings = torch.stack([id_embs[k] for k in labels]).to(args.device)
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    e_null = torch.nn.functional.normalize(embeddings.mean(dim=0, keepdim=True), dim=-1).squeeze(0)

    gallery = Gallery(labels=labels, embeddings=embeddings, e_null=e_null)
    out = Path(args.output)
    gallery.save(out)
    print(f"[attractors] saved {len(labels)} identity centroids -> {out}")


if __name__ == "__main__":
    main()
