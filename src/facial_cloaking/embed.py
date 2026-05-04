from __future__ import annotations

import torch
import open_clip
import csv
from pathlib import Path

from .paths import PROJECT_ROOT, SPLITS_DIR
from .data import load_image

def load_clip_model(device: str) -> tuple[torch.nn.Module, object] :
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(device).eval()

    return model, preprocess

def encode_image(model: torch.nn.Module, preprocess, img, device: str) -> torch.Tensor :
    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.encode_image(tensor)

    return torch.nn.functional.normalize(features, dim=-1).squeeze(0)

def _encode_paths(
    model: torch.nn.Module, preprocess,
    paths: list[Path], device : str,
    batch_size: int = 32
    )->torch.Tensor :

    images = [load_image(p) for p in paths]
    return encode_pil_images(model, preprocess, images, device, batch_size=batch_size)


def encode_pil_images(
    model: torch.nn.Module, preprocess,
    images: list,
    device: str,
    batch_size: int = 32,
) -> torch.Tensor:
    """Encode a list of PIL images to L2-normalized CLIP embeddings."""
    all_embeddings: list[torch.Tensor] = []

    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]

        tensors = [preprocess(img) for img in chunk]
        batch = torch.stack(tensors).to(device)

        with torch.no_grad():
            features = model.encode_image(batch)

        all_embeddings.append(torch.nn.functional.normalize(features, dim=-1))

    return torch.cat(all_embeddings, dim=0)

def _read_csv_identity(csv_path: Path)-> dict[str, list[Path]] :
    """
    Create a dictionary {identity, filepath} from csv created by build_splits.py
    """
    groups: dict[str, list[Path]] = {}

    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f) :
            identity = row["identity"].strip()
            img_path = PROJECT_ROOT / row["image_path"].strip()

            groups.setdefault(identity, []).append(img_path)

    return groups
    
def compute_identity_embeddings(model: torch.nn.Module, preprocess, csv_path: Path |str, device: str)->dict[str, torch.Tensor] :
    
    groups = _read_csv_identity(csv_path)

    identity_embeddings: dict[str, torch.Tensor] = {}

    for identity, paths in groups.items() :
        embeddings = _encode_paths(model, preprocess, paths, device)
        centroid = torch.nn.functional.normalize(embeddings.mean(dim=0, keepdim=True),dim=-1)
        identity_embeddings[identity] = centroid.squeeze(0)

    return identity_embeddings
    
def compute_null_attractor(model: torch.nn.Module, preprocess, clean_test_csv: Path | str, device: str)->torch.Tensor :
    
    identity_embeddings = compute_identity_embeddings(model, preprocess, clean_test_csv, device)

    centroids = torch.stack(list(identity_embeddings.values()))
    e_null = torch.nn.functional.normalize(centroids.mean(dim=0, keepdim=True), dim=-1)

    return e_null.squeeze(0)
