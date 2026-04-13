"""LFW data helpers: identity enumeration and image loading."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from PIL import Image

from .paths import LFW_ROOT


def list_identities(root: Path = LFW_ROOT) -> list[str]:
    """Return all identity folder names under the LFW root, sorted alphabetically."""
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def list_identity_images(identity: str, root: Path = LFW_ROOT) -> list[Path]:
    """Return the sorted list of .jpg paths for a given identity."""
    return sorted((root / identity).glob("*.jpg"))


def iter_identities_with_min_images(
    min_images: int, root: Path = LFW_ROOT
) -> Iterator[tuple[str, list[Path]]]:
    """Yield (identity, image_paths) for identities with at least `min_images` images."""
    for identity in list_identities(root):
        images = list_identity_images(identity, root)
        if len(images) >= min_images:
            yield identity, images


def load_image(path: Path) -> Image.Image:
    """Load an image as RGB PIL.Image."""
    return Image.open(path).convert("RGB")
