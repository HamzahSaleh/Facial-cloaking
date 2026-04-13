"""Smoke-test the development environment.

Prints torch / CUDA status, loads one LFW image, and runs it through a CLIP
ViT-B/32 encoder. If any step fails, fix the environment before proceeding.

Usage:
    python scripts/verify_env.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `import facial_cloaking` when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from facial_cloaking.data import iter_identities_with_min_images, load_image  # noqa: E402
from facial_cloaking.paths import LFW_ROOT  # noqa: E402


def main() -> int:
    print(f"torch version:    {torch.__version__}")
    print(f"cuda available:   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda device:      {torch.cuda.get_device_name(0)}")

    if not LFW_ROOT.exists():
        print(f"ERROR: LFW root not found at {LFW_ROOT}", file=sys.stderr)
        return 1

    # Grab the first identity that has at least one image.
    identity, images = next(iter_identities_with_min_images(min_images=1))
    sample_path = images[0]
    img: Image.Image = load_image(sample_path)
    print(f"sample identity:  {identity}")
    print(f"sample path:      {sample_path.name}")
    print(f"sample size:      {img.size}")

    # CLIP encoder smoke test.
    try:
        import open_clip
    except ImportError:
        print("ERROR: open_clip_torch not installed", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(device).eval()

    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(tensor)
    print(f"clip embed shape: {tuple(emb.shape)}")

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
