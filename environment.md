# Environment notes

## Python

Python 3.10 or newer is recommended.

## Virtual environment (Windows bash)

```bash
python -m venv venv
source venv/Scripts/activate
```

## PyTorch install

`requirements.txt` pins `torch>=2.2`, but it does **not** select a CUDA build —
the right wheel depends on your local NVIDIA driver. Install the matching torch
build *before* `pip install -r requirements.txt`:

| Setup            | Command                                                                                  |
|------------------|------------------------------------------------------------------------------------------|
| CPU only         | `pip install torch torchvision`                                                          |
| CUDA 12.1        | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`       |
| CUDA 11.8        | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`       |

Then run:

```bash
pip install -r requirements.txt
python scripts/verify_env.py
```

`verify_env.py` prints the torch version, whether CUDA is available, the size
of a sample LFW image, and the shape of a CLIP embedding. If any of those fail,
fix the environment before running `build_splits.py`.

## Notes on `open_clip_torch`

The first call to `open_clip.create_model_and_transforms("ViT-B-32",
pretrained="laion2b_s34b_b79k")` downloads the pretrained weights (~600 MB)
into your HuggingFace / open_clip cache. Subsequent runs use the cached copy.
