"""LLM-style image-edit providers used by the evaluation pipeline.

The evaluation pipeline scores arbitrary "method" folders against the
identity gallery, but to compare *editing resistance* across cloaking
methods we need every method to be edited under the **same** alteration
procedure. Otherwise differences between method scores reflect differences
between LLM runs (prompts, sampling temperature, network state) rather than
the cloak itself.

This module defines:

- ``EditProvider``: a small protocol every editor must implement
  (``name``, ``params`` dict, and a deterministic ``apply`` method).
- ``LocalStubEditor``: a fully deterministic, offline editor that mimics
  the kind of identity-flattening transform an LLM regenerator applies
  (downscale -> blur -> tonal shift -> JPEG re-encode). Its outputs depend
  only on the input image, the seed, and the named parameters, so the
  exact same files are produced on every run. This is the default so the
  pipeline always produces comparable numbers, even without API keys.
- ``run_edits``: applies one provider to every row of a CSV split,
  reading sources from a chosen "method" directory (or the original
  uncloaked images) and writing edited outputs to a target directory
  using the *same* filename layout as the rest of the pipeline.

Real LLM-backed providers can be added by implementing ``EditProvider``
and passing the same ``--seed`` / ``--prompt`` to keep runs reproducible.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Protocol, ClassVar

import numpy as np
from PIL import Image, ImageFilter

from .data import load_image
from .paths import PROJECT_ROOT


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------

class EditProvider(Protocol):
    """An LLM/image-edit backend with deterministic, recordable parameters."""

    name: str

    @property
    def params(self) -> dict: ...

    def apply(self, image: Image.Image, *, prompt: str, seed: int) -> Image.Image: ...


# ---------------------------------------------------------------------------
# Local deterministic stub editor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LocalStubEditor:
    """Deterministic offline edit that approximates an LLM regenerator.

    The transform is intentionally lossy in the identity-bearing
    mid-frequencies (downscale + blur), then re-encoded as JPEG. It is
    seeded by ``(seed, prompt, filename)`` so two runs with the same
    arguments produce byte-identical PNG/JPEG outputs.
    """

    downscale: int = 4
    blur_sigma: float = 1.0
    jpeg_quality: int = 90
    noise_sigma: float = 2.0

    name: str = "local_stub"

    @property
    def params(self) -> dict:
        return {
            "downscale": self.downscale,
            "blur_sigma": self.blur_sigma,
            "jpeg_quality": self.jpeg_quality,
            "noise_sigma": self.noise_sigma,
        }

    def _row_seed(self, seed: int, prompt: str, key: str) -> int:
        h = hashlib.sha256(
            f"{seed}|{prompt}|{key}".encode("utf-8")
        ).digest()
        return int.from_bytes(h[:8], "big", signed=False)

    def apply(self, image: Image.Image, *, prompt: str, seed: int) -> Image.Image:
        return self.apply_keyed(image, prompt=prompt, seed=seed, key="")

    def apply_keyed(
        self, image: Image.Image, *, prompt: str, seed: int, key: str
    ) -> Image.Image:
        img = image.convert("RGB")
        w, h = img.size

        # 1. Downscale -> upscale (kills high-frequency identity cues).
        small = img.resize(
            (max(1, w // self.downscale), max(1, h // self.downscale)),
            Image.BILINEAR,
        )
        recon = small.resize((w, h), Image.BICUBIC)

        # 2. Mild Gaussian blur on top of that.
        blurred = recon.filter(ImageFilter.GaussianBlur(radius=self.blur_sigma))

        # 3. Deterministic seeded noise (prompt+key folded into the seed).
        rng = np.random.default_rng(self._row_seed(seed, prompt, key))
        arr = np.asarray(blurred, dtype=np.float32)
        noise = rng.normal(0.0, self.noise_sigma, size=arr.shape).astype(np.float32)
        out = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)

        # 4. JPEG re-encode at a fixed quality (matches "regenerate as JPEG"
        # behaviour of most hosted editors).
        buf = BytesIO()
        Image.fromarray(out).save(buf, format="JPEG", quality=int(self.jpeg_quality))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

def get_provider(name: str) -> EditProvider:
    """Resolve a provider by name. Currently only the local stub is
    bundled; custom providers can be registered by importing this module
    and assigning into ``_PROVIDERS``.
    """
    if name in _PROVIDERS:
        return _PROVIDERS[name]
    raise SystemExit(
        f"unknown edit provider: {name!r}. available: {sorted(_PROVIDERS)}"
    )


@dataclass
class IPAdapterEditor:
    """Image editor using IP-Adapter (OpenCLIP-ViT-H/14 image conditioning).

    Generation is conditioned on the CLIP image embedding of the input.
    With ip_adapter_scale=1.0, text is ignored — output is driven entirely
    by the CLIP representation of the source image.

    Requires: pip install diffusers transformers accelerate
    First run downloads ~5 GB of weights to ~/.cache/huggingface.
    """
    name: str = "ipadapter"
    ip_adapter_scale: float = 1.0
    num_inference_steps: int = 30
    guidance_scale: float = 7.5

    # Class-level pipeline cache — shared across instances, not a dataclass field
    _pipe: ClassVar = None

    @property
    def params(self) -> dict:
        return {
            "ip_adapter_scale": self.ip_adapter_scale,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
        }

    def _get_pipeline(self):
        if IPAdapterEditor._pipe is None:
            import torch
            from diffusers import StableDiffusionPipeline

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype  = torch.float16 if device == "cuda" else torch.float32

            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
            ).to(device)

            pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="models",
                weight_name="ip-adapter_sd15.bin",
            )
            pipe.set_ip_adapter_scale(self.ip_adapter_scale)

            if device == "cpu":
                pipe.enable_attention_slicing()

            IPAdapterEditor._pipe = pipe

        return IPAdapterEditor._pipe

    def apply(self, image: Image.Image, *, prompt: str, seed: int) -> Image.Image:
        import torch

        pipe = self._get_pipeline()
        generator = torch.Generator(device=pipe.device.type).manual_seed(seed)

        result = pipe(
            prompt=prompt,
            ip_adapter_image=image.convert("RGB"),
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
            height=256,
            width=256,
        )
        return result.images[0]

_PROVIDERS: dict[str, EditProvider] = {
    "local_stub": LocalStubEditor(),
    "ipadapter": IPAdapterEditor(),
}


# ---------------------------------------------------------------------------
# Edit runner
# ---------------------------------------------------------------------------

@dataclass
class EditRunReport:
    provider: str
    params: dict
    prompt: str
    seed: int
    source: str
    csv: str
    out_dir: str
    n_inputs: int
    n_written: int
    skipped: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "params": self.params,
            "prompt": self.prompt,
            "seed": self.seed,
            "source": self.source,
            "csv": self.csv,
            "out_dir": self.out_dir,
            "n_inputs": self.n_inputs,
            "n_written": self.n_written,
            "skipped": self.skipped,
        }


def _read_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def resolve_source_path(row: dict, source_dir: Path | None) -> Path:
    """Where to read the *input* to the editor for this row.

    - ``source_dir is None`` -> the original uncloaked image
      (``row['image_path']``). Used when running the LLM directly on
      uncloaked references (the baseline edit).
    - otherwise -> ``source_dir / row['filename']`` (the cloaked output
      of some method).
    """
    if source_dir is None:
        return PROJECT_ROOT / row["image_path"].strip()
    return Path(source_dir) / row["filename"].strip()


def run_edits(
    *,
    csv_path: Path,
    source_dir: Path | None,
    out_dir: Path,
    provider: EditProvider,
    prompt: str,
    seed: int,
    overwrite: bool = False,
) -> EditRunReport:
    """Apply ``provider`` to every CSV row and write outputs to ``out_dir``.

    Output filenames mirror the ``filename`` column so the result is a
    drop-in "method folder" for ``scripts/evaluate.py``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(csv_path)
    skipped: list[str] = []
    n_written = 0
    for r in rows:
        fname = r["filename"].strip()
        src = resolve_source_path(r, source_dir)
        if not src.exists():
            skipped.append(str(src))
            continue
        dst = out_dir / fname
        if dst.exists() and not overwrite:
            n_written += 1
            continue
        with load_image(src) as img:
            if hasattr(provider, "apply_keyed"):
                edited = provider.apply_keyed(  # type: ignore[attr-defined]
                    img, prompt=prompt, seed=seed, key=fname
                )
            else:
                edited = provider.apply(img, prompt=prompt, seed=seed)
        edited.save(dst, "JPEG", quality=95)
        n_written += 1

    report = EditRunReport(
        provider=provider.name,
        params=dict(provider.params),
        prompt=prompt,
        seed=seed,
        source=str(source_dir) if source_dir is not None else "uncloaked",
        csv=str(csv_path),
        out_dir=str(out_dir),
        n_inputs=len(rows),
        n_written=n_written,
        skipped=skipped,
    )
    (out_dir / "_edit_run.json").write_text(json.dumps(report.to_dict(), indent=2))
    return report
