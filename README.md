# Manipulation-Resistant Face Cloaking for Image Editing Systems

**Project group:** Hamzah Saleh, Dakota Williams, Holden Roaten.

A lightweight face-cloaking method that makes a portrait less useful as a
reference for downstream face editing, identity-based matching, and identity
question answering, while keeping the image visually similar to the original.

Unlike prior additive-perturbation cloaks (Fawkes, LowKey), this method aims
for **irreversibility**: the perturbation is structurally entangled with the
identity-critical content of the image, so it cannot be estimated and
subtracted, denoised away, or JPEG-compressed out.

## Approach

Three mechanisms are combined into a single per-image optimization:

| Component                     | What it does                                                                                                                             | Why it resists removal                                                                  |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **Embedding collapse**        | Pushes every cloaked image toward a shared "null attractor" in CLIP space (the mean of all identity centroids).                          | Many-to-one mapping: there is no inverse from the null point back to the true identity. |
| **Frequency entanglement**    | Concentrates perturbation energy in mid-frequency DCT bands (bands 3–12) that carry identity cues (eye spacing, jawline, nose geometry). | Filtering the cloak out requires destroying the same bands that encode the face.        |
| **Non-linear channel mixing** | Replaces `x + δ` with `softplus(W · x + δ)` using a small learnable 3×3 mixing matrix `W`.                                               | No closed-form inverse; the additive delta cannot simply be subtracted.                 |

The full loss, optimization schedule, and evaluation surfaces are described in
[plan.md](plan.md).

## Repository layout

```
data/                        # LFW deep-funneled dataset (not committed)
src/facial_cloaking/         # importable package (paths, data, embed, ...)
scripts/                     # standalone scripts (env check, split builder, ...)
splits/                      # versioned CSV partition manifests
outputs/                     # generated artifacts (attractors, cloaked images, reports)
plan.md                      # design doc for the cloaking algorithm
environment.md               # environment / CUDA setup notes
```

## Setup

```bash
# 1. Create and activate a virtual environment (Windows bash shown)
python -m venv venv
source venv/Scripts/activate

# 2. Install dependencies
#    For a CUDA-enabled torch build, install torch first — see environment.md.
pip install -r requirements.txt

# 3. Download the LFW deep-funneled dataset into data/lfw-deepfunneled/
#    so that data/lfw-deepfunneled/lfw-deepfunneled/<identity>/*.jpg exists.

# 4. Smoke-test the environment (prints torch/CUDA status, loads one image,
#    runs a CLIP forward pass).
python scripts/verify_env.py

# 5. Build the LFW partitions used by every later experiment.
python scripts/build_splits.py
```

`build_splits.py` is deterministic — re-running it produces byte-identical
CSVs. See [environment.md](environment.md) for PyTorch / CUDA details.

## Data partitions

Built from LFW identities with at least 6 images. For each of 100 identities:

| Partition      | Per identity | Purpose                                |
| -------------- | ------------ | -------------------------------------- |
| `protected`    | 2            | inputs to the cloaking method          |
| `clean_test`   | 3            | retrieval queries + identity questions |
| `editing_eval` | 1            | reference image for the editing test   |

See [splits/split_manifest.json](splits/split_manifest.json) for the exact
configuration used.

## Pipeline

The full experiment pipeline (scripts marked *planned* are specified in
[plan.md](plan.md) but not yet implemented):

```
build_splits.py            -> splits/*.csv
compute_attractors.py  *   -> outputs/attractors.pt          (null + per-identity embeddings)
run_cloaking.py        *   -> outputs/cloaked/*.png          (+ cloaking_report.json)
evaluate.py            *   -> retrieval / Q&A / editing metrics
robustness_test.py     *   -> retrieval after JPEG / blur / filtering purification
```

Typical usage (once implemented):

```bash
python scripts/compute_attractors.py
python scripts/run_cloaking.py --epsilon 8 --steps 300 --lr 0.01
python scripts/evaluate.py
python scripts/robustness_test.py
```

## Package modules

- [src/facial_cloaking/paths.py](src/facial_cloaking/paths.py) — canonical
  project paths (`PROJECT_ROOT`, `LFW_ROOT`, `SPLITS_DIR`).
- [src/facial_cloaking/data.py](src/facial_cloaking/data.py) — identity
  enumeration, per-identity image listing, PIL image loading.
- [src/facial_cloaking/embed.py](src/facial_cloaking/embed.py) — CLIP
  ViT-B/32 wrapper: `load_clip_model`, `encode_image`,
  `compute_identity_embeddings`, `compute_null_attractor`.

Planned (see [plan.md](plan.md)): `dct_utils.py` (differentiable blockwise DCT
+ mid-frequency mask) and `cloak.py` (per-image optimization loop,
`CloakConfig`, `cloak_image`, `cloak_batch`).

## Evaluation surfaces

1. **Identity retrieval.** Cosine similarity of cloaked embeddings against all
   100 identity centroids. Report Rank-1 accuracy and mean cosine similarity
   to the true identity.
2. **Identity Q&A resistance.** Gap between cosine similarity to the true
   identity and the max cosine similarity to any other identity.
3. **Editing resistance.** Cosine similarity between an edited output and the
   original uncloaked identity, via an external editing pipeline.
4. **Visual quality.** SSIM and PSNR between cloaked and original.

A cloak is considered irreversible if retrieval accuracy stays near 0% after
JPEG compression, Gaussian blur, bilateral filtering, and bit-depth reduction
(see `robustness_test.py`).

## Industry-aligned default profile

Use this as the project default unless an experiment explicitly states a
different configuration.

### Frequency and transform

- Block transform: 8x8 DCT (JPEG-compatible)
- Coefficient order: JPEG zig-zag order
- Target band for frequency entanglement: indices 3-12 (mid-frequency)

### Color handling

- Working color split for frequency weighting: YCbCr
- Matrix convention: ITU-R BT.601
- Weighting policy: prioritize Y (luminance), weaker Cb/Cr weighting

### Perturbation budgets

- Primary run: epsilon = 8/255
- Required ablations: epsilon in {4/255, 16/255}

### Quality and identity metrics

- Visual quality: SSIM and PSNR
- Perceptual quality (recommended): LPIPS
- Identity metrics: Rank-1 retrieval and mean cosine to true identity

### Robustness stress tests

- JPEG recompression at quality 95, 75, and 50
- Gaussian blur (sigma 1.0 and 2.0)
- Bilateral filter
- Bit-depth reduction (6-bit and 4-bit)

### Reporting standard

- Fix random seeds and report them
- Report exact preprocessing and color-space transform used
- Report means and standard deviations across identities/images
- Report uncloaked baseline and cloaked deltas side-by-side

## Status

- [x] Environment, data splits, CLIP embedding utilities
- [ ] Null attractor + per-identity embedding precomputation script
- [ ] Differentiable DCT utilities
- [ ] Cloaking optimization loop
- [ ] Evaluation and robustness scripts
