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
data/                        # LFW deep-funneled dataset (not committed)*
src/facial_cloaking/         # importable package (paths, data, embed, ...)
scripts/                     # standalone scripts (env check, split builder, ...)
splits/                      # versioned CSV partition manifests
outputs/                     # generated artifacts (attractors, cloaked images, reports)
plan.md                      # design doc for the cloaking algorithm
environment.md               # environment / CUDA setup notes
```
*[LFW Deep-Funneled download via kaggle](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)
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

## Evaluation pipeline

Score any number of cloaking methods on three surfaces (identity retrieval,
Q&A gap, null-attractor gap) plus visual quality (SSIM / PSNR), and replay the
same scoring under purification attacks (JPEG, Gaussian blur, bilateral,
bit-depth reduction).

A "method" is just a folder of cloaked images named identically to the
`filename` column of `splits/protected.csv`. The literal name `uncloaked`
uses the original images and is the upper-bound retrieval reference.

### One-time setup

```bash
# Encode the clean_test split into a 100-identity gallery + null attractor.
python scripts/compute_attractors.py
# -> outputs/attractors.pt

# (Optional) generate two reference methods so the pipeline is runnable
# before any real cloak exists.
python scripts/make_baselines.py
# -> outputs/methods/noise/   outputs/methods/blur/
```

### Score from the CLI

```bash
# Compare uncloaked vs. the two reference methods, with a couple of attacks.
python scripts/evaluate.py \
    --methods uncloaked noise=outputs/methods/noise blur=outputs/methods/blur \
    --purify jpeg-75 blur-1.0 bits-4
# Writes a markdown table to stdout and outputs/eval_report.json.

# Same scoring, but with the full standard purification grid.
python scripts/robustness_test.py \
    --methods uncloaked noise=outputs/methods/noise
```

## Streamlit dashboard

A live UI over the same pipeline — useful for iterating on method × purification
combinations without re-loading CLIP each run.

```bash
streamlit run scripts/eval_dashboard.py
# Opens http://localhost:8501
```

Headless launch (no auto browser, no usage stats):

```bash
streamlit run scripts/eval_dashboard.py \
    --server.headless true \
    --server.port 8501 \
    --browser.gatherUsageStats false
```

Requirements: a successful `compute_attractors.py` run (the dashboard reads
`outputs/attractors.pt`) and at least one method folder under
`outputs/methods/` (or pick `uncloaked` only).

The dashboard has three tabs:

- **Score** — multi-select methods + purifications, click *Run* to populate a
  live table of `rank1 / mean_true_cos / qa_gap / null_gap / ssim / psnr` and
  a robustness sub-table; *Save JSON* writes the same payload as
  `scripts/evaluate.py`.
- **Drill-down** — pick a method and identity; see per-image cosine
  similarities and side-by-side thumbnails of the original, the cloaked
  candidate, and (if a purification is enabled) the purified candidate.
- **Compare** — pick two methods previously scored and view an A/B/Δ table
  for both core metrics and the per-purification rank-1.

CLIP and the gallery are cached across reruns (`@st.cache_resource`), so
toggling sidebar widgets does not re-load the model. CPU CLIP encodes ~200
images per method in roughly 1–3 minutes; GPU runs are seconds.

## AI Assistance
This project was developed with assistance from Claude (Anthropic). Claude was used throughout development for tasks including code review, debugging, architecture discussion, and documentation. All final design decisions, implementation, and evaluation were performed by the project authors.
