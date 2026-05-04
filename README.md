# Manipulation-Resistant Face Cloaking for Image Editing Systems

Project group: Hamzah Saleh, Dakota Williams, Holden Roaten.

A lightweight face-cloaking method that makes a portrait less useful as a
reference for downstream face editing, identity-based matching, and identity
question answering, while keeping the image visually similar to the original.

## Repository layout

```
data/                        # LFW deep-funneled dataset (not committed)
src/facial_cloaking/         # importable package
scripts/                     # standalone scripts (env check, split builder, ...)
splits/                      # versioned CSV partition manifests
```

## Setup

```bash
# 1. Create and activate a virtual environment (Windows bash shown)
python -m venv venv
source venv/Scripts/activate

# 2. Install dependencies
#    For CUDA-enabled torch, see environment.md first.
pip install -r requirements.txt

# 3. Smoke-test the environment
python scripts/verify_env.py

# 4. Build the LFW partitions used by every later experiment
python scripts/build_splits.py
```

`build_splits.py` is deterministic — re-running it produces byte-identical CSVs.

## Data partitions

Built from LFW identities with at least 6 images. For each of 100 identities:

| Partition       | Per identity | Purpose                                |
|-----------------|--------------|----------------------------------------|
| protected       | 2            | inputs to the cloaking method          |
| clean_test      | 3            | retrieval queries + identity questions |
| editing_eval    | 1            | reference image for the editing test   |

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
