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

## Evaluation pipeline

The evaluation pipeline lives in [src/facial_cloaking/eval_pipeline.py](src/facial_cloaking/eval_pipeline.py)
and is driven by three scripts under `scripts/`. It is **method-agnostic**:
each cloaking algorithm just has to drop its outputs into a flat directory
whose filenames match the `filename` column of `splits/protected.csv`.

### 1. Precompute the identity gallery (once)

The gallery is the per-identity CLIP centroid plus the null attractor.
It only has to be built once and is cached on disk.

```bash
python scripts/compute_attractors.py \
    --csv splits/clean_test.csv \
    --output outputs/attractors.pt
```

### 2. Produce method outputs

A "method" is a folder of cloaked images named exactly like the
`filename` column of the CSV split being scored. For local
sanity-checking before any cloaking algorithm is wired up, the repo
ships two reference baselines:

```bash
python scripts/make_baselines.py
# writes outputs/methods/noise/*.jpg  and  outputs/methods/blur/*.jpg
```

### 3. Score methods on the three identity surfaces + visual quality

```bash
python scripts/evaluate.py \
    --methods uncloaked \
              noise=outputs/methods/noise \
              blur=outputs/methods/blur \
    --csv splits/protected.csv \
    --attractors outputs/attractors.pt \
    --output outputs/eval_report.json
```

Add purifications inline to also report retrieval after each attack:

```bash
python scripts/evaluate.py \
    --methods uncloaked noise=outputs/methods/noise \
    --purify jpeg-75 blur-1.0 bits-4 bilateral
```

`uncloaked` is a special method name that uses the original images and
gives the upper-bound retrieval reference. The script prints a Markdown
table and writes a JSON report.

### 4. Standard robustness grid

Run the full purification grid from `plan.md` (JPEG 95/75/50, Gaussian
blur σ=1.0/2.0, bilateral, bit-depth 6/4) in one shot:

```bash
python scripts/robustness_test.py \
    --methods uncloaked noise=outputs/methods/noise blur=outputs/methods/blur \
    --output outputs/robustness_report.json
```

### 5. LLM / image-edit alterations (consistent across methods)

To compare cloaks on *editing resistance* every method must be altered
under the **same** edit procedure — same provider, same prompt, same
seed — otherwise score differences reflect LLM run-to-run variance
instead of cloak strength.

[scripts/run_edits.py](scripts/run_edits.py) enforces this. It reads
`splits/editing_eval.csv`, applies the chosen edit provider to every
row of the chosen source folder (or the uncloaked originals), and
writes a drop-in method folder with the same filename layout. A
`_edit_run.json` manifest is saved alongside each output folder
recording the provider, params, prompt, seed, source, and CSV used.

The default provider, `local_stub`, is fully deterministic and offline:
given the same `(image, prompt, seed, filename)` it produces
byte-identical output on every run, on every machine. This guarantees
the LLM-alteration column of the eval report is reproducible even
without API access. To plug a real LLM-backed editor in, implement the
`EditProvider` protocol in
[src/facial_cloaking/edits.py](src/facial_cloaking/edits.py) and add it
to the `_PROVIDERS` registry.

Recommended workflow:

```bash
# Edit the uncloaked baseline (what the LLM does to a clean portrait).
python scripts/run_edits.py \
    --source uncloaked \
    --out outputs/edits/uncloaked \
    --csv splits/editing_eval.csv \
    --seed 0

# Edit each cloaking method's outputs with the SAME provider/prompt/seed.
python scripts/run_edits.py \
    --source outputs/methods/noise \
    --out outputs/edits/noise \
    --csv splits/editing_eval.csv \
    --seed 0

python scripts/run_edits.py \
    --source outputs/methods/blur \
    --out outputs/edits/blur \
    --csv splits/editing_eval.csv \
    --seed 0

# Score the edited folders with the same gallery + metrics.
python scripts/evaluate.py \
    --methods uncloaked_edited=outputs/edits/uncloaked \
              noise_edited=outputs/edits/noise \
              blur_edited=outputs/edits/blur \
    --csv splits/editing_eval.csv \
    --output outputs/edit_eval_report.json
```

A cloak is considered to resist editing if its `*_edited` row has a
**lower** rank-1 and lower mean-true-cosine than `uncloaked_edited`
under the same provider/prompt/seed.

### Output layout

```
outputs/
  attractors.pt                # gallery (centroids + null attractor)
  methods/<name>/*.jpg         # cloaked images per method
  edits/<name>/*.jpg           # LLM-altered images per method
  edits/<name>/_edit_run.json  # provider/prompt/seed manifest
  eval_report.json             # scripts/evaluate.py output
  robustness_report.json       # scripts/robustness_test.py output
  edit_eval_report.json        # eval over edited methods
```
