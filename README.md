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
