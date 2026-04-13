"""Central path constants so every script resolves locations identically."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
LFW_ROOT = DATA_ROOT / "lfw-deepfunneled" / "lfw-deepfunneled"
SPLITS_DIR = PROJECT_ROOT / "splits"
