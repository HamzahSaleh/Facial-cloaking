"""Build the three LFW partitions used by every later experiment.

Outputs (under splits/):
    protected.csv      identity, image_path, filename, partition
    clean_test.csv     same schema
    editing_eval.csv   same schema
    split_manifest.json  reproducibility metadata

The script is deterministic. Re-running it produces byte-identical CSVs.

Usage:
    python scripts/build_splits.py
    python scripts/build_splits.py --num-identities 50 --seed 1
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `import facial_cloaking` when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from facial_cloaking.data import iter_identities_with_min_images  # noqa: E402
from facial_cloaking.paths import LFW_ROOT, PROJECT_ROOT, SPLITS_DIR  # noqa: E402

# Per-identity allocation. Sum == MIN_IMAGES_PER_IDENTITY.
PARTITION_SIZES = {
    "protected": 2,
    "clean_test": 3,
    "editing_eval": 1,
}
MIN_IMAGES_PER_IDENTITY = sum(PARTITION_SIZES.values())  # 6


def to_repo_relative(p: Path) -> str:
    """Return a forward-slash path relative to PROJECT_ROOT."""
    return p.resolve().relative_to(PROJECT_ROOT).as_posix()


def build_splits(num_identities: int, seed: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    selected: list[tuple[str, list[Path]]] = []
    for identity, images in iter_identities_with_min_images(MIN_IMAGES_PER_IDENTITY):
        selected.append((identity, images))
        if len(selected) >= num_identities:
            break

    if len(selected) < num_identities:
        raise RuntimeError(
            f"Only {len(selected)} identities have >= {MIN_IMAGES_PER_IDENTITY} "
            f"images; requested {num_identities}."
        )

    rows: dict[str, list[dict]] = {name: [] for name in PARTITION_SIZES}
    for identity, images in selected:
        # Per-identity deterministic shuffle, then slice.
        order = rng.permutation(len(images))
        cursor = 0
        for partition, count in PARTITION_SIZES.items():
            chosen = [images[i] for i in order[cursor : cursor + count]]
            cursor += count
            for path in chosen:
                rows[partition].append(
                    {
                        "identity": identity,
                        "image_path": to_repo_relative(path),
                        "filename": path.name,
                        "partition": partition,
                    }
                )

    return {name: pd.DataFrame(rows[name]) for name in PARTITION_SIZES}


def verify_invariants(frames: dict[str, pd.DataFrame], num_identities: int) -> None:
    # 1. No path appears in more than one partition.
    all_paths = pd.concat([df["image_path"] for df in frames.values()])
    duplicates = all_paths[all_paths.duplicated()].tolist()
    if duplicates:
        raise RuntimeError(f"Duplicate image_paths across partitions: {duplicates[:5]}")

    # 2. Every identity is in all three partitions.
    identities_per_partition = {
        name: set(df["identity"]) for name, df in frames.items()
    }
    common = set.intersection(*identities_per_partition.values())
    if len(common) != num_identities:
        missing = {
            name: sorted(set.union(*identities_per_partition.values()) - ids)
            for name, ids in identities_per_partition.items()
        }
        raise RuntimeError(
            f"Identity coverage mismatch (expected {num_identities} in all partitions). "
            f"Missing per partition (first few): "
            f"{ {k: v[:3] for k, v in missing.items()} }"
        )

    # 3. Every listed path exists on disk.
    for df in frames.values():
        for rel in df["image_path"]:
            full = PROJECT_ROOT / rel
            if not full.exists():
                raise RuntimeError(f"Listed image missing on disk: {full}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-identities", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not LFW_ROOT.exists():
        print(f"ERROR: LFW root not found at {LFW_ROOT}", file=sys.stderr)
        return 1

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    frames = build_splits(num_identities=args.num_identities, seed=args.seed)
    verify_invariants(frames, num_identities=args.num_identities)

    # Write CSVs (sorted for byte-stable output).
    counts: dict[str, int] = {}
    for name, df in frames.items():
        df_sorted = df.sort_values(["identity", "filename"]).reset_index(drop=True)
        out = SPLITS_DIR / f"{name}.csv"
        df_sorted.to_csv(out, index=False, lineterminator="\n")
        counts[name] = len(df_sorted)

    manifest = {
        "seed": args.seed,
        "min_images_per_identity": MIN_IMAGES_PER_IDENTITY,
        "num_identities": args.num_identities,
        "per_identity": PARTITION_SIZES,
        "lfw_root": LFW_ROOT.resolve().relative_to(PROJECT_ROOT).as_posix(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "counts": counts,
    }
    (SPLITS_DIR / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print(f"selected {args.num_identities} identities (min {MIN_IMAGES_PER_IDENTITY} images each)")
    for name, n in counts.items():
        print(f"  {name:<13} {n:>5} rows -> splits/{name}.csv")
    print(f"  manifest          -> splits/split_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
