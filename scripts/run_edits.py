"""Apply a deterministic LLM-style edit to a method's outputs.

Every method (including the uncloaked baseline) is edited under the SAME
provider, prompt, and seed so the resulting "rank-1 / mean-true-cos /
SSIM" numbers are directly comparable across methods.

Examples
--------
Edit the uncloaked references (baseline; what the LLM does to a clean
portrait):

    python scripts/run_edits.py \\
        --source uncloaked \\
        --out outputs/edits/uncloaked \\
        --csv splits/editing_eval.csv

Edit a cloaking method's outputs:

    python scripts/run_edits.py \\
        --source outputs/methods/noise \\
        --out outputs/edits/noise \\
        --csv splits/editing_eval.csv

Then score every edited folder with the regular eval pipeline:

    python scripts/evaluate.py --methods \\
        uncloaked_edited=outputs/edits/uncloaked \\
        noise_edited=outputs/edits/noise
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from facial_cloaking.edits import get_provider, run_edits
from facial_cloaking.paths import PROJECT_ROOT


def _parse_source(token: str) -> Path | None:
    if token == "uncloaked":
        return None
    p = Path(token)
    if not p.exists():
        raise SystemExit(f"source directory does not exist: {p}")
    return p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source", required=True,
        help="'uncloaked' (use original images from the CSV) or a path to a "
             "method's cloaked-output directory",
    )
    parser.add_argument("--out", required=True, help="output directory for edited images")
    parser.add_argument(
        "--csv", default=str(PROJECT_ROOT / "splits" / "editing_eval.csv"),
        help="CSV split that selects which images to edit",
    )
    parser.add_argument(
        "--provider", default="local_stub",
        help="edit provider name (default: local_stub, fully deterministic)",
    )
    parser.add_argument(
        "--prompt",
        default="regenerate this portrait while preserving the person's identity",
        help="text prompt passed to the provider; folded into the deterministic seed",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="overwrite existing edited files (default: skip if already present)",
    )
    args = parser.parse_args()

    source_dir = _parse_source(args.source)
    provider = get_provider(args.provider)

    print(
        f"[edits] provider={provider.name} prompt={args.prompt!r} seed={args.seed} "
        f"source={'uncloaked' if source_dir is None else source_dir}"
    )
    report = run_edits(
        csv_path=Path(args.csv),
        source_dir=source_dir,
        out_dir=Path(args.out),
        provider=provider,
        prompt=args.prompt,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(
        f"[edits] wrote {report.n_written}/{report.n_inputs} edits "
        f"to {report.out_dir} (skipped sources: {len(report.skipped)})"
    )


if __name__ == "__main__":
    main()
