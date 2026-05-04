"""Apply the standard purification grid (JPEG, Gaussian blur, bilateral,
bit-depth) to every method and report retrieval after each attack.

This is just ``evaluate.py`` with a fixed purification set, so the same JSON
report layout is reused.

Example:
    python scripts/robustness_test.py \
        --methods uncloaked noise=outputs/methods/noise blur=outputs/methods/blur
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from facial_cloaking.embed import load_clip_model
from facial_cloaking.eval_pipeline import (
    Gallery,
    MethodSpec,
    compare_methods,
    format_markdown_table,
)
from facial_cloaking.paths import PROJECT_ROOT
from facial_cloaking.purify import standard_robustness_grid


def _parse_method_spec(token: str) -> MethodSpec:
    if "=" not in token:
        if token == "uncloaked":
            return MethodSpec.uncloaked()
        raise SystemExit(
            f"method spec '{token}' must be 'name=path' (or the literal 'uncloaked')"
        )
    name, path = token.split("=", 1)
    return MethodSpec.folder(name.strip(), Path(path.strip()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--csv", default=str(PROJECT_ROOT / "splits" / "protected.csv"))
    parser.add_argument(
        "--attractors", default=str(PROJECT_ROOT / "outputs" / "attractors.pt")
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output", default=str(PROJECT_ROOT / "outputs" / "robustness_report.json")
    )
    args = parser.parse_args()

    attractors_path = Path(args.attractors)
    if not attractors_path.exists():
        raise SystemExit(
            f"missing {attractors_path}. Run scripts/compute_attractors.py first."
        )

    methods = [_parse_method_spec(t) for t in args.methods]
    purifications = standard_robustness_grid()

    print(f"[robustness] loading CLIP on {args.device} ...")
    model, preprocess = load_clip_model(args.device)

    print(f"[robustness] loading gallery from {attractors_path} ...")
    gallery = Gallery.load(attractors_path, device=args.device)

    print(
        f"[robustness] scoring {len(methods)} method(s) under "
        f"{len(purifications)} purifications on {args.csv} ..."
    )
    scores = compare_methods(
        methods,
        csv_path=Path(args.csv),
        gallery=gallery,
        model=model,
        preprocess=preprocess,
        device=args.device,
        purifications=purifications,
        batch_size=args.batch_size,
    )

    print()
    print(format_markdown_table(scores))
    print()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "csv": str(Path(args.csv)),
                "attractors": str(attractors_path),
                "device": args.device,
                "purifications": list(purifications.keys()),
                "scores": [s.to_dict() for s in scores],
            },
            indent=2,
        )
    )
    print(f"[robustness] wrote {out}")


if __name__ == "__main__":
    main()
