"""Score one or more cloaking methods on the three evaluation surfaces.

Method specs are passed as ``name=path`` pairs. The special name
``uncloaked`` does not require a path (it uses the original CSV image paths)
and gives the upper-bound retrieval reference.

Examples:
    python scripts/evaluate.py \
        --methods uncloaked noise=outputs/methods/noise blur=outputs/methods/blur \
        --csv splits/protected.csv --attractors outputs/attractors.pt

    # add a few purifications inline
    python scripts/evaluate.py \
        --methods uncloaked noise=outputs/methods/noise \
        --purify jpeg-75 blur-1.0 bits-4
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
from facial_cloaking.purify import parse_purification_spec


def _parse_method_spec(token: str) -> MethodSpec:
    if "=" not in token:
        if token == "uncloaked":
            return MethodSpec.uncloaked()
        raise SystemExit(
            f"method spec '{token}' must be 'name=path' (or the literal 'uncloaked')"
        )
    name, path = token.split("=", 1)
    name = name.strip()
    path = Path(path.strip())
    if not path.exists():
        raise SystemExit(f"method '{name}' directory does not exist: {path}")
    return MethodSpec.folder(name, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        nargs="+",
        required=True,
        help="space-separated method specs: 'name=path' or 'uncloaked'",
    )
    parser.add_argument("--csv", default=str(PROJECT_ROOT / "splits" / "protected.csv"))
    parser.add_argument(
        "--attractors", default=str(PROJECT_ROOT / "outputs" / "attractors.pt")
    )
    parser.add_argument(
        "--purify", nargs="*", default=[],
        help="optional purification specs (e.g. jpeg-75 blur-1.0 bits-4 bilateral)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output", default=str(PROJECT_ROOT / "outputs" / "eval_report.json")
    )
    args = parser.parse_args()

    attractors_path = Path(args.attractors)
    if not attractors_path.exists():
        raise SystemExit(
            f"missing {attractors_path}. Run scripts/compute_attractors.py first."
        )

    methods = [_parse_method_spec(t) for t in args.methods]

    purifications = {}
    for spec in args.purify:
        name, fn = parse_purification_spec(spec)
        purifications[name] = fn

    print(f"[evaluate] loading CLIP on {args.device} ...")
    model, preprocess = load_clip_model(args.device)

    print(f"[evaluate] loading gallery from {attractors_path} ...")
    gallery = Gallery.load(attractors_path, device=args.device)

    print(f"[evaluate] scoring {len(methods)} method(s) on {args.csv} ...")
    scores = compare_methods(
        methods,
        csv_path=Path(args.csv),
        gallery=gallery,
        model=model,
        preprocess=preprocess,
        device=args.device,
        purifications=purifications or None,
        batch_size=args.batch_size,
    )

    table = format_markdown_table(scores)
    print()
    print(table)
    print()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "csv": str(Path(args.csv)),
        "attractors": str(attractors_path),
        "device": args.device,
        "purifications": list(purifications.keys()),
        "scores": [s.to_dict() for s in scores],
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"[evaluate] wrote {out}")


if __name__ == "__main__":
    main()
