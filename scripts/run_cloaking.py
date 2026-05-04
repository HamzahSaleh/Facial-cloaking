"""Run the cloaking optimizer over splits/protected.csv and save cloaked images.

The output layout is the contract the eval pipeline + Streamlit dashboard
expect:

    outputs/methods/<method>/<filename>.jpg     # one per protected.csv row
    outputs/methods/<method>/_report.json       # run config + per-image metrics

`<filename>` matches the ``filename`` column of ``splits/protected.csv``
exactly so ``facial_cloaking.eval_pipeline.resolve_method_rows`` can pair
each cloaked image with its original.

Currently only ``--method nlcm`` (non-linear channel mixing) is wired up;
the embed-only and freq-only variants will be added by other contributors.

Examples:
    python scripts/run_cloaking.py --method nlcm --epsilon 8 --steps 300
    python scripts/run_cloaking.py --method nlcm --steps 50 --limit 10  # smoke test
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from facial_cloaking.cloak import CloakConfig, cloak_image_nlcm
from facial_cloaking.data import load_image
from facial_cloaking.embed import load_clip_model
from facial_cloaking.eval_pipeline import Gallery
from facial_cloaking.paths import PROJECT_ROOT


_METHODS = {
    "nlcm": cloak_image_nlcm,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=sorted(_METHODS.keys()),
        required=True,
        help="cloaking method to run (only 'nlcm' is implemented today)",
    )
    parser.add_argument("--csv", default=str(PROJECT_ROOT / "splits" / "protected.csv"))
    parser.add_argument(
        "--attractors", default=str(PROJECT_ROOT / "outputs" / "attractors.pt")
    )
    parser.add_argument(
        "--out-root", default=str(PROJECT_ROOT / "outputs" / "methods"),
        help="cloaked images go to <out_root>/<method>/<filename>",
    )
    parser.add_argument(
        "--epsilon", type=float, default=8.0,
        help="L_inf perturbation budget on the 0-255 pixel scale (default 8)",
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lambda-embed", type=float, default=1.0)
    parser.add_argument("--lambda-quality", type=float, default=2.0)
    parser.add_argument(
        "--limit", type=int, default=0,
        help="if > 0, cloak only the first N rows (smoke testing)",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()

    cloak_fn = _METHODS[args.method]

    attractors_path = Path(args.attractors)
    if not attractors_path.exists():
        raise SystemExit(
            f"missing {attractors_path}. Run scripts/compute_attractors.py first."
        )

    out_dir = Path(args.out_root) / args.method
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[cloak] loading CLIP on {args.device} ...")
    model, _ = load_clip_model(args.device)

    print(f"[cloak] loading gallery from {attractors_path} ...")
    gallery = Gallery.load(attractors_path, device=args.device)
    label_to_idx = {lab: i for i, lab in enumerate(gallery.labels)}

    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit > 0:
        rows = rows[: args.limit]

    config = CloakConfig(
        epsilon=args.epsilon / 255.0,
        n_steps=args.steps,
        lr=args.lr,
        lambda_embed=args.lambda_embed,
        lambda_quality=args.lambda_quality,
    )
    print(
        f"[cloak] method={args.method} epsilon={args.epsilon}/255 "
        f"steps={args.steps} lr={args.lr} "
        f"lambda_embed={args.lambda_embed} lambda_quality={args.lambda_quality}"
    )
    print(f"[cloak] cloaking {len(rows)} image(s) -> {out_dir}")

    per_image: list[dict] = []
    skipped: list[dict] = []
    t0 = time.time()

    for row in tqdm(rows, desc=args.method):
        identity = row["identity"].strip()
        filename = row["filename"].strip()
        src = PROJECT_ROOT / row["image_path"].strip()

        idx = label_to_idx.get(identity)
        if idx is None:
            skipped.append({"filename": filename, "reason": f"identity '{identity}' not in gallery"})
            continue

        e_id = gallery.embeddings[idx]
        e_null = gallery.e_null

        img = load_image(src)

        img_t0 = time.time()
        cloaked, metrics = cloak_fn(
            img, e_id, e_null, model, config=config, device=args.device,
        )
        img_dt = time.time() - img_t0

        out_path = out_dir / filename
        cloaked.save(out_path, "JPEG", quality=args.jpeg_quality)

        per_image.append({
            "filename": filename,
            "identity": identity,
            "wall_time_sec": round(img_dt, 3),
            "final_loss": metrics["final_loss"],
            "final_l_embed": metrics["final_l_embed"],
            "final_l_quality": metrics["final_l_quality"],
            "final_ssim_proxy": metrics["final_ssim_proxy"],
            "cos_to_identity": metrics["cos_to_identity"],
            "cos_to_null": metrics["cos_to_null"],
            "loss_history": metrics["loss_history"],
        })

    total_dt = time.time() - t0

    report = {
        "method": args.method,
        "csv": str(Path(args.csv)),
        "attractors": str(attractors_path),
        "device": args.device,
        "n_requested": len(rows),
        "n_cloaked": len(per_image),
        "n_skipped": len(skipped),
        "total_wall_time_sec": round(total_dt, 2),
        "config": {
            "epsilon_0_255": args.epsilon,
            "epsilon_0_1": args.epsilon / 255.0,
            "n_steps": args.steps,
            "lr": args.lr,
            "lambda_embed": args.lambda_embed,
            "lambda_quality": args.lambda_quality,
            "jpeg_quality": args.jpeg_quality,
        },
        "skipped": skipped,
        "per_image": per_image,
    }
    report_path = out_dir / "_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[cloak] wrote {len(per_image)} cloaked image(s) in {total_dt:.1f}s")
    print(f"[cloak] report -> {report_path}")


if __name__ == "__main__":
    main()
