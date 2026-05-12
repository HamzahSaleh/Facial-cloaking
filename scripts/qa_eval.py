"""
GPT-4o Vision Q&A identification assessment.

For each query image this script runs two verification calls:
  1. Positive pair  – query + a reference of the SAME identity  → expect "Yes"
  2. Negative pair  – query + a reference of a DIFFERENT identity → expect "No"

Metrics:
  tpr      – True Positive Rate  (same-person pairs answered "Yes")
  tnr      – True Negative Rate  (diff-person pairs answered "No")
  accuracy – (tpr + tnr) / 2

A well-cloaked image should have LOW tpr (GPT-4o cannot confirm the true
identity) and high tnr (it correctly rejects distractors).

Requires:
    pip install openai
    export OPENAI_API_KEY=sk-...

Examples:
    # Smoke test – 5 queries, uncloaked baseline
    python scripts/gpt4o_qa_eval.py --method uncloaked --limit 5 --dry-run

    # Evaluate the cloak
    python scripts/gpt4o_qa_eval.py \\
        --method cloak=outputs/methods/cloak \\
        --csv splits/protected.csv \\
        --output outputs/gpt4o_cloak.json

    # Evaluate multiple methods back-to-back
    python scripts/gpt4o_qa_eval.py \\
        --method uncloaked \\
        --method cloak=outputs/methods/cloak \\
        --method noise=outputs/methods/noise \\
        --output outputs/gpt4o_core.json

    # Evaluate ipadapter edited images
    python scripts/gpt4o_qa_eval.py \\
        --method uncloaked_edited=outputs/edits/ipadapter/uncloaked \\
        --method cloak_edited=outputs/edits/ipadapter/cloak \\
        --csv splits/editing_eval.csv \\
        --output outputs/gpt4o_ipadapter.json
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

PROJECT_ROOT = REPO_ROOT  # same directory; avoids importing torch via __init__


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a face verification assistant. "
    "You will be shown two portrait photographs. "
    "Your task is to determine whether they show the same individual. "
    "Respond with exactly one word: Yes or No. No explanation."
)

USER_PROMPT = (
    "Do these two portrait photographs show the same person? "
    "Respond with exactly one word: Yes or No."
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PairResult:
    identity: str
    filename: str
    ref_filename: str
    pair_type: str        # "positive" or "negative"
    response: str         # raw model response
    answer: bool          # True = model said "Yes"
    correct: bool         # whether that answer was correct
    latency_s: float = 0.0
    error: str = ""


@dataclass
class MethodResult:
    method: str
    n_queries: int
    n_positive: int
    n_negative: int
    tpr: float            # same-person pairs answered Yes
    tnr: float            # diff-person pairs answered No
    accuracy: float       # (tpr + tnr) / 2
    pairs: list[PairResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Keep pairs as dicts for JSON serialisation
        return d


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def group_by_identity(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["identity"].strip(), []).append(r)
    return groups


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def encode_image_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_message(b64: str, mime: str = "image/jpeg") -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "low"},
    }


# ---------------------------------------------------------------------------
# GPT-4o call (single verification pair)
# ---------------------------------------------------------------------------

async def verify_pair(
    client,
    img_a_path: Path,
    img_b_path: Path,
    *,
    model: str,
    semaphore: asyncio.Semaphore,
    dry_run: bool,
    rng: random.Random,
) -> tuple[str, float]:
    """
    Ask GPT-4o whether two images show the same person.
    Returns (raw_response_text, latency_seconds).
    """
    if dry_run:
        await asyncio.sleep(0.01)
        return rng.choice(["Yes", "No"]), 0.0

    b64_a = encode_image_b64(img_a_path)
    b64_b = encode_image_b64(img_b_path)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER_PROMPT},
                image_message(b64_a),
                image_message(b64_b),
            ],
        },
    ]

    async with semaphore:
        t0 = time.perf_counter()
        for attempt in range(6):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=5,
                    temperature=0.0,
                )
                latency = time.perf_counter() - t0
                return resp.choices[0].message.content.strip(), latency

            except Exception as e:
                # Import here to avoid hard dependency at module level
                try:
                    from openai import RateLimitError
                    is_rate_limit = isinstance(e, RateLimitError)
                except ImportError:
                    is_rate_limit = "rate" in str(e).lower() or "429" in str(e)

                if is_rate_limit:
                    # Respect Retry-After header if present, otherwise
                    # exponential backoff starting at 15s with ±3s jitter
                    retry_after = None
                    if hasattr(e, "response") and e.response is not None:
                        retry_after = e.response.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else (15 * (2 ** attempt))
                    wait += rng.uniform(-3, 3)          # jitter prevents thundering herd
                    wait = max(5.0, wait)
                    print(f"  [rate limit] attempt {attempt+1} — waiting {wait:.1f}s")
                else:
                    wait = (2 ** attempt) + rng.uniform(0, 1)
                    print(f"  [retry {attempt+1}] {type(e).__name__}: {e} — waiting {wait:.1f}s")

                await asyncio.sleep(wait)

        return "Error", time.perf_counter() - t0


def parse_answer(text: str) -> bool:
    """Return True if the model said Yes, False otherwise."""
    return text.strip().lower().startswith("y")


# ---------------------------------------------------------------------------
# Per-method evaluation
# ---------------------------------------------------------------------------

async def evaluate_method(
    method_name: str,
    candidate_dir: Path | None,
    query_rows: list[dict],
    gallery_groups: dict[str, list[dict]],
    *,
    client,
    model: str,
    semaphore: asyncio.Semaphore,
    dry_run: bool,
    seed: int,
    delay: float = 0.0,
) -> MethodResult:
    """
    For each query image, run one positive and one negative verification.

    Reference images come from the gallery split (clean_test.csv), which
    is disjoint from both protected and editing_eval.
    """
    rng = random.Random(seed)
    all_identities = sorted(gallery_groups.keys())

    tasks = []
    meta  = []

    for row in query_rows:
        identity = row["identity"].strip()
        filename = row["filename"].strip()

        # Resolve query path
        if candidate_dir is None:
            query_path = PROJECT_ROOT / row["image_path"].strip()
        else:
            query_path = candidate_dir / filename

        if not query_path.exists():
            print(f"  [skip] missing: {query_path}")
            continue

        # Positive reference — a different image of the same person from gallery
        pos_candidates = gallery_groups.get(identity, [])
        if not pos_candidates:
            print(f"  [skip] no gallery image for identity: {identity}")
            continue
        pos_row   = rng.choice(pos_candidates)
        pos_path  = PROJECT_ROOT / pos_row["image_path"].strip()

        # Negative reference — any image of a different person from gallery
        neg_ids = [i for i in all_identities if i != identity]
        neg_id   = rng.choice(neg_ids)
        neg_row  = rng.choice(gallery_groups[neg_id])
        neg_path = PROJECT_ROOT / neg_row["image_path"].strip()

        # Queue both pairs as coroutines
        tasks.append(verify_pair(client, query_path, pos_path,
                                 model=model, semaphore=semaphore,
                                 dry_run=dry_run, rng=rng))
        meta.append(("positive", identity, filename, pos_row["filename"].strip()))

        tasks.append(verify_pair(client, query_path, neg_path,
                                 model=model, semaphore=semaphore,
                                 dry_run=dry_run, rng=rng))
        meta.append(("negative", identity, filename, neg_row["filename"].strip()))

    total = len(tasks)
    done_count = 0

    async def _tracked(coro):
        nonlocal done_count
        result = await coro
        done_count += 1
        if done_count % 10 == 0 or done_count == total:
            print(f"  [{method_name}] {done_count}/{total} calls complete")
        return result

    print(f"  [{method_name}] running {total} verification calls …")

    if delay > 0:
        # Stagger launch times so requests don't all fire simultaneously,
        # but still run up to `concurrency` at once via the semaphore.
        async def _delayed_start(i: int, coro):
            await asyncio.sleep(i * delay)
            return await _tracked(coro)
        responses = await asyncio.gather(
            *[_delayed_start(i, c) for i, c in enumerate(tasks)]
        )
    else:
        responses = await asyncio.gather(*[_tracked(c) for c in tasks])

    pairs: list[PairResult] = []
    for (pair_type, identity, filename, ref_fname), (raw, latency) in zip(meta, responses):
        answered_yes = parse_answer(raw)
        correct = answered_yes if pair_type == "positive" else not answered_yes
        pairs.append(PairResult(
            identity=identity,
            filename=filename,
            ref_filename=ref_fname,
            pair_type=pair_type,
            response=raw,
            answer=answered_yes,
            correct=correct,
            latency_s=round(latency, 3),
            error="" if raw != "Error" else "api_error",
        ))

    positives = [p for p in pairs if p.pair_type == "positive" and not p.error]
    negatives = [p for p in pairs if p.pair_type == "negative" and not p.error]

    tpr = sum(p.correct for p in positives) / len(positives) if positives else float("nan")
    tnr = sum(p.correct for p in negatives) / len(negatives) if negatives else float("nan")
    acc = (tpr + tnr) / 2 if positives and negatives else float("nan")

    print(
        f"  [{method_name}] TPR={tpr:.3f}  TNR={tnr:.3f}  Acc={acc:.3f}"
        f"  ({len(positives)} pos / {len(negatives)} neg)"
    )

    return MethodResult(
        method=method_name,
        n_queries=len(query_rows),
        n_positive=len(positives),
        n_negative=len(negatives),
        tpr=round(tpr, 4),
        tnr=round(tnr, 4),
        accuracy=round(acc, 4),
        pairs=pairs,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_method(token: str) -> tuple[str, Path | None]:
    if "=" not in token:
        if token == "uncloaked":
            return "uncloaked", None
        raise SystemExit(
            f"method spec '{token}' must be 'name=path' or the literal 'uncloaked'"
        )
    name, path_str = token.split("=", 1)
    p = Path(path_str.strip())
    if not p.exists():
        raise SystemExit(f"method directory does not exist: {p}")
    return name.strip(), p


async def main_async(args: argparse.Namespace) -> None:
    # Load CSVs
    query_rows = read_csv(Path(args.csv))
    gallery_rows = read_csv(Path(args.gallery_csv))
    gallery_groups = group_by_identity(gallery_rows)

    if args.limit:
        query_rows = query_rows[: args.limit]

    methods = [_parse_method(t) for t in args.method]

    # OpenAI client
    if args.dry_run:
        client = None
        print("[gpt4o_eval] DRY RUN — no API calls will be made\n")
    else:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise SystemExit("openai package not found. Run: pip install openai")
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit(
                "No API key found. Set OPENAI_API_KEY or pass --api-key."
            )
        client = AsyncOpenAI(api_key=api_key, timeout=60.0)

    semaphore = asyncio.Semaphore(args.concurrency)

    all_results: list[MethodResult] = []
    for method_name, candidate_dir in methods:
        print(f"\n── method: {method_name}")
        result = await evaluate_method(
            method_name, candidate_dir,
            query_rows, gallery_groups,
            client=client,
            model=args.model,
            semaphore=semaphore,
            dry_run=args.dry_run,
            seed=args.seed,
            delay=args.delay,
        )
        all_results.append(result)

    # Summary table
    print("\n── Summary ─────────────────────────────")
    print(f"{'method':<25}  {'TPR':>6}  {'TNR':>6}  {'Acc':>6}")
    print("-" * 46)
    for r in all_results:
        print(f"{r.method:<25}  {r.tpr:>6.3f}  {r.tnr:>6.3f}  {r.accuracy:>6.3f}")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "csv": args.csv,
        "gallery_csv": args.gallery_csv,
        "model": args.model,
        "dry_run": args.dry_run,
        "seed": args.seed,
        "results": [r.to_dict() for r in all_results],
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n[gpt4o_eval] saved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--method", action="append", required=True, dest="method",
        help="'uncloaked' or 'name=path/to/folder'. Repeat for multiple methods.",
    )
    parser.add_argument(
        "--csv",
        default=str(PROJECT_ROOT / "splits" / "protected.csv"),
        help="query split CSV (default: splits/protected.csv)",
    )
    parser.add_argument(
        "--gallery-csv",
        default=str(PROJECT_ROOT / "splits" / "clean_test.csv"),
        help="reference image CSV, must be disjoint from --csv "
             "(default: splits/clean_test.csv)",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "gpt4o_eval.json"),
    )
    parser.add_argument(
        "--model", default="gpt-4o",
        help="OpenAI model name (default: gpt-4o)",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="OpenAI API key (falls back to OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=3,
        help="max simultaneous API requests (default: 3). "
             "Raise to 5-10 only on Tier 3+ accounts to avoid 429s.",
    )
    parser.add_argument(
        "--delay", type=float, default=0.0,
        help="seconds to sleep between launching each request (default: 0). "
             "Set to 0.5–1.0 if you continue to hit rate limits.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for reference image selection (default: 42)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="cap number of query images per method (for smoke testing)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="skip API calls and return random Yes/No answers (pipeline test)",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
