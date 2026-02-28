#!/usr/bin/env python3
"""
Phase 04: Quality Filtering.

Applies heuristic quality filters to remove low-quality documents.
Three filter presets are available:

  strict   -- Aggressive filtering, suitable for high-quality training sets
  moderate -- Balanced filtering (default), good for most use cases
  lenient  -- Light filtering, keeps more data for quantity

Each filter reports how many documents it removed, making it easy to
tune thresholds for your specific corpus.

Usage (standalone):
    python filter_quality.py \
        --input /data/deduped/deduped.jsonl \
        --output /data/filtered/filtered.jsonl \
        --preset moderate
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Filter presets
# ---------------------------------------------------------------------------

PRESETS = {
    "strict": {
        "min_words": 100,
        "max_words": 50_000,
        "min_alpha_ratio": 0.75,
        "max_symbol_ratio": 0.05,
        "max_repeated_line_ratio": 0.15,
        "max_bullet_ratio": 0.5,
        "min_avg_word_length": 3.0,
        "max_avg_word_length": 12.0,
    },
    "moderate": {
        "min_words": 50,
        "max_words": 100_000,
        "min_alpha_ratio": 0.60,
        "max_symbol_ratio": 0.10,
        "max_repeated_line_ratio": 0.30,
        "max_bullet_ratio": 0.70,
        "min_avg_word_length": 2.5,
        "max_avg_word_length": 15.0,
    },
    "lenient": {
        "min_words": 20,
        "max_words": 200_000,
        "min_alpha_ratio": 0.40,
        "max_symbol_ratio": 0.20,
        "max_repeated_line_ratio": 0.50,
        "max_bullet_ratio": 0.90,
        "min_avg_word_length": 2.0,
        "max_avg_word_length": 20.0,
    },
}


# ---------------------------------------------------------------------------
# Individual filter functions
# ---------------------------------------------------------------------------
# Each filter returns True if the document should be KEPT.

def filter_word_count(text: str, min_words: int, max_words: int) -> bool:
    """Keep documents within the word count range."""
    count = len(text.split())
    return min_words <= count <= max_words


def filter_alpha_ratio(text: str, min_ratio: float) -> bool:
    """Keep documents where at least min_ratio of characters are alphabetic."""
    if not text:
        return False
    alpha_count = sum(1 for ch in text if ch.isalpha())
    ratio = alpha_count / len(text)
    return ratio >= min_ratio


def filter_symbol_ratio(text: str, max_ratio: float) -> bool:
    """Reject documents with too many symbols (non-alpha, non-digit, non-space)."""
    if not text:
        return False
    symbol_count = sum(
        1 for ch in text
        if not ch.isalnum() and not ch.isspace()
    )
    ratio = symbol_count / len(text)
    return ratio <= max_ratio


def filter_repeated_lines(text: str, max_ratio: float) -> bool:
    """Reject documents where too many lines are repeated (boilerplate)."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) <= 1:
        return True

    line_counts = Counter(lines)
    repeated = sum(count - 1 for count in line_counts.values() if count > 1)
    ratio = repeated / len(lines)
    return ratio <= max_ratio


def filter_bullet_ratio(text: str, max_ratio: float) -> bool:
    """Reject documents that are mostly bullet/list items."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return True

    bullet_patterns = re.compile(r"^[\-\*\+\u2022\u25E6\u25AA\u2023]|\d+[\.\)]")
    bullet_count = sum(1 for ln in lines if bullet_patterns.match(ln))
    ratio = bullet_count / len(lines)
    return ratio <= max_ratio


def filter_avg_word_length(text: str, min_avg: float, max_avg: float) -> bool:
    """Reject documents with abnormal average word length."""
    words = text.split()
    if not words:
        return False
    avg = sum(len(w) for w in words) / len(words)
    return min_avg <= avg <= max_avg


# ---------------------------------------------------------------------------
# Filter registry
# ---------------------------------------------------------------------------

FILTER_REGISTRY = [
    ("word_count",      lambda t, p: filter_word_count(t, p["min_words"], p["max_words"])),
    ("alpha_ratio",     lambda t, p: filter_alpha_ratio(t, p["min_alpha_ratio"])),
    ("symbol_ratio",    lambda t, p: filter_symbol_ratio(t, p["max_symbol_ratio"])),
    ("repeated_lines",  lambda t, p: filter_repeated_lines(t, p["max_repeated_line_ratio"])),
    ("bullet_ratio",    lambda t, p: filter_bullet_ratio(t, p["max_bullet_ratio"])),
    ("avg_word_length", lambda t, p: filter_avg_word_length(t, p["min_avg_word_length"], p["max_avg_word_length"])),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 04: Heuristic quality filtering."
    )
    parser.add_argument(
        "--input", type=str,
        default=os.environ.get("INPUT_PATH", "/data/deduped/deduped.jsonl"),
        help="Input JSONL file path.",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.environ.get("OUTPUT_PATH", "/data/filtered/filtered.jsonl"),
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--preset", type=str, default="moderate",
        choices=PRESETS.keys(),
        help="Filter preset: strict, moderate (default), or lenient.",
    )
    parser.add_argument(
        "--min-words", type=int, default=None,
        help="Override preset min word count.",
    )
    parser.add_argument(
        "--max-words", type=int, default=None,
        help="Override preset max word count.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Build effective parameters from preset + overrides
    params = dict(PRESETS[args.preset])
    if args.min_words is not None:
        params["min_words"] = args.min_words
    if args.max_words is not None:
        params["max_words"] = args.max_words

    print("=" * 60)
    print("Phase 04: Quality Filtering")
    print("=" * 60)
    print(f"  Input:    {args.input}")
    print(f"  Output:   {args.output}")
    print(f"  Preset:   {args.preset}")
    print(f"  Params:")
    for k, v in sorted(params.items()):
        print(f"    {k}: {v}")
    print("=" * 60)

    t_start = time.time()

    # Per-filter removal counters
    filter_stats = {name: 0 for name, _ in FILTER_REGISTRY}
    input_count = 0
    output_count = 0

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            input_count += 1
            text = record.get("text", "")

            # Apply each filter in sequence
            passed = True
            for filter_name, filter_fn in FILTER_REGISTRY:
                if not filter_fn(text, params):
                    filter_stats[filter_name] += 1
                    passed = False
                    break  # First failing filter wins

            if passed:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_count += 1

            # Progress
            if input_count % 5000 == 0:
                print(f"  [{input_count:>7,}] kept={output_count:,}")

    # Report
    elapsed = time.time() - t_start
    total_removed = input_count - output_count
    pct_removed = (total_removed / input_count * 100) if input_count > 0 else 0

    print()
    print("=" * 60)
    print("Quality Filtering Complete")
    print("=" * 60)
    print(f"  Input documents:     {input_count:,}")
    print(f"  Output documents:    {output_count:,}")
    print(f"  Total removed:       {total_removed:,} ({pct_removed:.1f}%)")
    print()
    print("  Per-filter removal breakdown:")
    for filter_name, count in filter_stats.items():
        pct = (count / input_count * 100) if input_count > 0 else 0
        bar = "#" * int(pct * 2)
        print(f"    {filter_name:<20s}  {count:>6,} ({pct:5.1f}%)  {bar}")
    print()
    print(f"  Elapsed:             {elapsed:,.1f}s")
    print(f"  Output file:         {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
