#!/usr/bin/env python3
"""
Phase 06: Export and Validation.

Splits the classified dataset into train/val/test, validates each
record against the expected schema, and writes a statistics report.

Usage (standalone):
    python export_dataset.py \
        --input /data/classified/classified.jsonl \
        --output-dir /data/export

    # Custom split ratios:
    python export_dataset.py --input in.jsonl --output-dir /data/export \
        --train-ratio 0.90 --val-ratio 0.05
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = ["text"]
EXPECTED_FIELDS = [
    "text", "id", "title", "url", "language", "language_score",
    "word_count", "char_count", "domain", "quality_score",
]


def validate_record(record: dict, line_no: int) -> list[str]:
    """Validate a single record. Returns a list of error strings (empty = OK)."""
    errors = []

    # Required fields
    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append(f"line {line_no}: missing required field '{field}'")

    # Text must be a non-empty string
    text = record.get("text")
    if text is not None:
        if not isinstance(text, str):
            errors.append(f"line {line_no}: 'text' is {type(text).__name__}, expected str")
        elif not text.strip():
            errors.append(f"line {line_no}: 'text' is empty/whitespace-only")
        elif "\x00" in text:
            errors.append(f"line {line_no}: 'text' contains null bytes")

    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 06: Export, split, and validate the curated dataset."
    )
    parser.add_argument(
        "--input", type=str,
        default=os.environ.get("INPUT_PATH", "/data/classified/classified.jsonl"),
        help="Input JSONL file path.",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.environ.get("OUTPUT_DIR", "/data/export"),
        help="Output directory for split files (default: /data/export).",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.95,
        help="Fraction for training split (default: 0.95).",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.03,
        help="Fraction for validation split (default: 0.03).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible splits (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio < 0:
        print(
            f"ERROR: train_ratio ({args.train_ratio}) + val_ratio ({args.val_ratio}) > 1.0",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 06: Export and Validation")
    print("=" * 60)
    print(f"  Input:         {args.input}")
    print(f"  Output dir:    {output_dir}")
    print(f"  Split ratios:  train={args.train_ratio:.0%}  val={args.val_ratio:.0%}  test={test_ratio:.0%}")
    print(f"  Random seed:   {args.seed}")
    print("=" * 60)

    t_start = time.time()

    # Load and validate all records
    print("\nLoading and validating records...")
    records: list[dict] = []
    all_errors: list[str] = []

    with open(args.input, "r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                all_errors.append(f"line {line_no}: invalid JSON ({exc})")
                continue

            errors = validate_record(record, line_no)
            if errors:
                all_errors.extend(errors)
                continue

            records.append(record)

    total = len(records)
    print(f"  Loaded {total:,} valid records")
    if all_errors:
        print(f"  Validation errors: {len(all_errors)}")
        for err in all_errors[:10]:
            print(f"    - {err}")
        if len(all_errors) > 10:
            print(f"    ... and {len(all_errors) - 10} more")

    if total == 0:
        print("ERROR: No valid records to export.", file=sys.stderr)
        sys.exit(1)

    # Shuffle and split
    print("\nShuffling and splitting...")
    random.seed(args.seed)
    random.shuffle(records)

    train_end = int(total * args.train_ratio)
    val_end = train_end + int(total * args.val_ratio)

    splits = {
        "train": records[:train_end],
        "validation": records[train_end:val_end],
        "test": records[val_end:],
    }

    # Write splits
    for split_name, split_records in splits.items():
        out_path = output_dir / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as fout:
            for rec in split_records:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  {split_name}: {len(split_records):,} records -> {out_path}")

    # Compute statistics
    print("\nComputing statistics...")
    domain_dist: Counter = Counter()
    quality_scores: list[float] = []
    word_counts: list[int] = []
    total_chars = 0

    for rec in records:
        domain_dist[rec.get("domain", "unknown")] += 1
        qs = rec.get("quality_score", 0.0)
        if isinstance(qs, (int, float)):
            quality_scores.append(qs)
        wc = rec.get("word_count", 0)
        if isinstance(wc, int):
            word_counts.append(wc)
        total_chars += len(rec.get("text", ""))

    stats = {
        "pipeline": "nemo-curator-workshop",
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "total_records": total,
        "validation_errors": len(all_errors),
        "splits": {
            name: len(recs) for name, recs in splits.items()
        },
        "domain_distribution": dict(domain_dist.most_common()),
        "quality_score": {
            "mean": round(sum(quality_scores) / len(quality_scores), 4) if quality_scores else 0,
            "min": round(min(quality_scores), 4) if quality_scores else 0,
            "max": round(max(quality_scores), 4) if quality_scores else 0,
        },
        "word_count": {
            "total": sum(word_counts),
            "mean": round(sum(word_counts) / len(word_counts)) if word_counts else 0,
            "min": min(word_counts) if word_counts else 0,
            "max": max(word_counts) if word_counts else 0,
        },
        "total_characters": total_chars,
        "total_size_mb": round(total_chars / (1024 * 1024), 2),
    }

    stats_path = output_dir / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"  Statistics written to {stats_path}")

    # Final report
    elapsed = time.time() - t_start

    print()
    print("=" * 60)
    print("Export Complete")
    print("=" * 60)
    print(f"  Total records:       {total:,}")
    print(f"  Train split:         {len(splits['train']):,}")
    print(f"  Validation split:    {len(splits['validation']):,}")
    print(f"  Test split:          {len(splits['test']):,}")
    print(f"  Total size:          {stats['total_size_mb']:.1f} MB")
    print(f"  Total words:         {stats['word_count']['total']:,}")
    print(f"  Avg quality score:   {stats['quality_score']['mean']:.3f}")
    print()
    print("  Domain distribution:")
    for domain, count in domain_dist.most_common():
        pct = (count / total * 100) if total > 0 else 0
        print(f"    {domain:<28s}  {count:>6,} ({pct:5.1f}%)")
    print()
    print(f"  Validation errors:   {len(all_errors)}")
    print(f"  Elapsed:             {elapsed:,.1f}s")
    print(f"  Output directory:    {output_dir}")
    print("=" * 60)

    # Exit with error if there were validation problems
    if all_errors:
        print(
            f"\nWARNING: {len(all_errors)} validation errors were found. "
            "Review the errors above. Invalid records were excluded from the export.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
