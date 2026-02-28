#!/usr/bin/env python3
"""
Phase 03: Deduplication Pipeline.

Removes duplicate and near-duplicate documents using two passes:
  1. Exact dedup  -- MD5 hash of the text field (fast, CPU-only)
  2. Fuzzy dedup  -- MinHash LSH via datasketch (finds near-duplicates)

Usage (standalone):
    python dedup_pipeline.py \
        --input /data/extracted/extracted.jsonl \
        --output /data/deduped/deduped.jsonl

    # Skip fuzzy dedup (exact only, much faster):
    python dedup_pipeline.py --input in.jsonl --output out.jsonl --exact-only
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 03: Exact + fuzzy deduplication."
    )
    parser.add_argument(
        "--input", type=str,
        default=os.environ.get("INPUT_PATH", "/data/extracted/extracted.jsonl"),
        help="Input JSONL file path.",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.environ.get("OUTPUT_PATH", "/data/deduped/deduped.jsonl"),
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--jaccard-threshold", type=float, default=0.8,
        help="Jaccard similarity threshold for fuzzy dedup (default: 0.8).",
    )
    parser.add_argument(
        "--num-perm", type=int, default=128,
        help="Number of permutations for MinHash (default: 128).",
    )
    parser.add_argument(
        "--exact-only", action="store_true",
        help="Run only exact dedup (skip fuzzy dedup).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pass 1: Exact dedup via MD5
# ---------------------------------------------------------------------------

def exact_dedup(records: list[dict]) -> tuple[list[dict], int]:
    """Remove exact duplicates based on MD5 hash of the 'text' field.

    Returns (deduplicated_records, num_removed).
    """
    seen_hashes: set[str] = set()
    unique: list[dict] = []
    removed = 0

    for rec in records:
        text = rec.get("text", "")
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()

        if text_hash in seen_hashes:
            removed += 1
            continue

        seen_hashes.add(text_hash)
        unique.append(rec)

    return unique, removed


# ---------------------------------------------------------------------------
# Pass 2: Fuzzy dedup via MinHash LSH
# ---------------------------------------------------------------------------

def fuzzy_dedup(
    records: list[dict],
    threshold: float = 0.8,
    num_perm: int = 128,
) -> tuple[list[dict], int]:
    """Remove near-duplicate documents using MinHash LSH.

    Requires the ``datasketch`` library. Falls back gracefully if
    unavailable, returning the input unchanged.

    Returns (deduplicated_records, num_removed).
    """
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        print(
            "WARNING: datasketch not installed. Fuzzy dedup skipped.\n"
            "  Install with: pip install datasketch",
            file=sys.stderr,
        )
        return records, 0

    print(f"  Building MinHash signatures (num_perm={num_perm})...")
    t0 = time.time()

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes: list[tuple[int, MinHash]] = []

    for idx, rec in enumerate(records):
        text = rec.get("text", "")
        mh = MinHash(num_perm=num_perm)

        # Create shingles (5-character n-grams)
        shingle_size = 5
        for i in range(max(0, len(text) - shingle_size + 1)):
            shingle = text[i : i + shingle_size]
            mh.update(shingle.encode("utf-8"))

        minhashes.append((idx, mh))

        # Insert into LSH index -- duplicates of the key are skipped
        try:
            lsh.insert(str(idx), mh)
        except ValueError:
            # Key already exists (should not happen with unique idx)
            pass

    print(f"  MinHash signatures built in {time.time() - t0:.1f}s")
    print(f"  Querying LSH for near-duplicate clusters...")

    # Find connected components of near-duplicates
    t1 = time.time()
    to_remove: set[int] = set()
    checked_pairs: set[tuple[int, int]] = set()

    for idx, mh in minhashes:
        if idx in to_remove:
            continue

        candidates = lsh.query(mh)
        for cand_str in candidates:
            cand_idx = int(cand_str)
            if cand_idx == idx or cand_idx in to_remove:
                continue

            pair = (min(idx, cand_idx), max(idx, cand_idx))
            if pair in checked_pairs:
                continue
            checked_pairs.add(pair)

            # Verify with actual Jaccard estimation
            similarity = mh.jaccard(minhashes[cand_idx][1])
            if similarity >= threshold:
                # Keep the first one (lower index), mark the other
                to_remove.add(cand_idx)

    print(f"  LSH query completed in {time.time() - t1:.1f}s")

    # Build output, preserving original order
    unique = [rec for idx, rec in enumerate(records) if idx not in to_remove]
    return unique, len(to_remove)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 03: Deduplication Pipeline")
    print("=" * 60)
    print(f"  Input:              {args.input}")
    print(f"  Output:             {args.output}")
    print(f"  Jaccard threshold:  {args.jaccard_threshold}")
    print(f"  MinHash perms:      {args.num_perm}")
    print(f"  Mode:               {'exact only' if args.exact_only else 'exact + fuzzy'}")
    print("=" * 60)

    # Load all records into memory (workshop-scale data fits in RAM)
    print("\nLoading input records...")
    t_start = time.time()
    records: list[dict] = []

    with open(args.input, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    input_count = len(records)
    print(f"  Loaded {input_count:,} records in {time.time() - t_start:.1f}s\n")

    if input_count == 0:
        print("ERROR: No records loaded. Check that the input file is valid JSONL.",
              file=sys.stderr)
        sys.exit(1)

    # Pass 1: Exact dedup
    print("Pass 1: Exact deduplication (MD5)...")
    records, exact_removed = exact_dedup(records)
    after_exact = len(records)
    pct_exact = (exact_removed / input_count * 100) if input_count > 0 else 0
    print(f"  Removed: {exact_removed:,} exact duplicates ({pct_exact:.1f}%)")
    print(f"  Remaining: {after_exact:,} documents\n")

    # Pass 2: Fuzzy dedup
    fuzzy_removed = 0
    if not args.exact_only:
        print("Pass 2: Fuzzy deduplication (MinHash LSH)...")
        records, fuzzy_removed = fuzzy_dedup(
            records,
            threshold=args.jaccard_threshold,
            num_perm=args.num_perm,
        )
        after_fuzzy = len(records)
        pct_fuzzy = (fuzzy_removed / after_exact * 100) if after_exact > 0 else 0
        print(f"  Removed: {fuzzy_removed:,} fuzzy duplicates ({pct_fuzzy:.1f}%)")
        print(f"  Remaining: {after_fuzzy:,} documents\n")

    # Write output
    print("Writing deduplicated records...")
    with open(args.output, "w", encoding="utf-8") as fout:
        for rec in records:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Final report
    elapsed = time.time() - t_start
    total_removed = exact_removed + fuzzy_removed
    pct_total = (total_removed / input_count * 100) if input_count > 0 else 0

    print()
    print("=" * 60)
    print("Deduplication Complete")
    print("=" * 60)
    print(f"  Input documents:       {input_count:,}")
    print(f"  Exact dupes removed:   {exact_removed:,}")
    print(f"  Fuzzy dupes removed:   {fuzzy_removed:,}")
    print(f"  Total removed:         {total_removed:,} ({pct_total:.1f}%)")
    print(f"  Output documents:      {len(records):,}")
    print(f"  Elapsed:               {elapsed:,.1f}s")
    print(f"  Output file:           {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
