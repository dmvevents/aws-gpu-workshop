#!/usr/bin/env python3
"""
Download Wikipedia articles from HuggingFace and save as JSONL.

This script streams a Wikipedia dataset from HuggingFace's `wikimedia/wikipedia`
collection (or any compatible dataset) and writes each article as a JSON line.
Designed for the NeMo Curator workshop: downloads a manageable subset (~50K
articles) that can be processed in a 90-minute session.

Usage (standalone):
    pip install datasets
    python download_wikipedia.py --output /data/raw/wikipedia_50k.jsonl --max-records 50000

Usage (in K8s):
    Injected via ConfigMap and executed by the download Job.
    Parameters are set via environment variables DATASET_SIZE and OUTPUT_PATH.

The output JSONL has one record per line with the schema:
    {"id": str, "title": str, "text": str, "url": str, "timestamp": str}
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Wikipedia articles from HuggingFace and save as JSONL."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.environ.get("OUTPUT_PATH", "/data/raw/wikipedia_50k.jsonl"),
        help="Output JSONL file path (default: /data/raw/wikipedia_50k.jsonl). "
        "Also configurable via OUTPUT_PATH env var.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=int(os.environ.get("DATASET_SIZE", "50000")),
        help="Maximum number of records to download (default: 50000). "
        "Also configurable via DATASET_SIZE env var.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=os.environ.get(
            "DATASET_NAME", "wikimedia/wikipedia"
        ),
        help="HuggingFace dataset name (default: wikimedia/wikipedia).",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=os.environ.get("DATASET_CONFIG", "20231101.en"),
        help="Dataset configuration/subset name (default: 20231101.en).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ---- Lazy import so --help works without datasets installed ----
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "ERROR: The 'datasets' library is required. Install it with:\n"
            "  pip install datasets",
            file=sys.stderr,
        )
        sys.exit(1)

    # ---- Graceful shutdown on SIGINT / SIGTERM -----------------------
    # If the user (or K8s) interrupts, we flush the current file and
    # print stats for however many records were written.  The partial
    # output is valid JSONL.
    shutdown_requested = False

    def _handle_signal(signum: int, _frame) -> None:  # noqa: ANN001
        nonlocal shutdown_requested
        sig_name = signal.Signals(signum).name
        print(f"\n[SIGNAL] Received {sig_name} -- finishing current record and exiting.")
        shutdown_requested = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # ---- Prepare output directory ------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Wikipedia Download for NeMo Curator Workshop")
    print("=" * 60)
    print(f"  Dataset:       {args.dataset_name} ({args.dataset_config})")
    print(f"  Max records:   {args.max_records:,}")
    print(f"  Output:        {output_path}")
    print("=" * 60)
    print()

    # ---- Stream the dataset ------------------------------------------
    # streaming=True avoids downloading the full ~21 GB dump to disk.
    # We iterate and pick only the fields we need.
    print("Connecting to HuggingFace and starting stream...")
    t_start = time.time()

    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    records_written = 0
    total_bytes = 0

    with open(output_path, "w", encoding="utf-8") as fout:
        for item in dataset:
            if shutdown_requested:
                break
            if records_written >= args.max_records:
                break

            # Build a normalised record.  The wikimedia/wikipedia schema has
            # 'id', 'url', 'title', 'text'.  We add a timestamp field
            # (extracted from the dataset config name, e.g. "20231101.en")
            # for provenance tracking.  For other datasets the fields may
            # differ -- we fall back gracefully.
            record = {
                "id": str(item.get("id", records_written)),
                "title": item.get("title", ""),
                "text": item.get("text", ""),
                "url": item.get("url", ""),
                "timestamp": args.dataset_config.split(".")[0],  # e.g. "20231101"
            }

            line = json.dumps(record, ensure_ascii=False)
            fout.write(line + "\n")

            total_bytes += len(line.encode("utf-8")) + 1  # +1 for newline
            records_written += 1

            # Progress reporting every 5000 records
            if records_written % 5000 == 0:
                elapsed = time.time() - t_start
                rate = records_written / elapsed if elapsed > 0 else 0
                mb = total_bytes / (1024 * 1024)
                print(
                    f"  [{records_written:>7,} / {args.max_records:,}]  "
                    f"{mb:,.1f} MB written  |  {rate:,.0f} records/s  |  "
                    f"{elapsed:,.0f}s elapsed"
                )

    # ---- Final statistics --------------------------------------------
    elapsed = time.time() - t_start
    mb = total_bytes / (1024 * 1024)
    avg_len = total_bytes / records_written if records_written > 0 else 0

    print()
    print("=" * 60)
    print("Download Complete")
    print("=" * 60)
    print(f"  Records written:    {records_written:,}")
    print(f"  Total size:         {mb:,.1f} MB")
    print(f"  Avg record size:    {avg_len:,.0f} bytes")
    print(f"  Elapsed time:       {elapsed:,.1f}s")
    print(f"  Throughput:         {records_written / elapsed:,.0f} records/s" if elapsed > 0 else "")
    print(f"  Output file:        {output_path}")
    if shutdown_requested:
        print("  NOTE: Download was interrupted -- output is partial but valid JSONL.")
    print("=" * 60)


if __name__ == "__main__":
    main()
