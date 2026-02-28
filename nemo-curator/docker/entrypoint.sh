#!/usr/bin/env bash
# Entrypoint for the NeMo Curator Workshop container.
# Shows usage when run without arguments or with --help.
# Passes through to exec for any other command.

set -euo pipefail

if [ "$#" -eq 0 ] || [ "${1}" = "--help" ]; then
    cat <<'USAGE'
================================================================
  NeMo Curator Workshop - Data Curation Pipeline
================================================================

Available pipeline scripts (run in order):

  1. Download raw data:
     python /workspace/scripts/download_wikipedia.py \
         --output /data/raw/wikipedia_50k.jsonl --max-records 50000

  2. Extract and clean text:
     python /workspace/scripts/extract_text.py \
         --input /data/raw/wikipedia_50k.jsonl \
         --output /data/extracted/extracted.jsonl

  3. Deduplicate:
     python /workspace/scripts/dedup_pipeline.py \
         --input /data/extracted/extracted.jsonl \
         --output /data/deduped/deduped.jsonl

  4. Quality filter:
     python /workspace/scripts/filter_quality.py \
         --input /data/deduped/deduped.jsonl \
         --output /data/filtered/filtered.jsonl

  5. Domain classification:
     python /workspace/scripts/classify_domains.py \
         --input /data/filtered/filtered.jsonl \
         --output /data/classified/classified.jsonl

  6. Export and validate:
     python /workspace/scripts/export_dataset.py \
         --input /data/classified/classified.jsonl \
         --output-dir /data/export

Run the full pipeline:
  python /workspace/scripts/download_wikipedia.py --output /data/raw/wikipedia_50k.jsonl && \
  python /workspace/scripts/extract_text.py --input /data/raw/wikipedia_50k.jsonl --output /data/extracted/extracted.jsonl && \
  python /workspace/scripts/dedup_pipeline.py --input /data/extracted/extracted.jsonl --output /data/deduped/deduped.jsonl && \
  python /workspace/scripts/filter_quality.py --input /data/deduped/deduped.jsonl --output /data/filtered/filtered.jsonl && \
  python /workspace/scripts/classify_domains.py --input /data/filtered/filtered.jsonl --output /data/classified/classified.jsonl && \
  python /workspace/scripts/export_dataset.py --input /data/classified/classified.jsonl --output-dir /data/export

For interactive exploration:
  docker run --rm -it -v ./data:/data nemo-curator-workshop bash

================================================================
USAGE
    exit 0
fi

# Pass through to the provided command
exec "$@"
