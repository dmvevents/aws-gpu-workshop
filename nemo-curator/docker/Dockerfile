# NeMo Curator Workshop Container
#
# Provides all dependencies for running the data curation pipeline
# locally or inside Kubernetes.  Uses python:3.11-slim as the base
# to keep the image small and avoid NVIDIA licensing constraints
# on the nvcr.io base image.
#
# Build:
#   docker build -t nemo-curator-workshop .
#
# Run (interactive):
#   docker run --rm -it -v $(pwd)/data:/data nemo-curator-workshop bash
#
# Run (single stage):
#   docker run --rm -v $(pwd)/data:/data nemo-curator-workshop \
#       python /workspace/scripts/extract_text.py --input /data/raw/input.jsonl --output /data/extracted/output.jsonl

FROM python:3.11-slim

LABEL maintainer="NeMo Curator Workshop"
LABEL description="Data curation pipeline for LLM training data preparation"

# ── System dependencies ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ──────────────────────────────────────────────
# Install in dependency order so layer caching works well.
# Build context is workshop/nemo-curator/ (set by build.sh).
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# ── Download fastText language-identification model ──────────────────
# lid.176.ftz is a compact (917 KB) model covering 176 languages.
RUN mkdir -p /models && \
    curl -fsSL -o /models/lid.176.ftz \
        https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz

ENV FASTTEXT_MODEL=/models/lid.176.ftz

# ── Copy workshop scripts ───────────────────────────────────────────
# Build context is workshop/nemo-curator/ (set by build.sh).
WORKDIR /workspace
COPY scripts/ /workspace/scripts/

# ── Data directory (mount point for PVC or local volume) ─────────────
RUN mkdir -p /data/raw /data/extracted /data/deduped /data/filtered \
             /data/classified /data/export

# ── Default entrypoint: show usage ───────────────────────────────────
COPY docker/entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["--help"]
