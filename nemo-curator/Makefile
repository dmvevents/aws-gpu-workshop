# NeMo Curator Workshop - Makefile
#
# Convenience targets for building, running, and managing the
# data curation pipeline both locally and on Kubernetes.
#
# Usage:
#   make build       Build the workshop Docker image
#   make download    Download raw data (local Docker)
#   make curate      Run the full curation pipeline (local Docker)
#   make k8s-setup   Create K8s PVC and ConfigMap
#   make k8s-run     Run the full pipeline as a K8s Job
#   make clean       Remove local data artifacts
#   make all         download + curate (local)

.PHONY: all build download curate run-extract run-dedup run-filter \
        run-classify run-export k8s-setup k8s-run k8s-logs k8s-clean \
        clean help

# ── Configuration ─────────────────────────────────────────────────────
IMAGE_NAME     ?= nemo-curator-workshop
IMAGE_TAG      ?= latest
DATA_DIR       ?= $(CURDIR)/data
DATASET_SIZE   ?= 50000
FILTER_PRESET  ?= moderate
JACCARD_THRESH ?= 0.8
TRAIN_RATIO    ?= 0.95
NAMESPACE      ?= default

# Docker run template with common options
DOCKER_RUN = docker run --rm \
    -v $(DATA_DIR):/data \
    -e PYTHONUNBUFFERED=1 \
    $(IMAGE_NAME):$(IMAGE_TAG)

# ── Meta targets ──────────────────────────────────────────────────────

all: download curate  ## Run download + full curation pipeline

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Docker build ──────────────────────────────────────────────────────

build:  ## Build the workshop Docker image
	@echo "Building $(IMAGE_NAME):$(IMAGE_TAG)..."
	bash docker/build.sh $(IMAGE_TAG)

# ── Local pipeline (Docker) ──────────────────────────────────────────

download:  ## Download raw Wikipedia data
	@mkdir -p $(DATA_DIR)/raw
	$(DOCKER_RUN) python /workspace/scripts/download_wikipedia.py \
		--output /data/raw/wikipedia.jsonl \
		--max-records $(DATASET_SIZE)

run-extract:  ## Run Phase 02: Text extraction
	@mkdir -p $(DATA_DIR)/extracted
	$(DOCKER_RUN) python /workspace/scripts/extract_text.py \
		--input /data/raw/wikipedia.jsonl \
		--output /data/extracted/extracted.jsonl

run-dedup:  ## Run Phase 03: Deduplication
	@mkdir -p $(DATA_DIR)/deduped
	$(DOCKER_RUN) python /workspace/scripts/dedup_pipeline.py \
		--input /data/extracted/extracted.jsonl \
		--output /data/deduped/deduped.jsonl \
		--jaccard-threshold $(JACCARD_THRESH)

run-filter:  ## Run Phase 04: Quality filtering
	@mkdir -p $(DATA_DIR)/filtered
	$(DOCKER_RUN) python /workspace/scripts/filter_quality.py \
		--input /data/deduped/deduped.jsonl \
		--output /data/filtered/filtered.jsonl \
		--preset $(FILTER_PRESET)

run-classify:  ## Run Phase 05: Domain classification
	@mkdir -p $(DATA_DIR)/classified
	$(DOCKER_RUN) python /workspace/scripts/classify_domains.py \
		--input /data/filtered/filtered.jsonl \
		--output /data/classified/classified.jsonl

run-export:  ## Run Phase 06: Export and validation
	@mkdir -p $(DATA_DIR)/export
	$(DOCKER_RUN) python /workspace/scripts/export_dataset.py \
		--input /data/classified/classified.jsonl \
		--output-dir /data/export \
		--train-ratio $(TRAIN_RATIO)

curate: run-extract run-dedup run-filter run-classify run-export  ## Run full curation pipeline (phases 02-06)
	@echo ""
	@echo "==== Curation pipeline complete ===="
	@echo "Output: $(DATA_DIR)/export/"
	@ls -lh $(DATA_DIR)/export/ 2>/dev/null || true

# ── Kubernetes deployment ────────────────────────────────────────────

k8s-setup:  ## Create K8s PVC and pipeline ConfigMap
	kubectl apply -f k8s/data-pvc.yaml
	kubectl create configmap workshop-pipeline-scripts \
		--from-file=download_wikipedia.py=scripts/download_wikipedia.py \
		--from-file=extract_text.py=scripts/extract_text.py \
		--from-file=dedup_pipeline.py=scripts/dedup_pipeline.py \
		--from-file=filter_quality.py=scripts/filter_quality.py \
		--from-file=classify_domains.py=scripts/classify_domains.py \
		--from-file=export_dataset.py=scripts/export_dataset.py \
		--namespace=$(NAMESPACE) \
		--dry-run=client -o yaml | kubectl apply -f -

k8s-run: k8s-setup  ## Run full pipeline as a K8s Job
	kubectl delete job workshop-curation-pipeline --ignore-not-found --namespace=$(NAMESPACE)
	kubectl apply -f k8s/curation-pipeline-job.yaml --namespace=$(NAMESPACE)
	@echo ""
	@echo "Job submitted. Watch logs with:"
	@echo "  make k8s-logs"

k8s-logs:  ## Tail logs from the K8s pipeline Job
	kubectl logs -f job/workshop-curation-pipeline --namespace=$(NAMESPACE)

k8s-clean:  ## Delete K8s pipeline resources
	kubectl delete job workshop-curation-pipeline --ignore-not-found --namespace=$(NAMESPACE)
	kubectl delete configmap workshop-pipeline-scripts --ignore-not-found --namespace=$(NAMESPACE)
	@echo "K8s resources cleaned. PVC retained (delete manually if needed)."

# ── Cleanup ──────────────────────────────────────────────────────────

clean:  ## Remove local data artifacts
	@echo "Removing local pipeline data from $(DATA_DIR)..."
	rm -rf $(DATA_DIR)/extracted $(DATA_DIR)/deduped \
	       $(DATA_DIR)/filtered $(DATA_DIR)/classified $(DATA_DIR)/export
	@echo "Done. Raw data in $(DATA_DIR)/raw/ preserved."
	@echo "To remove raw data too: rm -rf $(DATA_DIR)/raw"

clean-all: clean  ## Remove ALL local data including raw downloads
	rm -rf $(DATA_DIR)/raw
	@echo "All data removed."
