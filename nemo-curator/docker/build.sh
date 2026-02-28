#!/usr/bin/env bash
# Build the NeMo Curator Workshop Docker image.
#
# Usage:
#   ./build.sh                    # Build with default tag
#   ./build.sh v2                 # Build with custom tag
#   ./build.sh v2 --push          # Build and push to ECR
#
# The build context is the workshop/nemo-curator/ root so that the
# Dockerfile can COPY the scripts/ directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSHOP_DIR="$(dirname "${SCRIPT_DIR}")"

# ── Configuration ────────────────────────────────────────────────────
IMAGE_NAME="${IMAGE_NAME:-nemo-curator-workshop}"
TAG="${1:-latest}"
PUSH="${2:-}"

# ECR registry (override with ECR_REGISTRY env var)
ECR_REGISTRY="${ECR_REGISTRY:-}"

echo "=============================================="
echo "  NeMo Curator Workshop - Docker Build"
echo "=============================================="
echo "  Image:     ${IMAGE_NAME}:${TAG}"
echo "  Context:   ${WORKSHOP_DIR}"
echo "  Dockerfile: ${SCRIPT_DIR}/Dockerfile"
if [ -n "${ECR_REGISTRY}" ]; then
    echo "  Registry:  ${ECR_REGISTRY}"
fi
echo "=============================================="
echo ""

# ── Build ────────────────────────────────────────────────────────────
echo "Building image..."
docker build \
    -t "${IMAGE_NAME}:${TAG}" \
    -f "${SCRIPT_DIR}/Dockerfile" \
    "${WORKSHOP_DIR}"

echo ""
echo "Build complete: ${IMAGE_NAME}:${TAG}"
docker images "${IMAGE_NAME}:${TAG}" --format "  Size: {{.Size}}"
echo ""

# ── Optional: tag and push to ECR ────────────────────────────────────
if [ "${PUSH}" = "--push" ]; then
    if [ -z "${ECR_REGISTRY}" ]; then
        echo "ERROR: Set ECR_REGISTRY env var to push. Example:"
        echo "  export ECR_REGISTRY=058264135704.dkr.ecr.us-east-2.amazonaws.com"
        exit 1
    fi

    FULL_TAG="${ECR_REGISTRY}/${IMAGE_NAME}:${TAG}"
    echo "Tagging as ${FULL_TAG}..."
    docker tag "${IMAGE_NAME}:${TAG}" "${FULL_TAG}"

    echo "Logging in to ECR..."
    aws ecr get-login-password --region "${AWS_REGION:-us-east-2}" \
        | docker login --username AWS --password-stdin "${ECR_REGISTRY}"

    echo "Pushing ${FULL_TAG}..."
    docker push "${FULL_TAG}"

    echo ""
    echo "Pushed: ${FULL_TAG}"
fi

echo "=============================================="
echo "  BUILD COMPLETE"
echo "=============================================="
