# Module 2: Single-Node Disaggregated Inference

**Time:** 15 minutes

## Overview

Deploy Dynamo with separate prefill and decode workers on a single node. This
validates the disaggregation pipeline before adding cross-node complexity.

## Step 1: Deploy Frontend + Workers

The Multi-DGD pattern uses three separate DynamoGraphDeployments sharing a
`dynamoNamespace`. Each DGD uses a different `DYN_SYSTEM_PORT` to avoid
hostPort conflicts.

```bash
kubectl apply -f deployments/single-node.yaml
```

Key environment variables:
```yaml
# All DGDs must share the same namespace
DYN_NAMESPACE: "my-dynamo-test"

# Each DGD needs a unique system port
DYN_SYSTEM_PORT: "9090"   # Frontend
DYN_SYSTEM_PORT: "9091"   # Prefill
DYN_SYSTEM_PORT: "9190"   # Decode
```

## Step 2: Wait for Model Loading

TRT-LLM PyTorch backend JIT-compiles CUDA kernels on first run. This takes
15-20 minutes on P5.

```bash
# Watch pod status
watch kubectl get pods -l app.kubernetes.io/part-of=dynamo-test

# Check model loading progress
kubectl logs -f <prefill-pod> | grep -E "TRT-LLM|model|engine|download"
```

## Step 3: Verify NIXL Activation

```bash
kubectl logs <prefill-pod> | grep NixlTransferAgent
# Expected:
#   [TensorRT-LLM][INFO] NixlTransferAgent::NixlTransferAgent using NIXL backend: LIBFABRIC
```

## Step 4: Test Inference

```bash
kubectl port-forward svc/<frontend-service> 8000:8000 &

curl -s http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"Hello","max_tokens":20}'
```

Expected response with `prompt_tokens`, `completion_tokens`, and generated text.

## Key Environment Variables

```yaml
# NIXL LIBFABRIC backend
NIXL_BACKEND: "LIBFABRIC"
TRTLLM_NIXL_KVCACHE_BACKEND: "LIBFABRIC"

# EFA provider
FI_PROVIDER: "efa"
FI_EFA_USE_DEVICE_RDMA: "1"
FI_EFA_ENABLE_SHM: "0"           # Critical: glibc 2.39 SHM bug
FI_EFA_ENABLE_SHM_TRANSFER: "0"

# NIXL configuration
NIXL_SKIP_TOPOLOGY_CHECK: "1"    # Required in containers
NIXL_LIBFABRIC_MAX_RAILS: "1"    # Match GPU count
```

## Next

[Module 3: Cross-Node with EFA](../03-cross-node/README.md)
