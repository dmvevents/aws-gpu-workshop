# Module 3: Cross-Node Disaggregated Inference with EFA

**Time:** 20 minutes

## Overview

Deploy prefill and decode workers on separate P5.48xlarge nodes, using NIXL
LIBFABRIC over EFA RDMA for KV cache transfer. This is the production
configuration for disaggregated inference at scale.

## Architecture

```
   Node 1 (P5.48xlarge)              Node 2 (P5.48xlarge)
   ┌───────────────────┐             ┌───────────────────┐
   │ Frontend + Prefill │    NIXL    │   Decode Worker   │
   │   1x H100          │◄──────────►│   1x H100         │
   │   1x EFA           │ LIBFABRIC  │   1x EFA          │
   └───────────────────┘   EFA RDMA  └───────────────────┘
```

## Step 1: Deploy Cross-Node Configuration

```bash
kubectl apply -f deployments/trtllm-crossnode-libfabric.yaml
```

This creates 3 DGDs:
- **Frontend** — any node, no GPU
- **Prefill Worker** — pinned to Node 1 via `nodeName`
- **Decode Worker** — pinned to Node 2 via `nodeName`

> **Critical:** Edit the YAML to set your node names:
> ```yaml
> nodeName: <your-node-1-name>  # prefill
> nodeName: <your-node-2-name>  # decode
> ```

## Step 2: Memory Requirement (128Gi)

Worker pods **must** have 128Gi memory. With 64Gi, the pod is OOMKilled:

```
Last State:     Terminated
  Reason:       OOMKilled
  Exit Code:    137
```

**Why:** NIXL LIBFABRIC enumerates all 32 EFA devices on P5, creating queue
pairs and memory registrations for each. Combined with TRT-LLM model loading,
this exceeds 64Gi.

## Step 3: Wait for Startup

TRT-LLM engine compilation takes ~20 minutes on first run (PyTorch JIT).

```bash
# Monitor pods
watch kubectl get pods | grep trtllm

# Check GPU memory (engine loaded when ~69GB used)
kubectl exec <prefill-pod> -- nvidia-smi --query-gpu=memory.used --format=csv
```

## Step 4: Verify Cross-Node EFA Handshake

This is the key validation — look for EFA handshake messages between nodes:

```bash
# On prefill worker (Node 1)
kubectl logs <prefill-pod> | grep "HANDSHAKE received"
# Expected:
#   HANDSHAKE received from peer with explicit fi_addr 33
#   Received peer host id: <node-2-instance-id>

# On decode worker (Node 2)
kubectl logs <decode-pod> | grep "HANDSHAKE received"
# Expected:
#   HANDSHAKE received from peer with explicit fi_addr 33
#   Received peer host id: <node-1-instance-id>
```

**8 handshakes on each side** confirms cross-node RDMA connections are
established via NIXL LIBFABRIC over EFA.

## Step 5: Test Inference

```bash
kubectl port-forward svc/<frontend-service> 8000:8000 &

# Single request
curl -s http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"What is the capital of France?","max_tokens":30}'

# Batch test (10 concurrent)
for i in $(seq 1 10); do
  curl -s http://localhost:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"Qwen/Qwen3-0.6B","prompt":"Explain distributed computing","max_tokens":50}' &
done
wait
```

## Step 6: Verify NIXL Backend Activation

```bash
# Check both workers show LIBFABRIC
kubectl logs <prefill-pod> | grep "NixlTransferAgent"
kubectl logs <decode-pod> | grep "NixlTransferAgent"
# Both should show: NixlTransferAgent::NixlTransferAgent using NIXL backend: LIBFABRIC

# Check cache transceiver config
kubectl logs <decode-pod> | grep "cache_transceiver_config"
# Should show: CacheTransceiverConfig(backend='NIXL', ...)
```

## Validated Results

| Metric | Result |
|--------|--------|
| NIXL LIBFABRIC activation | PASS |
| Cross-node EFA handshake | 8 bidirectional |
| Single request latency | 147ms |
| Batch (10 concurrent) | All successful |
| Memory (128Gi) | No OOM |

## Troubleshooting

### OOMKilled (Exit Code 137)
Increase memory to 128Gi. See Step 2.

### Startup probe failures
The Dynamo operator probes port 9090 by default. If `DYN_SYSTEM_PORT` differs,
the probe fails but has 720 retries (2 hours). The pod stays Running but not
Ready. Inference still works via port-forward.

### HPC-X / OpenMPI conflict
The base container ships HPC-X which conflicts with EFA OpenMPI. The deployment
YAML includes a fix that unsets HPCX_* vars and pins Amazon OpenMPI.

### No EFA HW counters in pods
`/sys/class/infiniband/*/ports/1/hw_counters/` doesn't exist inside containers.
Use the libfabric handshake logs as proof of EFA activation instead.

## Next

[Module 4: Performance Benchmarking](../04-performance/README.md)
