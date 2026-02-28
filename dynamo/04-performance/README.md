# Module 4: Performance Benchmarking

**Time:** 10 minutes

## Overview

Compare UCX and LIBFABRIC backends for KV cache transfer, and understand the
performance characteristics of disaggregated inference on EFA.

## Benchmarking Plan

| Test | Backend | Patches | Model | TP |
|------|---------|---------|-------|----|
| Baseline UCX | UCX | None | Qwen3-0.6B | 1 |
| Baseline LIBFABRIC | LIBFABRIC | None | Qwen3-0.6B | 1 |
| Patched LIBFABRIC | LIBFABRIC | 11 stability | Qwen3-0.6B | 1 |
| Optimized LIBFABRIC | LIBFABRIC | 11 + 3 perf | Qwen3-0.6B | 1 |
| Multi-GPU | LIBFABRIC | All | Llama-3.1-8B | 1 |
| Production | LIBFABRIC | All | Llama-3.1-70B | 8 |

## Metrics to Collect

- **Average latency** (ms) — end-to-end request time
- **P99 latency** (ms) — tail latency under load
- **TTFT** (ms) — time to first token
- **Throughput** (tok/s) — tokens per second under concurrency
- **EFA traffic** — libfabric handshake count, connection establishment time

## Running a Benchmark

```bash
# Simple latency test
kubectl port-forward svc/<frontend> 8000:8000 &

# Single request timing
time curl -s http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"Hello","max_tokens":100}'

# Concurrent load test
for concurrency in 1 5 10 20; do
  echo "=== Concurrency: $concurrency ==="
  for i in $(seq 1 $concurrency); do
    curl -s -o /dev/null -w "%{time_total}\n" \
      http://localhost:8000/v1/completions \
      -H 'Content-Type: application/json' \
      -d '{"model":"Qwen/Qwen3-0.6B","prompt":"Explain AI","max_tokens":50}' &
  done
  wait
done
```

## Previous Validated Results (NIXL 0.8.0 Patches)

| Framework | Metric | Before (stock) | After (patched+optimized) | Gain |
|-----------|--------|----------------|---------------------------|------|
| TRT-LLM | Avg latency | 403ms | 337ms | 16% |
| TRT-LLM | P99 latency | 632ms | 366ms | 42% |
| vLLM | TTFT avg | 3,517ms | 2,092ms | 40% |
| vLLM | Throughput | 301 tok/s | 487 tok/s | 62% |

These results were from NIXL 0.8.0 with all 13 patches + 3 performance
optimizations (batch CQ reads, lock-free rails, MRRC). Results on NIXL 0.9.0
stock are pending.

## Switching Backends

To switch from LIBFABRIC to UCX, change these environment variables:

```yaml
# Remove LIBFABRIC env vars and set:
NIXL_BACKEND: "UCX"
UCX_TLS: "tcp,cuda_copy,cuda_ipc,sm,self"
UCX_NET_DEVICES: "all"
```

## Performance Notes

- **First run penalty:** TRT-LLM PyTorch backend JIT-compiles kernels (~20 min)
- **CUDA graph capture:** After JIT, CUDA graphs are captured for batch sizes 1-16
- **GPU memory:** Qwen3-0.6B uses ~69GB on GPU 0 (model + KV cache)
- **EFA device enumeration:** NIXL scans all 32 EFA devices even with MAX_RAILS=1
