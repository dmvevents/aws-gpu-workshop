# Module 4: Performance Benchmarking

**Time:** 15 minutes

## Overview

Benchmark disaggregated inference with AIPerf (NVIDIA's new benchmarking tool,
successor to GenAI-Perf) and compare UCX vs LIBFABRIC backends for KV cache
transfer.

## Tool: AIPerf (Replaces GenAI-Perf)

AIPerf is the successor to GenAI-Perf. Key differences:
- **No Triton dependency** — standalone Python package
- **Better metrics** — TTFT, TTST (time to second token), ITL, ICL, goodput/SLO
- **Production load patterns** — Poisson/Gamma arrivals, concurrency bursts
- **Real-time dashboard** — TUI with live metrics

> **Note:** GenAI-Perf (from `triton-inference-server/perf_analyzer`) is
> deprecated. Use AIPerf for all new benchmarking work.

### 1. Install AIPerf

```bash
pip install aiperf
aiperf --version
# Expected: 0.5.0 or later
```

### 2. Port-forward the frontend

```bash
kubectl port-forward svc/<frontend-service> 8000:8000 &
```

### 3. Verify endpoint

```bash
curl -s http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"Hello","max_tokens":10}'
```

### 4. Run AIPerf benchmark (streaming)

```bash
aiperf profile \
  --model Qwen/Qwen3-0.6B \
  --url http://localhost:8000 \
  --endpoint-type completions \
  --streaming \
  --concurrency 10 \
  --request-count 100 \
  --isl 512 \
  --osl 128 \
  --ui simple
```

Expected output includes:
- **TTFT** (Time to First Token) — dominated by prefill + KV transfer
- **ITL** (Inter-Token Latency) — per-token decode step
- **Output throughput** (tok/s)
- **Request throughput** (req/s)
- **E2E latency** (ms) with P50/P90/P99 percentiles

### 5. Concurrency sweep

Run at different concurrency levels to find the throughput-latency tradeoff:

```bash
for CONC in 1 5 10 20 50; do
  echo "=== Concurrency: $CONC ==="
  aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --url http://localhost:8000 \
    --endpoint-type completions \
    --streaming \
    --concurrency $CONC \
    --request-count 100 \
    --isl 512 \
    --osl 128 \
    --ui none
  echo ""
done
```

### 6. Rate-based benchmark (production-like)

```bash
aiperf profile \
  --model Qwen/Qwen3-0.6B \
  --url http://localhost:8000 \
  --endpoint-type completions \
  --streaming \
  --request-rate 20 \
  --arrival-pattern poisson \
  --request-count 500 \
  --concurrency 32 \
  --isl 512 \
  --osl 128
```

### 7. With goodput/SLO measurement

```bash
aiperf profile \
  --model Qwen/Qwen3-0.6B \
  --url http://localhost:8000 \
  --endpoint-type completions \
  --streaming \
  --concurrency 10 \
  --request-count 200 \
  --goodput "request_latency:2000 inter_token_latency:50"
```

## Alternative: vLLM benchmark_serving.py

For comparison with the AWS workshop's original benchmark:

```bash
kubectl run -it --rm bench --image vllm/vllm-openai:v0.14.1 --command -- bash

# Inside the pod:
pip install pandas datasets
python3 benchmarks/benchmark_serving.py \
  --backend openai \
  --host <frontend-svc>.default.svc.cluster.local \
  --port 8000 \
  --model Qwen/Qwen3-0.6B \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 128 \
  --num-prompts 100 \
  --endpoint /v1/completions \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,99
```

## Benchmark Plan

### Phase 1: Core Comparisons

| Test | Model | TP | Backend | Expected Impact |
|------|-------|----|---------|----|
| B-01 | Qwen3-0.6B | TP1 | UCX (stock) | Baseline |
| B-04 | Qwen3-0.6B | TP1 | LIBFABRIC (patched) | EFA RDMA improvement |
| B-05 | Llama-3.1-8B | TP1 | UCX (stock) | Baseline |
| B-08 | Llama-3.1-8B | TP1 | LIBFABRIC (patched) | EFA RDMA improvement |
| B-14 | Llama-3.1-70B | TP4 | LIBFABRIC (patched) | Production config |
| B-16 | Llama-3.1-70B | TP8 | LIBFABRIC (patched) | Max TP config |

### Key Metrics to Compare

| Metric | UCX | LIBFABRIC | Why It Matters |
|--------|-----|-----------|----------------|
| TTFT | Higher (TCP overhead) | Lower (RDMA bypass) | User-perceived responsiveness |
| ITL | Similar (decode-bound) | Similar | Decode step doesn't change |
| Throughput | Lower | Higher | More requests/second at scale |
| P99 latency | Higher | Lower | Tail latency from KV transfer stalls |

## Previous Validated Results (NIXL 0.8.0 Patches)

| Framework | Metric | Before (stock) | After (patched+optimized) | Gain |
|-----------|--------|----------------|---------------------------|------|
| TRT-LLM | Avg latency | 403ms | 337ms | 16% |
| TRT-LLM | P99 latency | 632ms | 366ms | 42% |
| vLLM | TTFT avg | 3,517ms | 2,092ms | 40% |
| vLLM | Throughput | 301 tok/s | 487 tok/s | 62% |

## AIPerf vs GenAI-Perf Migration

| GenAI-Perf | AIPerf | Notes |
|-----------|--------|-------|
| `genai-perf profile` | `aiperf profile` | Same subcommand |
| `--endpoint-type chat` | `--endpoint-type chat` | Same |
| `--max-threads N` | `--workers-max N` | Renamed |
| `--synthetic-input-tokens-mean` | `--isl` | Shortened |
| `--output-tokens-mean` | `--osl` | Shortened |
| Requires Triton `perf_analyzer` | Standalone | No C++ dependency |
| TTFT metric | TTFT + TTFO (for reasoning models) | More granular |
| N/A | `--goodput` | New: SLO-based measurement |
| N/A | `--arrival-pattern poisson` | New: production-like load |
| N/A | `--ui dashboard` | New: real-time TUI |

## Performance Notes

- **First run penalty:** TRT-LLM PyTorch backend JIT-compiles kernels (~20 min)
- **CUDA graph capture:** After JIT, CUDA graphs are captured for batch sizes 1-16
- **GPU memory:** Qwen3-0.6B uses ~69GB on GPU 0 (model + KV cache)
- **Memory requirement:** 128Gi minimum per worker on P5 (32 EFA device enumeration)
- **EFA device enumeration:** NIXL scans all 32 EFA devices even with MAX_RAILS=1
