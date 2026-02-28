# Section 02: vLLM Model Serving

## Objective

Deploy TechPulse Media's fine-tuned model using vLLM with tensor parallelism on the Ray cluster. By the end of this section, you will have an OpenAI-compatible API endpoint serving inference requests with PagedAttention and continuous batching.

**Estimated time: 20 minutes**

---

## Background: How vLLM Works

vLLM is a high-throughput LLM inference engine. Two innovations make it significantly faster than naive implementations:

### PagedAttention

Traditional inference engines pre-allocate contiguous GPU memory for each request's KV (key-value) cache. This wastes memory because:
- The final sequence length is unknown at request time
- Contiguous allocation leads to fragmentation

PagedAttention borrows the concept of virtual memory paging from operating systems. It divides the KV cache into fixed-size blocks and manages them with a page table:

```
Traditional (contiguous):
  Request A: [==KV Cache===========...unused...]   # Wastes 60%
  Request B: [==KV Cache=======...unused........]   # Wastes 70%
  Request C: Cannot fit -- OOM!

PagedAttention (paged):
  Block table:
    Request A -> [blk 0][blk 3][blk 7]
    Request B -> [blk 1][blk 4]
    Request C -> [blk 2][blk 5][blk 8]     # Fits!
    Free pool -> [blk 6][blk 9][blk 10]...

  GPU Memory: [0][1][2][3][4][5][6][7][8][9][10]...
               A  B  C  A  B  C  -  A  C  -  -
```

This eliminates fragmentation and allows vLLM to serve 2-4x more concurrent requests than naive implementations.

### Continuous Batching

Traditional batching waits for a batch to fill up, processes it, and returns all results. Continuous batching interleaves requests at the iteration level:

```
Traditional batching:
  Iteration 1: [A, B, C, D]    # All 4 requests
  Iteration 2: [A, B, C, D]    # A finishes, but waits for D
  ...
  Iteration 8: [_, _, _, D]    # Only D is still generating
  -> A, B, C blocked until D finishes

Continuous batching:
  Iteration 1: [A, B, C, D]
  Iteration 2: [A, B, C, D]    # A finishes -> immediately return
  Iteration 3: [E, B, C, D]    # E starts in A's slot
  Iteration 4: [E, B, C, D]    # B finishes -> immediately return
  Iteration 5: [E, F, C, D]    # F starts in B's slot
  -> Requests return as soon as they finish
```

This improves both latency (requests are not blocked by slow neighbors) and throughput (new requests start immediately as slots open up).

---

## Deployment Option 1: Standalone vLLM (Single GPU)

The simplest deployment runs vLLM directly as a Kubernetes Job with one GPU. This is suitable for models that fit in a single GPU's memory (7B models on A100 40GB, 14B models on H100 80GB).

### Kubernetes Manifest

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: techpulse-vllm-inference
  labels:
    app: techpulse-inference
spec:
  backoffLimit: 0
  template:
    metadata:
      labels:
        app: techpulse-inference
    spec:
      restartPolicy: Never
      containers:
        - name: vllm
          image: vllm/vllm-openai:v0.11.0
          command:
            - python
            - -m
            - vllm.entrypoints.openai.api_server
          args:
            - --model
            - TechPulse/techpulse-7b-v1      # Your fine-tuned model
            - --host
            - "0.0.0.0"
            - --port
            - "8000"
            - --tensor-parallel-size
            - "1"
            - --gpu-memory-utilization
            - "0.90"
            - --max-model-len
            - "4096"
            - --trust-remote-code
          env:
            - name: HF_HOME
              value: "/shared/hf_cache"
            - name: VLLM_USE_V1
              value: "1"
          ports:
            - containerPort: 8000
              name: http
          resources:
            limits:
              nvidia.com/gpu: 1
              memory: 64Gi
            requests:
              nvidia.com/gpu: 1
              memory: 32Gi
          volumeMounts:
            - name: shared
              mountPath: /shared
            - name: shm
              mountPath: /dev/shm
      volumes:
        - name: shared
          persistentVolumeClaim:
            claimName: fsx-pvc
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: 32Gi
---
apiVersion: v1
kind: Service
metadata:
  name: techpulse-inference-svc
spec:
  selector:
    app: techpulse-inference
  ports:
    - name: http
      port: 8000
      targetPort: 8000
  type: ClusterIP
```

### Testing the Deployment

```bash
# Deploy
kubectl apply -f vllm-standalone.yaml

# Wait for pod to be ready (model download + initialization)
kubectl wait --for=condition=ready pod \
  -l app=techpulse-inference \
  --timeout=600s

# Port-forward the service
kubectl port-forward svc/techpulse-inference-svc 8000:8000 &

# Test with curl (OpenAI-compatible API)
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TechPulse/techpulse-7b-v1",
    "messages": [
      {"role": "user", "content": "Explain the difference between TCP and UDP in two sentences."}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }' | python3 -m json.tool

# Check model info
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

---

## Deployment Option 2: Tensor Parallelism with Ray Serve

For models too large for a single GPU, or when you need lower latency from splitting the model across GPUs, use tensor parallelism. The model's layers are partitioned across multiple GPUs, and NCCL handles the inter-GPU communication.

### How Tensor Parallelism Works

```
                 Tensor Parallelism (TP=2)
                 =========================

  Input Tokens
       |
       v
  +----+----+                +----+----+
  |  GPU 0  |    NCCL       |  GPU 1  |
  | Shard A |  <-------->   | Shard B |
  |         |  all-reduce    |         |
  | Layers  |  all-gather    | Layers  |
  | 0-31    |  (<1ms each)   | 0-31    |
  | (half)  |                | (half)  |
  +---------+                +---------+
       |                          |
       +--------Combine-----------+
                   |
                   v
             Output Logits
```

Each GPU holds half the model weights for every layer. At each layer, the GPUs compute their portion and synchronize using NCCL collectives. This halves the per-GPU memory requirement and nearly halves latency for compute-bound operations.

### Ray Serve Deployment with TP=2

The recommended approach uses vLLM's multiprocessing executor inside a Ray Serve deployment. This avoids placement group complications in vLLM v1:

```yaml
# This is the Ray Serve deployment configuration.
# Submit it to the Ray cluster using the Ray CLI or SDK.
#
# Key insight: Use distributed_executor_backend="mp" (multiprocessing)
# instead of "ray" to avoid placement group issues in vLLM v1.
```

To deploy this on your Ray cluster, create the following deployment configuration and submit it:

```bash
# From the Ray head pod
kubectl exec $HEAD -- python3 << 'DEPLOY_EOF'
import ray
from ray import serve
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from fastapi import FastAPI
import uuid, time

app_fastapi = FastAPI()

@serve.deployment(
    name="techpulse-7b-tp2",
    num_replicas=1,
    ray_actor_options={"num_gpus": 2},    # Must match tensor_parallel_size
    max_ongoing_requests=16,
)
@serve.ingress(app_fastapi)
class VLLMDeployment:
    def __init__(self):
        engine_args = AsyncEngineArgs(
            model="TechPulse/techpulse-7b-v1",
            tensor_parallel_size=2,
            distributed_executor_backend="mp",   # Multiprocessing -- required!
            trust_remote_code=True,
            enforce_eager=True,
            gpu_memory_utilization=0.85,
            max_model_len=4096,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @app_fastapi.post("/v1/chat/completions")
    async def chat(self, request: dict):
        messages = request.get("messages", [])
        max_tokens = request.get("max_tokens", 256)
        temperature = request.get("temperature", 0.7)

        # Format prompt (simplified -- use your chat template)
        prompt = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        )

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
        )

        request_id = str(uuid.uuid4())
        results = []
        async for output in self.engine.generate(prompt, sampling_params, request_id):
            results.append(output)

        final = results[-1]
        return {
            "id": f"chatcmpl-{request_id[:8]}",
            "model": "TechPulse/techpulse-7b-v1",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": final.outputs[0].text,
                },
                "finish_reason": final.outputs[0].finish_reason,
            }],
        }

# Deploy
ray.init(address="auto")
serve.run(VLLMDeployment.bind(), route_prefix="/")
print("Deployment ready at http://localhost:8000")
DEPLOY_EOF
```

### Why Multiprocessing Executor?

vLLM v1 introduced a subprocess-based `EngineCore` for process isolation. This creates a problem with Ray's placement group mechanism:

```
Ray executor (BROKEN with vLLM v1):
  Ray Serve Actor (2 GPUs allocated)
    -> EngineCore subprocess (loses placement group context)
      -> Ray Worker Actor 0 (cannot find placement group)
      -> Ray Worker Actor 1 (cannot find placement group)
    Result: ValueError: "Current node has no GPU available"

Multiprocessing executor (WORKS):
  Ray Serve Actor (2 GPUs allocated)
    -> CUDA_VISIBLE_DEVICES=0,1 (inherited by subprocess)
      -> Worker Process 0 (GPU 0)
      -> Worker Process 1 (GPU 1)
      -> NCCL communication between workers
    Result: Successful TP=2 inference
```

The multiprocessing executor bypasses Ray entirely for worker management, using Python's `multiprocessing` module to spawn workers that inherit GPU access from the parent Ray Serve actor.

**Limitation:** Multiprocessing executor works within a single node only. For multi-node tensor parallelism (TP=16 across 2 nodes), you would need the Ray executor with upstream fixes or use standalone vLLM without Ray Serve.

---

## KV Cache Management

The KV cache stores the key-value attention states for all active requests. Its size directly determines how many requests vLLM can serve concurrently.

### gpu_memory_utilization

This parameter controls what fraction of GPU memory vLLM reserves for the KV cache (after loading the model):

```
Total GPU Memory:  80 GB (H100)
Model Weights:    -14 GB (7B model in BF16)
Activations:      - 2 GB (intermediate computations)
Available:         64 GB

gpu_memory_utilization=0.90:
  Total reserved: 80 * 0.90 = 72 GB
  KV cache:       72 - 14 - 2 = 56 GB
  -> ~45,000 tokens of KV cache (model-dependent)

gpu_memory_utilization=0.60:
  Total reserved: 80 * 0.60 = 48 GB
  KV cache:       48 - 14 - 2 = 32 GB
  -> ~25,000 tokens of KV cache
```

**Guidelines:**
- **Inference only:** Use `0.85-0.95` for maximum concurrent requests
- **Colocated with training:** Use `0.50-0.70` to leave room for training activations
- **Development/testing:** Use `0.40-0.50` for safe memory headroom

### Checking KV Cache Capacity

```bash
# vLLM logs the cache capacity at startup
kubectl logs <vllm-pod> | grep "KV cache"

# Example output:
# INFO: GPU KV cache capacity: 45,322 tokens in 2,831 blocks
# INFO: Maximum concurrent requests: ~88 (at 512 tokens average)
```

### Block Size and Memory Efficiency

vLLM allocates KV cache in blocks (default 16 tokens per block). Smaller blocks reduce waste but increase management overhead:

```
Request with 100 tokens:
  Block size 16: ceil(100/16) = 7 blocks -> 112 tokens allocated (12% waste)
  Block size 64: ceil(100/64) = 2 blocks -> 128 tokens allocated (28% waste)
```

For most workloads, the default block size is optimal.

---

## Benchmarking Throughput and Latency

### Quick Benchmark with curl

```bash
# Time a single request
time curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TechPulse/techpulse-7b-v1",
    "messages": [{"role": "user", "content": "What is Kubernetes?"}],
    "max_tokens": 200
  }' > /dev/null

# Expected: 1-3 seconds for a 7B model on A100
```

### Load Testing with Multiple Concurrent Requests

```bash
# Install hey (HTTP load generator)
# Or use any HTTP benchmarking tool

# Prepare a request payload
cat > /tmp/bench_payload.json << 'EOF'
{
  "model": "TechPulse/techpulse-7b-v1",
  "messages": [{"role": "user", "content": "Explain microservices in one paragraph."}],
  "max_tokens": 150,
  "temperature": 0.7
}
EOF

# Run 50 requests with 10 concurrent connections
hey -n 50 -c 10 \
  -m POST \
  -H "Content-Type: application/json" \
  -D /tmp/bench_payload.json \
  http://localhost:8000/v1/chat/completions

# Key metrics to watch:
#   Requests/sec    -> Throughput
#   Latency (p50)   -> Median response time
#   Latency (p99)   -> Tail latency
#   Errors          -> Should be 0
```

### Expected Performance (Order of Magnitude)

| Configuration | Model | Throughput | Latency (p50) |
|---------------|-------|------------|---------------|
| 1x A100, TP=1 | 7B | ~8-12 req/s | ~400ms |
| 2x A100, TP=2 | 7B | ~15-20 req/s | ~250ms |
| 1x H100, TP=1 | 7B | ~15-20 req/s | ~200ms |
| 8x H100, TP=8 | 70B | ~5-8 req/s | ~800ms |

These numbers vary significantly based on input/output length, batch size, and model architecture. Always benchmark with your actual workload.

---

## Scaling: Multiple Replicas

To increase throughput beyond what a single model instance can handle, deploy multiple replicas. Each replica loads a full copy of the model on its own set of GPUs:

```python
@serve.deployment(
    name="techpulse-7b-tp2",
    num_replicas=4,                         # 4 replicas
    ray_actor_options={"num_gpus": 2},      # 2 GPUs each
    max_ongoing_requests=16,
)
# Total GPUs: 4 replicas x 2 GPUs = 8 GPUs
# Throughput: ~4x single replica
```

Ray Serve automatically load-balances requests across replicas.

---

## Choosing TP Size

| Model Size | Recommended TP | GPUs per Replica | Notes |
|------------|---------------|-------------------|-------|
| 1-3B | 1 | 1 | Fits easily on any GPU |
| 7-8B | 1 or 2 | 1-2 | TP=1 on H100, TP=2 on A100 for headroom |
| 13-14B | 2 | 2 | TP=2 on A100 80GB, TP=1 on H100 |
| 30-34B | 4 | 4 | |
| 70B | 8 | 8 (full node) | Single node TP |
| 405B | 16+ | Multi-node | Requires pipeline parallelism |

**Rule of thumb:** Choose the smallest TP that fits the model in memory with enough KV cache for your target concurrency. Larger TP reduces latency but adds NCCL overhead and consumes more GPU resources.

---

## Checkpoint: Verify Serving

```bash
# 1. Model is loaded and serving
curl -s http://localhost:8000/v1/models | python3 -m json.tool

# 2. Inference works correctly
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TechPulse/techpulse-7b-v1",
    "messages": [{"role": "user", "content": "Hello, what can you help me with?"}],
    "max_tokens": 50
  }' | python3 -m json.tool

# 3. Check GPU memory usage
kubectl exec $HEAD -- nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
```

---

## Common Issues

### CUDA Out of Memory

**Symptom:** `torch.cuda.OutOfMemoryError: CUDA out of memory`

**Fixes:**
- Reduce `gpu_memory_utilization` (try 0.70, then increase)
- Reduce `max_model_len` (shorter sequences use less KV cache)
- Increase `tensor_parallel_size` to spread the model across more GPUs
- Check that no other processes are using GPU memory: `nvidia-smi`

### Model Download Timeout

**Symptom:** Pod runs for a long time without serving, then times out.

**Fix:** Pre-download model weights to the shared filesystem:
```bash
kubectl exec $HEAD -- python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('TechPulse/techpulse-7b-v1', cache_dir='/shared/hf_cache')
print('Download complete')
"
```

### TP=2 Fails with "No current placement group found"

**Symptom:** `ValueError: Current node has no GPU available`

**Fix:** Ensure you are using `distributed_executor_backend="mp"` (multiprocessing), not `"ray"`. This is a known vLLM v1 issue with Ray Serve placement groups. See the explanation above.

### Slow First Request

**Symptom:** First request takes 30-60 seconds, subsequent requests are fast.

**Cause:** vLLM compiles CUDA graphs on the first request (when `enforce_eager=False`). This is a one-time cost.

**Options:**
- Accept the warmup time (production recommendation)
- Set `enforce_eager=True` to skip CUDA graph compilation (slightly slower steady-state)

---

Next: [Section 03: GRPO/RLHF Training with NeMo RL](../03-grpo-training/)
