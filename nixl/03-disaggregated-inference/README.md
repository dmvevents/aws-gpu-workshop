# Section 3: Disaggregated Inference with NIXL

**Estimated time:** 15 minutes

## The Problem: Monolithic Inference Bottleneck

In a standard LLM inference deployment, a single GPU (or TP group) handles
both phases of autoregressive generation:

1. **Prefill**: Process the entire input prompt in parallel, populating the
   KV cache. This is compute-bound and benefits from high FLOPS utilization.
2. **Decode**: Generate tokens one at a time, each requiring a KV cache
   lookup. This is memory-bandwidth-bound and under-utilizes GPU compute.

```
Monolithic Inference (single instance):

  User Prompt (2048 tokens)
         |
         v
  +----------------------------------------------+
  |  GPU 0-7 (TP=8)                              |
  |                                              |
  |  [Prefill: 200ms] --> [Decode: 50ms/token]  |
  |  High compute       Low compute              |
  |  100% FLOPS util    10% FLOPS util           |
  |                     Memory-bandwidth bound    |
  +----------------------------------------------+

  Problem: During decode, expensive GPU compute sits idle.
  Problem: New prefill requests must wait for decode to finish.
```

Because prefill saturates GPU compute while decode barely uses it, running
both on the same GPUs leads to poor utilization and scheduling conflicts.
Long-context prefills block decode steps, causing latency spikes.

## Disaggregated Prefill/Decode Architecture

The solution is to separate prefill and decode onto different sets of GPUs,
each optimized for its workload:

```
Disaggregated Inference:

  User Prompt                    Token Stream
       |                              ^
       v                              |
  +-----------+    KV Cache     +-----------+
  |  Prefill  |===============>|  Decode   |
  |  Node(s)  |    via NIXL    |  Node(s)  |
  |           |    (RDMA)      |           |
  |  TP=8     |                |  TP=8     |
  |  Compute  |                |  Memory   |
  |  optimized|                |  optimized|
  +-----------+                +-----------+
       |                              |
  NCCL (intra-TP)             NCCL (intra-TP)
```

**Benefits:**

- **Independent scaling**: Scale prefill and decode pools separately based
  on workload characteristics.
- **Better utilization**: Prefill GPUs stay compute-saturated. Decode GPUs
  can serve more concurrent requests since they are not blocked by prefill.
- **Lower latency**: Decode tokens stream continuously without prefill
  interruptions.
- **Cost optimization**: Different instance types can be used. Prefill
  benefits from compute-heavy GPUs; decode benefits from high memory
  bandwidth GPUs.

**The key challenge**: The KV cache generated during prefill must be
transferred to the decode node before decode can begin. For a model like
Llama 3.1 70B with 128K context, the KV cache can be 4-8 GB. This transfer
must happen quickly (tens of milliseconds) to avoid negating the latency
benefits of disaggregation.

This is exactly what NIXL was built to solve.

## KV Cache Transfer: How It Works

### KV Cache Structure

A transformer KV cache is organized as:

```
KV Cache shape: [num_layers, 2, num_kv_heads, seq_len, head_dim]
                      |      |       |            |         |
                      |      |       |            |     dimension per head
                      |      |   attention heads  |
                      |      K and V matrices     |
                   model layers              tokens processed

Example: Llama 3.1 70B, 4K context, TP=8
  num_layers = 80
  num_kv_heads = 8 (GQA, per TP shard = 1)
  seq_len = 4096
  head_dim = 128
  dtype = float16

  Per-layer KV size = 2 * 1 * 4096 * 128 * 2 bytes = 2 MB
  Total KV cache = 80 * 2 MB = 160 MB

Example: DeepSeek R1 671B, 32K context, TP=8
  Total KV cache per TP shard can exceed 4 GB
```

### Block-Based KV Cache (vLLM PagedAttention)

vLLM uses paged memory management for KV caches, similar to virtual memory.
The KV cache is divided into fixed-size blocks, and a page table maps
logical sequence positions to physical GPU memory blocks:

```
Logical KV Cache:                Physical GPU Memory:
+--------+--------+--------+    +--------+--------+--------+--------+
| Blk 0  | Blk 1  | Blk 2  |    | Blk 7  | Blk 3  | Blk 12 | Blk 0  |
| tok 0-15| tok 16-31| tok 32-47|    | (free) | (req B)| (free) | (req A)|
+--------+--------+--------+    +--------+--------+--------+--------+

Page Table for Request A:
  Logical block 0 --> Physical block 3
  Logical block 1 --> Physical block 7
  Logical block 2 --> Physical block 12
```

When transferring a KV cache from prefill to decode, NIXL transfers the
physical blocks and the decode node reconstructs the page table.

### Transfer Flow

```
Prefill Node                          Decode Node
+--------------+                      +--------------+
|              |                      |              |
| 1. Run       |                      |              |
|    prefill   |                      |              |
|    (prompt)  |                      |              |
|              |                      |              |
| 2. KV cache  |                      |              |
|    in GPU    |                      |              |
|    memory    |    NIXL RDMA WRITE   |              |
|    blocks    |--------------------->| 3. Receive   |
|              |    (multi-rail,      |    KV cache  |
|              |     GDR, async)      |    blocks    |
|              |                      |              |
| 4. Send      |    Notification      |              |
|    notif     |--------------------->| 5. Notif     |
|    "done"    |                      |    received  |
|              |                      |              |
|              |                      | 6. Start     |
|              |                      |    decode    |
|              |                      |    (tokens)  |
+--------------+                      +--------------+
```

### NIXL API for KV Cache Transfer

Using NIXL's index-based transfer API, which maps naturally to vLLM's block
structure:

```python
# Both prefill and decode instances pre-register their full KV cache pool:
kv_pool = torch.zeros(
    (num_blocks, block_size, num_kv_heads, head_dim),
    dtype=torch.float16, device="cuda:0"
)
reg_handle = agent.register_memory(kv_pool)

# Build descriptor list (one entry per block):
block_descs = agent.get_xfer_descs(
    [kv_pool[i] for i in range(num_blocks)]
)

# Prepare descriptor lists once (this validates everything):
local_prepared = agent.prep_xfer_dlist("", block_descs)
remote_prepared = agent.prep_xfer_dlist("decode-node", remote_block_descs)

# At transfer time, select specific blocks by index:
# "Transfer local blocks [3, 7, 12] to remote blocks [0, 1, 2]"
handle = agent.make_prepped_xfer(
    "WRITE",
    local_prepared,  [3, 7, 12],      # Source block indices
    remote_prepared, [0, 1, 2],        # Destination block indices
    b"req-12345-kv-ready"              # Notification
)

# Post the transfer (async, non-blocking):
agent.transfer(handle)

# Check completion:
while agent.check_xfer_state(handle) != "DONE":
    # Do other work while transfer proceeds in background
    pass
```

The index-based approach (`prep_xfer_dlist` + `make_prepped_xfer`) is
important for performance because:

1. Descriptor validation happens once at preparation time, not per-transfer.
2. The block-to-block mapping can change each request without re-preparation.
3. Multiple small block transfers are batched into a single RDMA operation
   spanning the selected indices.

## Integration with vLLM

vLLM integrates NIXL through its KV connector framework. The `NixlConnector`
class handles KV cache transfer between prefill (producer) and decode
(consumer) instances.

### vLLM Configuration

```python
# Prefill instance:
engine_args = AsyncEngineArgs(
    model="meta-llama/Llama-3.1-70B",
    tensor_parallel_size=8,
    kv_transfer_config={
        "kv_connector": "NixlConnector",
        "kv_role": "kv_producer",
        "kv_buffer_device": "cuda",
        "kv_connector_extra_config": {
            "backends": ["UCX"]        # Or ["Libfabric"] for EFA
        }
    }
)

# Decode instance:
engine_args = AsyncEngineArgs(
    model="meta-llama/Llama-3.1-70B",
    tensor_parallel_size=8,
    kv_transfer_config={
        "kv_connector": "NixlConnector",
        "kv_role": "kv_consumer",
        "kv_buffer_device": "cuda",
        "kv_connector_extra_config": {
            "backends": ["UCX"]        # Must match producer backend
        }
    }
)
```

### What NixlConnector Does Internally

```
vLLM Startup:
  1. Create nixl_agent (one per inference worker)
  2. Register the KV cache pool with NIXL
  3. Exchange metadata (via ETCD or direct connection)
  4. Build and prepare descriptor lists for all KV blocks

Per Request (Prefill side):
  1. Run prefill, populating KV cache blocks
  2. Determine which physical blocks contain this request's KV data
  3. Create transfer using make_prepped_xfer with block indices
  4. Post transfer (async RDMA WRITE to decode node)
  5. Notification sent on completion

Per Request (Decode side):
  1. Poll for notifications (get_new_notifs)
  2. On notification: KV cache blocks have arrived in local GPU memory
  3. Update local page table to point to the received blocks
  4. Begin autoregressive decode
```

## The Hybrid Architecture: NCCL + NIXL

The recommended production architecture uses both NCCL and NIXL, each for
what it does best:

```
                    Prefill Pool                    Decode Pool
            +--------------------------+    +--------------------------+
            |     Prefill Instance 0   |    |     Decode Instance 0    |
            |                          |    |                          |
            |  GPU 0 <--NCCL--> GPU 1  |    |  GPU 0 <--NCCL--> GPU 1 |
            |  GPU 2 <--NCCL--> GPU 3  |    |  GPU 2 <--NCCL--> GPU 3 |
            |  GPU 4 <--NCCL--> GPU 5  |    |  GPU 4 <--NCCL--> GPU 5 |
            |  GPU 6 <--NCCL--> GPU 7  |    |  GPU 6 <--NCCL--> GPU 7 |
            |      (TP=8, NVLink)      |    |      (TP=8, NVLink)     |
            |                          |    |                          |
            +-----------+--------------+    +------------+-------------+
                        |                                |
                        |       NIXL (RDMA/EFA)          |
                        +-------- KV Cache ------------->+
                        |                                |
            +-----------+--------------+    +------------+-------------+
            |     Prefill Instance 1   |    |     Decode Instance 1    |
            |      (TP=8, NVLink)      |    |      (TP=8, NVLink)     |
            +--------------------------+    +--------------------------+

  NCCL: Intra-instance tensor parallelism
        - All-reduce, all-gather across TP ranks
        - Uses NVLink (intra-node) or IB/EFA (inter-node)
        - Sub-millisecond latency, every model layer

  NIXL: Inter-instance KV cache transfer
        - Point-to-point RDMA WRITE
        - Uses EFA/IB with multi-rail and GDR
        - Tens of milliseconds, once per request
```

### Why Not NIXL for Everything?

| Aspect | NCCL (TP) | NIXL (KV Transfer) |
|--------|-----------|-------------------|
| Pattern | Collective (all-to-all) | Point-to-point |
| Frequency | Every layer of every forward pass | Once per request |
| Message size | Small (MBs, activations) | Large (GBs, KV cache) |
| Latency target | < 1 ms | 10-100 ms acceptable |
| Operations | all-reduce, all-gather, reduce-scatter | read, write |
| Topology | Fixed TP group | Dynamic prefill/decode mapping |

NIXL does not implement collective operations. Building all-reduce from
point-to-point transfers would add significant latency overhead compared to
NCCL's highly optimized ring and tree algorithms. NCCL also leverages NVLink
for intra-node communication (900 GB/s on DGX H100), which NIXL does not
use.

## Performance Considerations

### Transfer Latency Budget

For disaggregated inference to be beneficial, the KV cache transfer time must
be less than the time saved by removing prefill from the decode path:

```
Monolithic: TTFT = T_prefill + T_first_decode_step
Disaggregated: TTFT = T_prefill + T_kv_transfer + T_first_decode_step

Benefit condition: T_kv_transfer < T_scheduling_savings

Typical values (Llama 70B, 4K context, P4d):
  T_prefill = 200 ms
  T_kv_transfer = 15-30 ms (160 MB over ~7 GB/s)
  T_scheduling_savings = 50-200 ms (no more prefill blocking decode)

For long context (32K+) the KV cache grows, but so do scheduling savings.
```

### Factors Affecting Transfer Performance

| Factor | Impact | How to Optimize |
|--------|--------|-----------------|
| **Number of EFA rails** | Linear bandwidth scaling | Use all available EFAs (4 on P4d, 32 on P5) |
| **GPU Direct RDMA** | Eliminates host copy | Ensure nvidia-peermem loaded, GPU-EFA co-located |
| **Block size** | Larger = better bandwidth utilization | Match vLLM block size (typically 16-128 tokens) |
| **Number of blocks per transfer** | Amortizes setup overhead | Batch multiple blocks into single transfer |
| **Concurrent transfers** | Keeps network pipe full | Overlap transfers for different requests |
| **Topology mapping** | Avoids cross-NUMA traffic | Libfabric plugin handles this automatically |

### Benchmarking KV Cache Transfers

NIXL includes KVBench, a benchmark specifically designed to measure KV cache
transfer performance for real model architectures:

```bash
# Generate benchmark plan for DeepSeek R1 with TP=1, PP=8:
python kvbench/main.py plan \
    --model ./examples/model_deepseek_r1.yaml \
    --model_config ./examples/block-tp1-pp8.yaml \
    --backend UCX \
    --etcd-endpoints "http://etcd-server:2379"

# This generates nixlbench commands with realistic block sizes
# and batch counts derived from the model architecture.
```

KVBench calculates the exact IO sizes based on model parameters (number of
layers, KV heads, head dimension, sequence length, page size, TP/PP
configuration) and generates `nixlbench` commands that simulate the actual
transfer patterns.

## Beyond KV Cache: Other NIXL Use Cases

While KV cache transfer for disaggregated inference is the primary use case,
NIXL's design supports additional inference data movement patterns:

### KV Cache Offloading to Storage

For long-context requests, KV caches can be offloaded to NVMe or S3 and
retrieved when needed:

```
Active Request:      GPU HBM  (fast access during decode)
Paused Request:      NVMe SSD (via GDS, microsecond-level retrieval)
Archived Request:    S3       (via OBJ backend, for resumable sessions)

NIXL handles all three tiers through its unified descriptor API.
The application uses the same transfer primitives regardless of
whether the destination is another GPU, a local disk, or S3.
```

### Model Shard Loading

When scaling up inference instances, new nodes need to load model weights.
NIXL can transfer model shards directly from a node that already has them
loaded in GPU memory, which can be faster than reading from storage:

```
Existing Node (GPU HBM) --NIXL RDMA--> New Node (GPU HBM)
                                       (faster than S3 download)
```

### Speculative Decoding Coordination

In speculative decoding, a small "draft" model generates candidate tokens
and a large "verifier" model validates them. If the draft and verifier run
on different nodes, NIXL can transfer the verification data:

```
Draft Node --> candidate tokens + KV updates --> Verifier Node
                        (via NIXL)
```

## Deployment on EKS

A typical EKS deployment for disaggregated inference with NIXL:

```
Kubernetes Cluster (EKS)
+--------------------------------------------------------------+
|                                                              |
|  ETCD StatefulSet (metadata coordination)                    |
|  +--------+                                                  |
|  | etcd-0 |  <-- Agents publish/fetch metadata here          |
|  +--------+                                                  |
|                                                              |
|  Prefill Deployment (HPA: scale on queue depth)              |
|  +-------------------+  +-------------------+                |
|  | prefill-0         |  | prefill-1         |                |
|  | 8x A100/H100      |  | 8x A100/H100     |                |
|  | 4x EFA            |  | 4x EFA           |                |
|  | vLLM kv_producer  |  | vLLM kv_producer |                |
|  +-------------------+  +-------------------+                |
|                                                              |
|  Decode Deployment (HPA: scale on active decodes)            |
|  +-------------------+  +-------------------+                |
|  | decode-0          |  | decode-1          |                |
|  | 8x A100/H100      |  | 8x A100/H100     |                |
|  | 4x EFA            |  | 4x EFA           |                |
|  | vLLM kv_consumer  |  | vLLM kv_consumer |                |
|  +-------------------+  +-------------------+                |
|                                                              |
|  Router Service (directs prefill/decode traffic)             |
|  +-------------------+                                       |
|  | router            |                                       |
|  +-------------------+                                       |
|                                                              |
+--------------------------------------------------------------+
```

Key infrastructure requirements:

- **EFA-enabled node groups**: Use P4d.24xlarge or P5.48xlarge instances
  with the EFA Kubernetes device plugin installed.
- **ETCD cluster**: For NIXL agent metadata coordination. Can be shared
  with other services but should be low-latency.
- **Security groups**: Self-referencing inbound AND outbound rules for SRD.
- **Primary VPC CIDR**: Nodes must be on the primary CIDR for SRD to work.
- **nvidia-peermem**: Kernel module must be loaded on all nodes for GPU
  Direct RDMA.

## Telemetry and Monitoring

NIXL includes a telemetry system for monitoring transfer performance in
production:

```bash
# Enable telemetry via environment variables:
export NIXL_TELEMETRY_ENABLE=1
export NIXL_TELEMETRY_BUFFER_SIZE=4096
export NIXL_TELEMETRY_DIR=/tmp/nixl_telemetry
```

Key metrics exported:

| Metric | Description |
|--------|-------------|
| `agent_tx_bytes` | Bytes transmitted per TX request |
| `agent_rx_bytes` | Bytes received per RX request |
| `agent_xfer_time` | Transfer time from start to completion (microseconds) |
| `agent_xfer_post_time` | Time from start to backend posting (microseconds) |
| `agent_memory_registered` | Memory registered per API call (bytes) |

These metrics can be consumed by a Python or C++ telemetry reader, or
exported to Prometheus using the experimental Prometheus exporter plugin.

## Summary

| Concept | Key Takeaway |
|---------|--------------|
| Disaggregation | Separate prefill and decode for better GPU utilization |
| KV Cache Transfer | NIXL RDMA transfers KV cache blocks between instances |
| vLLM Integration | NixlConnector handles registration, metadata exchange, and block-level transfers |
| Hybrid Architecture | NCCL for intra-instance TP; NIXL for inter-instance KV transfer |
| Performance | Transfer time must be less than scheduling savings; multi-rail and GDR are critical |
| EKS Deployment | EFA device plugin, ETCD for coordination, primary VPC CIDR, security group rules |

---

Previous: [Section 2: NIXL on AWS EFA](../02-efa-integration/README.md) |
Back to: [Workshop Overview](../README.md)
