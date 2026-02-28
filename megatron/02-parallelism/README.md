# 02 - Understanding Parallelism Strategies

## Objective

Choose the right combination of Tensor Parallelism (TP), Pipeline Parallelism (PP), Data Parallelism (DP), and Expert Parallelism (EP) for TechPulse Media's model and GPU cluster. This section explains how each parallelism dimension works, what it costs in terms of communication and memory, and how to select the right configuration for different model sizes and GPU counts.

## Background: Why Parallelism Matters

A 7B-parameter dense model in BF16 requires roughly 14GB just for the model weights. Add gradients (14GB), optimizer states (56GB for Adam with FP32 master weights), and activations, and a single GPU cannot fit the training state. Parallelism splits this load across multiple GPUs.

The four dimensions of parallelism in Megatron-LM are:

```
                  ┌─────────────────────────────────┐
                  │          Total GPUs              │
                  │   = TP x PP x DP (x EP for MoE) │
                  └─────────┬───────────────────────┘
                            │
           ┌────────────────┼────────────────┐
           │                │                │
     ┌─────┴─────┐   ┌─────┴─────┐   ┌──────┴─────┐
     │  Tensor   │   │ Pipeline  │   │    Data     │
     │ Parallel  │   │ Parallel  │   │  Parallel   │
     │   (TP)    │   │   (PP)    │   │   (DP)      │
     │           │   │           │   │             │
     │ Splits    │   │ Splits    │   │ Replicates  │
     │ layers    │   │ stages    │   │ model, each │
     │ across    │   │ across    │   │ GPU gets    │
     │ GPUs      │   │ nodes     │   │ different   │
     │ within    │   │           │   │ data batch  │
     │ a node    │   │           │   │             │
     └───────────┘   └───────────┘   └─────────────┘
```

## Tensor Parallelism (TP)

### What It Does

Tensor Parallelism splits individual layer weight matrices across GPUs. For a transformer's self-attention layer, the Q, K, V projection matrices are split column-wise, and the output projection is split row-wise. Each GPU computes its portion of the matrix multiplication, then the results are combined via AllReduce.

### How It Works

Consider a weight matrix W of shape [hidden_size, hidden_size] with TP=2:

```
Full weight matrix W (4096 x 4096):
┌──────────────────────────────┐
│                              │
│         4096 x 4096          │
│                              │
└──────────────────────────────┘

Split with TP=2:
┌──────────────┐ ┌──────────────┐
│  GPU 0       │ │  GPU 1       │
│  4096 x 2048 │ │  4096 x 2048 │
└──────────────┘ └──────────────┘

Each GPU:
1. Receives the full input activation X
2. Computes partial output: Y_i = X @ W_i
3. AllReduce to combine: Y = Y_0 + Y_1
```

### Communication Pattern

TP requires **AllReduce** after every transformer layer's attention and MLP blocks. This means:

- **2 AllReduce operations per transformer layer** (one for attention, one for MLP)
- Communication volume per AllReduce: `batch_size * seq_len * hidden_size * 2 bytes` (BF16)
- Must happen on **high-bandwidth interconnect** (NVLink within a node)

### When to Use TP

| Scenario | Recommendation |
|----------|---------------|
| Model fits on one GPU | TP=1 (no tensor parallelism needed) |
| Model too large for one GPU, within one node | TP=2, 4, or 8 (use NVLink bandwidth) |
| Cross-node communication | Avoid TP across nodes (too slow without NVLink) |

### Key Rule

**TP should stay within a single node.** NVLink provides 600-900 GB/s bandwidth within a node (A100/H100), while EFA provides 50-400 GB/s across nodes. Running TP across nodes degrades throughput significantly due to the AllReduce after every layer.

## Pipeline Parallelism (PP)

### What It Does

Pipeline Parallelism splits the model vertically by assigning consecutive groups of transformer layers to different GPUs (or nodes). GPU 0 gets layers 0-15, GPU 1 gets layers 16-31, and so on. Data flows through the pipeline in micro-batches.

### How It Works

For a 32-layer model with PP=2:

```
Stage 0 (GPU 0-3):          Stage 1 (GPU 4-7):
┌───────────────────┐        ┌───────────────────┐
│ Embedding         │        │ Layers 16-31      │
│ Layers 0-15       │  ───>  │ LM Head           │
│                   │        │                   │
└───────────────────┘        └───────────────────┘
  Forward pass                 Forward pass
  sends activations            computes loss
  to Stage 1                   sends gradients back
```

### The Pipeline Bubble

The fundamental cost of PP is the **pipeline bubble** -- idle time when stages wait for data from upstream or gradients from downstream. Megatron mitigates this with micro-batch scheduling:

```
Time -->

Stage 0: [F1][F2][F3][F4][  ][  ][  ][  ][B4][B3][B2][B1]
Stage 1: [  ][F1][F2][F3][F4][  ][  ][B4][B3][B2][B1][  ]
                                   ^
                          Pipeline bubble (idle)

F = forward micro-batch, B = backward micro-batch
```

The bubble fraction is approximately `(PP - 1) / num_microbatches`. More micro-batches reduce the bubble but increase memory for in-flight activations.

### Communication Pattern

PP requires **point-to-point** communication (Send/Recv) between adjacent pipeline stages:

- Activations sent forward: `batch_size * seq_len * hidden_size * 2 bytes`
- Gradients sent backward: same size
- Happens once per micro-batch per stage boundary
- Can run over **EFA** (cross-node) because bandwidth demand is lower than TP

### When to Use PP

| Scenario | Recommendation |
|----------|---------------|
| Model fits within TP on one node | PP=1 (no pipeline parallelism needed) |
| Model too large for one node even with TP=8 | PP=2+ (split across nodes) |
| Many micro-batches available | PP works well (small bubble fraction) |
| Very few micro-batches | Avoid PP (large bubble, wasted compute) |

## Data Parallelism (DP)

### What It Does

Data Parallelism replicates the model on each GPU (or group of GPUs, after TP/PP). Each replica processes a different mini-batch of data. After the backward pass, gradients are synchronized across replicas via AllReduce, and all replicas apply the same optimizer step.

### How It Works

```
DP Rank 0:                    DP Rank 1:
┌───────────────────┐         ┌───────────────────┐
│ Full model copy   │         │ Full model copy   │
│ Batch: [0:256]    │         │ Batch: [256:512]  │
│                   │         │                   │
│ Forward + Backward│         │ Forward + Backward│
│ Local gradients   │         │ Local gradients   │
└────────┬──────────┘         └────────┬──────────┘
         │                             │
         └──────── AllReduce ──────────┘
                   gradients
                      │
              Synchronized update
```

### DP Is Implicit

In Megatron-LM, DP is calculated from the other parallelism dimensions:

```
DP = total_GPUs / (TP * PP)
```

For MoE models with Expert Parallelism:

```
DP = total_GPUs / (TP * PP * EP)
```

You do not set DP explicitly -- it is whatever GPUs remain after TP, PP, and EP are assigned.

### Communication Pattern

- **AllReduce** of gradients once per training step (not per layer like TP)
- Communication volume: `num_params * 2 bytes` (BF16 gradients)
- Can overlap with backward computation (gradient bucketing)
- Works well over **EFA** because it is less latency-sensitive than TP

### Distributed Optimizer (ZeRO Stage 1)

Megatron supports a **distributed optimizer** that shards optimizer states across DP ranks:

```python
optimizer = OptimizerConfig(
    use_distributed_optimizer=True,  # ZeRO Stage 1
    ...
)
```

This reduces per-GPU memory for optimizer states from `params * 12 bytes` (Adam FP32) to `params * 12 / DP bytes`. For a 7B model with DP=8, this saves ~84GB to ~10.5GB per GPU.

## Expert Parallelism (EP) -- Mixture of Experts

### What It Does

For Mixture of Experts (MoE) models, Expert Parallelism distributes experts across GPUs. Each GPU holds a subset of experts, and tokens are routed to the appropriate GPU via All-to-All communication.

### How It Works

For a model with 64 experts and EP=8:

```
GPU 0: Experts 0-7      GPU 4: Experts 32-39
GPU 1: Experts 8-15     GPU 5: Experts 40-47
GPU 2: Experts 16-23    GPU 6: Experts 48-55
GPU 3: Experts 24-31    GPU 7: Experts 56-63

Token routing:
1. Router selects top-K experts for each token
2. All-to-All sends tokens to the GPU holding the selected expert
3. Each GPU processes its local expert computation
4. All-to-All sends results back to the original GPU
```

### Critical Rule: Expert Divisibility

```
num_experts % expert_parallel_size == 0    # MUST be true
```

If this is not satisfied, Megatron crashes at startup with an assertion error. The check is in `megatron/core/transformer/moe/moe_layer.py`:

```python
assert self.config.num_moe_experts % ep_size == 0
self.num_local_experts = self.config.num_moe_experts // ep_size
```

### Valid EP Configurations

| Num Experts | Valid EP Values | Recommended |
|-------------|-----------------|-------------|
| 8 | 1, 2, 4, 8 | 8 |
| 16 | 1, 2, 4, 8, 16 | 8 |
| 32 | 1, 2, 4, 8, 16, 32 | 8 |
| 64 | 1, 2, 4, 8, 16, 32, 64 | 8 |
| 62 | 1, 2, 31, 62 | 2 (limited choices!) |

**Recommendation:** Always use power-of-2 expert counts (8, 16, 32, 64, 128) for flexibility in EP selection.

### Communication Pattern

- **All-to-All** communication twice per MoE layer (dispatch + combine)
- Communication volume depends on tokens routed per expert and hidden size
- Higher than TP/DP communication; benefits from high-bandwidth interconnect

## Calculating Parallelism for Your Setup

### The Master Formula

```
total_GPUs = TP * PP * DP * EP

Where:
  TP = tensor_parallel_size
  PP = pipeline_parallel_size
  EP = expert_parallel_size (1 for dense models)
  DP = total_GPUs / (TP * PP * EP)    (implicit, must be >= 1)
```

### Global Batch Size Constraint

```
global_batch_size % (micro_batch_size * DP) == 0    # MUST be true
```

If this constraint is violated, Megatron exits with:

```
Error: global batch size (X) is not divisible by micro batch size (Y) times data parallel size (Z)
```

### Memory Estimation Per GPU

For a dense transformer model in BF16 with Adam optimizer:

```
Per-GPU memory (approximate):

Model weights:     params_per_gpu * 2 bytes        (BF16)
Gradients:         params_per_gpu * 2 bytes        (BF16)
Optimizer states:  params_per_gpu * 12 bytes / DP  (FP32 Adam, with distributed optimizer)
Activations:       batch * seq_len * hidden * num_layers_per_gpu * ~10 bytes

Where:
  params_per_gpu = total_params / (TP * PP)
```

## Decision Table: Common Configurations

### Dense Models (No MoE)

| Model Size | GPUs | TP | PP | DP | micro_bs | global_bs | Notes |
|-----------|------|----|----|-----|----------|-----------|-------|
| 1.5B | 2 (1 node) | 1 | 1 | 2 | 4 | 64 | Single node, data parallel only |
| 7B | 8 (1 node) | 2 | 1 | 4 | 2 | 256 | TP=2 for weight splitting |
| 7B | 16 (2 nodes) | 2 | 1 | 8 | 2 | 512 | TP within node, DP across nodes |
| 13B | 16 (2 nodes) | 4 | 1 | 4 | 1 | 256 | Higher TP for larger layers |
| 30B | 16 (2 nodes) | 8 | 2 | 1 | 1 | 128 | Full TP + PP, no DP |
| 70B | 32 (4 nodes) | 8 | 4 | 1 | 1 | 256 | 4 pipeline stages |
| 70B | 64 (8 nodes) | 8 | 4 | 2 | 1 | 512 | DP=2 for throughput |

### MoE Models

| Model | Experts | GPUs | TP | PP | EP | DP | Notes |
|-------|---------|------|----|----|----|-----|-------|
| 8x3B MoE | 8 | 8 (1 node) | 1 | 1 | 8 | 1 | 1 expert per GPU |
| 8x3B MoE | 8 | 16 (2 nodes) | 1 | 1 | 8 | 2 | DP=2 for throughput |
| 64x3B MoE | 64 | 64 (8 nodes) | 1 | 1 | 8 | 8 | 8 experts/GPU, DP=8 |
| 64x3B MoE | 64 | 128 (16 nodes) | 2 | 1 | 8 | 8 | TP=2 + EP=8 |

### TechPulse Media's Configuration

TechPulse Media has 2x P4d.24xlarge (16x A100 40GB) and wants to continue pre-training a 7B model:

```
Model:      7B parameters (dense)
GPUs:       16 (2 nodes x 8 GPUs)
GPU memory: 40GB per A100

Recommended:
  TP = 2    (split layers across 2 GPUs within each node)
  PP = 1    (no pipeline parallelism -- model fits with TP=2)
  DP = 8    (16 / (2 * 1) = 8 data parallel replicas)
  EP = 1    (dense model, no experts)

Batch size:
  micro_batch_size = 2
  DP = 8
  global_batch_size = 2 * 8 * grad_accum_steps
  global_batch_size = 512  (with grad_accum = 32)
```

Memory breakdown per GPU:

```
Model weights:    7B / 2 (TP) = 3.5B params * 2 bytes = 7 GB
Gradients:        3.5B * 2 bytes = 7 GB
Optimizer:        3.5B * 12 bytes / 8 (DP) = 5.25 GB  (distributed optimizer)
Activations:      ~8 GB (batch=2, seq=4096, hidden=4096, 32 layers / 1 PP)
Buffer/overhead:  ~4 GB
Total:            ~31 GB  (fits in 40GB A100)
```

## Sequence Parallelism (SP)

When TP > 1, Megatron supports **Sequence Parallelism** as an optimization. SP distributes the LayerNorm and Dropout computations (which are not split by TP) across the TP group along the sequence dimension:

```python
strategy = MegatronStrategy(
    tensor_model_parallel_size=2,
    sequence_parallel=True,  # Requires TP > 1
)
```

Benefits:
- Reduces activation memory by a factor of TP for LayerNorm and Dropout
- Replaces AllReduce with more memory-efficient AllGather + ReduceScatter
- No additional communication cost (same total bytes transferred)

**Rule:** Always enable SP when TP > 1. There is no downside.

## Context Parallelism (CP)

For very long sequences (32K+ tokens), Context Parallelism splits the sequence across GPUs. Each GPU processes a portion of the sequence, with ring attention handling cross-GPU attention computation:

```python
strategy = MegatronStrategy(
    context_parallel_size=2,  # Split sequence across 2 GPUs
)
```

**When to use:** Only when sequence length exceeds what fits in GPU memory with other parallelism dimensions. For TechPulse Media's 4096-token sequences, CP is not needed.

## Quick Batch Size Calculator

When node count changes (hardware failure, scaling up/down), the global batch size must be recalculated:

```bash
# Calculate valid batch sizes
NODES=2
GPUS_PER_NODE=8
TP=2
PP=1
MICRO_BATCH=2

TOTAL_GPUS=$((NODES * GPUS_PER_NODE))
DP=$((TOTAL_GPUS / (TP * PP)))
DIVISOR=$((MICRO_BATCH * DP))

echo "Total GPUs:     $TOTAL_GPUS"
echo "Data Parallel:  $DP"
echo "Divisor:        $DIVISOR"
echo ""
echo "Valid global_batch_size values:"
for i in 1 2 4 8 16 32; do
    echo "  $((DIVISOR * i))"
done
```

Output for TechPulse Media (2 nodes, TP=2):

```
Total GPUs:     16
Data Parallel:  8
Divisor:        16
Valid global_batch_size values:
  16
  32
  64
  128
  256
  512
```

## Summary: How to Choose

```
Step 1: Can the model fit on one GPU?
  YES → TP=1, PP=1, maximize DP
  NO  → Go to Step 2

Step 2: Can the model fit with TP within one node?
  TP=2 → 2x memory per GPU
  TP=4 → 4x memory per GPU
  TP=8 → 8x memory per GPU (full node)
  If YES → PP=1, DP = total_GPUs / TP
  If NO  → Go to Step 3

Step 3: Model needs more than one node
  Set TP=8 (full NVLink within node)
  Set PP=N where N = number of nodes needed for model to fit
  DP = total_GPUs / (TP * PP)

Step 4: For MoE models, add EP
  EP should divide evenly into num_experts
  EP is typically 8 (one EP group per node)
  DP = total_GPUs / (TP * PP * EP)

Step 5: Verify constraints
  - DP >= 1
  - global_batch_size % (micro_batch * DP) == 0
  - Memory per GPU < GPU memory limit
  - For MoE: num_experts % EP == 0
```

## Verification Checklist

- [ ] TP is set to 1, 2, 4, or 8 (must be power of 2, within one node)
- [ ] PP divides evenly into the number of transformer layers
- [ ] DP >= 1 (total_GPUs / (TP * PP * EP) must be at least 1)
- [ ] global_batch_size is divisible by (micro_batch_size * DP)
- [ ] (MoE) num_experts is divisible by EP
- [ ] Estimated per-GPU memory fits within GPU memory limit
- [ ] SP is enabled when TP > 1

## Next Step

Proceed to [03 - Launch Training](../03-launch/README.md) to deploy the training job on EKS with the parallelism configuration chosen here.
