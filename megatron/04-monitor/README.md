# 04 - Monitoring and Troubleshooting

## Objective

Monitor TechPulse Media's training job health, understand loss curves and learning rate schedules, diagnose common failures (hangs, NCCL errors, OOM), and manage checkpoints for fault recovery. This section is based on production experience running MoE models at 512-GPU scale on AWS.

## Training Metrics

### Loss Curve

The most important signal during training is the loss curve. Megatron logs the loss at every `--log-interval` steps:

```
 iteration      100/ 10000 | consumed samples:     51200 | elapsed time per iteration (ms): 2345.2 | learning rate: 9.500E-06 | global batch size:   512 | lm loss: 3.4521E+00 | loss scale: 1.0 | grad norm: 1.234 | num zeros: 0.0 | params norm: 543.21
```

### What to Look For

| Metric | Healthy Range | Warning Sign |
|--------|--------------|--------------|
| **lm loss** | Decreasing over time | Flat, increasing, or NaN |
| **grad norm** | 0.1 - 10.0 | > 100 (exploding gradients), 0.0 (dead gradients) |
| **learning rate** | Following schedule (warmup then decay) | Stuck at 0 or max |
| **loss scale** | 1.0 (BF16) | Decreasing rapidly (FP16 underflow) |
| **num zeros** | < 5% of total params | > 50% (vanishing gradients) |
| **elapsed time** | Stable per iteration | Spiking (communication stalls or data loading) |

### Typical Loss Curve Phases

```
Loss
 4.0 ┤
     │\
 3.5 ┤ \
     │  \          Phase 1: Rapid initial descent
 3.0 ┤   \         (first 500-1000 steps)
     │    \
 2.5 ┤     \___
     │         \___      Phase 2: Steady improvement
 2.0 ┤             \___  (bulk of training)
     │                 \_____
 1.5 ┤                       \_______
     │                               Phase 3: Plateau
 1.0 ┤                               (diminishing returns)
     └──────────────────────────────────────
     0    2000   4000   6000   8000   10000
                    Training Steps
```

### Warning: Loss Spikes

Occasional small loss spikes (10-20% above trend) are normal and typically caused by:
- Learning rate warmup completing
- Data domain transitions (if blending multiple datasets)
- Gradient noise from stochastic sampling

Large persistent spikes or NaN loss indicate:
- Learning rate too high -- reduce by 2-5x
- Gradient explosion -- reduce `--clip-grad` or check data quality
- Data corruption -- verify preprocessed binary files

## Learning Rate Schedules

Megatron supports several learning rate schedules. The most common for continued pre-training:

### Cosine Annealing (Recommended)

```
LR
 1e-5 ┤    ╭──────╮
      │   ╱        ╲
      │  ╱          ╲
      │ ╱            ╲
 5e-6 ┤╱              ╲
      │                ╲
      │                 ╲
 1e-6 ┤                  ╲_______________
      └─────────────────────────────────
      0    warmup     mid-point      end
         (100 steps)
```

Configuration:
```
--lr=1e-5
--lr-decay-style=cosine
--lr-warmup-iters=100
--min-lr=1e-6
```

### WSD (Warmup-Stable-Decay)

Used for continued pre-training where you want a long stable phase:

```
LR
 1e-5 ┤    ┌──────────────────────┐
      │   ╱                        ╲
      │  ╱                          ╲
      │ ╱                            ╲
 1e-6 ┤                               ╲___
      └─────────────────────────────────
      0   warmup    stable phase    decay
```

Configuration:
```
--lr=1e-5
--lr-decay-style=WSD
--lr-warmup-iters=100
--lr-decay-iters=1000
--min-lr=1e-6
```

## GPU Utilization Monitoring

### nvidia-smi Continuous Monitoring

```bash
# From inside the master pod
nvidia-smi dmon -s u -d 5

# Output every 5 seconds:
# gpu   sm   mem   enc   dec   jpg   ofa
#   0   87%  62%    0%    0%    0%    0%
#   1   85%  60%    0%    0%    0%    0%
#   ...
```

### Interpreting GPU Metrics

| Metric | Meaning | Target |
|--------|---------|--------|
| **SM%** | GPU compute utilization | 70-95% during training |
| **Mem%** | Memory bandwidth utilization | 40-70% |
| **GPU Temp** | Junction temperature | < 83C (throttles at 83C on A100) |
| **Power** | Power draw | Near TDP (400W A100, 700W H100) |

### Low GPU Utilization Diagnosis

If SM% is below 50%, check these in order:

```
Step 1: Is data loading the bottleneck?
  ├── Check: CPU utilization on data workers
  ├── Fix: Increase --num-workers (8 → 16)
  └── Fix: Move data to faster storage (FSx IOPS tier)

Step 2: Is communication the bottleneck?
  ├── Check: Is EFA active? (NCCL logs show NET/OFI)
  ├── Fix: Reduce TP if running cross-node (TP should be within-node)
  └── Fix: Increase micro_batch_size (amortizes communication)

Step 3: Is pipeline bubble the problem? (PP > 1)
  ├── Check: Idle time between forward and backward phases
  ├── Fix: Increase number of micro-batches per step
  └── Fix: Reduce PP if possible (increase TP instead)

Step 4: Is memory the bottleneck?
  ├── Check: GPU memory usage near limit
  ├── Fix: Enable activation checkpointing
  └── Fix: Enable distributed optimizer
```

## NCCL Debugging

### Common NCCL Errors

#### Error: Timeout

```
torch.distributed.DistributedBackendError: NCCL error in: ..., unhandled system error, NCCL version 2.27.3
ncclSystemError: System call (e.g. socket, malloc) or external library call failed
```

**Causes and fixes:**

| Scenario | Diagnosis | Fix |
|----------|-----------|-----|
| Timeout during init | Rendezvous cannot connect | Check security groups, hostNetwork, MASTER_ADDR |
| Timeout during training | One rank is slow or hung | Check GPU health (`nvidia-smi -q -d ECC`), increase NCCL_TIMEOUT |
| Timeout loading checkpoint | FSx I/O too slow for resharding | Increase TORCH_DISTRIBUTED_TIMEOUT to 1800 |

#### Error: Bootstrap No Socket Interface

```
NCCL WARN Bootstrap : no socket interface found
```

**Cause:** `NCCL_SOCKET_IFNAME=eth0` is set, but P4d/P5 instances use `ens*` interfaces.

**Fix:** Remove `NCCL_SOCKET_IFNAME` entirely. Let NCCL auto-detect.

#### Error: NVLink Peer Access

```
CUDA error: an illegal memory access was encountered (error 226)
Invalid access of peer GPU memory over nvlink
```

**Cause:** NVLS (NVLink SHARP) fails on some node configurations.

**Fix:** `NCCL_NVLS_ENABLE=0`

### Enabling NCCL Debug Logging

For debugging connectivity issues, temporarily increase NCCL verbosity:

```yaml
env:
  - name: NCCL_DEBUG
    value: "INFO"
  - name: NCCL_DEBUG_SUBSYS
    value: "INIT,NET"
```

This produces detailed logs showing:
- Which network interface NCCL selected
- Whether aws-ofi-nccl plugin loaded
- EFA device discovery
- Ring and tree topology construction

**Warning:** `NCCL_DEBUG=INFO` generates enormous log volume (gigabytes per hour at scale). Use only for debugging, then switch back to `WARN`.

### NCCL RAS (Remote Access Server)

NCCL 2.27+ includes a built-in Remote Access Server for diagnosing distributed training issues:

```bash
# Enable RAS (usually enabled by default)
export NCCL_RAS_ENABLE=1

# Query RAS status during training
nccl-ras-query --host <master-ip> --port 23456
```

Output during healthy training:
```
Group  Comms  Nodes  Ranks  Status   Errors
0      1      2      16     RUNNING  OK
```

Output during a hang:
```
Group  Comms  Nodes  Ranks  Status   Errors
0      1      2      16     RUNNING  MISMATCH
Warnings: Communicator ranks have different AllReduce operation counts
  14 ranks have launched up to operation 1234
   2 ranks have launched up to operation 1233
```

The mismatch indicates which ranks are lagging -- useful for identifying the problematic GPU or node.

## Checkpoint Management

### Checkpoint Structure

Megatron saves checkpoints in this structure:

```
/shared/checkpoints/techpulse-cpt/
  iter_0000500/
    mp_rank_00/         # TP rank 0
      model_optim_rng.pt
    mp_rank_01/         # TP rank 1
      model_optim_rng.pt
    latest_checkpointed_iteration.txt
  iter_0001000/
    mp_rank_00/
      model_optim_rng.pt
    mp_rank_01/
      model_optim_rng.pt
    latest_checkpointed_iteration.txt
```

Each checkpoint contains:
- **Model weights**: Split according to TP/PP configuration
- **Optimizer states**: Adam first and second moments
- **RNG state**: Random number generator state for exact reproducibility
- **Iteration count**: For resume tracking

### Checkpoint Size Estimation

```
Checkpoint size per save (approximate):

Model weights:       total_params * 2 bytes (BF16)
Optimizer states:    total_params * 12 bytes (Adam FP32)
RNG + metadata:      ~10 MB
Total:               total_params * 14 bytes

Example for 7B model:
  7B * 14 bytes = 98 GB per checkpoint
```

### Checkpoint Frequency Trade-offs

| Save Interval | Disk Usage | Recovery Loss | Use Case |
|---------------|-----------|---------------|----------|
| Every 100 steps | Very high | Minimal (lose ~100 steps) | Development, debugging |
| Every 500 steps | Moderate | Acceptable (lose ~500 steps) | Production training |
| Every 1000 steps | Low | Higher (lose ~1000 steps) | Long training runs with stable hardware |
| Every 5000 steps | Minimal | Significant | Very long runs on reliable hardware |

For TechPulse Media's 10,000-step training:
- `--save-interval=500` gives 20 checkpoints
- Keep the 3 most recent with `save_top_k=3` to manage disk usage
- Total disk: 3 * 98 GB = ~294 GB on FSx

### Resuming from Checkpoint

Megatron automatically resumes from the latest checkpoint when `--load` points to the checkpoint directory:

```bash
--load=/shared/checkpoints/techpulse-cpt
```

It reads `latest_checkpointed_iteration.txt` to find the most recent complete checkpoint, loads the model, optimizer, and RNG state, and continues training from the next iteration.

**Important:** When changing parallelism (e.g., TP=2 to TP=4), the checkpoint must be **resharded**. This happens automatically with NeMo's checkpoint loading strategy but can be slow with many GPUs on shared storage. Set `TORCH_DISTRIBUTED_TIMEOUT=1800` to prevent timeouts during resharding.

### Checkpoint Cleanup

To avoid filling up FSx, implement a retention policy. Keep only the N most recent checkpoints:

```bash
# List checkpoints sorted by iteration
ls -d /shared/checkpoints/techpulse-cpt/iter_* | sort -t_ -k2 -n

# Keep only the 3 most recent, delete older ones
ls -d /shared/checkpoints/techpulse-cpt/iter_* | sort -t_ -k2 -n | \
  head -n -3 | xargs rm -rf
```

The NeMo `ModelCheckpoint` callback handles this automatically with `save_top_k`:

```python
ModelCheckpoint(
    every_n_train_steps=500,
    save_top_k=3,    # Keep 3 best/most recent
    monitor="reduced_train_loss",
)
```

## Diagnosing Training Hangs

Training hangs are the most common failure mode in distributed training. The process appears to be running (no crash, no error) but makes no progress.

### Step-by-Step Diagnosis

```
Step 1: Check GPU activity
  $ nvidia-smi dmon -s u -d 1
  If SM% = 0 on all GPUs → hang confirmed
  If SM% > 0 on some GPUs → partial hang (rank desync)

Step 2: Check which ranks are stuck
  $ NCCL RAS query (if available)
  Or check logs for last reported iteration per rank

Step 3: Get Python stack trace
  $ PID=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | head -1)
  $ py-spy dump --pid $PID

Step 4: Check for known patterns
  Pattern: "token_dispatch" in stack → MoE routing hang (all-to-all)
  Pattern: "synchronize" in stack → CUDA sync waiting for NCCL
  Pattern: "gc.collect" in stack → GC-induced deadlock
  Pattern: "allgather" in stack → Checkpoint resharding timeout

Step 5: Check hardware
  $ nvidia-smi -q -d ECC     # Check for ECC errors
  $ dmesg | grep -i xid       # Check for GPU XID errors
  $ dmesg | grep -i efa       # Check for EFA errors
```

### Common Hang Patterns

| Pattern | Stack Trace Indicator | Cause | Fix |
|---------|----------------------|-------|-----|
| MoE token dispatch | `token_dispatcher.py:token_dispatch` | All-to-All deadlock | Upgrade aws-ofi-nccl to 1.17.2+, check EP divisibility |
| NCCL AllReduce | `nccl_allreduce` | Rank desynchronization | Check GC callback, increase NCCL_TIMEOUT |
| Triton compilation | `triton/compiler` | FSx cache lock contention | Set TRITON_CACHE_DIR to local /tmp |
| Data loading | `DataLoader.__next__` | FSx I/O stall | Reduce num_workers, check FSx throughput |
| Checkpoint save | `torch.save` | FSx write bottleneck | Reduce checkpoint frequency, increase FSx capacity |

### GC-Induced Hangs (Production Lesson)

A common and subtle cause of training hangs at scale is Python's garbage collector running unsynchronized across ranks. When the GC callback triggers:

```
Rank 0: [Step 49] [GC pauses 2 seconds] [Step 50] ──> NCCL AllReduce (waiting for Rank 1)
Rank 1: [Step 49] [Step 50] [NCCL AllReduce] ──> waiting for Rank 0 (still in GC)
Result: DEADLOCK
```

**Fix:** Disable NeMo's `GarbageCollectionCallback`. If memory pressure requires GC, use a synchronized version that calls `dist.barrier()` before and after `gc.collect()`:

```python
def safe_garbage_collect():
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    gc.collect()
    torch.cuda.empty_cache()
    if dist.is_initialized():
        dist.barrier()
```

## Monitoring Checklist During Training

Use this as a periodic health check while training is running:

```
Every 10 minutes:
  [ ] Loss is decreasing (check training logs)
  [ ] GPU SM% is 70-95% (nvidia-smi dmon)
  [ ] No NCCL warnings in logs (grep for WARN/ERROR)
  [ ] Iteration time is stable (no spikes)

Every hour:
  [ ] Checkpoints are being written (ls checkpoint dir)
  [ ] Disk usage is within limits (df -h /shared)
  [ ] No GPU ECC errors (nvidia-smi -q -d ECC)
  [ ] Loss curve follows expected shape

On any anomaly:
  [ ] Check NCCL RAS status
  [ ] Check dmesg for hardware errors
  [ ] Check FSx throughput (CloudWatch metrics)
  [ ] Get py-spy stack dump if process appears hung
```

## Summary of Critical AWS-Specific Lessons

These lessons come from production experience running Megatron-LM at scale on AWS:

| Issue | Impact | Root Cause | Resolution |
|-------|--------|------------|------------|
| aws-ofi-nccl v1.14.0 deadlock | Random training hangs | Missing domain lock in `reg_mr()` | Upgrade to v1.17.2+ |
| Triton cache on FSx | Hang at first forward pass | 512 processes contending for file locks | `TRITON_CACHE_DIR=/tmp/...` |
| GarbageCollectionCallback | Hang at step 50 (or GC interval) | Unsynchronized GC across ranks | Disable or use barrier-synchronized GC |
| OptimizerMonitor with MoE | 20-40% throughput loss | Iterates all 3,840+ expert parameters per step | Disable for MoE models |
| NCCL_NVLS_ENABLE | NVLink peer access crash | NVLS initialization failure on some nodes | Set to 0 |
| NCCL_SOCKET_IFNAME=eth0 | Bootstrap failure on P5 | P5 uses ens* interfaces, not eth0 | Do not set this variable |
| NCCL_PROTO / NCCL_ALGO | Suboptimal performance | Disables aws-ofi-nccl tuner | Do not set these variables |
| Default NCCL timeout (600s) | Timeout during checkpoint resharding | FSx I/O + 512-GPU AllGather exceeds 10 min | Set NCCL_TIMEOUT=1800 |

## What Comes Next

After training completes:

1. **Evaluate** the checkpoint on held-out data to verify domain knowledge acquisition
2. **Convert** the checkpoint to HuggingFace format for downstream SFT or serving
3. **Fine-tune** with SFT (Supervised Fine-Tuning) on domain-specific instruction data
4. **Align** with DPO/GRPO for response quality (see the NeMo RL workshop section)
5. **Deploy** for inference with vLLM or TensorRT-LLM

The trained checkpoint from this workshop feeds directly into the SFT and alignment pipeline stages.

## References

- [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)
- [Megatron-Core Documentation](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core)
- [NeMo Documentation](https://docs.nvidia.com/nemo/index.html)
- [AWS OFI NCCL](https://github.com/aws/aws-ofi-nccl)
- [EFA Cheatsheet](https://github.com/aws-samples/awsome-distributed-training/blob/main/1.architectures/efa-cheatsheet.md)
- [NCCL RAS Documentation](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2273/user-guide/docs/troubleshooting/ras.html)
- [NVIDIA Resiliency Extension](https://github.com/NVIDIA/nvidia-resiliency-ext)
