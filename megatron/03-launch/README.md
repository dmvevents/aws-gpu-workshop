# 03 - Launching Distributed Training on EKS

## Objective

Deploy TechPulse Media's multi-node Megatron-LM training job on Amazon EKS. This section covers building the container image, configuring NCCL and EFA for AWS networking, writing Kubernetes manifests, and launching distributed training with `torchrun`.

## Step 1: Container Image

Training with Megatron-LM on EKS requires a container image with the following stack:

```
┌─────────────────────────────────────────┐
│  Application Layer                      │
│  ├── NeMo / Megatron-LM                │
│  ├── Megatron-Core                      │
│  ├── Transformer Engine                 │
│  └── Training script + configs          │
├─────────────────────────────────────────┤
│  Communication Layer                    │
│  ├── NCCL 2.27+                        │
│  ├── aws-ofi-nccl 1.17+               │
│  └── libfabric (EFA provider)          │
├─────────────────────────────────────────┤
│  Compute Layer                          │
│  ├── PyTorch 2.x                       │
│  ├── CUDA 12.x                         │
│  └── cuDNN, cuBLAS                     │
└─────────────────────────────────────────┘
```

### Dockerfile

```dockerfile
FROM nvcr.io/nvidia/pytorch:25.01-py3

# Install RDMA and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libibverbs1 libibverbs-dev librdmacm1 librdmacm-dev \
    rdma-core ibverbs-utils ibverbs-providers \
    libhwloc15 libhwloc-dev \
    build-essential autoconf automake libtool wget git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Build libfabric with EFA support
ARG LIBFABRIC_VERSION="2.3.0"
RUN cd /tmp && \
    wget -q "https://github.com/ofiwg/libfabric/releases/download/v${LIBFABRIC_VERSION}/libfabric-${LIBFABRIC_VERSION}.tar.bz2" && \
    tar xjf libfabric-${LIBFABRIC_VERSION}.tar.bz2 && \
    cd libfabric-${LIBFABRIC_VERSION} && \
    ./configure --prefix=/opt/amazon/efa \
      --enable-efa=yes --with-cuda=/usr/local/cuda --enable-cuda-dlopen && \
    make -j$(nproc) && make install && \
    echo "/opt/amazon/efa/lib" > /etc/ld.so.conf.d/efa.conf && \
    ldconfig && cd / && rm -rf /tmp/libfabric-*

# Build aws-ofi-nccl (NCCL-to-EFA bridge)
ARG AWS_OFI_NCCL_VERSION="1.17.1"
RUN cd /tmp && \
    git clone --depth 1 --branch v${AWS_OFI_NCCL_VERSION} \
      https://github.com/aws/aws-ofi-nccl.git && \
    cd aws-ofi-nccl && ./autogen.sh && \
    ./configure --prefix=/opt/aws-ofi-nccl \
      --with-libfabric=/opt/amazon/efa --with-cuda=/usr/local/cuda && \
    make -j$(nproc) && make install && \
    echo "/opt/aws-ofi-nccl/lib" > /etc/ld.so.conf.d/aws-ofi-nccl.conf && \
    ldconfig && cd / && rm -rf /tmp/aws-ofi-nccl

# Install NeMo and Megatron-Core
RUN pip install --no-cache-dir \
    nemo_toolkit[nlp] megatron-core transformer-engine[pytorch] einops \
    --extra-index-url https://pypi.nvidia.com

# Set networking environment
ENV FI_PROVIDER=efa \
    FI_EFA_USE_DEVICE_RDMA=1 \
    FI_EFA_USE_HUGE_PAGE=0 \
    LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH} \
    TRITON_CACHE_DIR=/tmp/triton_cache

# Verify installation
RUN python3 -c "import torch; print(f'PyTorch {torch.__version__}')" && \
    ls /opt/aws-ofi-nccl/lib/libnccl-net*.so && \
    echo "Build verified"

WORKDIR /workspace
CMD ["/bin/bash"]
```

### Build and Push

```bash
# Build
docker build -t megatron-training:latest -f Dockerfile.megatron .

# Tag for ECR
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-2
REPO=${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/megatron-training

aws ecr get-login-password --region ${REGION} | \
  docker login --username AWS --password-stdin ${REPO}

docker tag megatron-training:latest ${REPO}:latest
docker push ${REPO}:latest
```

## Step 2: NCCL and EFA Environment Variables

This is the most critical configuration section. Incorrect environment variables are the number one cause of training hangs, silent performance degradation, and startup failures on AWS GPU instances.

### Variables You MUST Set

| Variable | Value | Why |
|----------|-------|-----|
| `FI_PROVIDER` | `efa` | Select EFA as the libfabric provider |
| `FI_EFA_USE_HUGE_PAGE` | `0` | Prevents memory allocation failures during fork/GC |
| `NCCL_TIMEOUT` | `1800` | 30-minute timeout for large collectives (default 10 min is too short) |
| `NCCL_NVLS_ENABLE` | `0` | Disables NVLink SHARP -- fixes peer access errors on some nodes |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Reduces CUDA memory fragmentation for large models |
| `TORCH_DISTRIBUTED_TIMEOUT` | `1800` | Must match NCCL_TIMEOUT -- PyTorch's ProcessGroup watchdog |
| `TRITON_CACHE_DIR` | `/tmp/triton_cache` | Uses local disk, NOT shared filesystem (see below) |

### Variables You Must NOT Set

| Variable | Why NOT |
|----------|---------|
| `NCCL_SOCKET_IFNAME=eth0` | P4d/P5 instances do not use `eth0` for EFA. Causes "Bootstrap: no socket interface found" |
| `NCCL_PROTO` | Disables the aws-ofi-nccl tuner's algorithm selection -- causes suboptimal protocol choices |
| `NCCL_ALGO` | Same as above -- disables tuner |
| `NCCL_P2P_DISABLE=1` | Disables NVLink GPU-to-GPU transfers. 10-100x performance loss |
| `CUDA_LAUNCH_BLOCKING=1` | Forces synchronous CUDA execution. Debug only -- 10-100x slower |
| `FI_EFA_USE_DEVICE_RDMA` | Auto-enabled in libfabric >= 1.18.0. Manual setting can conflict |
| `FI_EFA_FORK_SAFE` | Auto-managed by aws-ofi-nccl >= 1.13.0 |

### Why Triton Cache Must Be Local

When Triton (used by FlashAttention and fused kernels) compiles JIT kernels, it writes them to a cache directory. If this directory is on a shared filesystem (FSx), hundreds of processes contend for file locks simultaneously:

```
Process 0: acquire lock → write kernel → release lock
Process 1: acquire lock → BLOCKED waiting for Process 0
Process 2: acquire lock → BLOCKED waiting for Process 1
...
Process 511: acquire lock → BLOCKED waiting for Process 510

Result: training hangs at first forward pass (Triton compilation)
```

Setting `TRITON_CACHE_DIR=/tmp/triton_cache` puts the cache on local NVMe. Each pod compiles independently -- slightly redundant but much faster than contended shared storage.

## Step 3: Kubernetes Manifest -- PyTorchJob

The Kubeflow Training Operator provides the `PyTorchJob` CRD for distributed PyTorch training. It automatically handles:
- Worker pod creation and cleanup
- `MASTER_ADDR` and `MASTER_PORT` environment injection
- `WORLD_SIZE` and `RANK` assignment
- Pod restart on failure

### PyTorchJob Manifest

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: megatron-cpt-techpulse
  namespace: megatron-workshop
spec:
  nprocPerNode: "8"       # GPUs per node
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        metadata:
          labels:
            app: megatron-training
            role: master
        spec:
          hostNetwork: true
          dnsPolicy: ClusterFirstWithHostNet
          containers:
            - name: megatron
              image: <your-ecr>/megatron-training:latest
              command:
                - torchrun
                - --nproc_per_node=8
                - --nnodes=2
                - --node_rank=0
                - --master_addr=$(MASTER_ADDR)
                - --master_port=$(MASTER_PORT)
                - /workspace/train_megatron.py
                # --- Model configuration ---
                - --num-layers=32
                - --hidden-size=4096
                - --num-attention-heads=32
                - --seq-length=4096
                - --max-position-embeddings=4096
                # --- Parallelism ---
                - --tensor-model-parallel-size=2
                - --pipeline-model-parallel-size=1
                # --- Training ---
                - --micro-batch-size=2
                - --global-batch-size=512
                - --train-iters=10000
                - --lr=1e-5
                - --lr-decay-style=cosine
                - --min-lr=1e-6
                - --weight-decay=0.1
                - --clip-grad=1.0
                - --bf16
                # --- Data ---
                - --data-path=/shared/data/megatron/techpulse_text_document
                - --tokenizer-type=HuggingFaceTokenizer
                - --tokenizer-model=/shared/tokenizer/llama-3.1-8b
                - --split=98,2,0
                # --- Checkpointing ---
                - --save=/shared/checkpoints/techpulse-cpt
                - --save-interval=500
                - --load=/shared/checkpoints/techpulse-cpt
                # --- Logging ---
                - --log-interval=10
                - --eval-interval=500
                - --eval-iters=20
              env:
                - name: FI_PROVIDER
                  value: "efa"
                - name: FI_EFA_USE_HUGE_PAGE
                  value: "0"
                - name: NCCL_TIMEOUT
                  value: "1800"
                - name: NCCL_NVLS_ENABLE
                  value: "0"
                - name: NCCL_DEBUG
                  value: "WARN"
                - name: TORCH_DISTRIBUTED_TIMEOUT
                  value: "1800"
                - name: PYTORCH_CUDA_ALLOC_CONF
                  value: "expandable_segments:True"
                - name: TRITON_CACHE_DIR
                  value: "/tmp/triton_cache"
              resources:
                requests:
                  nvidia.com/gpu: 8
                  vpc.amazonaws.com/efa: 4
                  memory: "200Gi"
                  cpu: "80"
                limits:
                  nvidia.com/gpu: 8
                  vpc.amazonaws.com/efa: 4
                  memory: "200Gi"
                  cpu: "96"
              volumeMounts:
                - name: shared
                  mountPath: /shared
                - name: dshm
                  mountPath: /dev/shm
          volumes:
            - name: shared
              persistentVolumeClaim:
                claimName: fsx-shared-pvc
            - name: dshm
              emptyDir:
                medium: Memory
                sizeLimit: "128Gi"
    Worker:
      replicas: 1
      restartPolicy: OnFailure
      template:
        metadata:
          labels:
            app: megatron-training
            role: worker
        spec:
          hostNetwork: true
          dnsPolicy: ClusterFirstWithHostNet
          containers:
            - name: megatron
              image: <your-ecr>/megatron-training:latest
              command:
                - torchrun
                - --nproc_per_node=8
                - --nnodes=2
                - --node_rank=1
                - --master_addr=$(MASTER_ADDR)
                - --master_port=$(MASTER_PORT)
                - /workspace/train_megatron.py
                # (same args as Master)
              env:
                # (same env vars as Master)
              resources:
                # (same resource requests as Master)
              volumeMounts:
                - name: shared
                  mountPath: /shared
                - name: dshm
                  mountPath: /dev/shm
          volumes:
            - name: shared
              persistentVolumeClaim:
                claimName: fsx-shared-pvc
            - name: dshm
              emptyDir:
                medium: Memory
                sizeLimit: "128Gi"
```

### Key Manifest Decisions Explained

**`hostNetwork: true`**: Required for EFA. EFA devices are bound to the host network namespace, and pod-level networking adds overhead. With `hostNetwork`, NCCL can directly access EFA devices.

**`dnsPolicy: ClusterFirstWithHostNet`**: When using `hostNetwork`, DNS resolution defaults to the host. This policy ensures Kubernetes service DNS still works.

**`/dev/shm` volume**: PyTorch uses shared memory for inter-process communication within a node. The default 64MB is insufficient -- mount a memory-backed emptyDir of 128GB.

**`vpc.amazonaws.com/efa: 4`**: Requests 4 EFA devices per pod. P4d has 4 EFA NICs; P5 has 32.

**`NCCL_DEBUG: WARN`**: Use `WARN` for production training. Switch to `INFO` only when debugging networking issues (it produces enormous log volume).

## Step 4: torchrun Parameters

`torchrun` is PyTorch's distributed launcher. It replaces the older `torch.distributed.launch` and provides elastic training support.

### Parameter Reference

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `--nproc_per_node` | `8` | Number of GPU processes per node |
| `--nnodes` | `2` | Total number of nodes |
| `--node_rank` | `0` (master), `1` (worker) | This node's rank in the cluster |
| `--master_addr` | Auto-injected by PyTorchJob | IP address of the master node |
| `--master_port` | Auto-injected by PyTorchJob | Port for rendezvous (default 29500) |
| `--rdzv_backend` | `c10d` (default) | Rendezvous backend |

### How PyTorchJob Injects Environment

The Training Operator automatically sets these environment variables in each pod:

```
MASTER_ADDR=megatron-cpt-techpulse-master-0    # DNS name of master pod
MASTER_PORT=23456                                # Rendezvous port
WORLD_SIZE=16                                    # Total processes (2 * 8)
RANK=0                                           # This process's global rank
```

You reference `$(MASTER_ADDR)` and `$(MASTER_PORT)` in the torchrun command, and Kubernetes substitutes the actual values at runtime.

## Step 5: NeMo-Based Training Script Alternative

Instead of raw Megatron-LM command-line arguments, you can use NeMo's Python API for more configurability. Here is the equivalent training configuration:

```python
# train_nemo.py
import torch
from nemo.collections.llm.gpt.model import GPTConfig, GPTModel
from nemo.collections.llm.gpt.data import PreTrainingDataModule
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning import Trainer, MegatronStrategy, MegatronMixedPrecision, NeMoLogger
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.lightning.pytorch.callbacks import ModelCheckpoint
from megatron.core.optimizer import OptimizerConfig
from nemo.lightning.pytorch.optim.lr_scheduler import CosineAnnealingScheduler

# Tokenizer
tokenizer = get_nmt_tokenizer(
    library="huggingface",
    model_name="/shared/tokenizer/llama-3.1-8b",
    use_fast=True,
)

# Data
data = PreTrainingDataModule(
    paths={
        "train": [1.0, "/shared/data/megatron/techpulse_text_document"],
        "validation": [1.0, "/shared/data/megatron/techpulse_text_document"],
    },
    seq_length=4096,
    micro_batch_size=2,
    global_batch_size=512,
    tokenizer=tokenizer,
    num_workers=8,
)

# Optimizer
optimizer_config = OptimizerConfig(
    optimizer="adam",
    lr=1e-5,
    adam_beta1=0.9,
    adam_beta2=0.95,
    weight_decay=0.1,
    clip_grad=1.0,
    use_distributed_optimizer=True,
    bf16=True,
)
scheduler = CosineAnnealingScheduler(
    max_steps=10000,
    warmup_steps=100,
    min_lr=1e-6,
)
optimizer = MegatronOptimizerModule(config=optimizer_config, lr_scheduler=scheduler)

# Strategy (parallelism)
strategy = MegatronStrategy(
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=1,
    sequence_parallel=True,     # Enable with TP > 1
    pipeline_dtype=torch.bfloat16,
    ckpt_async_save=False,
    gradient_as_bucket_view=True,
)

# Precision
precision = MegatronMixedPrecision(
    precision="bf16-mixed",
    params_dtype=torch.bfloat16,
    pipeline_dtype=torch.bfloat16,
    autocast_enabled=False,
    grad_reduce_in_fp32=True,
)

# Callbacks
callbacks = [
    ModelCheckpoint(
        every_n_train_steps=500,
        save_top_k=3,
        monitor="reduced_train_loss",
        dirpath="/shared/checkpoints/techpulse-cpt",
    ),
]

# Trainer
trainer = Trainer(
    devices=8,
    num_nodes=2,
    max_steps=10000,
    accelerator="gpu",
    strategy=strategy,
    callbacks=callbacks,
    log_every_n_steps=10,
    val_check_interval=500,
    limit_val_batches=20,
    plugins=precision,
)

# Model
model = GPTModel(config=GPTConfig(), optim=optimizer, tokenizer=tokenizer)

# Train
trainer.fit(model, data)
```

## Step 6: Deploy the Training Job

```bash
# Apply the PyTorchJob manifest
kubectl apply -f megatron-cpt-pytorchjob.yaml

# Verify pods are created
kubectl get pods -n megatron-workshop -l app=megatron-training

# Expected output:
# megatron-cpt-techpulse-master-0   1/1   Running   0   30s
# megatron-cpt-techpulse-worker-0   1/1   Running   0   30s
```

## Step 7: Verify EFA and NCCL Connectivity

Once pods are running, verify that NCCL is using EFA (not falling back to TCP sockets):

```bash
# Stream master pod logs
MASTER=$(kubectl get pods -n megatron-workshop \
  -l app=megatron-training,role=master \
  -o jsonpath='{.items[0].metadata.name}')

kubectl logs -f $MASTER -n megatron-workshop 2>&1 | head -100
```

**What to look for:**

Good (EFA active):
```
NET/OFI Initializing aws-ofi-nccl 1.17.2
NET/OFI Selected Provider is efa
NET/OFI Using RDMA for GPU Direct communication
```

Bad (EFA not working, fell back to TCP):
```
NET/Plugin: Could not find: libnccl-net-aws-ofi-nccl.so
NET/Socket: Using [0]eth0:10.1.1.203<0>
```

If you see the "bad" output, check:
1. `LD_LIBRARY_PATH` includes `/opt/aws-ofi-nccl/lib` and `/opt/amazon/efa/lib`
2. EFA device plugin is installed on the nodes
3. `vpc.amazonaws.com/efa` resource is requested in the pod spec

## Step 8: Verify GPU Utilization

```bash
# Exec into master pod and check GPU usage
kubectl exec -n megatron-workshop $MASTER -- nvidia-smi

# For continuous monitoring
kubectl exec -n megatron-workshop $MASTER -- nvidia-smi dmon -s u -d 5
```

Expected during active training:
```
# gpu   sm   mem   enc   dec
    0   87%  62%    0%    0%
    1   85%  60%    0%    0%
    2   88%  63%    0%    0%
    ...
```

SM utilization should be 70-95% during training. If below 50%, check for:
- Data loading bottleneck (increase `num_workers`)
- NCCL communication overhead (check parallelism configuration)
- Pipeline bubble (increase micro-batches if using PP)

## Common Launch Issues

| Problem | Symptom | Fix |
|---------|---------|-----|
| EFA not detected | Logs show `NET/Socket` instead of `NET/OFI` | Check `LD_LIBRARY_PATH`, EFA device plugin, pod resource requests |
| Rendezvous timeout | `NCCL error: timeout` during initialization | Increase `NCCL_TIMEOUT`, check `hostNetwork: true`, verify security groups allow all traffic between nodes |
| OOM at startup | `CUDA out of memory` | Reduce `micro_batch_size`, increase TP, enable distributed optimizer |
| Batch size error | `global batch size not divisible` | Recalculate: `global_bs % (micro_bs * DP) == 0` |
| Triton hang | All processes stuck during first forward pass | Verify `TRITON_CACHE_DIR=/tmp/...` (local, not shared FSx) |
| Bootstrap error | `no socket interface found` | Remove `NCCL_SOCKET_IFNAME` from environment |
| Slow training | Low GPU utilization, high communication wait | Check TP is within-node only, verify EFA is active (not TCP fallback) |

## Verification Checklist

- [ ] Container image built with aws-ofi-nccl and libfabric
- [ ] Image pushed to ECR and accessible from EKS
- [ ] PyTorchJob manifest has `hostNetwork: true` and EFA resource requests
- [ ] `/dev/shm` mounted with sufficient size (128Gi)
- [ ] NCCL logs confirm EFA provider is active (`NET/OFI Selected Provider is efa`)
- [ ] All GPUs visible and utilized (nvidia-smi shows activity)
- [ ] Training log shows loss decreasing over steps
- [ ] Checkpoints being written to FSx at `save-interval`

## Next Step

Proceed to [04 - Monitoring and Troubleshooting](../04-monitor/README.md) to learn how to track training health, debug common failures, and manage checkpoints.
