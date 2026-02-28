# Section 04: Production Deployment Patterns

## Objective

This section covers the operational patterns needed to run Ray + vLLM inference and GRPO training in a production environment. It addresses auto-scaling, health monitoring, EFA networking, and a comprehensive debugging guide drawn from real deployment experience.

**Estimated time: 10 minutes (reference material)**

---

## Ray Serve for Auto-Scaling Inference

### Autoscaling Configuration

Ray Serve can automatically adjust the number of model replicas based on request load. This is configured through the `autoscaling_config` in the deployment:

```python
@serve.deployment(
    name="techpulse-7b",
    autoscaling_config={
        "min_replicas": 1,          # Always keep at least 1 replica
        "max_replicas": 4,          # Scale up to 4 replicas
        "target_ongoing_requests": 8,  # Target 8 concurrent requests per replica
        "upscale_delay_s": 30,      # Wait 30s before adding replicas
        "downscale_delay_s": 300,   # Wait 5 min before removing replicas
        "metrics_interval_s": 10,   # Check metrics every 10s
    },
    ray_actor_options={"num_gpus": 2},
    max_ongoing_requests=16,        # Hard cap per replica
)
class VLLMDeployment:
    ...
```

### How Autoscaling Works

```
Low traffic (1 replica active):
  Requests: |||                        (3 concurrent)
  Replicas: [Replica 0: 3/8 target]
  Action:   No change (below target)

Medium traffic (scaling up):
  Requests: |||||||||||||               (13 concurrent)
  Replicas: [Replica 0: 8/8]  [Replica 1: 5/8]
  Action:   Upscaled from 1 -> 2 after 30s sustained load

High traffic (at max):
  Requests: |||||||||||||||||||||||||||||| (30+ concurrent)
  Replicas: [R0: 8/8] [R1: 8/8] [R2: 8/8] [R3: 8/8]
  Action:   At max_replicas=4, requests queue

Traffic drops (scaling down):
  Replicas: [R0: 2/8] [R1: 0/8] [R2: 0/8] [R3: 0/8]
  Action:   After 300s idle, scale down 4 -> 1
```

### KubeRay Autoscaling for Worker Nodes

The KubeRay RayCluster can also autoscale worker nodes to provide GPU resources as Ray Serve demands them:

```yaml
workerGroupSpecs:
  - groupName: gpu-workers
    replicas: 1            # Start with 1 worker
    minReplicas: 1         # Keep at least 1
    maxReplicas: 4         # Scale up to 4 workers
    rayStartParams:
      num-gpus: '8'
```

When Ray Serve needs more GPUs for additional replicas, KubeRay requests new worker pods. The EKS Cluster Autoscaler or Karpenter provisions the underlying GPU nodes.

**Scaling timeline:**
1. Ray Serve detects high load (10s metrics interval)
2. Ray Serve requests new replica (~30s upscale delay)
3. Ray finds insufficient GPUs, requests new Ray worker
4. KubeRay creates new worker pod
5. Cluster Autoscaler provisions new GPU node (3-10 min for GPU instances)
6. Worker pod starts, joins Ray cluster
7. Replica initializes (model loading: 1-3 min with FSx cache)

**Total cold-start time: 5-15 minutes.** This is dominated by GPU node provisioning. Pre-warming strategies include maintaining a minimum number of replicas and using Karpenter provisioners with warm pools.

---

## Health Checks and Liveness Probes

### vLLM Health Endpoint

vLLM exposes a `/health` endpoint that returns 200 when the engine is ready:

```yaml
containers:
  - name: vllm
    livenessProbe:
      httpGet:
        path: /health
        port: 8000
      initialDelaySeconds: 120    # Model loading takes time
      periodSeconds: 30
      timeoutSeconds: 10
      failureThreshold: 3
    readinessProbe:
      httpGet:
        path: /health
        port: 8000
      initialDelaySeconds: 60
      periodSeconds: 10
      timeoutSeconds: 5
      failureThreshold: 3
    startupProbe:
      httpGet:
        path: /health
        port: 8000
      initialDelaySeconds: 30
      periodSeconds: 10
      failureThreshold: 30       # Allow up to 5 min for startup
```

### Ray Serve Health

Ray Serve provides its own health checks through the head node:

```bash
# Check Ray Serve deployment status
kubectl exec $HEAD -- python3 -c "
from ray import serve
import ray
ray.init(address='auto')
status = serve.status()
for app_name, app_status in status.applications.items():
    print(f'App: {app_name}')
    print(f'  Status: {app_status.status}')
    for dep_name, dep_status in app_status.deployments.items():
        print(f'  Deployment: {dep_name}')
        print(f'    Status: {dep_status.status}')
        print(f'    Replicas: {dep_status.replica_states}')
ray.shutdown()
"
```

### GPU Health Monitoring

For production workloads, monitor GPU health proactively:

```bash
# Check for GPU errors (XID errors indicate hardware problems)
kubectl exec <pod> -- nvidia-smi --query-gpu=index,name,temperature.gpu,ecc.errors.uncorrected.volatile.total --format=csv

# Check for ECC errors (memory corruption)
kubectl exec <pod> -- nvidia-smi --query-gpu=index,ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total --format=csv

# Monitor GPU utilization over time
kubectl exec <pod> -- nvidia-smi dmon -s u -d 5
# Columns: gpu, sm%, mem%, enc%, dec%
```

---

## EFA Networking for Multi-Node Workloads

### What Is EFA?

Elastic Fabric Adapter (EFA) is AWS's high-performance network interface for GPU instances. It provides RDMA-like capabilities through the SRD (Scalable Reliable Datagram) protocol, enabling GPU-to-GPU communication without CPU involvement.

```
Without EFA:
  GPU -> PCIe -> CPU -> TCP/IP -> NIC -> Network -> NIC -> CPU -> PCIe -> GPU
  Latency: ~100-500 us, Bandwidth: ~12.5 GB/s (100 Gbps)

With EFA (SRD + GPUDirect RDMA):
  GPU -> PCIe -> EFA NIC -> Network -> EFA NIC -> PCIe -> GPU
  Latency: ~5-10 us, Bandwidth: ~25-100 GB/s (200-800 Gbps)
```

### EFA Configuration for NCCL

NCCL uses EFA through the aws-ofi-nccl plugin (Open Fabrics Interface bridge):

```yaml
env:
  # Tell NCCL to use the OFI plugin
  - name: NCCL_NET_PLUGIN
    value: "ofi"

  # Tell libfabric to use the EFA provider
  - name: FI_PROVIDER
    value: "efa"

  # Enable GPUDirect RDMA (GPU memory registered directly with EFA)
  - name: FI_EFA_USE_DEVICE_RDMA
    value: "1"

  # Required for forking processes (vLLM, Ray)
  - name: FI_EFA_FORK_SAFE
    value: "1"
  - name: RDMAV_FORK_SAFE
    value: "1"

  # GDRCopy for small message acceleration
  - name: FI_HMEM_CUDA_USE_GDRCOPY
    value: "1"

  # Network interface selection (exclude virtual interfaces)
  - name: NCCL_SOCKET_IFNAME
    value: "^lo,docker,veth,eni"
```

### Requesting EFA Resources in Kubernetes

EFA devices are exposed as extended resources by the EFA device plugin:

```yaml
resources:
  limits:
    nvidia.com/gpu: 8
    vpc.amazonaws.com/efa: 4     # Number of EFA interfaces
  requests:
    nvidia.com/gpu: 8
    vpc.amazonaws.com/efa: 4
```

The number of EFA interfaces varies by instance type:

| Instance Type | EFA Interfaces | Bandwidth |
|---------------|---------------|-----------|
| p4d.24xlarge | 4 | 400 Gbps |
| p5.48xlarge | 32 | 3,200 Gbps |
| p5en.48xlarge | 32 | 3,200 Gbps |

### Verifying EFA Is Working

```bash
# Check EFA devices are visible
kubectl exec <pod> -- ls /dev/infiniband/
# Expected: uverbs0 uverbs1 uverbs2 uverbs3

# Check libfabric sees EFA
kubectl exec <pod> -- fi_info -p efa -t FI_EP_RDM 2>&1 | head -20

# Check NCCL is using EFA (in training logs)
kubectl logs <pod> | grep "NET/OFI"
# Expected: NET/OFI Selected provider is efa
```

### Critical EFA Gotchas

These issues are documented from real production deployments and can cause silent failures:

**1. Security Group Self-Referencing Egress Rule**

EFA SRD traffic requires the security group to have an egress rule that references itself. A `0.0.0.0/0` outbound rule is NOT sufficient because Nitro evaluates SRD egress by security group membership, not by IP/CIDR.

**Symptom:** NCCL hangs at initialization. `tx_pkts` increments but `rx_pkts` stays at 0.

**Fix:**
```bash
# Add self-referencing egress rule
aws ec2 authorize-security-group-egress \
  --group-id sg-xxxx \
  --protocol -1 \
  --source-group sg-xxxx
```

**2. Primary VPC CIDR Only**

EFA SRD does NOT work on secondary VPC CIDRs. The Nitro fabric silently drops SRD packets from secondary CIDRs.

**Symptom:** Same as above -- `tx_pkts > 0`, `rx_pkts = 0`.

**Verification:**
```bash
# Check which CIDR block the node is using
kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.addresses[?(@.type=="InternalIP")].address}{"\n"}{end}'

# If IPs are in a secondary CIDR (e.g., 100.64.x.x), EFA SRD will not work
```

**3. hostNetwork Required**

For EFA to function properly with NCCL, pods must use `hostNetwork: true`. This gives the pod direct access to the host's EFA devices and IP address.

---

## Common Issues and Debugging Guide

### Systematic Debugging Approach

When a deployment fails, follow this sequence:

```
1. Pod status?     kubectl get pods -l app=<label>
   |
   +-- Pending?    Check GPU/EFA resources, node taints, anti-affinity
   +-- CrashLoop?  Check container logs for startup errors
   +-- Running?    Continue to step 2
   |
2. Ray cluster?    kubectl exec $HEAD -- ray status
   |
   +-- Missing nodes?  Check master IP discovery, network connectivity
   +-- All nodes?      Continue to step 3
   |
3. Model loaded?   kubectl logs <pod> | grep "Model loaded\|KV cache"
   |
   +-- Download error?  Check HF_HOME, network access, disk space
   +-- OOM?             Reduce gpu_memory_utilization or increase TP
   +-- Loaded?          Continue to step 4
   |
4. Inference works? curl http://localhost:8000/v1/models
   |
   +-- Connection refused?  Serve not started, port-forward wrong
   +-- 500 error?           Check vLLM logs for engine errors
   +-- 200 OK?              Deployment healthy
```

### Issue: NCCL Timeouts in Multi-Node Training

**Symptom:**
```
[Rank 0] NCCL INFO Bootstrap : Using eth0:10.1.2.3<0>
[Rank 0] NCCL INFO NET/OFI Selected Provider is efa
... (hangs for minutes) ...
[Rank 0] NCCL WARN Timeout waiting for connect from peer
```

**Debugging steps:**

```bash
# 1. Check EFA hardware counters (most important diagnostic)
kubectl exec <pod> -- bash -c '
for dev in /sys/class/infiniband/rdmap*/ports/1/hw_counters; do
  echo "=== $(dirname $(dirname $dev)) ==="
  echo "tx_pkts: $(cat $dev/tx_pkts)"
  echo "rx_pkts: $(cat $dev/rx_pkts)"
done
'
# If tx_pkts > 0 but rx_pkts = 0: network issue (SG, CIDR, routing)
# If both are 0: EFA not being used (check FI_PROVIDER, device plugin)

# 2. Test basic connectivity between nodes
kubectl exec <rank0-pod> -- ping -c 3 <rank1-ip>

# 3. Test EFA specifically
kubectl exec <pod> -- fi_pingpong -p efa -e rdm <peer-ip>
```

### Issue: OOM (Out of Memory)

**Symptom:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory.
Tried to allocate X GiB. GPU 0 has Y GiB total capacity.
```

**Memory budget for a 7B model on A100 40GB:**

```
Model weights (BF16):     ~14 GB
KV cache (vLLM):          ~20 GB (at gpu_memory_utilization=0.85)
Training activations:     ~4 GB  (with activation checkpointing)
Optimizer state (AdamW):  ~2 GB  (with LoRA, much less than full params)
Overhead (CUDA, NCCL):    ~2 GB
                          ------
Total:                    ~42 GB  -> Does NOT fit in 40 GB!

Solutions:
  Option A: Reduce gpu_memory_utilization to 0.65 (KV cache: ~12 GB, total: ~34 GB)
  Option B: Enable LoRA (optimizer state: 0.2 GB instead of 28 GB)
  Option C: Use TP=2 (splits model across 2 GPUs)
  Option D: Use H100 80GB (2x the memory)
```

### Issue: Stale Master IP from Previous Run

**Symptom:** Workers in a new training job try to connect to an IP from a previous run that no longer exists.

**Root cause:** The master IP file on FSx (`/shared/.grpo-master-ip`) persists across runs. If the previous run's rank 0 pod had a different IP, workers will try to connect to a dead address.

**Prevention:** The manifests in Section 01 and 03 handle this by having rank 0 delete and rewrite the file. But if rank 0 crashes before writing, the stale file persists.

**Recovery:**
```bash
# Option 1: Delete the file from any pod with FSx mounted
kubectl exec <any-pod-with-fsx> -- rm -f /shared/.grpo-master-ip

# Option 2: Use a unique file per job
# In the manifest args, use:
MASTER_FILE="/shared/.master-ip-${JOB_NAME}"
```

### Issue: Ray Workers Deregistered During Training

**Symptom:** Training suddenly fails with "worker died" or "actor died" errors.

**Debugging:**
```bash
# Check if the worker pod is still running
kubectl get pods -l component=worker

# Check worker pod events
kubectl describe pod <worker-pod> | grep -A 10 Events

# Common causes:
# - OOM kill: Check `kubectl describe pod` for OOMKilled
# - Node preemption: Check node events
# - Health check failure: Check liveness probe configuration
```

### Issue: vLLM Slow After Checkpoint Load

**Symptom:** After GRPO loads a new checkpoint, the next generation phase is very slow.

**Cause:** vLLM needs to rebuild CUDA graphs after model weights change. This is a one-time cost per checkpoint load.

**Mitigation:** Set `enforce_eager=True` in the vLLM config to avoid CUDA graph compilation. This slightly reduces steady-state performance but eliminates the recompilation delay.

---

## Production Checklist

Before deploying to production, verify each item:

### Infrastructure
- [ ] GPU nodes have EFA device plugin installed
- [ ] Security groups have self-referencing egress rule for SRD
- [ ] Nodes are on the primary VPC CIDR
- [ ] FSx Lustre PVC is provisioned and accessible from all nodes
- [ ] `hostNetwork: true` is set for multi-node workloads
- [ ] `/dev/shm` is mounted as Memory-backed tmpfs (128Gi+)

### Model Serving
- [ ] `gpu_memory_utilization` is tuned for your model and concurrency target
- [ ] Liveness, readiness, and startup probes are configured
- [ ] Model weights are pre-cached on FSx (no cold-start download)
- [ ] Autoscaling min/max replicas match your traffic pattern
- [ ] `distributed_executor_backend="mp"` for TP>1 with Ray Serve

### GRPO Training
- [ ] NCCL timeouts set to 7200s (prevents false failures during initialization)
- [ ] Master IP file path is unique per job or cleaned between runs
- [ ] Checkpoint directory is on FSx (not local disk)
- [ ] `gpu_memory_utilization` for vLLM is set lower (0.5-0.7) for colocated mode
- [ ] LoRA enabled for memory-efficient training on smaller GPU instances

### Monitoring
- [ ] Ray dashboard accessible via port-forward or ingress
- [ ] GPU utilization and memory monitored (nvidia-smi, DCGM)
- [ ] Training metrics logged (W&B, TensorBoard, or MLflow)
- [ ] Alerts configured for pod restarts, OOM events, and GPU errors

---

## Further Reading

- [Ray Serve Documentation](https://docs.ray.io/en/latest/serve/) -- Autoscaling, deployment patterns, and configuration
- [vLLM Documentation](https://docs.vllm.ai/) -- Engine configuration, PagedAttention, and performance tuning
- [NeMo RL GRPO Guide](https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/grpo.md) -- GRPO algorithm, datasets, and loss functions
- [AWS EFA User Guide](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html) -- EFA setup, troubleshooting, and supported instances
- [KubeRay Documentation](https://ray-project.github.io/kuberay/) -- RayCluster CRD, RayJob, and operator configuration
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/) -- Environment variables, debugging, and performance tuning

---

Back to: [Workshop Overview](../)
