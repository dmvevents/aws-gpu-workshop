# Section 2: NIXL on AWS EFA

**Estimated time:** 15 minutes

## EFA and the SRD Protocol

AWS Elastic Fabric Adapter (EFA) is a custom network interface available on
GPU instances (P4d, P5, P5e, Trn1, etc.) that provides high-bandwidth,
low-latency networking using the Scalable Reliable Datagram (SRD) protocol.
SRD is a proprietary transport implemented in the Nitro hardware that
provides:

- **Multi-path routing**: Packets for a single flow are sprayed across
  multiple network paths, achieving higher aggregate bandwidth than any
  single path.
- **Reliability without head-of-line blocking**: Lost packets are
  retransmitted without stalling other in-flight packets (unlike TCP).
- **No ordering guarantees**: SRD does not guarantee message delivery order.
  Applications must handle reordering at a higher layer.

EFA exposes itself to userspace through the libfabric API (OpenFabrics
Interfaces), using the `efa` provider. NIXL's Libfabric backend plugin
speaks directly to this provider for GPU-to-GPU RDMA transfers.

```
+------------------------------------------------------------------+
|  NIXL Libfabric Backend Plugin                                   |
+------------------------------------------------------------------+
|  libfabric API  (fi_write, fi_read, fi_cq_read, fi_mr_reg)      |
+------------------------------------------------------------------+
|  EFA Provider  (libfabric efa provider)                          |
+------------------------------------------------------------------+
|  EFA Device Driver  (/dev/infiniband/uverbs0, etc.)              |
+------------------------------------------------------------------+
|  AWS Nitro Hardware  (SRD protocol, multi-path, in-network)      |
+------------------------------------------------------------------+
```

### EFA Characteristics You Must Know

These are verified hardware behaviors, not theoretical:

| Property | Value | Implication |
|----------|-------|-------------|
| **Endpoint type** | FI_EP_RDM (Reliable Datagram) | Connection-less, message-based |
| **Atomics** | NOT supported (no FI_ATOMIC) | Cannot use fi_atomicmsg; will fail |
| **Message ordering** | None (msg_order=[]) | Application must handle reordering |
| **FI_FENCE** | Unreliable (partial effect) | Do not depend on fence for ordering |
| **CQ completion** | LOCAL (sender-side) | Completion means sent, not received |
| **Same-node loopback** | NOT supported | Packets to self are silently dropped; libfabric uses SHM for same-node |
| **Secondary VPC CIDR** | NOT supported for SRD | Nitro fabric silently drops SRD packets on secondary CIDRs |
| **Security groups** | Self-referencing EGRESS rule required | 0.0.0.0/0 outbound is NOT sufficient for SRD |

The last three points are particularly dangerous because they cause silent
failures: tx_pkts increment but rx_pkts stay at zero, with no error messages.

### EFA on P4d vs P5

| Instance | EFA Devices | Bandwidth per EFA | Total Network | GPUs |
|----------|-------------|-------------------|---------------|------|
| P4d.24xlarge | 4x EFA | 100 Gbps each | 400 Gbps | 8x A100 40GB |
| P5.48xlarge | 32x EFA | 100 Gbps each | 3,200 Gbps | 8x H100 80GB |

On P5 instances, the 32 EFA devices are organized as 4 EFAs per GPU,
enabling true multi-rail bandwidth scaling.

## The Libfabric Backend Plugin

NIXL's Libfabric plugin (`src/plugins/libfabric/`) is purpose-built for EFA.
It was co-developed by NVIDIA and Amazon, as evidenced by the dual copyright
in the source headers. Its key architectural components are:

```
+------------------------------------------------------------------+
|                  nixlLibfabricEngine                              |
|                                                                  |
|  +----------------------------+  +-----------------------------+ |
|  |  nixlLibfabricRailManager  |  |  nixlLibfabricTopology      | |
|  |                            |  |                             | |
|  |  Rail 0: rdmap0s28-efa0    |  |  GPU 0 --> EFA {0,1,2,3}   | |
|  |  Rail 1: rdmap16s28-efa1   |  |  GPU 1 --> EFA {4,5,6,7}   | |
|  |  Rail 2: rdmap32s28-efa2   |  |  GPU 2 --> EFA {8,9,10,11} | |
|  |  ...                       |  |  ...                        | |
|  |  Rail N: rdmapXsY-efaN     |  |  (hwloc-based mapping)     | |
|  +----------------------------+  +-----------------------------+ |
|                                                                  |
|  +----------------------------+  +-----------------------------+ |
|  |  Connection Manager        |  |  Request Pool               | |
|  |                            |  |                             | |
|  |  Per-agent connections     |  |  Pre-allocated request      | |
|  |  Multi-rail address maps   |  |  handles with atomic        | |
|  |  State tracking & reconnect|  |  completion counters        | |
|  +----------------------------+  +-----------------------------+ |
+------------------------------------------------------------------+
```

### Multi-Rail RDMA

The Libfabric plugin automatically discovers all available EFA devices and
creates one "rail" per device. When a transfer is issued, the plugin
distributes the work across multiple rails for maximum bandwidth:

```
Transfer: 4 GB KV cache from Prefill GPU 0 --> Decode GPU 0

Prefill Node                           Decode Node
+----------+                           +----------+
|  GPU 0   |                           |  GPU 0   |
|  (HBM)   |                           |  (HBM)   |
+----+-----+                           +----+-----+
     |                                      |
     |  GDR (GPUDirect RDMA)                |  GDR
     |                                      |
+----v-----+    SRD multi-path      +-------v--+
| EFA 0    |========================| EFA 0    |  ~1 GB via rail 0
| EFA 1    |========================| EFA 1    |  ~1 GB via rail 1
| EFA 2    |========================| EFA 2    |  ~1 GB via rail 2
| EFA 3    |========================| EFA 3    |  ~1 GB via rail 3
+----------+                        +----------+

Total bandwidth: 4x single-rail = approaching 400 Gbps on P4d
```

### Topology-Aware GPU-to-EFA Mapping

The plugin uses `hwloc` (Hardware Locality) to discover the physical
topology of the machine and map each GPU to its closest EFA devices:

```
P5.48xlarge Topology (simplified):

NUMA Node 0                          NUMA Node 1
+-----------+  +-----------+         +-----------+  +-----------+
| GPU 0     |  | GPU 1     |         | GPU 4     |  | GPU 5     |
| EFA 0-3   |  | EFA 4-7   |         | EFA 16-19 |  | EFA 20-23 |
+-----------+  +-----------+         +-----------+  +-----------+

NUMA Node 2                          NUMA Node 3
+-----------+  +-----------+         +-----------+  +-----------+
| GPU 2     |  | GPU 3     |         | GPU 6     |  | GPU 7     |
| EFA 8-11  |  | EFA 12-15 |         | EFA 24-27 |  | EFA 28-31 |
+-----------+  +-----------+         +-----------+  +-----------+
```

When GPU 0 initiates a transfer, the plugin selects EFA devices 0-3 (those
on the same PCIe bus / NUMA node) rather than distant EFA devices. This
minimizes PCIe hops and maximizes GPU Direct RDMA performance.

The topology mapping is stored in the `nixlLibfabricTopology` class and used
by `nixlLibfabricRailManager` for rail selection.

### GPU Direct RDMA (GDR)

On EFA-equipped instances with NVIDIA GPUs, the Libfabric plugin uses GPU
Direct RDMA to transfer data directly between GPU HBM and the network
adapter without staging through host memory:

```
Without GDR:                         With GDR:
GPU HBM --> Host DRAM --> EFA        GPU HBM ---------> EFA
         (PCIe copy)    (RDMA)                (direct DMA)

2 copies, higher latency             1 copy, lower latency
```

GDR requires:
- The `nvidia-peermem` kernel module loaded
- CUDA-aware libfabric build (with `--with-cuda`)
- The EFA device and GPU on the same PCIe root complex (for best performance)

The plugin handles CUDA context management internally through the
`nixlLibfabricCudaCtx` class, which tracks and sets the correct CUDA context
when the progress thread processes completions on different GPU devices.

## Memory Registration on EFA

When you register a GPU buffer with the Libfabric backend, the following
happens internally:

1. **Per-rail MR registration**: The buffer is registered with `fi_mr_reg()`
   on each selected rail (EFA device), producing a memory region handle and
   a remote access key per rail.

2. **Rail selection**: Based on the GPU device ID and the topology map, the
   plugin selects which rails (EFA devices) are topologically close to the
   GPU. Only those rails are used for transfers involving this buffer.

3. **Metadata generation**: The registration produces a
   `nixlLibfabricPrivateMetadata` object containing:
   - Buffer address and length
   - Device ID
   - Per-rail memory region handles (`fid_mr *`)
   - Per-rail remote access keys (`uint64_t`)
   - Per-rail source endpoint names
   - Selected rail indices (topology-based)

This metadata is later serialized into the agent's public metadata for
exchange with remote agents.

```
register_memory(gpu_tensor)
         |
         v
    For each selected rail (based on GPU topology):
         |
         +--> fi_mr_reg(domain, buf, len, FI_REMOTE_READ | FI_REMOTE_WRITE, ...)
         |         |
         |         +--> mr_handle, remote_key
         |
         +--> Store in nixlLibfabricPrivateMetadata
         |
         v
    Return registration handle
```

## Security Group Requirements for EFA SRD

EFA's SRD protocol has specific security group requirements that differ from
standard TCP/UDP networking. The Nitro hardware evaluates SRD egress rules
by security group membership, not by IP/CIDR:

```
REQUIRED Security Group Rules:

Inbound:
  +------------------------------------------------------------+
  | Type          | Protocol | Port | Source                    |
  |---------------|----------|------|---------------------------|
  | All traffic   | All      | All  | Self (sg-xxxx)            |
  | SSH           | TCP      | 22   | Your IP / bastion SG      |
  +------------------------------------------------------------+

Outbound:
  +------------------------------------------------------------+
  | Type          | Protocol | Port | Destination               |
  |---------------|----------|------|---------------------------|
  | All traffic   | All      | All  | Self (sg-xxxx)            |
  +------------------------------------------------------------+

CRITICAL: The outbound self-referencing rule is REQUIRED for SRD.
A rule with destination 0.0.0.0/0 is NOT sufficient.
Missing this rule causes the same silent failure as secondary
CIDR issues: tx_pkts > 0, rx_pkts = 0.
```

Verify your security group:

```bash
# Get the cluster security group
CLUSTER_SG=$(aws eks describe-cluster --name my-cluster \
  --query 'cluster.resourcesVpcConfig.clusterSecurityGroupId' \
  --output text)

# Check self-referencing inbound rule
aws ec2 describe-security-groups --group-ids $CLUSTER_SG \
  --query 'SecurityGroups[0].IpPermissions[].UserIdGroupPairs[?GroupId==`'$CLUSTER_SG'`].GroupId'

# Check self-referencing outbound rule (CRITICAL for SRD)
aws ec2 describe-security-groups --group-ids $CLUSTER_SG \
  --query 'SecurityGroups[0].IpPermissionsEgress[].UserIdGroupPairs[?GroupId==`'$CLUSTER_SG'`].GroupId'
```

## VPC CIDR Considerations

SRD only works on the primary VPC CIDR. If your EKS nodes are placed on a
secondary CIDR (common when the primary CIDR is exhausted), SRD packets will
be silently dropped by the Nitro fabric:

```
Primary CIDR: 10.0.0.0/16     --> SRD works
Secondary CIDR: 100.64.0.0/16 --> SRD silently dropped

Symptom: tx_pkts incrementing, rx_pkts = 0
No error messages in any log.
```

Always verify SRD connectivity by checking hardware counters:

```bash
# On the sending node:
cat /sys/class/infiniband/rdmap*/ports/1/hw_counters/tx_pkts

# On the receiving node:
cat /sys/class/infiniband/rdmap*/ports/1/hw_counters/rx_pkts

# If tx_pkts > 0 but rx_pkts = 0, check:
# 1. Security group self-referencing egress rule
# 2. VPC CIDR (primary vs secondary)
# 3. EFA device assignment to pods (device plugin)
```

## Running NIXL with EFA on EKS

### EFA Device Plugin

On EKS, the AWS EFA Kubernetes device plugin exposes EFA interfaces as
allocatable resources:

```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: nixl-worker
    resources:
      requests:
        vpc.amazonaws.com/efa: 4    # Request 4 EFA devices
      limits:
        vpc.amazonaws.com/efa: 4
```

### Environment Variables

The NIXL Libfabric plugin needs to find EFA devices. Key environment
variables:

```bash
# Libfabric provider selection
export FI_PROVIDER=efa        # Use EFA provider

# Debug logging (for troubleshooting)
export FI_LOG_LEVEL=warn      # Or: info, debug, trace
export FI_LOG_PROV=efa

# NIXL plugin directory
export NIXL_PLUGIN_DIR=/usr/local/lib/x86_64-linux-gnu/plugins

# NIXL log level
export NIXL_LOG_LEVEL=info    # Or: debug
```

### Using the Libfabric Backend

To use EFA through NIXL's Libfabric backend:

```python
from nixl._api import nixl_agent, nixl_agent_config

# Initialize agent with Libfabric backend
config = nixl_agent_config(
    enable_prog_thread=True,
    enable_listen_thread=True,
    listen_port=5555,
    backends=["Libfabric"]      # Use Libfabric instead of UCX
)

agent = nixl_agent("efa-worker-0", config)
```

Or with vLLM's KV transfer configuration:

```python
kv_transfer_config = {
    "kv_connector": "NixlConnector",
    "kv_role": "kv_producer",           # or "kv_consumer"
    "kv_buffer_device": "cuda",
    "kv_connector_extra_config": {
        "backends": ["Libfabric"]       # Use EFA via Libfabric
    }
}
```

## Benchmarking with NIXLBench

NIXLBench supports EFA through the Libfabric backend. Use it to validate
your EFA setup and measure transfer performance:

```bash
# On Node 1 (target):
nixlbench \
    --etcd_endpoints http://etcd-server:2379 \
    --backend Libfabric \
    --initiator_seg_type VRAM \
    --target_seg_type VRAM \
    --start_block_size 65536 \
    --max_block_size 67108864 \
    --num_iter 1000

# On Node 2 (initiator) -- start within 60 seconds:
nixlbench \
    --etcd_endpoints http://etcd-server:2379 \
    --backend Libfabric \
    --initiator_seg_type VRAM \
    --target_seg_type VRAM \
    --start_block_size 65536 \
    --max_block_size 67108864 \
    --num_iter 1000
```

Expected bandwidth on P4d.24xlarge with 4 EFA devices: approximately
7-12 GB/s for large (>1 MB) VRAM-to-VRAM transfers.

## Debugging EFA Issues

When things go wrong, follow this checklist in order:

**1. Check hardware counters first (always start here):**

```bash
# Are packets being sent?
cat /sys/class/infiniband/rdmap0s28-efa0/ports/1/hw_counters/tx_pkts

# Are packets being received?
cat /sys/class/infiniband/rdmap0s28-efa0/ports/1/hw_counters/rx_pkts

# tx > 0, rx = 0 means: SG rule, CIDR, or routing issue
# tx = 0 means: device not configured, or wrong device selected
```

**2. Verify EFA devices are visible:**

```bash
fi_info -p efa                    # List EFA fabric interfaces
fi_info -p efa -t FI_EP_RDM -v   # Detailed RDM endpoint info
ibv_devices                       # List RDMA devices
```

**3. Check libfabric provider logs:**

```bash
export FI_LOG_LEVEL=debug
export FI_LOG_PROV=efa
# Run your NIXL application and examine stderr
```

**4. Verify GPU Direct RDMA:**

```bash
lsmod | grep nvidia_peermem       # Module must be loaded
nvidia-smi topo -m                # Check GPU-to-NIC topology
                                  # PIX/PXB = same PCIe switch (good)
                                  # SYS = cross-socket (slower)
```

## Summary

| Topic | Key Point |
|-------|-----------|
| EFA Protocol | SRD: multi-path, reliable, unordered, local CQ completion |
| NIXL Backend | Libfabric plugin with multi-rail, topology-aware, GDR support |
| Multi-Rail | Automatic striping across all topologically-close EFA devices |
| Security Groups | Self-referencing EGRESS rule required (0.0.0.0/0 is insufficient) |
| VPC CIDR | SRD only works on primary CIDR; secondary CIDR silently fails |
| Debugging | Always check hw_counters (tx_pkts/rx_pkts) first |
| Benchmarking | Use NIXLBench with `--backend Libfabric` to validate setup |

---

Previous: [Section 1: NIXL Architecture](../01-architecture/README.md) |
Next: [Section 3: Disaggregated Inference](../03-disaggregated-inference/README.md)
