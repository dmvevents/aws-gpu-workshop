# Section 1: NIXL Architecture

**Estimated time:** 15 minutes

## What is NIXL and Why It Exists

Traditional GPU communication libraries like NCCL were designed for training
workloads where every GPU participates in collective operations (all-reduce,
all-gather) at every layer of the forward and backward pass. These collectives
are symmetric, frequent, and latency-critical within a single computation
graph.

Inference workloads are fundamentally different:

- **Asymmetric communication**: A prefill node generates a KV cache and sends
  it to a decode node. There is no all-reduce; it is strictly point-to-point.
- **Heterogeneous memory tiers**: KV caches may originate in GPU HBM, get
  staged in host DRAM, cached on local NVMe, or stored in S3 for long-context
  reuse.
- **Dynamic scaling**: Inference systems add and remove nodes based on demand.
  The set of communicating peers changes at runtime.
- **Large bulk transfers**: A single KV cache transfer for a 70B-parameter
  model with 128K context can be several gigabytes, not the small per-layer
  synchronizations typical of training.

NIXL was created by NVIDIA to address exactly these patterns. It provides a
unified abstraction for point-to-point data movement across all memory and
storage types that matter for inference, with automatic backend selection,
one-sided RDMA semantics, and first-class support for dynamic agent
discovery.

```
Training Communication (NCCL)        Inference Communication (NIXL)
+------+  all-reduce  +------+      +--------+   KV cache   +--------+
| GPU0 |<------------>| GPU1 |      |Prefill |------------->| Decode |
| GPU2 |<------------>| GPU3 |      | Node   |   (RDMA)     | Node   |
+------+  (symmetric) +------+      +--------+              +--------+
  Every layer, all GPUs                One-time, point-to-point,
  Low latency (<1ms)                   High bandwidth, large payload
  Fixed topology                       Dynamic scaling
```

## NIXL vs NCCL vs NVSHMEM

| Characteristic | NCCL | NVSHMEM | NIXL |
|----------------|------|---------|------|
| **Primary use case** | Training collectives | GPU-initiated RDMA | Inference data transfer |
| **Communication pattern** | Collective (all-reduce, all-gather) | Put/Get from GPU kernels | Point-to-point (read/write) |
| **Who initiates** | Host CPU orchestrates | GPU threads directly | Host CPU (Transfer Agent) |
| **Memory types** | GPU only | GPU only | GPU, CPU, NVMe, S3, Block |
| **Topology** | Fixed process group | Fixed PE set | Dynamic agents (add/remove) |
| **Backend selection** | Automatic (NVLink, IB) | Manual | Automatic per-transfer |
| **Notification** | Implicit (collective) | Fence/barrier | Explicit per-transfer |
| **Metadata exchange** | Out-of-band (MPI/sockets) | Bootstrap | ETCD, side-channel, or listener |

**When to use each:**

- **NCCL**: Tensor-parallel and data-parallel collectives within a training job
  or within a single inference instance that uses TP across local GPUs.
- **NVSHMEM**: When GPU threads need to directly read/write remote GPU memory
  without host CPU involvement (e.g., custom CUDA kernels for communication).
- **NIXL**: Transferring KV caches between disaggregated inference instances,
  moving data between GPU and storage tiers, any point-to-point inference
  data movement.

The recommended production pattern is to use NCCL for intra-instance tensor
parallelism and NIXL for inter-instance KV cache transfer.

## Core Architecture: The Transfer Agent

Every NIXL-using process creates exactly one `nixl_agent`. The agent is the
central object that manages all communication state:

```
+-----------------------------------------------------------------+
|                        nixl_agent                               |
|                                                                 |
|  +-----------------------+                                      |
|  |   Backend Engines     |    UCX (IB, RoCE, TCP)               |
|  |   (Plugins)           |    Libfabric (EFA)                   |
|  |                       |    GDS (GPUDirect Storage)           |
|  |                       |    POSIX, OBJ (S3), GUSLI, etc.     |
|  +-----------------------+                                      |
|                                                                 |
|  +-----------------------+                                      |
|  |   Memory Registry     |    Registered VRAM regions           |
|  |                       |    Registered DRAM regions           |
|  |                       |    Registered FILE/OBJ regions       |
|  +-----------------------+                                      |
|                                                                 |
|  +-----------------------+                                      |
|  |   Remote Agent Cache  |    Cached metadata from peer agents  |
|  |                       |    Connection state per backend      |
|  +-----------------------+                                      |
|                                                                 |
|  +-----------------------+                                      |
|  |   Notification Queue  |    Incoming notifications from peers |
|  +-----------------------+                                      |
+-----------------------------------------------------------------+
```

### Agent Configuration

```python
from nixl._api import nixl_agent, nixl_agent_config

config = nixl_agent_config(
    enable_prog_thread=True,    # Background thread for async progress
    enable_listen_thread=True,  # Listen for metadata from remote agents
    listen_port=5555,           # Port for metadata listener
    backends=["UCX"]            # Which backend plugins to initialize
)

agent = nixl_agent("worker-0", config)
```

Key configuration parameters:

- **enable_prog_thread**: Enables a background thread that drives
  asynchronous transfer completion. Without it, the application must call
  status-check functions to make progress.
- **enable_listen_thread**: Starts a TCP listener so remote agents can push
  their metadata directly (alternative to using ETCD).
- **backends**: List of backend plugin names to instantiate. Common choices
  are `"UCX"` for InfiniBand/RoCE, `"Libfabric"` for EFA, and `"GDS"` for
  GPUDirect Storage.
- **num_threads**: Number of threads for multi-threaded backends (UCX, GDS_MT).

## Memory Sections and Registration

Before any data can be transferred, the memory regions involved must be
registered with NIXL. Registration serves two purposes:

1. **Pin memory** so the network hardware can DMA directly to/from it
   (avoiding page faults during transfer).
2. **Generate metadata** (remote keys, addresses) that remote agents need
   to perform one-sided RDMA reads or writes.

### Memory Types

| Type | Constant | Description | Example Backend |
|------|----------|-------------|-----------------|
| VRAM | `VRAM_SEG` | GPU HBM (High Bandwidth Memory) | UCX, Libfabric |
| DRAM | `DRAM_SEG` | Host CPU memory | UCX, Libfabric |
| FILE | `FILE_SEG` | File on local/remote filesystem | GDS, POSIX |
| BLOCK | `BLK_SEG` | Block device (NVMe) | GUSLI |
| OBJ | `OBJ_SEG` | Object storage (S3, Azure Blob) | OBJ, AZURE_BLOB |

### Registration with Tensors

The simplest way to register memory is to pass a PyTorch tensor:

```python
import torch

# Allocate GPU memory
kv_cache = torch.zeros((num_layers, 2, num_heads, seq_len, head_dim),
                       dtype=torch.float16, device="cuda:0")

# Register the entire allocation as a single contiguous region.
# Fewer, larger registrations are better -- reduces kernel calls
# and internal lookups.
reg_handle = agent.register_memory(kv_cache)
```

### Registration with Raw Descriptors

For non-tensor memory, you can pass raw (address, length, device_id) tuples:

```python
# Register a region of host memory
import ctypes
buf = ctypes.create_string_buffer(1024 * 1024)  # 1 MB
reg_handle = agent.register_memory(
    [(ctypes.addressof(buf), len(buf), 0)],
    mem_type="DRAM"
)
```

### Descriptor Lists

After registration, you create descriptor lists that describe which parts of
registered memory participate in a transfer. Each descriptor is an
(address, length, device_id) triple:

```python
# Build transfer descriptors from individual KV cache layers
layer_tensors = [kv_cache[i] for i in range(num_layers)]
xfer_descs = agent.get_xfer_descs(layer_tensors)
```

Descriptors can also be constructed from NumPy Nx3 arrays for bulk descriptor
creation with lower overhead:

```python
import numpy as np

descs = np.array([
    [tensor.data_ptr(), tensor.nelement() * tensor.element_size(), gpu_id],
    # ... more descriptors
], dtype=np.uint64)

xfer_descs = agent.get_xfer_descs(descs, mem_type="cuda")
```

## Backend Plugin Architecture

NIXL's plugin system has two API layers:

```
+-------------------------------------------------------------------+
|  Application / Inference Framework                                |
+-------------------------------------------------------------------+
|  NIXL North Bound API (NB API)                                    |
|  register_memory, initialize_xfer, transfer, check_xfer_state    |
+-------------------------------------------------------------------+
|  NIXL Transfer Agent                                              |
|  Backend selection, metadata management, handle bookkeeping       |
+-------------------------------------------------------------------+
|  NIXL South Bound API (SB API)    <-- Plugin interface            |
|  registerMem, prepXfer, postXfer, checkXfer, getNotifs, ...       |
+-------------------------------------------------------------------+
|  UCX Plugin  |  Libfabric Plugin  |  GDS Plugin  |  OBJ Plugin   |
+-------------------------------------------------------------------+
|  Hardware: InfiniBand / RoCE / EFA / NVMe / S3                    |
+-------------------------------------------------------------------+
```

### How Backend Selection Works

When you create a transfer request, NIXL automatically selects the best
backend based on:

1. **Source and destination memory types**: If source is VRAM and destination
   is FILE, only GDS can handle it directly.
2. **Common backends on both agents**: Both sides must have the same backend
   plugin loaded for network transfers.
3. **Registered memory coverage**: The backend must have both source and
   destination regions registered.

If multiple backends qualify, NIXL uses a preference list or picks the first
match. You can also force a specific backend:

```python
# Force UCX backend for this transfer
handle = agent.initialize_xfer(
    "READ", local_descs, remote_descs, "remote-agent",
    backend="UCX"
)
```

### Plugin Capabilities

Each backend plugin declares its capabilities:

| Capability | UCX | Libfabric | GDS | POSIX | OBJ |
|------------|-----|-----------|-----|-------|-----|
| supportsLocal() | Yes | Yes | Yes | Yes | Yes |
| supportsRemote() | Yes | Yes | No | No | No |
| supportsNotif() | Yes | Yes | No | No | No |
| VRAM | Yes | Yes | Yes | No | No |
| DRAM | Yes | Yes | No | Yes | No |
| FILE | No | No | Yes | Yes | No |
| OBJ | No | No | No | No | Yes |

Storage backends (GDS, POSIX, OBJ) are always "local" from NIXL's perspective
because they communicate with a local storage client, not a remote NIXL agent.

## Metadata Exchange

NIXL uses one-sided RDMA semantics (read/write), which means the remote CPU
does not participate in the data transfer itself. But for the initiating side
to know *where* to read or write, it needs the remote agent's metadata:
connection information and memory registration keys.

### Exchange Methods

**Method 1: Direct side-channel (listener thread)**

```python
# On Node A (target): start listener
agent_a = nixl_agent("target", nixl_agent_config(
    enable_listen_thread=True, listen_port=5555))

# On Node B (initiator): fetch metadata from target
agent_b.fetch_remote_metadata("target", "10.0.1.100", 5555)
agent_b.send_local_metadata("10.0.1.100", 5555)
```

**Method 2: Central metadata service (ETCD)**

```python
# Set environment variables
# NIXL_ETCD_ENDPOINTS=http://etcd-server:2379

# Each agent publishes its metadata automatically
agent.send_local_metadata()   # Publishes to ETCD

# Fetch remote agent metadata from ETCD
agent.fetch_remote_metadata("remote-agent-name")
```

### What Metadata Contains

The serialized metadata includes:

- **Per backend**: Connection information (addresses, keys, endpoint names)
- **Per registered memory region**: Remote access keys and buffer addresses
- **Agent identification**: Unique agent name

Metadata exchange happens once during initialization. After exchange, all
transfers proceed without further coordination on the control path.

## Transfer Lifecycle

A NIXL transfer follows this lifecycle:

```
  register_memory()        Pins memory, generates backend metadata
         |
         v
  get_agent_metadata()     Serializes local metadata for exchange
         |
         v
  load_remote_metadata()   Loads peer metadata, enables one-sided access
         |
         v
  initialize_xfer()        Validates descriptors, selects backend, creates handle
         |                 (or: prep_xfer_dlist + make_prepped_xfer for index-based)
         v
  transfer(handle)         Posts the async transfer to the backend
         |
         v
  check_xfer_state(handle) Polls for completion: "PROC" -> "DONE" or "ERR"
         |
         v
  release_xfer_handle()    Frees transfer resources
         |
         v
  deregister_memory()      Unpins memory, releases backend metadata
```

### Two Styles of Transfer Creation

**Style 1: initialize_xfer (ad-hoc descriptors)**

Use when source/destination addresses are determined at transfer time:

```python
handle = agent.initialize_xfer(
    "READ",              # Operation: READ or WRITE
    local_descs,         # Local descriptor list
    remote_descs,        # Remote descriptor list
    "remote-agent",      # Target agent name
    b"transfer-complete" # Optional notification message
)
```

**Style 2: prep_xfer_dlist + make_prepped_xfer (index-based)**

Use when you have a fixed set of blocks and want to select subsets by index
at transfer time. This avoids re-validating descriptors on every transfer:

```python
# Prepare both sides once
local_side = agent.prep_xfer_dlist("", local_descs)
remote_side = agent.prep_xfer_dlist("remote-agent", remote_descs)

# Create transfers by selecting indices
handle = agent.make_prepped_xfer(
    "READ",
    local_side,  [0, 1, 2],     # Read local blocks 0,1,2
    remote_side, [5, 3, 1],     # From remote blocks 5,3,1
    b"kv-batch-42"              # Notification
)
```

This index-based approach is what vLLM's NixlConnector uses for KV cache
transfer, where KV cache blocks are pre-registered and transfers select
which blocks to move based on the request's page table.

### Transfer Reposting

A completed transfer handle can be reposted without re-preparation:

```python
# First transfer
agent.transfer(handle)
while agent.check_xfer_state(handle) != "DONE":
    pass

# Repost the same transfer (e.g., after remote data is updated)
agent.transfer(handle, b"repost-notification")
while agent.check_xfer_state(handle) != "DONE":
    pass
```

This is efficient for repeated transfers between the same memory locations,
such as periodic model parameter synchronization.

## Notifications

NIXL notifications are small messages delivered to the target agent upon
transfer completion. They serve as the signaling mechanism in the
disaggregated inference pipeline:

```python
# Initiator side: attach notification to transfer
handle = agent.initialize_xfer(
    "WRITE", local_descs, remote_descs, "decoder",
    b"request-id:12345"  # Notification message (bytes)
)
agent.transfer(handle)

# Target side: poll for notifications
while True:
    notifs = agent.get_new_notifs()
    if "prefiller" in notifs:
        for msg in notifs["prefiller"]:
            request_id = msg.decode()
            # KV cache for this request has arrived; schedule decode
```

Notifications are delivered through the same backend that performed the
transfer (UCX or Libfabric). They are not ordered across different
transfers.

## Dynamic Agent Management

NIXL supports adding and removing agents at runtime, which is essential for
inference auto-scaling:

```python
# Adding a new agent: exchange metadata with existing agents
new_agent = nixl_agent("new-decoder", config)
# ... register memory, exchange metadata ...

# Removing an agent: invalidate its metadata on connected peers
agent.remove_remote_agent("failed-decoder")
# This triggers disconnect for all backends connected to that agent
# and purges cached metadata
```

When using ETCD, agents can discover each other automatically without
explicit IP/port exchange.

## Summary

| Concept | Key Takeaway |
|---------|--------------|
| Agent | One per process; manages all communication state |
| Memory Registration | Required before transfer; pins memory, generates RDMA keys |
| Backend Plugins | Modular transports; auto-selected based on memory types |
| Metadata Exchange | One-time control-path operation; enables one-sided RDMA |
| Transfers | Async, non-blocking; two styles (ad-hoc or index-based) |
| Notifications | Transfer-completion signals; drive the inference pipeline |
| Dynamic Scaling | Add/remove agents at runtime via metadata invalidation |

---

Next: [Section 2: NIXL on AWS EFA](../02-efa-integration/README.md)
