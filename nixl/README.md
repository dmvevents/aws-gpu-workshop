# GPU-to-GPU Communication with NIXL on EKS

**Estimated time:** 45 minutes

## Overview

NVIDIA Inference Xfer Library (NIXL) is a high-performance point-to-point
communication library purpose-built for AI inference workloads. Unlike NCCL,
which focuses on collective operations for training, NIXL addresses the data
movement challenges unique to distributed inference: transferring KV caches
between disaggregated prefill and decode nodes, moving data between
heterogeneous memory tiers (GPU HBM, host DRAM, NVMe, object storage), and
doing so with the low latency and high bandwidth that production serving
demands.

This workshop covers NIXL's architecture, how it integrates with AWS Elastic
Fabric Adapter (EFA) through its libfabric backend, and how it enables
disaggregated inference patterns in frameworks like vLLM and NVIDIA Dynamo.

## What You Will Learn

- How NIXL's Transfer Agent abstraction unifies GPU memory, host memory, and
  storage behind a single API for one-sided RDMA operations
- The role of backend plugins (UCX, libfabric/EFA, GPUDirect Storage) and how
  NIXL selects the optimal transport automatically
- How memory registration, metadata exchange, and asynchronous transfers work
  in practice using the Python API
- Multi-rail RDMA and topology-aware GPU-to-NIC mapping on EFA-equipped
  instances (P4d, P5)
- How vLLM's NixlConnector uses NIXL to transfer KV caches between prefill
  and decode instances

## Prerequisites

- Familiarity with GPU computing concepts (CUDA, GPU memory)
- Basic understanding of RDMA and network fabrics
- Experience with Kubernetes / EKS is helpful but not required
- Access to an AWS account with EFA-enabled instances (P4d.24xlarge or
  P5.48xlarge) for hands-on sections

## Workshop Sections

| # | Section | Time | Description |
|---|---------|------|-------------|
| 1 | [NIXL Architecture](01-architecture/README.md) | 15 min | Core concepts: agents, memory sections, backend plugins, metadata exchange, transfer primitives |
| 2 | [NIXL on AWS EFA](02-efa-integration/README.md) | 15 min | Libfabric backend, multi-rail RDMA, topology-aware mapping, security group requirements, QP budget |
| 3 | [Disaggregated Inference](03-disaggregated-inference/README.md) | 15 min | Prefill/decode separation, KV cache transfer with vLLM, NCCL + NIXL hybrid architecture |

## Architecture at a Glance

```
+---------------------------------------------------------------+
|                    Inference Framework                         |
|              (vLLM / NVIDIA Dynamo / TensorRT-LLM)            |
+---------------------------------------------------------------+
                            |
                     NIXL Python API
                   (nixl_agent, transfers)
                            |
+---------------------------------------------------------------+
|                     NIXL Transfer Agent                       |
|                                                               |
|  +------------------+  +------------------+  +--------------+ |
|  | Memory Sections  |  | Backend Selector |  |   Metadata   | |
|  |                  |  |                  |  |   Handler    | |
|  | VRAM  (GPU HBM)  |  | UCX (IB/RoCE)   |  |              | |
|  | DRAM  (Host)     |  | Libfabric (EFA)  |  | Side-channel | |
|  | FILE  (NVMe)     |  | GDS (GPUDirect)  |  | or ETCD      | |
|  | OBJ   (S3)       |  | POSIX / GUSLI   |  |              | |
|  +------------------+  +------------------+  +--------------+ |
+---------------------------------------------------------------+
                            |
              Hardware / Network Fabric
        (NVLink, InfiniBand, EFA SRD, NVMe-oF, TCP)
```

## Key Terminology

| Term | Definition |
|------|------------|
| **NIXL Agent** | A per-process instance that manages memory registrations, connections, and transfers. Each inference worker creates one agent. |
| **Memory Section** | A set of registered address ranges (segments) of a given type: VRAM, DRAM, FILE, BLOCK, or OBJ. |
| **Backend Plugin** | A loadable transport module (UCX, Libfabric, GDS, etc.) that implements NIXL's South Bound API for actual data movement. |
| **Descriptor List** | An ordered list of (address, length, device_id) tuples describing source or destination buffers for a transfer. |
| **Transfer Handle** | An opaque reference to a prepared and posted asynchronous transfer, used to check completion status. |
| **Metadata** | Serialized connection and memory registration information exchanged between agents so that one-sided (RDMA) operations can proceed without remote CPU involvement. |
| **Notification** | A small message piggy-backed on transfer completion, delivered to the target agent to signal that data has arrived. |

## References

- [NIXL GitHub Repository](https://github.com/ai-dynamo/nixl) (Apache 2.0)
- [NIXL Architecture Overview](https://github.com/ai-dynamo/nixl/blob/main/docs/nixl.md)
- [NIXL Python API Reference](https://github.com/ai-dynamo/nixl/blob/main/docs/python_api.md)
- [NIXL Backend Plugin Guide](https://github.com/ai-dynamo/nixl/blob/main/docs/BackendGuide.md)
- [NIXLBench Benchmark Guide](https://github.com/ai-dynamo/nixl/blob/main/benchmark/nixlbench/README.md)
- [AWS EFA Documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html)
- [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo)
