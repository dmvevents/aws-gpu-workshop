# Disaggregated Inference with NVIDIA Dynamo on AWS EKS

**Estimated time:** 60 minutes

## Overview

NVIDIA Dynamo is an open-source inference runtime that separates the prefill
and decode phases of LLM inference, enabling each to scale independently across
GPU nodes. Combined with AWS Elastic Fabric Adapter (EFA) and NIXL's LIBFABRIC
backend, Dynamo achieves high-bandwidth, low-latency KV cache transfer between
nodes using RDMA.

This workshop walks through deploying disaggregated inference on Amazon EKS with
P5.48xlarge instances (H100 GPUs + 32 EFA adapters), progressing from single-node
to cross-node operation with NIXL LIBFABRIC over EFA RDMA.

## What You Will Learn

- How Dynamo's DynamoGraphDeployment (DGD) CRD orchestrates frontend, prefill,
  and decode workers across Kubernetes
- Setting up the Dynamo operator, etcd, and NATS on EKS
- Deploying disaggregated inference on a single node
- Cross-node KV cache transfer using NIXL LIBFABRIC over EFA RDMA
- Verifying EFA activation via libfabric handshake logs
- Memory requirements and common pitfalls (128Gi minimum on P5)
- Performance benchmarking: UCX vs LIBFABRIC backends

## Prerequisites

- Amazon EKS cluster with P5.48xlarge (or P4d.24xlarge) nodes
- EFA device plugin installed
- GDRCopy installer DaemonSet running
- kubectl and Helm configured
- HuggingFace token for model downloads

## Workshop Sections

| # | Section | Time | Description |
|---|---------|------|-------------|
| 1 | [Cluster Setup](01-setup/README.md) | 15 min | Dynamo operator, etcd, NATS, EFA device plugin |
| 2 | [Single-Node Disagg](02-single-node/README.md) | 15 min | Prefill + decode on same node, verify NIXL activation |
| 3 | [Cross-Node with EFA](03-cross-node/README.md) | 20 min | Two-node deployment with LIBFABRIC over RDMA |
| 4 | [Performance](04-performance/README.md) | 10 min | Benchmarking, UCX vs LIBFABRIC, multi-rail |

## Component Versions (Dynamo v0.9.0)

| Component | Version |
|-----------|---------|
| Dynamo | v0.9.0 |
| NIXL | 0.9.0 |
| TRT-LLM | 1.3.0rc1 / 1.3.0rc3 |
| vLLM | 0.14.1 |
| CUDA | 13.1 |
| EFA / libfabric | 2.3.1amzn3.0 |

## Key Finding: Memory Requirement

Worker pods require **128Gi memory** on P5.48xlarge. With 64Gi, the pod is
OOMKilled during NIXL LIBFABRIC initialization because NIXL enumerates all 32
EFA devices on P5, allocating queue pairs and memory registrations for each.

## Architecture

```
                    ┌─────────────────────┐
                    │   Client Requests   │
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   Dynamo Frontend   │
                    │   (HTTP API, no GPU)│
                    └────┬───────────┬────┘
                         │           │
              ┌──────────┘           └──────────┐
              ▼                                  ▼
    ┌──────────────────┐             ┌──────────────────┐
    │  Prefill Worker  │             │  Decode Worker   │
    │  Node 1 (P5)     │   NIXL     │  Node 2 (P5)     │
    │  8x H100         │◄──────────►│  8x H100         │
    │  32x EFA         │ LIBFABRIC  │  32x EFA         │
    └──────────────────┘  EFA RDMA  └──────────────────┘
```

## References

- [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo)
- [NIXL](https://github.com/ai-dynamo/nixl)
- [AWS EFA](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html)
- [Dynamo Fixes Repo](https://github.com/dmvevents/dynamo-disagg-fixes) (patches + test infrastructure)
