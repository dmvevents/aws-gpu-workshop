# Distributed Training with Megatron-LM on EKS

## Overview

This workshop section teaches you how to train large language models using [NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM) on Amazon EKS with multi-dimensional parallelism. You will learn how to prepare data in Megatron's binary format, configure tensor, pipeline, data, and expert parallelism, launch distributed training jobs on Kubernetes, and monitor training health in production.

## Use Case: TechPulse Media

**TechPulse Media** has completed data curation (the previous NeMo Curator workshop section) and now holds a clean, deduplicated corpus of 50K+ technology articles. They want to continue pre-training (CPT) an open-source base model on this domain-specific corpus so it learns technology terminology, product knowledge, and writing style specific to their vertical.

Their infrastructure:
- **EKS cluster** with 2 nodes of P4d.24xlarge (8x A100 40GB each, 16 GPUs total)
- **FSx for Lustre** for shared storage (model checkpoints, training data)
- **EFA networking** for high-bandwidth GPU-to-GPU communication across nodes

Their training plan:
1. Convert the curated JSONL corpus to Megatron's indexed binary format
2. Configure parallelism to fit a 7B-parameter model across 16 GPUs
3. Launch distributed training as a Kubernetes PyTorchJob
4. Monitor loss curves, GPU utilization, and NCCL health
5. Manage checkpoints for fault recovery and downstream fine-tuning

## Pipeline Architecture

```
Curated JSONL Corpus (from NeMo Curator)
       |
       v
+-------------------------+
| 01 - Data Preparation   |  Tokenize JSONL into Megatron indexed binary format
|                         |  Configure data blending for multi-domain training
+-------------------------+
       |
       v
+-------------------------+
| 02 - Parallelism        |  Choose TP/PP/DP/EP for model size and GPU count
|                         |  Understand memory and communication tradeoffs
+-------------------------+
       |
       v
+-------------------------+
| 03 - Launch Training    |  Deploy PyTorchJob on EKS with EFA networking
|                         |  Configure torchrun, NCCL, and environment variables
+-------------------------+
       |
       v
+-------------------------+
| 04 - Monitor & Debug    |  Track loss curves, GPU utilization, NCCL health
|                         |  Manage checkpoints, diagnose common failures
+-------------------------+
       |
       v
  Trained Model Checkpoint
  (ready for SFT / evaluation)
```

## What You Will Learn

- How Megatron-LM's indexed binary format enables high-throughput data loading at scale
- The four dimensions of parallelism (Tensor, Pipeline, Data, Expert) and when to use each
- How to calculate global batch size, data parallel size, and memory requirements
- How to deploy multi-node GPU training jobs on Kubernetes with EFA networking
- How to configure NCCL environment variables correctly for AWS (and which ones to avoid)
- How to monitor training health, debug hangs, and manage checkpoints
- MoE (Mixture of Experts) training with Expert Parallelism

## Prerequisites

Before starting this section, ensure you have:

| Requirement | Details |
|-------------|---------|
| **EKS Cluster** | Running cluster with GPU nodes (P4d.24xlarge or P5.48xlarge) |
| **GPU Node Pool** | Node group with NVIDIA device plugin and EFA device plugin installed |
| **FSx for Lustre** | Shared PVC for training data, checkpoints, and model weights |
| **kubectl** | Configured and authenticated against your EKS cluster |
| **Curated Dataset** | JSONL corpus from the NeMo Curator workshop (or equivalent) |
| **Container Image** | Megatron-LM container with EFA networking (see section 03) |
| **Completed Curation** | Output from the NeMo Curator pipeline (50K+ documents in JSONL) |

## Estimated Time

| Section | Duration |
|---------|----------|
| 01 - Data Preparation | 15 min |
| 02 - Parallelism Strategy | 15 min |
| 03 - Launch Training | 20 min |
| 04 - Monitor and Debug | 10 min |
| **Total** | **~60 minutes** |

## Directory Structure

```
workshop/megatron/
  README.md              <-- You are here
  01-data-prep/
    README.md            Tokenization, binary format, data blending
  02-parallelism/
    README.md            TP, PP, DP, EP strategy and decision tables
  03-launch/
    README.md            K8s manifests, EFA config, torchrun parameters
  04-monitor/
    README.md            Loss curves, GPU monitoring, checkpoint management
```

## Key Concepts

### Megatron-LM

Megatron-LM is NVIDIA's open-source framework for training large transformer models with efficient multi-dimensional parallelism. It provides optimized implementations of tensor parallelism (splitting individual layers across GPUs), pipeline parallelism (splitting model stages across nodes), and expert parallelism (distributing MoE experts across GPUs). It integrates with NVIDIA's Transformer Engine for mixed-precision training with FP8 support.

### Why Megatron-LM on EKS?

Training models beyond 1B parameters requires distributing computation across multiple GPUs and multiple nodes. Megatron-LM handles the parallelism; EKS handles the orchestration. Together, they provide:

- **Elastic scaling**: Add or remove GPU nodes without rewriting training code
- **Fault tolerance**: Kubernetes restarts failed pods; Megatron resumes from checkpoints
- **Resource isolation**: Multiple teams can share a GPU cluster with namespace-level quotas
- **Reproducibility**: All configuration is declarative (YAML manifests, environment variables)

### How This Connects to Data Curation

The NeMo Curator workshop produced a clean JSONL corpus. This workshop converts that JSONL into Megatron's high-performance binary format and uses it for training. The pipeline is:

```
NeMo Curator Output     Megatron Data Prep       Megatron Training
(curated JSONL)    -->  (indexed .bin/.idx)  -->  (distributed GPUs)
```

## Hardware Reference

| Instance | GPUs | GPU Memory | EFA NICs | Interconnect | Best For |
|----------|------|------------|----------|-------------|----------|
| g5.8xlarge | 1x A10G | 24GB | 0 | None | Development, small models (<3B) |
| p4d.24xlarge | 8x A100 | 40GB | 4 | 400 Gbps | Medium models (7B-13B) |
| p4de.24xlarge | 8x A100 | 80GB | 4 | 400 Gbps | Large models (13B-30B) |
| p5.48xlarge | 8x H100 | 80GB | 32 | 3200 Gbps | Very large models (30B+), MoE |

## Next Step

Proceed to [01 - Data Preparation](./01-data-prep/README.md) to convert TechPulse Media's curated corpus into Megatron's binary training format.
