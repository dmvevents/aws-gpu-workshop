# Scalable Inference & RLHF with Ray and vLLM on EKS

## Overview

This workshop section teaches you how to deploy a production-grade LLM inference stack and run reinforcement learning from human feedback (RLHF) alignment training using **Ray**, **vLLM**, and **NVIDIA NeMo RL** on Amazon EKS. You will work through a realistic end-to-end scenario: deploying a fine-tuned model for scalable inference, then running GRPO alignment training to improve its reasoning capabilities.

## Use Case: TechPulse Media

**TechPulse Media** has completed fine-tuning a 7B-parameter open-source model on their curated technology corpus (see the NeMo Curator workshop section). Now they need to:

1. **Serve the model** to their editorial team and API customers with low latency and high throughput
2. **Align the model** using GRPO (Group Relative Policy Optimization) so that it produces more accurate, well-reasoned responses to technical queries
3. **Scale inference** across multiple GPUs using tensor parallelism for their larger 70B model variant

Their infrastructure runs on Amazon EKS with GPU-equipped nodes (A100 or H100). They need a solution that handles both inference serving and training within the same Kubernetes cluster.

## Workshop Architecture

```
                     TechPulse Media - Inference & RLHF Stack
                     ==========================================

  Section 01                Section 02                Section 03
  Ray Cluster               vLLM Serving              GRPO Training
  Setup                     (Inference)               (Alignment)
  +-----------+             +-------------+           +---------------+
  | Ray Head  |             | vLLM Engine |           | NeMo RL       |
  | Dashboard |             | OpenAI API  |           | Generation +  |
  | GCS       |             | TP=2 across |           | Training loop |
  +-----------+             | 2 GPUs      |           | Math rewards  |
       |                    +-------------+           +---------------+
       |                         |                          |
  +-----------+             +----+--------+           +-----+----------+
  | Ray       |             | PagedAttn   |           | vLLM inference |
  | Workers   |             | Continuous  |           | DTensor train  |
  | (GPU)     |             | Batching    |           | Checkpointing  |
  +-----------+             +-------------+           +----------------+
       |                         |                          |
  =====+=========================+=========================+========
       |                         |                          |
  +----+---------------------------------------------------------+
  |                    Amazon EKS Cluster                        |
  |   GPU Nodes (A100/H100)  |  FSx Lustre  |  EFA Networking   |
  +-------------------------------------------------------------+

  Section 04: Production patterns -- autoscaling, health checks, EFA, debugging
```

## What You Will Learn

1. **Ray cluster fundamentals** -- head/worker architecture, Indexed Jobs for multi-node clusters, GPU resource allocation, and the Ray dashboard
2. **vLLM model serving** -- PagedAttention, continuous batching, tensor parallelism, the OpenAI-compatible API, and KV cache tuning
3. **GRPO/RLHF training** -- Group Relative Policy Optimization with NeMo RL, colocated vs. dedicated generation, math verification rewards, and key hyperparameters
4. **Production deployment** -- Ray Serve auto-scaling, health checks, EFA networking for multi-node, and common debugging patterns (NCCL timeouts, OOM, stale master IP)

## Prerequisites

Before starting this section, ensure you have:

| Requirement | Details |
|-------------|---------|
| **EKS Cluster** | Running cluster with at least 2 GPU nodes |
| **GPU Nodes** | p4d.24xlarge (8x A100 40GB) or p5.48xlarge (8x H100 80GB) |
| **KubeRay Operator** | Installed via Helm (v1.1+) |
| **FSx for Lustre** | PVC provisioned for shared model weights and checkpoints |
| **EFA Device Plugin** | Installed for high-performance inter-node networking |
| **NVIDIA Device Plugin** | GPU resources visible in `kubectl describe node` |
| **kubectl** | Configured and authenticated against your EKS cluster |
| **Container Image** | Pre-built image with Ray, vLLM, and NeMo RL (see Section 01) |

## Sections

| Section | Title | Duration | Description |
|---------|-------|----------|-------------|
| [01-ray-cluster](01-ray-cluster/) | Setting Up Ray on Kubernetes | 20 min | Deploy a Ray cluster using Indexed Jobs, configure GPU resources, access the dashboard |
| [02-vllm-serving](02-vllm-serving/) | vLLM Model Serving | 20 min | Deploy vLLM with tensor parallelism, benchmark throughput and latency |
| [03-grpo-training](03-grpo-training/) | GRPO/RLHF Training with NeMo RL | 25 min | Run GRPO alignment training, configure rewards, monitor progress |
| [04-production](04-production/) | Production Deployment Patterns | 10 min | Auto-scaling, health checks, EFA networking, debugging |

**Total estimated time: 75 minutes**

## Technology Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| **Ray** | 2.49+ | Distributed computing framework |
| **vLLM** | 0.11+ | LLM inference engine with PagedAttention |
| **NeMo RL** | main | GRPO reinforcement learning framework |
| **NCCL** | 2.27+ | GPU collective communication |
| **EFA** | libfabric 2.1+ | AWS high-speed networking |
| **KubeRay** | 1.1+ | Kubernetes operator for Ray clusters |
| **PyTorch** | 2.9+ | Deep learning framework |

## Cost Estimates

| Configuration | Instance | GPUs | Hourly Cost (on-demand) |
|---------------|----------|------|-------------------------|
| **Development** | 1x g5.8xlarge | 1x A10G | ~$1.20/hr |
| **Standard** | 2x p4d.24xlarge | 16x A100 40GB | ~$65/hr |
| **Production** | 2x p5.48xlarge | 16x H100 80GB | ~$196/hr |

Spot instances can reduce costs by 60-70% for training workloads.

## Navigating This Workshop

Each section builds on the previous one but can also be completed independently if you already have the prerequisites in place. Section 01 is required for all subsequent sections since it establishes the Ray cluster. Sections 02 and 03 can be done in either order. Section 04 is a reference guide.

Begin with [Section 01: Setting Up Ray on Kubernetes](01-ray-cluster/).
