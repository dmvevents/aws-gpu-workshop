# Data Curation with NeMo Curator on EKS

## Overview

This workshop section teaches you how to build a production-grade data curation pipeline using [NVIDIA NeMo Curator](https://github.com/NVIDIA/NeMo-Curator) running on Amazon EKS with GPU-accelerated nodes. You will work through a realistic end-to-end scenario: preparing web-scraped content for LLM fine-tuning.

## Use Case: TechPulse Media

**TechPulse Media** is a tech news aggregator that collects articles from thousands of technology blogs, news sites, and forums. They want to fine-tune an open-source LLM to power their intelligent search and summarization features. However, their raw web corpus is noisy: it contains duplicates, boilerplate HTML, low-quality pages, non-English content, and off-topic articles.

Your job is to build a curation pipeline that transforms 500K+ raw web documents into a clean, deduplicated, high-quality training dataset of approximately 50K documents focused on technology topics.

## Pipeline Architecture

The pipeline follows NeMo Curator's recommended curation flow:

```
Raw Web Corpus (S3)
       |
       v
+------------------+
| 01 - Download    |  Fetch Wikipedia dumps or Common Crawl samples
+------------------+
       |
       v
+------------------+
| 02 - Extract     |  HTML to text, language ID, initial cleaning
+------------------+
       |
       v
+------------------+
| 03 - Dedup       |  Exact hash dedup + fuzzy MinHash LSH dedup (GPU)
+------------------+
       |
       v
+------------------+
| 04 - Filter      |  Heuristic + classifier-based quality filtering (GPU)
+------------------+
       |
       v
+------------------+
| 05 - Classify    |  Domain classification + quality scoring (GPU)
+------------------+
       |
       v
+------------------+
| 06 - Export      |  JSONL export, validation, upload to S3/HuggingFace
+------------------+
       |
       v
  Curated Dataset
  (~50K documents)
```

## What You Will Learn

- How to deploy NeMo Curator jobs on Kubernetes with GPU scheduling
- Exact and fuzzy deduplication at scale using GPU-accelerated MinHash LSH
- Quality filtering with both heuristic rules and neural classifiers
- Domain classification to keep only relevant content
- Best practices for building reproducible curation pipelines

## Prerequisites

Before starting this section, ensure you have:

| Requirement | Details |
|-------------|---------|
| **EKS Cluster** | Running cluster with at least 2 GPU nodes (g5.xlarge or larger) |
| **GPU Node Pool** | Node group with NVIDIA GPU Operator installed and `nvidia.com/gpu` resources visible |
| **S3 Bucket** | An S3 bucket for storing raw and curated data |
| **kubectl** | Configured and authenticated against your EKS cluster |
| **Helm** | v3.x installed for deploying any supporting charts |
| **AWS CLI** | Configured with permissions for S3 and ECR |
| **Namespace** | A dedicated Kubernetes namespace (created in step 00) |

## Estimated Time

| Section | Duration |
|---------|----------|
| 00 - Setup | 10 min |
| 01 - Download | 10 min |
| 02 - Extract | 15 min |
| 03 - Dedup | 20 min |
| 04 - Filter | 15 min |
| 05 - Classify | 10 min |
| 06 - Export | 10 min |
| **Total** | **~90 minutes** |

## Directory Structure

```
workshop/nemo-curator/
  README.md              <-- You are here
  00-setup/
    README.md            Setup and prerequisites verification
  01-download/
    README.md            Data download pipeline
  02-extract/
    README.md            Text extraction and language ID
  03-dedup/
    README.md            Exact and fuzzy deduplication
  04-filter/
    README.md            Quality filtering
  05-classify/
    README.md            Topic classification and scoring
  06-export/
    README.md            Export, validation, and upload
```

## Key Concepts

### NeMo Curator

NeMo Curator is NVIDIA's open-source library for scalable data curation. It provides GPU-accelerated implementations of common data processing steps including deduplication, filtering, and classification. It integrates with Dask for distributed processing and cuDF for GPU DataFrames.

### Why GPU-Accelerated Curation?

Traditional CPU-based curation pipelines struggle at scale. Operations like fuzzy deduplication (MinHash LSH) and neural quality classification are compute-intensive. By running these on GPUs via EKS, you can process millions of documents in minutes rather than hours.

### Pipeline Reproducibility

Each stage in this workshop reads from and writes to well-defined paths, logs its configuration, and produces summary statistics. This makes the pipeline fully reproducible and auditable -- a requirement for production ML systems.

## Next Step

Proceed to [00 - Setup](./00-setup/README.md) to verify your cluster and create the workshop namespace.
