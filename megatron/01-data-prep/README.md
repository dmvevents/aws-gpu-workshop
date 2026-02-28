# 01 - Data Preparation for Megatron-LM

## Objective

Convert TechPulse Media's curated JSONL corpus into Megatron-LM's indexed binary format. This binary format enables high-throughput, zero-copy data loading during distributed training -- a critical requirement when hundreds of GPUs need to read training data simultaneously without becoming I/O-bound.

## Background

Megatron-LM does not read raw text files during training. Instead, it uses a custom indexed binary format consisting of two files per dataset:

| File | Extension | Contents |
|------|-----------|----------|
| **Binary data** | `.bin` | Tokenized sequences packed as contiguous int16/int32 arrays |
| **Index** | `.idx` | Byte offsets and sequence lengths for random access |

This format allows Megatron to memory-map the binary file and seek directly to any document or sequence position without parsing JSON, counting newlines, or loading the entire dataset into RAM. At scale (billions of tokens), this difference in I/O efficiency determines whether your GPUs sit idle waiting for data or stay saturated with compute.

## Step 1: Understand the Input Format

The NeMo Curator pipeline produced JSONL files where each line is a JSON object with (at minimum) a `text` field:

```json
{"text": "NVIDIA announced the H200 GPU featuring 141GB of HBM3e memory...", "title": "NVIDIA H200 Launch", "source": "techpulse", "quality_score": 0.92}
{"text": "Kubernetes 1.30 introduces in-place resource resize for pods...", "title": "K8s 1.30 Features", "source": "techpulse", "quality_score": 0.88}
```

Megatron's preprocessing tool reads the `text` field from each JSON line, tokenizes it, and writes the binary output.

## Step 2: Choose a Tokenizer

The tokenizer must match the base model you plan to train or continue pre-training. Common choices:

| Base Model | Tokenizer | Vocab Size | Type |
|------------|-----------|------------|------|
| Llama 3.x | `meta-llama/Llama-3.1-8B` | 128,256 | BPE (tiktoken) |
| Qwen 2.5 / Qwen 3 | `Qwen/Qwen2.5-7B` | 151,936 | BPE (tiktoken) |
| Mistral | `mistralai/Mistral-7B-v0.3` | 32,768 | BPE (SentencePiece) |
| GPT-NeoX | `EleutherAI/gpt-neox-20b` | 50,257 | BPE |

For TechPulse Media, we will use the Llama 3.1 tokenizer since we are continuing pre-training from Llama-3.1-8B:

```bash
# Download the tokenizer (requires HuggingFace access)
huggingface-cli download meta-llama/Llama-3.1-8B \
  --include "tokenizer*" "special_tokens_map.json" \
  --local-dir /shared/tokenizer/llama-3.1-8b
```

## Step 3: Run Megatron's Preprocessing Tool

Megatron-LM provides `tools/preprocess_data.py` to convert JSONL to binary format. Here is the command:

```bash
python tools/preprocess_data.py \
  --input /shared/data/curated/techpulse_corpus.jsonl \
  --output-prefix /shared/data/megatron/techpulse \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model /shared/tokenizer/llama-3.1-8b \
  --json-keys text \
  --workers 16 \
  --chunk-size 1000 \
  --append-eod
```

### Parameter Reference

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `--input` | Path to JSONL | Source corpus from NeMo Curator |
| `--output-prefix` | Path prefix | Produces `<prefix>_text_document.bin` and `.idx` |
| `--tokenizer-type` | `HuggingFaceTokenizer` | Use any HuggingFace tokenizer |
| `--tokenizer-model` | Path or model name | Local path or HF model ID |
| `--json-keys` | `text` | Which JSON field(s) to tokenize |
| `--workers` | `16` | Parallel tokenization workers |
| `--chunk-size` | `1000` | Documents per worker chunk |
| `--append-eod` | (flag) | Append end-of-document token between documents |

### Output Files

After preprocessing completes, you will have:

```
/shared/data/megatron/
  techpulse_text_document.bin    # Binary tokenized data (~200MB for 50K docs)
  techpulse_text_document.idx    # Index file (~2MB)
```

### Running as a Kubernetes Job

To run preprocessing on the EKS cluster (recommended for large corpora):

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: megatron-preprocess
  namespace: megatron-workshop
spec:
  backoffLimit: 2
  template:
    metadata:
      labels:
        app: megatron
        stage: preprocess
    spec:
      restartPolicy: OnFailure
      containers:
        - name: preprocessor
          image: nvcr.io/nvidia/pytorch:25.01-py3
          command:
            - python
            - /opt/megatron-lm/tools/preprocess_data.py
            - --input
            - /shared/data/curated/techpulse_corpus.jsonl
            - --output-prefix
            - /shared/data/megatron/techpulse
            - --tokenizer-type
            - HuggingFaceTokenizer
            - --tokenizer-model
            - /shared/tokenizer/llama-3.1-8b
            - --json-keys
            - text
            - --workers
            - "16"
            - --chunk-size
            - "1000"
            - --append-eod
          resources:
            requests:
              cpu: "16"
              memory: "32Gi"
            limits:
              cpu: "32"
              memory: "64Gi"
          volumeMounts:
            - name: shared
              mountPath: /shared
      volumes:
        - name: shared
          persistentVolumeClaim:
            claimName: fsx-shared-pvc
```

> **Note:** Preprocessing is CPU-intensive, not GPU-intensive. No GPUs are requested in this job.

## Step 4: Verify the Preprocessed Data

After the job completes, verify the output:

```bash
# Check that both files exist
ls -lh /shared/data/megatron/techpulse_text_document.*

# Expected output:
# -rw-r--r-- 1 root root 198M  techpulse_text_document.bin
# -rw-r--r-- 1 root root 1.8M  techpulse_text_document.idx
```

You can inspect the index file to verify document counts and token statistics:

```python
import numpy as np

# Read the index header
with open("/shared/data/megatron/techpulse_text_document.idx", "rb") as f:
    magic = f.read(9)         # b'MMIDIDX\x00\x00'
    version = np.frombuffer(f.read(8), dtype=np.int64)[0]
    dtype_code = np.frombuffer(f.read(1), dtype=np.uint8)[0]
    num_sequences = np.frombuffer(f.read(8), dtype=np.int64)[0]
    num_documents = np.frombuffer(f.read(8), dtype=np.int64)[0]

print(f"Version:       {version}")
print(f"Dtype code:    {dtype_code}")   # 3 = int16, 5 = int32
print(f"Num sequences: {num_sequences}")
print(f"Num documents: {num_documents}")
```

Expected output for 50K documents:

```
Version:       1
Dtype code:    5
Num sequences: 50000
Num documents: 50000
```

## Step 5: Data Blending for Multi-Domain Training

TechPulse Media's corpus may contain multiple domains (news articles, blog posts, API documentation, forum discussions). Megatron supports **data blending** -- mixing multiple preprocessed datasets with configurable weights during training.

### Preparing Multiple Datasets

Preprocess each domain separately:

```bash
# Domain 1: News articles
python tools/preprocess_data.py \
  --input /shared/data/curated/news_articles.jsonl \
  --output-prefix /shared/data/megatron/news \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model /shared/tokenizer/llama-3.1-8b \
  --json-keys text --workers 16 --append-eod

# Domain 2: Technical documentation
python tools/preprocess_data.py \
  --input /shared/data/curated/tech_docs.jsonl \
  --output-prefix /shared/data/megatron/techdocs \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model /shared/tokenizer/llama-3.1-8b \
  --json-keys text --workers 16 --append-eod

# Domain 3: Forum discussions
python tools/preprocess_data.py \
  --input /shared/data/curated/forums.jsonl \
  --output-prefix /shared/data/megatron/forums \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model /shared/tokenizer/llama-3.1-8b \
  --json-keys text --workers 16 --append-eod
```

### Configuring Blend Weights

In the Megatron training command, specify blended datasets with weights:

```bash
--data-path \
  0.5 /shared/data/megatron/news_text_document \
  0.3 /shared/data/megatron/techdocs_text_document \
  0.2 /shared/data/megatron/forums_text_document
```

The weights control sampling probability. In this example:
- 50% of training batches come from news articles
- 30% from technical documentation
- 20% from forum discussions

### Blend Weight Guidelines

| Strategy | When to Use | Example Weights |
|----------|-------------|-----------------|
| **Proportional** | Domains are equally important | Equal weights (0.33 each) |
| **Upweight target domain** | Optimizing for specific domain | Target 0.6, others split remaining 0.4 |
| **Token-count balanced** | Datasets differ greatly in size | Weight inversely proportional to token count |
| **Curriculum** | Quality varies by domain | Higher weight for higher quality domains |

### Alternative: YAML Blend Configuration

For complex blending setups, use a YAML configuration file:

```yaml
# blend_config.yaml
- dataset_prefix: /shared/data/megatron/news_text_document
  dataset_weight: 0.5
  dataset_split: train

- dataset_prefix: /shared/data/megatron/techdocs_text_document
  dataset_weight: 0.3
  dataset_split: train

- dataset_prefix: /shared/data/megatron/forums_text_document
  dataset_weight: 0.2
  dataset_split: train

- dataset_prefix: /shared/data/megatron/news_text_document
  dataset_weight: 1.0
  dataset_split: validation
```

This can be loaded programmatically in the training script using NeMo's `PreTrainingDataModule`:

```python
from nemo.collections.llm.gpt.data import PreTrainingDataModule

data = PreTrainingDataModule(
    paths=blend_config,  # parsed from YAML
    seq_length=4096,
    micro_batch_size=2,
    global_batch_size=512,
    tokenizer=tokenizer,
    num_workers=8,
)
```

## Step 6: Sequence Length and Packing Considerations

### Sequence Length

The `--seq-length` parameter in Megatron controls the maximum context window for training. Documents longer than this are truncated; shorter documents are packed together (separated by EOD tokens).

| Sequence Length | Memory Impact | Use Case |
|----------------|---------------|----------|
| 2048 | Lower | Small models, limited GPU memory |
| 4096 | Medium | Standard pre-training, good balance |
| 8192 | Higher | Long-context applications |
| 32768+ | Very high | Requires sequence parallelism or ring attention |

### Document Packing

Megatron automatically packs multiple short documents into a single training sequence (separated by EOD tokens). This is efficient because it minimizes padding waste. For a 4096-token sequence length:

```
[Doc1 tokens (800)] [EOD] [Doc2 tokens (1200)] [EOD] [Doc3 tokens (2000)] [EOD] [PAD x 96]
|<---------------------------- 4096 tokens total ------------------------------>|
```

This packing happens at the data loader level and does not require any preprocessing changes. The `--append-eod` flag during preprocessing inserts the boundary markers.

## Verification Checklist

- [ ] Tokenizer downloaded and verified (correct model, correct vocab size)
- [ ] Preprocessing job completed without errors
- [ ] Both `.bin` and `.idx` files exist with non-zero size
- [ ] Document count matches expected number from curated corpus
- [ ] (If blending) All domain datasets preprocessed with the same tokenizer
- [ ] (If blending) Blend weights sum to 1.0 per split (train, validation)

## Common Issues

| Problem | Cause | Fix |
|---------|-------|-----|
| `tokenizer not found` | Missing tokenizer files | Re-download from HuggingFace, ensure all files present |
| `KeyError: 'text'` | JSONL field name mismatch | Check `--json-keys` matches your JSON field name |
| Empty `.bin` file | Input JSONL is empty or malformed | Verify input with `head -1 input.jsonl \| python3 -m json.tool` |
| Slow preprocessing | Too few workers | Increase `--workers` (up to CPU count) |
| OOM during preprocessing | Extremely long documents | Increase pod memory limits or pre-split long documents |

## Next Step

Proceed to [02 - Parallelism Strategy](../02-parallelism/README.md) to choose the right parallelism configuration for TechPulse Media's model and GPU cluster.
