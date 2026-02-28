# 03 - Deduplication

## Objective

Remove duplicate and near-duplicate documents from the extracted corpus using both exact and fuzzy deduplication. This stage uses NeMo Curator's GPU-accelerated MinHash LSH implementation to efficiently find similar documents at scale.

## Background

Duplicate content is one of the largest quality problems in web-scraped corpora. TechPulse Media's raw feeds often contain:

- **Exact duplicates**: The same article syndicated across multiple sites
- **Near duplicates**: Slightly modified copies (different headers/footers, minor edits)
- **Boilerplate overlap**: Template-heavy pages that share 80%+ of their content

Training an LLM on duplicated data wastes compute, amplifies memorization, and degrades output diversity. NeMo Curator addresses this with a two-pass deduplication pipeline:

1. **Exact dedup** (fast, hash-based) -- Removes byte-identical documents
2. **Fuzzy dedup** (GPU-accelerated MinHash LSH) -- Removes near-identical documents

## Step 1: Exact Deduplication

Exact dedup computes a hash (MD5 or SHA256) of each document's text field and removes records with duplicate hashes. This is fast and runs on CPU.

### How It Works

```
Document A: hash("The quick brown fox...") = a1b2c3
Document B: hash("The quick brown fox...") = a1b2c3  <-- DUPLICATE
Document C: hash("A different article...")  = d4e5f6
```

Documents A and B produce the same hash, so one is removed. Only the first occurrence is kept.

### Job Manifest for Exact Dedup

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: nemo-curator-exact-dedup
  namespace: nemo-curator-workshop
spec:
  backoffLimit: 2
  template:
    metadata:
      labels:
        app: nemo-curator
        stage: exact-dedup
    spec:
      restartPolicy: OnFailure
      containers:
        - name: exact-dedup
          image: nvcr.io/nvidia/nemo-curator:v0.6.0
          command:
            - python
            - /scripts/exact_dedup.py
            - --input-dir
            - /data/extracted
            - --output-dir
            - /data/exact-deduped
            - --hash-method
            - md5
            - --id-field
            - url
            - --text-field
            - text
          resources:
            requests:
              cpu: "4"
              memory: "16Gi"
            limits:
              cpu: "8"
              memory: "32Gi"
          volumeMounts:
            - name: data
              mountPath: /data
            - name: scripts
              mountPath: /scripts
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: nemo-curator-data
        - name: scripts
          configMap:
            name: dedup-scripts
```

### Submit and Monitor

```bash
kubectl apply -f exact-dedup-job.yaml

kubectl logs -f job/nemo-curator-exact-dedup -n nemo-curator-workshop

kubectl wait --for=condition=complete job/nemo-curator-exact-dedup \
  -n nemo-curator-workshop --timeout=600s
```

### Core NeMo Curator API

```python
from nemo_curator.modules import ExactDuplicates

exact_dedup = ExactDuplicates(
    id_field="url",
    text_field="text",
    hash_method="md5",
)

# Returns a DocumentDataset with duplicates removed
deduped_dataset = exact_dedup(dataset)
```

## Step 2: Fuzzy Deduplication (GPU-Accelerated)

Fuzzy dedup uses MinHash Locality-Sensitive Hashing (LSH) to find documents that are similar but not identical. This is where GPU acceleration provides the most value -- computing MinHash signatures and performing LSH bucketing on GPU is 10-50x faster than CPU.

### How MinHash LSH Works

1. **Shingling**: Each document is split into overlapping character n-grams (shingles)
2. **MinHash signatures**: Each document's shingle set is compressed into a fixed-size signature using multiple hash functions
3. **LSH bucketing**: Signatures are divided into bands; documents sharing a band are candidate pairs
4. **Jaccard similarity**: Candidate pairs are compared to compute actual similarity
5. **Connected components**: Documents above the similarity threshold are grouped; all but one per group are removed

```
Document A: "The quick brown fox jumps over the lazy dog"
Document B: "The quick brown fox leaps over the lazy dog"  (1 word changed)
Document C: "Completely different article about technology"

Jaccard(A, B) = 0.85  --> Near-duplicate, remove one
Jaccard(A, C) = 0.02  --> Not similar, keep both
```

### Job Manifest for Fuzzy Dedup

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: nemo-curator-fuzzy-dedup
  namespace: nemo-curator-workshop
spec:
  backoffLimit: 2
  template:
    metadata:
      labels:
        app: nemo-curator
        stage: fuzzy-dedup
    spec:
      restartPolicy: OnFailure
      containers:
        - name: fuzzy-dedup
          image: nvcr.io/nvidia/nemo-curator:v0.6.0
          command:
            - python
            - /scripts/fuzzy_dedup.py
            - --input-dir
            - /data/exact-deduped
            - --output-dir
            - /data/fuzzy-deduped
            - --seed
            - "42"
            - --char-ngrams
            - "5"
            - --num-hashes
            - "128"
            - --num-buckets
            - "64"
            - --jaccard-threshold
            - "0.8"
            - --id-field
            - url
            - --text-field
            - text
          resources:
            requests:
              cpu: "4"
              memory: "16Gi"
              nvidia.com/gpu: "1"
            limits:
              cpu: "8"
              memory: "32Gi"
              nvidia.com/gpu: "1"
          volumeMounts:
            - name: data
              mountPath: /data
            - name: scripts
              mountPath: /scripts
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: nemo-curator-data
        - name: scripts
          configMap:
            name: dedup-scripts
```

> **Key**: This job requests `nvidia.com/gpu: 1` for GPU-accelerated MinHash computation.

### Submit and Monitor

```bash
kubectl apply -f fuzzy-dedup-job.yaml

kubectl logs -f job/nemo-curator-fuzzy-dedup -n nemo-curator-workshop

kubectl wait --for=condition=complete job/nemo-curator-fuzzy-dedup \
  -n nemo-curator-workshop --timeout=900s
```

### Core NeMo Curator API

```python
from nemo_curator.modules import FuzzyDuplicates, FuzzyDuplicatesConfig

fuzzy_config = FuzzyDuplicatesConfig(
    seed=42,
    char_ngrams=5,
    num_hashes=128,
    num_buckets=64,
    jaccard_threshold=0.8,
    id_field="url",
    text_field="text",
)

fuzzy_dedup = FuzzyDuplicates(config=fuzzy_config)

# Uses GPU if available, falls back to CPU
deduped_dataset = fuzzy_dedup(dataset)
```

### Understanding the Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `char_ngrams` | 5 | Size of character n-grams (shingles). Larger = more context, fewer false positives |
| `num_hashes` | 128 | Number of MinHash signatures. More = better accuracy, more memory |
| `num_buckets` | 64 | Number of LSH bands. More = higher recall, more candidate pairs |
| `jaccard_threshold` | 0.8 | Minimum similarity to consider duplicates. 0.8 is standard for web data |

## Step 3: Verify Deduplication Results

```bash
kubectl run data-inspector --rm -it --restart=Never \
  --image=nvcr.io/nvidia/nemo-curator:v0.6.0 \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "inspector",
        "image": "nvcr.io/nvidia/nemo-curator:v0.6.0",
        "command": ["bash"],
        "stdin": true,
        "tty": true,
        "volumeMounts": [{"name": "data", "mountPath": "/data"}]
      }],
      "volumes": [{
        "name": "data",
        "persistentVolumeClaim": {"claimName": "nemo-curator-data"}
      }]
    }
  }' \
  -n nemo-curator-workshop
```

Inside the pod:

```bash
echo "=== Document counts at each stage ==="
echo -n "After extraction:  "; wc -l /data/extracted/*.jsonl | tail -1
echo -n "After exact dedup: "; wc -l /data/exact-deduped/*.jsonl | tail -1
echo -n "After fuzzy dedup: "; wc -l /data/fuzzy-deduped/*.jsonl | tail -1

echo ""
echo "=== Dedup removal rate ==="
EXTRACTED=$(wc -l < /data/extracted/*.jsonl)
FINAL=$(wc -l < /data/fuzzy-deduped/*.jsonl)
echo "Removed: $(( EXTRACTED - FINAL )) documents ($(( (EXTRACTED - FINAL) * 100 / EXTRACTED ))%)"
```

**Expected results:**
- Exact dedup removes 1-5% of documents (Wikipedia has few exact duplicates)
- Fuzzy dedup removes an additional 5-15% (overlapping content, stub articles)
- Total dedup reduction: 10-20% for Wikipedia; 30-60% is typical for web crawls

## Step 4: Inspect Duplicate Clusters (Optional)

The fuzzy dedup step produces a duplicate cluster log. Inspect it to understand what was removed:

```bash
# View duplicate clusters
python3 -c "
import json
with open('/data/fuzzy-deduped/duplicate_clusters.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 5: break
        cluster = json.loads(line)
        print(f'Cluster {i}: {len(cluster[\"documents\"])} documents')
        for doc in cluster['documents'][:3]:
            print(f'  - {doc[\"url\"]} (similarity: {doc.get(\"jaccard_score\", \"N/A\")})')
        print()
"
```

This helps verify that the dedup is working correctly -- documents in the same cluster should be genuinely similar.

## Expected Outputs

| Artifact | Location | Description |
|----------|----------|-------------|
| Exact-deduped JSONL | `/data/exact-deduped/*.jsonl` | Documents after hash dedup |
| Fuzzy-deduped JSONL | `/data/fuzzy-deduped/*.jsonl` | Documents after MinHash LSH dedup |
| Duplicate clusters | `/data/fuzzy-deduped/duplicate_clusters.jsonl` | Log of detected duplicate groups |

## Verification Checklist

- [ ] Exact dedup job completed successfully
- [ ] Fuzzy dedup job completed successfully (GPU was used)
- [ ] Document count decreased at each stage
- [ ] Fuzzy dedup used GPU (check logs for CUDA/cuDF references)
- [ ] Duplicate cluster log exists and shows reasonable groupings
- [ ] Remaining documents are unique (no duplicates in final output)

## Next Step

Proceed to [04 - Filter](../04-filter/README.md) to apply quality filters and remove low-quality documents.
