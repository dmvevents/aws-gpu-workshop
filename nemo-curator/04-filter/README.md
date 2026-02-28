# 04 - Quality Filtering

## Objective

Apply heuristic and classifier-based quality filters to remove low-quality documents from the deduplicated corpus. This stage ensures that only well-written, informative, and appropriately-sized documents proceed to the classification stage.

## Background

Even after deduplication, TechPulse Media's corpus contains documents that would harm LLM training quality:

- **Too short**: Stub articles, error pages, redirect notices
- **Too long**: Massive data tables, auto-generated listings
- **Low quality**: Garbled text, OCR errors, machine-generated spam
- **Wrong format**: Lists of links, navigation-only pages, cookie notices
- **Repetitive**: Documents with excessive repeated phrases or characters

NeMo Curator provides two categories of filters:

1. **Heuristic filters** (CPU) -- Rule-based checks on document statistics
2. **Classifier filters** (GPU) -- Neural models that score document quality

## Step 1: Understand the Filtering Pipeline

```
Deduplicated corpus
       |
       v
+---------------------------+
| Heuristic Filters (CPU)   |
|  - Word count range        |
|  - Mean word length        |
|  - Symbol-to-word ratio    |
|  - Bullet/ellipsis ratio   |
|  - Repeated line fraction  |
|  - Top n-gram fraction     |
+---------------------------+
       |
       v
+---------------------------+
| Classifier Filters (GPU)  |
|  - Quality classifier      |
|  - Perplexity filter       |
+---------------------------+
       |
       v
Quality-filtered corpus
```

## Step 2: Heuristic Filters

Heuristic filters apply fast, rule-based checks. These are inspired by the filtering criteria used for datasets like C4, RefinedWeb, and The Pile.

### Filter Definitions

| Filter | Condition | Rationale |
|--------|-----------|-----------|
| Word count (min) | >= 50 words | Remove stubs, error pages |
| Word count (max) | <= 100,000 words | Remove data dumps, generated lists |
| Mean word length | 3-10 characters | Remove garbled text, code-heavy pages |
| Symbol-to-word ratio | < 0.1 | Remove pages heavy with symbols/special chars |
| Bullet line ratio | < 0.9 | Remove pages that are entirely bullet lists |
| Ellipsis line ratio | < 0.3 | Remove truncated/clickbait content |
| Repeated line fraction | < 0.3 | Remove boilerplate-heavy pages |
| Top 2-gram fraction | < 0.2 | Remove repetitive text (SEO spam, auto-generated) |
| Top 3-gram fraction | < 0.18 | Additional repetition check |
| Top 4-gram fraction | < 0.16 | Additional repetition check |

### NeMo Curator API for Heuristic Filters

```python
from nemo_curator.filters import (
    WordCountFilter,
    MeanWordLengthFilter,
    SymbolsToWordsFilter,
    BulletsFilter,
    EllipsisFilter,
    RepeatedLinesFilter,
    RepeatedParagraphsFilter,
    TopNGramsFilter,
)
from nemo_curator import Sequential, ScoreFilter

heuristic_pipeline = Sequential([
    ScoreFilter(WordCountFilter(min_words=50, max_words=100000)),
    ScoreFilter(MeanWordLengthFilter(min_mean=3.0, max_mean=10.0)),
    ScoreFilter(SymbolsToWordsFilter(max_ratio=0.1)),
    ScoreFilter(BulletsFilter(max_ratio=0.9)),
    ScoreFilter(EllipsisFilter(max_ratio=0.3)),
    ScoreFilter(RepeatedLinesFilter(max_fraction=0.3)),
    ScoreFilter(TopNGramsFilter(n=2, max_fraction=0.2)),
    ScoreFilter(TopNGramsFilter(n=3, max_fraction=0.18)),
    ScoreFilter(TopNGramsFilter(n=4, max_fraction=0.16)),
])
```

## Step 3: Classifier-Based Filters

For documents that pass heuristic checks, we apply a neural quality classifier. NeMo Curator includes a pre-trained quality classifier that scores documents on a scale of 0 (low quality) to 1 (high quality).

### Quality Classifier

The classifier is based on a fine-tuned language model that has been trained to distinguish high-quality text (Wikipedia, books, curated articles) from low-quality text (spam, auto-generated, garbled).

```python
from nemo_curator.classifiers import QualityClassifier

quality_classifier = QualityClassifier(
    model_path="nemo_curator/classifiers/quality",
    batch_size=64,
    text_field="text",
    pred_column="quality_score",
    max_chars=2000,    # Classify based on first 2000 chars
)

# This runs on GPU automatically if available
scored_dataset = quality_classifier(dataset)

# Filter to keep only high-quality documents
high_quality = scored_dataset.filter(lambda row: row["quality_score"] >= 0.7)
```

### Perplexity Filter (Optional)

A KenLM-based perplexity filter can identify documents that are statistically unusual compared to a reference corpus:

```python
from nemo_curator.filters import PerplexityFilter

perplexity_filter = PerplexityFilter(
    model_path="kenlm_models/en.arpa.bin",
    max_perplexity=1500,    # Remove very high perplexity (garbled text)
    min_perplexity=10,      # Remove very low perplexity (repetitive text)
)
```

## Step 4: Create the Filter Job Manifest

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: nemo-curator-filter
  namespace: nemo-curator-workshop
spec:
  backoffLimit: 2
  template:
    metadata:
      labels:
        app: nemo-curator
        stage: filter
    spec:
      restartPolicy: OnFailure
      containers:
        - name: filter
          image: nvcr.io/nvidia/nemo-curator:v0.6.0
          command:
            - python
            - /scripts/filter.py
            - --input-dir
            - /data/fuzzy-deduped
            - --output-dir
            - /data/filtered
            - --min-words
            - "50"
            - --max-words
            - "100000"
            - --quality-threshold
            - "0.7"
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
            name: filter-scripts
```

## Step 5: Submit and Monitor

```bash
kubectl apply -f filter-job.yaml

kubectl logs -f job/nemo-curator-filter -n nemo-curator-workshop

kubectl wait --for=condition=complete job/nemo-curator-filter \
  -n nemo-curator-workshop --timeout=900s
```

## Step 6: Verify Filtering Results

Inspect the output and review filtering statistics:

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
echo "=== Document counts ==="
echo -n "Before filtering: "; wc -l /data/fuzzy-deduped/*.jsonl | tail -1
echo -n "After filtering:  "; wc -l /data/filtered/*.jsonl | tail -1

echo ""
echo "=== Quality score distribution ==="
python3 -c "
import json, sys
scores = []
with open('/data/filtered/filter_scores.jsonl') as f:
    for line in f:
        doc = json.loads(line)
        scores.append(doc.get('quality_score', 0))

print(f'Documents scored: {len(scores)}')
print(f'Mean quality score: {sum(scores)/len(scores):.3f}')
print(f'Min: {min(scores):.3f}, Max: {max(scores):.3f}')

# Histogram
buckets = [0]*10
for s in scores:
    b = min(int(s * 10), 9)
    buckets[b] += 1
for i, count in enumerate(buckets):
    bar = '#' * (count * 40 // max(buckets)) if max(buckets) > 0 else ''
    print(f'  {i/10:.1f}-{(i+1)/10:.1f}: {bar} ({count})')
"
```

## Step 7: Review Per-Filter Removal Statistics

The filter script logs how many documents each filter removed:

```bash
kubectl logs job/nemo-curator-filter -n nemo-curator-workshop | grep -E "Filter|Removed|Remaining"
```

**Expected output example:**

```
WordCountFilter: Removed 23 documents (2.8%)
MeanWordLengthFilter: Removed 5 documents (0.6%)
SymbolsToWordsFilter: Removed 2 documents (0.2%)
RepeatedLinesFilter: Removed 8 documents (1.0%)
TopNGramsFilter(2): Removed 12 documents (1.5%)
QualityClassifier: Removed 45 documents (5.7%)
---
Total removed: 95 documents (11.8%)
Remaining: 705 documents
```

## Step 8: Examine Edge Cases

Look at documents that were just above and just below the quality threshold to validate the filtering:

```bash
python3 -c "
import json

# Load scored documents
docs = []
with open('/data/filtered/filter_scores.jsonl') as f:
    for line in f:
        docs.append(json.loads(line))

# Sort by quality score
docs.sort(key=lambda d: d.get('quality_score', 0))

print('=== Lowest quality docs that PASSED (just above threshold) ===')
passed = [d for d in docs if d.get('quality_score', 0) >= 0.7]
for d in passed[:3]:
    print(f'Score: {d[\"quality_score\"]:.3f} | {d[\"text\"][:150]}...')
    print()

print('=== Highest quality docs that FAILED (just below threshold) ===')
failed = [d for d in docs if d.get('quality_score', 0) < 0.7]
for d in failed[-3:]:
    print(f'Score: {d[\"quality_score\"]:.3f} | {d[\"text\"][:150]}...')
    print()
"
```

This manual inspection helps validate that the threshold is appropriate for TechPulse Media's use case.

## Expected Outputs

| Artifact | Location | Description |
|----------|----------|-------------|
| Filtered JSONL | `/data/filtered/*.jsonl` | Documents passing all quality filters |
| Filter scores | `/data/filtered/filter_scores.jsonl` | All documents with their quality scores |
| Filter stats | Job logs | Per-filter removal counts |

## Verification Checklist

- [ ] Filter job completed successfully
- [ ] Document count reduced by 10-30% compared to deduped corpus
- [ ] Quality scores present on all documents
- [ ] No extremely short (<50 words) or extremely long (>100K words) documents remain
- [ ] Manual inspection of edge cases shows reasonable filtering decisions
- [ ] GPU was used for classifier-based filtering (check logs for CUDA references)

## Next Step

Proceed to [05 - Classify](../05-classify/README.md) to apply domain classification and quality scoring.
