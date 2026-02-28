# 02 - Text Extraction and Language Identification

## Objective

Extract clean text from the raw downloaded documents, identify the language of each document, and perform initial text cleaning. This stage transforms noisy raw web content into standardized, language-tagged text documents ready for deduplication.

## Background

TechPulse Media's raw corpus contains documents with varying levels of noise: HTML remnants, boilerplate navigation text, encoding artifacts, and non-English content. This extraction phase normalizes the corpus by:

1. Stripping any residual HTML/markup
2. Normalizing Unicode and whitespace
3. Identifying the language of each document
4. Filtering out non-English documents
5. Adding metadata fields for downstream processing

## Step 1: Understand the Extraction Pipeline

NeMo Curator provides several text extraction and cleaning utilities:

- **`DocumentExtractor`** -- Strips HTML tags, scripts, and styles from raw content
- **`UnicodeReformatter`** -- Normalizes Unicode characters (NFKC normalization)
- **`LanguageIdentifier`** -- Uses fastText-based language identification models

The pipeline runs as a Dask-based distributed job. For this workshop, we use a single-node configuration; in production, you would scale to multiple workers.

## Step 2: Create the Extraction Job Manifest

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: nemo-curator-extract
  namespace: nemo-curator-workshop
spec:
  backoffLimit: 2
  template:
    metadata:
      labels:
        app: nemo-curator
        stage: extract
    spec:
      restartPolicy: OnFailure
      containers:
        - name: extractor
          image: nvcr.io/nvidia/nemo-curator:v0.6.0
          command:
            - python
            - /scripts/extract.py
            - --input-dir
            - /data/raw/wikipedia
            - --output-dir
            - /data/extracted
            - --language
            - en
            - --min-lang-score
            - "0.7"
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
            name: extract-scripts
```

> **Note:** The `extract.py` script will be provided by the pipeline scripts agent. It uses NeMo Curator's `DocumentExtractor`, `UnicodeReformatter`, and `LanguageIdentifier` classes.

## Step 3: What the Extraction Script Does

The extraction script performs these operations in sequence:

```
Raw JSONL documents
       |
       v
+----------------------+
| 1. Load documents    |  Read JSONL into Dask DataFrame
+----------------------+
       |
       v
+----------------------+
| 2. HTML extraction   |  Remove tags, scripts, styles, boilerplate
+----------------------+
       |
       v
+----------------------+
| 3. Unicode normalize |  NFKC normalization, fix encoding artifacts
+----------------------+
       |
       v
+----------------------+
| 4. Whitespace clean  |  Collapse runs, strip leading/trailing, normalize newlines
+----------------------+
       |
       v
+----------------------+
| 5. Language ID       |  fastText model, keep only docs with lang_score >= 0.7
+----------------------+
       |
       v
+----------------------+
| 6. Add metadata      |  char_count, word_count, extraction_timestamp
+----------------------+
       |
       v
Cleaned JSONL documents
```

### Core NeMo Curator APIs Used

```python
from nemo_curator import Sequential
from nemo_curator.modules import ExactDuplicates
from nemo_curator.filters import WordCountFilter
from nemo_curator.utils.distributed_utils import get_client

# Example: building an extraction pipeline
pipeline = Sequential([
    DocumentExtractor(),
    UnicodeReformatter(),
])

dataset = pipeline(dataset)
```

## Step 4: Submit the Extraction Job

```bash
kubectl apply -f extract-job.yaml
```

Monitor progress:

```bash
kubectl logs -f job/nemo-curator-extract -n nemo-curator-workshop
```

Wait for completion:

```bash
kubectl wait --for=condition=complete job/nemo-curator-extract \
  -n nemo-curator-workshop --timeout=600s
```

## Step 5: Verify Extracted Data

Inspect the output:

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
# Count extracted documents
wc -l /data/extracted/*.jsonl

# Check that language scores are present
head -3 /data/extracted/*.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    doc = json.loads(line)
    print(f'Language: {doc.get(\"language\", \"N/A\")}')
    print(f'Lang score: {doc.get(\"language_score\", \"N/A\")}')
    print(f'Word count: {doc.get(\"word_count\", \"N/A\")}')
    print(f'Text preview: {doc.get(\"text\", \"\")[:200]}')
    print('---')
"

# Check for any remaining HTML artifacts
grep -c '<[a-zA-Z]' /data/extracted/*.jsonl || echo "No HTML tags found (good)"
```

## Step 6: Review Extraction Statistics

The extraction script outputs summary statistics at the end of its run. Check the logs:

```bash
kubectl logs job/nemo-curator-extract -n nemo-curator-workshop | tail -20
```

**Expected statistics:**
- Input documents: ~1,000
- Documents after language filter: ~900-950 (some may be filtered as non-English)
- Average document length: 2,000-5,000 characters
- Language distribution: >95% English

## Step 7: Inspect the Data Schema

After extraction, each document should have the following schema:

```json
{
  "text": "Clean extracted text content...",
  "title": "Article Title",
  "url": "https://en.wikipedia.org/wiki/...",
  "source": "wikipedia",
  "language": "en",
  "language_score": 0.98,
  "char_count": 4523,
  "word_count": 782,
  "extraction_timestamp": "2024-01-15T10:30:00Z"
}
```

Key fields added during extraction:
- `language` -- ISO 639-1 language code detected by fastText
- `language_score` -- Confidence of language detection (0.0 to 1.0)
- `char_count` -- Number of characters in cleaned text
- `word_count` -- Number of whitespace-delimited tokens

## Expected Outputs

| Artifact | Location | Description |
|----------|----------|-------------|
| Extracted JSONL | `/data/extracted/*.jsonl` | Cleaned, language-tagged documents |
| Extraction logs | Job logs | Statistics on document counts and filtering |

## Verification Checklist

- [ ] Extraction job completed successfully
- [ ] Output JSONL files exist at `/data/extracted/`
- [ ] All documents contain `language`, `language_score`, `word_count` fields
- [ ] No residual HTML tags in text content
- [ ] Non-English documents filtered out (language_score < 0.7 removed)
- [ ] Document count is ~900-950 (slight reduction from filtering)

## Next Step

Proceed to [03 - Dedup](../03-dedup/README.md) to remove duplicate and near-duplicate documents.
