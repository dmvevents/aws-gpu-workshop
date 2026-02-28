# 06 - Export and Validation

## Objective

Export the curated dataset to JSONL format suitable for LLM training, run validation checks to ensure data integrity, generate summary statistics, and upload the final dataset to S3 or HuggingFace Hub.

## Background

TechPulse Media's curation pipeline is complete. The corpus has been downloaded, extracted, deduplicated, filtered, and classified. This final stage prepares the dataset for consumption by the training pipeline:

1. Select and rename fields for the training format
2. Apply final sampling/balancing if needed
3. Validate data integrity (no nulls, no truncation, schema consistency)
4. Generate a dataset card with statistics
5. Upload to S3 and/or HuggingFace Hub

## Step 1: Define the Export Schema

The training pipeline expects documents in a specific format. Map the curation fields to the training schema:

| Curation Field | Training Field | Required |
|----------------|---------------|----------|
| `text` | `text` | Yes |
| `title` | `metadata.title` | No |
| `url` | `metadata.source_url` | No |
| `domain` | `metadata.domain` | No |
| `quality_prob` | `metadata.quality_score` | No |
| `word_count` | `metadata.word_count` | No |

### Target Export Format

```json
{
  "text": "Full article text for training...",
  "metadata": {
    "title": "Article Title",
    "source_url": "https://...",
    "domain": "artificial_intelligence",
    "quality_score": 0.87,
    "word_count": 782,
    "source": "wikipedia",
    "curation_pipeline": "nemo-curator-v0.6.0",
    "curation_date": "2024-01-15"
  }
}
```

## Step 2: Create the Export Job Manifest

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: nemo-curator-export
  namespace: nemo-curator-workshop
spec:
  backoffLimit: 2
  template:
    metadata:
      labels:
        app: nemo-curator
        stage: export
    spec:
      restartPolicy: OnFailure
      containers:
        - name: exporter
          image: nvcr.io/nvidia/nemo-curator:v0.6.0
          command:
            - python
            - /scripts/export.py
            - --input-dir
            - /data/classified
            - --output-dir
            - /data/export
            - --format
            - jsonl
            - --split-ratio
            - "0.95,0.03,0.02"
            - --seed
            - "42"
          resources:
            requests:
              cpu: "2"
              memory: "8Gi"
            limits:
              cpu: "4"
              memory: "16Gi"
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
            name: export-scripts
```

## Step 3: Submit and Monitor

```bash
kubectl apply -f export-job.yaml

kubectl logs -f job/nemo-curator-export -n nemo-curator-workshop

kubectl wait --for=condition=complete job/nemo-curator-export \
  -n nemo-curator-workshop --timeout=300s
```

## Step 4: Verify Exported Files

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
echo "=== Exported files ==="
ls -lh /data/export/

echo ""
echo "=== Split sizes ==="
for split in train validation test; do
  if [ -f "/data/export/${split}.jsonl" ]; then
    count=$(wc -l < "/data/export/${split}.jsonl")
    size=$(du -h "/data/export/${split}.jsonl" | cut -f1)
    echo "  ${split}: ${count} documents (${size})"
  fi
done
```

**Expected output:**
```
=== Exported files ===
train.jsonl
validation.jsonl
test.jsonl
dataset_stats.json

=== Split sizes ===
  train: ~670 documents
  validation: ~21 documents
  test: ~14 documents
```

## Step 5: Validate Data Integrity

Run comprehensive validation checks on the exported dataset:

```bash
python3 -c "
import json, sys

def validate_split(filepath, split_name):
    errors = []
    doc_count = 0
    text_lengths = []

    with open(filepath) as f:
        for line_num, line in enumerate(f, 1):
            doc_count += 1

            # Check valid JSON
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f'Line {line_num}: Invalid JSON')
                continue

            # Check required fields
            if 'text' not in doc:
                errors.append(f'Line {line_num}: Missing \"text\" field')
                continue

            text = doc['text']

            # Check text is non-empty
            if not text or not text.strip():
                errors.append(f'Line {line_num}: Empty text')
                continue

            # Check text is string
            if not isinstance(text, str):
                errors.append(f'Line {line_num}: text is {type(text).__name__}, expected str')
                continue

            # Check for null bytes
            if '\x00' in text:
                errors.append(f'Line {line_num}: Contains null bytes')

            # Check metadata exists
            if 'metadata' not in doc:
                errors.append(f'Line {line_num}: Missing metadata')

            text_lengths.append(len(text))

    # Report
    print(f'--- {split_name} ---')
    print(f'Documents: {doc_count}')
    print(f'Errors: {len(errors)}')
    if errors:
        for e in errors[:5]:
            print(f'  ! {e}')
        if len(errors) > 5:
            print(f'  ... and {len(errors) - 5} more')
    if text_lengths:
        avg_len = sum(text_lengths) / len(text_lengths)
        print(f'Avg text length: {avg_len:.0f} chars')
        print(f'Min: {min(text_lengths)}, Max: {max(text_lengths)}')
    print()

    return len(errors) == 0

all_valid = True
for split in ['train', 'validation', 'test']:
    filepath = f'/data/export/{split}.jsonl'
    try:
        valid = validate_split(filepath, split)
        all_valid = all_valid and valid
    except FileNotFoundError:
        print(f'--- {split} --- FILE NOT FOUND')
        all_valid = False

if all_valid:
    print('ALL SPLITS PASSED VALIDATION')
else:
    print('VALIDATION FAILED - see errors above')
    sys.exit(1)
"
```

## Step 6: Generate Dataset Statistics

Create a comprehensive statistics report:

```bash
python3 -c "
import json
from collections import Counter

stats = {
    'total_documents': 0,
    'splits': {},
    'domain_distribution': Counter(),
    'quality_distribution': {'high': 0, 'medium': 0, 'low': 0},
    'text_length_stats': {},
}

for split in ['train', 'validation', 'test']:
    filepath = f'/data/export/{split}.jsonl'
    try:
        docs = []
        with open(filepath) as f:
            for line in f:
                docs.append(json.loads(line))

        text_lengths = [len(d['text']) for d in docs]
        word_counts = [d.get('metadata', {}).get('word_count', 0) for d in docs]

        stats['splits'][split] = {
            'count': len(docs),
            'avg_text_length': sum(text_lengths) / len(text_lengths) if text_lengths else 0,
            'total_chars': sum(text_lengths),
            'total_words': sum(word_counts),
        }
        stats['total_documents'] += len(docs)

        for doc in docs:
            meta = doc.get('metadata', {})
            domain = meta.get('domain', 'unknown')
            stats['domain_distribution'][domain] += 1

            qs = meta.get('quality_score', 0)
            if qs >= 0.8:
                stats['quality_distribution']['high'] += 1
            elif qs >= 0.6:
                stats['quality_distribution']['medium'] += 1
            else:
                stats['quality_distribution']['low'] += 1

    except FileNotFoundError:
        pass

# Convert Counter to dict for JSON serialization
stats['domain_distribution'] = dict(stats['domain_distribution'])

# Write stats file
with open('/data/export/dataset_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

# Print summary
print(json.dumps(stats, indent=2))
"
```

## Step 7: Upload to S3

Sync the exported dataset to S3:

```bash
kubectl run s3-upload --rm -it --restart=Never \
  --image=amazon/aws-cli:latest \
  --env="AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}" \
  --env="AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}" \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "s3-upload",
        "image": "amazon/aws-cli:latest",
        "command": ["aws", "s3", "sync", "/data/export/", "'"${S3_BUCKET}"'/curated/"],
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

Verify the upload:

```bash
aws s3 ls ${S3_BUCKET}/curated/ --human-readable
```

**Expected output:**
```
2024-01-15 10:45:00  2.1 MiB train.jsonl
2024-01-15 10:45:00   68 KiB validation.jsonl
2024-01-15 10:45:00   45 KiB test.jsonl
2024-01-15 10:45:00  1.2 KiB dataset_stats.json
```

## Step 8: (Optional) Upload to HuggingFace Hub

If TechPulse Media wants to share the curated dataset on HuggingFace:

```bash
# Inside a pod with huggingface_hub installed
pip install huggingface_hub

python3 -c "
from huggingface_hub import HfApi, login

# Login with your HF token
login(token='hf_YOUR_TOKEN_HERE')

api = HfApi()

# Create the dataset repo
api.create_repo(
    repo_id='techpulse-media/tech-news-curated',
    repo_type='dataset',
    private=True,
)

# Upload the files
for split_file in ['train.jsonl', 'validation.jsonl', 'test.jsonl', 'dataset_stats.json']:
    api.upload_file(
        path_or_fileobj=f'/data/export/{split_file}',
        path_in_repo=split_file,
        repo_id='techpulse-media/tech-news-curated',
        repo_type='dataset',
    )

print('Dataset uploaded to HuggingFace Hub')
"
```

## Step 9: Pipeline Summary

Review the complete pipeline metrics:

```bash
python3 -c "
stages = [
    ('01 - Download',    'raw/wikipedia',   'Raw web documents'),
    ('02 - Extract',     'extracted',        'After text extraction + lang ID'),
    ('03a - Exact Dedup', 'exact-deduped',   'After hash-based dedup'),
    ('03b - Fuzzy Dedup', 'fuzzy-deduped',   'After MinHash LSH dedup'),
    ('04 - Filter',      'filtered',         'After quality filtering'),
    ('05 - Classify',    'classified',       'After domain classification'),
    ('06 - Export',      'export',           'Final curated dataset'),
]

print('=' * 70)
print('TECHPULSE MEDIA - DATA CURATION PIPELINE SUMMARY')
print('=' * 70)
print(f'{\"Stage\":<25s} {\"Documents\":>10s} {\"Reduction\":>10s}')
print('-' * 70)

prev_count = None
for stage_name, data_dir, description in stages:
    import glob
    files = glob.glob(f'/data/{data_dir}/*.jsonl')
    count = 0
    for f in files:
        with open(f) as fh:
            count += sum(1 for _ in fh)

    reduction = ''
    if prev_count is not None and prev_count > 0:
        pct = (prev_count - count) * 100 / prev_count
        reduction = f'-{pct:.1f}%'

    print(f'{stage_name:<25s} {count:>10,d} {reduction:>10s}   {description}')
    prev_count = count

print('-' * 70)
print('Pipeline complete. Dataset ready for LLM training.')
"
```

**Expected output example:**

```
======================================================================
TECHPULSE MEDIA - DATA CURATION PIPELINE SUMMARY
======================================================================
Stage                      Documents  Reduction
----------------------------------------------------------------------
01 - Download                  1,000              Raw web documents
02 - Extract                     945      -5.5%   After text extraction + lang ID
03a - Exact Dedup                920      -2.6%   After hash-based dedup
03b - Fuzzy Dedup                800     -13.0%   After MinHash LSH dedup
04 - Filter                      705     -11.9%   After quality filtering
05 - Classify                    705      -0.0%   After domain classification
06 - Export                      705      -0.0%   Final curated dataset
----------------------------------------------------------------------
Pipeline complete. Dataset ready for LLM training.
```

## Expected Outputs

| Artifact | Location | Description |
|----------|----------|-------------|
| `train.jsonl` | `/data/export/train.jsonl` | Training split (~95% of data) |
| `validation.jsonl` | `/data/export/validation.jsonl` | Validation split (~3%) |
| `test.jsonl` | `/data/export/test.jsonl` | Test split (~2%) |
| `dataset_stats.json` | `/data/export/dataset_stats.json` | Full statistics report |
| S3 copy | `s3://<bucket>/curated/` | Cloud backup of all exports |

## Verification Checklist

- [ ] Export job completed successfully
- [ ] Three split files exist: train, validation, test
- [ ] All documents pass JSON validation (no parse errors)
- [ ] All documents have non-empty `text` field
- [ ] All documents have `metadata` with expected fields
- [ ] Dataset statistics file generated and reviewed
- [ ] Data uploaded to S3
- [ ] Pipeline summary shows expected reduction at each stage

## Cleanup

When you are finished with the workshop, clean up the resources:

```bash
# Delete all jobs
kubectl delete jobs --all -n nemo-curator-workshop

# Delete the PVC (WARNING: this deletes all pipeline data)
kubectl delete pvc nemo-curator-data -n nemo-curator-workshop

# Delete the namespace
kubectl delete namespace nemo-curator-workshop
```

## Congratulations

You have completed the NeMo Curator data curation workshop. You now know how to:

- Download and extract web content at scale
- Remove exact and near-duplicate documents using GPU-accelerated MinHash LSH
- Apply heuristic and neural quality filters
- Classify documents by topic domain
- Export validated datasets for LLM training
- Run the entire pipeline on Kubernetes with GPU scheduling

### What is Next

- **Scale up**: Process millions of documents by adding more GPU workers and using Dask distributed scheduling
- **Customize filters**: Write domain-specific heuristic filters for your use case
- **Train a classifier**: Fine-tune the domain classifier on your own labeled data
- **Integrate with training**: Feed the curated dataset directly into NeMo Framework or HuggingFace TRL for LLM fine-tuning
- **Automate**: Wrap the pipeline in an Argo Workflow or Kubeflow Pipeline for scheduled curation runs
