# 05 - Topic Classification and Quality Scoring

## Objective

Classify the filtered documents by topic domain and assign fine-grained quality scores using NeMo Curator's GPU-accelerated classifiers. This stage ensures TechPulse Media's curated dataset is properly categorized and that the highest-quality documents are prioritized for training.

## Background

TechPulse Media wants their fine-tuned LLM to excel at specific technology domains. Not all articles in the corpus are equally relevant or valuable. This classification stage accomplishes two goals:

1. **Domain classification** -- Tag each document with its primary topic (e.g., "artificial intelligence", "cybersecurity", "cloud computing") so TechPulse Media can control the topic distribution in their training mix
2. **Quality scoring** -- Assign a fine-grained quality score using a reward model, enabling TechPulse Media to prioritize the best content

## Step 1: Understand the Classification Pipeline

```
Quality-filtered corpus
       |
       v
+-----------------------------+
| Domain Classifier (GPU)     |
|  - Multi-label topic tags   |
|  - Confidence scores        |
+-----------------------------+
       |
       v
+-----------------------------+
| Quality Scorer (GPU)        |
|  - Reward model scoring     |
|  - Educational value score  |
+-----------------------------+
       |
       v
+-----------------------------+
| Topic Distribution Report   |
|  - Category counts          |
|  - Score distributions      |
+-----------------------------+
       |
       v
Classified & scored corpus
```

## Step 2: Domain Classification

NeMo Curator provides domain classifiers that can tag documents with topic labels. For TechPulse Media, we define the following technology domains:

| Domain | Description | Examples |
|--------|-------------|----------|
| `artificial_intelligence` | AI, ML, deep learning, LLMs | GPT, training, neural networks |
| `cybersecurity` | Security, privacy, threats | Vulnerabilities, encryption, zero-trust |
| `cloud_computing` | Cloud platforms, infrastructure | AWS, Azure, Kubernetes, serverless |
| `software_engineering` | Dev practices, tools, languages | CI/CD, Rust, microservices |
| `hardware` | Chips, devices, networking | GPUs, ARM, 5G, quantum computing |
| `data_science` | Analytics, data engineering | Pandas, Spark, data pipelines |
| `business_tech` | Enterprise IT, SaaS, startups | Funding, acquisitions, digital transformation |
| `other` | Non-technology content | Catch-all for off-topic articles |

### NeMo Curator API for Domain Classification

```python
from nemo_curator.classifiers import DomainClassifier

domain_classifier = DomainClassifier(
    model_path="nemo_curator/classifiers/domain",
    labels=[
        "artificial_intelligence",
        "cybersecurity",
        "cloud_computing",
        "software_engineering",
        "hardware",
        "data_science",
        "business_tech",
        "other",
    ],
    batch_size=64,
    text_field="text",
    pred_column="domain",
    prob_column="domain_prob",
    max_chars=2000,
)

classified_dataset = domain_classifier(dataset)
```

Each document receives:
- `domain` -- The predicted primary topic label
- `domain_prob` -- Confidence score for the prediction (0.0 to 1.0)

## Step 3: Quality Scoring with Reward Models

Beyond the binary quality filter from the previous stage, we now assign a fine-grained quality score. NeMo Curator supports quality scoring with reward models that evaluate:

- **Writing quality** -- Grammar, coherence, readability
- **Informativeness** -- Depth of content, factual density
- **Educational value** -- How useful the content is for learning

### NeMo Curator API for Quality Scoring

```python
from nemo_curator.classifiers import QualityClassifier

quality_scorer = QualityClassifier(
    model_path="nemo_curator/classifiers/quality",
    batch_size=64,
    text_field="text",
    pred_column="quality_label",    # "high", "medium", "low"
    prob_column="quality_prob",     # Continuous score 0-1
    max_chars=2000,
)

scored_dataset = quality_scorer(classified_dataset)
```

## Step 4: Create the Classification Job Manifest

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: nemo-curator-classify
  namespace: nemo-curator-workshop
spec:
  backoffLimit: 2
  template:
    metadata:
      labels:
        app: nemo-curator
        stage: classify
    spec:
      restartPolicy: OnFailure
      containers:
        - name: classifier
          image: nvcr.io/nvidia/nemo-curator:v0.6.0
          command:
            - python
            - /scripts/classify.py
            - --input-dir
            - /data/filtered
            - --output-dir
            - /data/classified
            - --domain-model
            - nemo_curator/classifiers/domain
            - --quality-model
            - nemo_curator/classifiers/quality
            - --batch-size
            - "64"
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
            name: classify-scripts
```

## Step 5: Submit and Monitor

```bash
kubectl apply -f classify-job.yaml

kubectl logs -f job/nemo-curator-classify -n nemo-curator-workshop

kubectl wait --for=condition=complete job/nemo-curator-classify \
  -n nemo-curator-workshop --timeout=900s
```

## Step 6: Verify Classification Results

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
# Inspect classified documents
head -3 /data/classified/*.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    doc = json.loads(line)
    print(f'Title: {doc.get(\"title\", \"N/A\")}')
    print(f'Domain: {doc.get(\"domain\", \"N/A\")} (confidence: {doc.get(\"domain_prob\", \"N/A\")})')
    print(f'Quality: {doc.get(\"quality_label\", \"N/A\")} (score: {doc.get(\"quality_prob\", \"N/A\")})')
    print(f'Text preview: {doc.get(\"text\", \"\")[:150]}...')
    print('---')
"
```

## Step 7: Analyze Topic Distribution

Understanding the topic distribution helps TechPulse Media decide if they need to adjust their data collection strategy:

```bash
python3 -c "
import json
from collections import Counter

domains = Counter()
quality_by_domain = {}

with open('/data/classified/classified.jsonl') as f:
    for line in f:
        doc = json.loads(line)
        domain = doc.get('domain', 'unknown')
        quality = doc.get('quality_prob', 0)
        domains[domain] += 1
        quality_by_domain.setdefault(domain, []).append(quality)

print('=== Topic Distribution ===')
total = sum(domains.values())
for domain, count in domains.most_common():
    pct = count * 100 / total
    avg_q = sum(quality_by_domain[domain]) / len(quality_by_domain[domain])
    bar = '#' * int(pct)
    print(f'  {domain:30s} {count:5d} ({pct:5.1f}%) avg_quality={avg_q:.3f} {bar}')

print(f'\nTotal documents: {total}')
"
```

**Expected output example:**

```
=== Topic Distribution ===
  software_engineering            185 ( 26.2%) avg_quality=0.823 ##########################
  artificial_intelligence         142 ( 20.1%) avg_quality=0.851 ####################
  business_tech                    98 ( 13.9%) avg_quality=0.789 #############
  cloud_computing                  87 ( 12.3%) avg_quality=0.812 ############
  data_science                     72 ( 10.2%) avg_quality=0.834 ##########
  cybersecurity                    58 (  8.2%) avg_quality=0.798 ########
  hardware                         42 (  5.9%) avg_quality=0.776 #####
  other                            21 (  3.0%) avg_quality=0.712 ###

Total documents: 705
```

## Step 8: Filter by Domain (Optional)

If TechPulse Media wants to focus on specific domains, they can filter at this stage:

```bash
python3 -c "
import json

# Keep only tech-relevant domains (exclude 'other')
target_domains = {
    'artificial_intelligence',
    'cybersecurity',
    'cloud_computing',
    'software_engineering',
    'hardware',
    'data_science',
    'business_tech',
}

kept = 0
removed = 0

with open('/data/classified/classified.jsonl') as fin, \
     open('/data/classified/tech_only.jsonl', 'w') as fout:
    for line in fin:
        doc = json.loads(line)
        if doc.get('domain') in target_domains:
            fout.write(line)
            kept += 1
        else:
            removed += 1

print(f'Kept: {kept}, Removed (off-topic): {removed}')
"
```

## Expected Outputs

| Artifact | Location | Description |
|----------|----------|-------------|
| Classified JSONL | `/data/classified/classified.jsonl` | Documents with domain + quality labels |
| Tech-only JSONL | `/data/classified/tech_only.jsonl` | (Optional) Domain-filtered subset |
| Classification logs | Job logs | Timing and batch processing stats |

## Document Schema After Classification

Each document now has the complete set of fields:

```json
{
  "text": "Clean article text...",
  "title": "Article Title",
  "url": "https://...",
  "source": "wikipedia",
  "language": "en",
  "language_score": 0.98,
  "char_count": 4523,
  "word_count": 782,
  "quality_score": 0.85,
  "domain": "artificial_intelligence",
  "domain_prob": 0.92,
  "quality_label": "high",
  "quality_prob": 0.87
}
```

## Verification Checklist

- [ ] Classification job completed successfully
- [ ] All documents have `domain` and `domain_prob` fields
- [ ] All documents have `quality_label` and `quality_prob` fields
- [ ] Topic distribution is reasonable (no single domain >50% unless expected)
- [ ] High-confidence classifications (domain_prob > 0.8) match their content on manual inspection
- [ ] GPU was used for classification (check logs)

## Next Step

Proceed to [06 - Export](../06-export/README.md) to export the curated dataset and validate it for training.
