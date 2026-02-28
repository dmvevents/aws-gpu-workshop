# 01 - Data Download

## Objective

Download a raw web corpus that TechPulse Media will curate for LLM fine-tuning. You will use NeMo Curator's built-in download utilities to fetch either Wikipedia dumps or Common Crawl samples, then store the raw data on shared storage for downstream processing.

## Background

TechPulse Media aggregates technology news from across the web. For this workshop, we simulate their raw corpus using publicly available data sources:

- **Wikipedia** (recommended for workshop): Well-structured, moderate size, fast to download
- **Common Crawl**: Realistic web-scale data, but larger and slower to process

We will download a Wikipedia snapshot and treat it as TechPulse Media's incoming content feed.

## Step 1: Create the Download Job Manifest

Create a Kubernetes Job that runs NeMo Curator's Wikipedia download utility. The job will fetch a recent Wikipedia dump and store it on the shared PVC.

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: nemo-curator-download
  namespace: nemo-curator-workshop
spec:
  backoffLimit: 2
  template:
    metadata:
      labels:
        app: nemo-curator
        stage: download
    spec:
      restartPolicy: OnFailure
      containers:
        - name: downloader
          image: nvcr.io/nvidia/nemo-curator:v0.6.0
          command:
            - python
            - -m
            - nemo_curator.scripts.download_wikipedia
            - --output-dir
            - /data/raw/wikipedia
            - --language
            - en
            - --dump-date
            - "20240101"
            - --url-limit
            - "1000"           # Limit to 1000 articles for workshop speed
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
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: nemo-curator-data
```

> **Note:** We use `--url-limit 1000` to keep the download fast for the workshop. In production, TechPulse Media would download the full dump or use their own crawl data.

## Step 2: Submit the Download Job

```bash
kubectl apply -f download-job.yaml
```

Monitor the job progress:

```bash
kubectl logs -f job/nemo-curator-download -n nemo-curator-workshop
```

Wait for the job to complete:

```bash
kubectl wait --for=condition=complete job/nemo-curator-download \
  -n nemo-curator-workshop --timeout=600s
```

## Step 3: Verify Downloaded Data

Launch a temporary pod to inspect the downloaded files:

```bash
kubectl run data-inspector --rm -it --restart=Never \
  --image=nvcr.io/nvidia/nemo-curator:v0.6.0 \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "data-inspector",
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

Inside the pod, verify the data:

```bash
# Count downloaded files
find /data/raw/wikipedia -name "*.jsonl" | wc -l

# Check total line count (each line = one document)
wc -l /data/raw/wikipedia/*.jsonl

# Inspect the first record
head -1 /data/raw/wikipedia/*.jsonl | python3 -m json.tool
```

**Expected output:**
- One or more `.jsonl` files in `/data/raw/wikipedia/`
- Approximately 1,000 document records
- Each record should contain at minimum a `text` field and a `url` or `title` field

## Step 4: Inspect Data Format

Each downloaded document should have this structure:

```json
{
  "text": "Full article text content...",
  "title": "Article Title",
  "url": "https://en.wikipedia.org/wiki/...",
  "language": "en",
  "source": "wikipedia",
  "download_date": "2024-01-01"
}
```

Check that all expected fields are present:

```bash
head -5 /data/raw/wikipedia/*.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    doc = json.loads(line)
    print(f'Fields: {sorted(doc.keys())}')
    print(f'Text length: {len(doc.get(\"text\", \"\"))} chars')
    print('---')
"
```

## Step 5: (Alternative) Using Common Crawl

If you prefer a more realistic web corpus, you can download Common Crawl WARC files instead. Replace the download command with:

```yaml
command:
  - python
  - -m
  - nemo_curator.scripts.download_common_crawl
  - --output-dir
  - /data/raw/common_crawl
  - --start-snapshot
  - "2024-01"
  - --end-snapshot
  - "2024-01"
  - --url-limit
  - "5000"
```

> **Warning:** Common Crawl downloads are significantly larger. Allow extra time and storage for this option.

## Step 6: Backup Raw Data to S3

Back up the raw data to S3 so it can be restored if the PVC is lost:

```bash
kubectl run s3-sync --rm -it --restart=Never \
  --image=amazon/aws-cli:latest \
  --env="AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}" \
  --env="AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}" \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "s3-sync",
        "image": "amazon/aws-cli:latest",
        "command": ["aws", "s3", "sync", "/data/raw/", "'"${S3_BUCKET}"'/raw/"],
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

## Expected Outputs

| Artifact | Location | Description |
|----------|----------|-------------|
| Raw JSONL files | `/data/raw/wikipedia/*.jsonl` | ~1,000 Wikipedia articles |
| S3 backup | `s3://<bucket>/nemo-curator-workshop/raw/` | Copy of raw data |

## Verification Checklist

- [ ] Download job completed successfully (`kubectl get job`)
- [ ] JSONL files exist on the PVC at `/data/raw/wikipedia/`
- [ ] Each record contains `text`, `title`, and `url` fields
- [ ] Approximately 1,000 documents downloaded
- [ ] Raw data backed up to S3

## Next Step

Proceed to [02 - Extract](../02-extract/README.md) to extract and clean the text content.
