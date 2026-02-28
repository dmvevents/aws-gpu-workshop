# 00 - Setup and Prerequisites

## Objective

Verify that your EKS cluster is properly configured for running NeMo Curator GPU workloads, create the workshop namespace, and provision shared storage for the pipeline.

## Step 1: Verify Cluster Access

Confirm that kubectl can reach your EKS cluster and that you have admin permissions:

```bash
kubectl cluster-info
kubectl get nodes -o wide
```

You should see at least two nodes with GPU capacity. Verify GPU resources are schedulable:

```bash
kubectl get nodes -l nvidia.com/gpu.present=true \
  -o custom-columns="NAME:.metadata.name,GPU:.status.capacity.nvidia\.com/gpu,INSTANCE:.metadata.labels.node\.kubernetes\.io/instance-type"
```

**Expected output:** At least 2 nodes showing `nvidia.com/gpu: 1` (or more) with instance types like `g5.xlarge`, `g5.2xlarge`, or `p4d.24xlarge`.

## Step 2: Verify NVIDIA GPU Operator

Confirm the NVIDIA GPU Operator is running and all device plugin pods are healthy:

```bash
kubectl get pods -n gpu-operator --no-headers | head -20
```

All pods should be in `Running` or `Completed` state. If you see `CrashLoopBackOff` or `Pending`, consult the GPU Operator troubleshooting guide before proceeding.

Run a quick GPU smoke test:

```bash
kubectl run gpu-test --rm -it --restart=Never \
  --image=nvcr.io/nvidia/cuda:12.4.0-base-ubuntu22.04 \
  --limits=nvidia.com/gpu=1 \
  -- nvidia-smi
```

**Expected output:** `nvidia-smi` output showing the GPU model, driver version, and CUDA version.

## Step 3: Create Workshop Namespace

Create a dedicated namespace to keep workshop resources isolated:

```bash
kubectl create namespace nemo-curator-workshop
kubectl label namespace nemo-curator-workshop purpose=workshop
```

Set it as your default context:

```bash
kubectl config set-context --current --namespace=nemo-curator-workshop
```

## Step 4: Create S3 Access Secret

The pipeline reads raw data from and writes curated data to S3. Create a Kubernetes secret with your AWS credentials (or use IRSA if configured):

**Option A: Explicit credentials (dev/workshop only)**

```bash
kubectl create secret generic aws-credentials \
  --namespace=nemo-curator-workshop \
  --from-literal=AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
  --from-literal=AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
  --from-literal=AWS_DEFAULT_REGION="us-east-1"
```

**Option B: IAM Roles for Service Accounts (IRSA) -- recommended for production**

```bash
kubectl create serviceaccount nemo-curator-sa \
  --namespace=nemo-curator-workshop

kubectl annotate serviceaccount nemo-curator-sa \
  --namespace=nemo-curator-workshop \
  eks.amazonaws.com/role-arn=arn:aws:iam::<ACCOUNT_ID>:role/<ROLE_NAME>
```

## Step 5: Provision Shared Storage (PVC)

Create a PersistentVolumeClaim for intermediate pipeline data. This PVC will be mounted by all pipeline stages so they can pass data between steps.

Create the file `pvc.yaml`:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: nemo-curator-data
  namespace: nemo-curator-workshop
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: fsx-lustre-sc    # Use your cluster's RWX storage class
  resources:
    requests:
      storage: 100Gi
```

Apply it:

```bash
kubectl apply -f pvc.yaml
```

Verify the PVC is bound:

```bash
kubectl get pvc nemo-curator-data -n nemo-curator-workshop
```

**Expected output:** STATUS should be `Bound`. If it remains `Pending`, check that your storage class exists (`kubectl get sc`) and that the provisioner is running.

## Step 6: Set Environment Variables

Set the following shell variables that will be used throughout the workshop:

```bash
export NAMESPACE="nemo-curator-workshop"
export S3_BUCKET="s3://<YOUR_BUCKET_NAME>/nemo-curator-workshop"
export PVC_NAME="nemo-curator-data"
export NEMO_CURATOR_IMAGE="nvcr.io/nvidia/nemo-curator:v0.6.0"
export DATA_DIR="/data"   # Mount path inside pods
```

## Step 7: Verify Setup

Run the final verification:

```bash
echo "=== Cluster ==="
kubectl get nodes --no-headers | wc -l

echo "=== GPU Nodes ==="
kubectl get nodes -l nvidia.com/gpu.present=true --no-headers | wc -l

echo "=== Namespace ==="
kubectl get namespace ${NAMESPACE}

echo "=== PVC ==="
kubectl get pvc ${PVC_NAME} -n ${NAMESPACE} -o jsonpath='{.status.phase}'

echo "=== S3 Bucket ==="
aws s3 ls ${S3_BUCKET}/ 2>/dev/null && echo "OK" || echo "Bucket not yet populated (OK for now)"
```

**Expected output:**
- At least 2 total nodes
- At least 2 GPU nodes
- Namespace exists
- PVC status is `Bound`
- S3 bucket is accessible (may be empty)

## Troubleshooting

| Issue | Resolution |
|-------|------------|
| `nvidia.com/gpu` not visible on nodes | Verify GPU Operator is installed: `helm list -n gpu-operator` |
| PVC stuck in `Pending` | Check storage class: `kubectl get sc` and provisioner pods |
| S3 access denied | Verify IAM permissions include `s3:GetObject`, `s3:PutObject`, `s3:ListBucket` |
| `kubectl` connection refused | Run `aws eks update-kubeconfig --name <cluster-name> --region <region>` |

## Next Step

Proceed to [01 - Download](../01-download/README.md) to fetch the raw dataset.
