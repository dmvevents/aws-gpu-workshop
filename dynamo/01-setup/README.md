# Module 1: EKS Cluster Setup for Dynamo

**Time:** 15 minutes

## Prerequisites

An EKS cluster with GPU node pools. This workshop uses P5.48xlarge nodes with
EFA networking on SageMaker HyperPod EKS.

## Step 1: Install Dynamo Operator

The Dynamo operator manages DynamoGraphDeployment (DGD) custom resources that
orchestrate frontend, prefill, and decode workers.

```bash
export RELEASE_VERSION=0.9.0
export NAMESPACE=default

# Install CRDs
helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-crds-${RELEASE_VERSION}.tgz
helm install dynamo-crds dynamo-crds-${RELEASE_VERSION}.tgz --namespace ${NAMESPACE}

# Install platform (includes etcd + NATS)
helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-${RELEASE_VERSION}.tgz
helm install dynamo-platform dynamo-platform-${RELEASE_VERSION}.tgz --namespace ${NAMESPACE}

# Wait for readiness
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=dynamo-platform \
  -n ${NAMESPACE} --timeout=300s
```

## Step 2: Verify Platform Components

```bash
kubectl get pods | grep dynamo-platform
# Expected:
#   dynamo-platform-dynamo-operator-controller-manager-xxx   2/2     Running
#   dynamo-platform-etcd-0                                   1/1     Running
#   dynamo-platform-nats-0                                   2/2     Running
```

## Step 3: Verify EFA Device Plugin

```bash
# Check EFA devices are allocatable
kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.allocatable.vpc\.amazonaws\.com/efa}{"\n"}{end}'
# Expected: 32 EFA devices per P5 node
```

## Step 4: Verify GDRCopy

```bash
# Check GDRCopy installer DaemonSet
kubectl get ds | grep gdrcopy
# Should show 1 per GPU node
```

## Step 5: Create HuggingFace Token Secret

```bash
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=hf_your_token_here
```

## Verification

All components running:
- [ ] Dynamo operator (2/2 Running)
- [ ] etcd (1/1 Running)
- [ ] NATS (2/2 Running)
- [ ] EFA devices allocatable (32 per P5 node)
- [ ] GDRCopy DaemonSet running
- [ ] HF token secret created

## Next

[Module 2: Single-Node Disaggregated Inference](../02-single-node/README.md)
