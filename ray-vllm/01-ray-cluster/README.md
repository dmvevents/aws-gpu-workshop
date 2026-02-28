# Section 01: Setting Up Ray on Kubernetes

## Objective

Deploy a multi-node Ray cluster on Amazon EKS with GPU resources. By the end of this section, you will have a running Ray cluster with a dashboard, ready to serve as the foundation for vLLM inference and GRPO training.

**Estimated time: 20 minutes**

---

## Background: What Is Ray?

Ray is a distributed computing framework designed for scaling Python applications across multiple machines. In the context of LLM workloads, Ray provides:

- **Resource management** -- Allocates GPUs, CPUs, and memory across a cluster
- **Actor model** -- Long-lived stateful processes (used by vLLM inference engines)
- **Task model** -- Stateless function execution (used for data preprocessing)
- **Ray Serve** -- A scalable model serving framework built on top of Ray actors
- **Ray Dashboard** -- A web UI for monitoring cluster state, GPU usage, and job progress

### Head and Worker Architecture

A Ray cluster has two types of nodes:

```
                    Ray Cluster Architecture
                    ========================

  +-------------------+          +-------------------+
  |    Ray Head       |          |   Ray Worker(s)   |
  |                   |          |                   |
  |  GCS (state)      |  <---->  |  GPU resources    |
  |  Dashboard (:8265)|          |  CPU resources    |
  |  Client API       |          |  Memory           |
  |  Job submission   |          |                   |
  |  Autoscaler       |          |  Runs actors and  |
  |                   |          |  tasks scheduled   |
  |  Optional: GPU    |          |  by the head      |
  +-------------------+          +-------------------+
         |                              |
         +--------Shared Storage--------+
                  (FSx Lustre)
```

**Ray Head** runs the Global Control Store (GCS), the dashboard, and coordinates scheduling. It does not need a GPU unless you want it to run inference or training workloads.

**Ray Workers** provide compute resources (GPUs, CPUs). When you submit a deployment or task, Ray schedules it onto workers that have sufficient resources.

---

## Method 1: Indexed Job Pattern (Recommended for Training)

The Indexed Job pattern is the most robust way to deploy a multi-node Ray cluster for training workloads. It uses Kubernetes Indexed Jobs where each pod knows its rank via `JOB_COMPLETION_INDEX`, and the rank-0 pod becomes the Ray head.

### Why Indexed Jobs?

KubeRay is excellent for long-lived serving clusters, but for training jobs that start, run, and finish, the Indexed Job pattern offers advantages:

- Pods are created simultaneously (no staggered startup)
- Each pod has a deterministic rank (`JOB_COMPLETION_INDEX`)
- The job completes when all pods finish (natural lifecycle)
- Works well with shared filesystems for master IP discovery
- No CRD dependency -- uses only standard Kubernetes APIs

### Master IP Discovery via Shared Filesystem

The key coordination challenge is: how does rank 1 find rank 0's IP address? The Indexed Job pattern solves this by writing the master IP to a shared filesystem:

```
Rank 0 (Head):
  1. Get own IP: MY_IP=$(hostname -I | awk '{print $1}')
  2. Write to shared file: echo "$MY_IP" > /shared/.master-ip
  3. Start Ray head: ray start --head --port=6379

Rank 1+ (Workers):
  1. Poll shared file until it exists
  2. Read master IP: MASTER_IP=$(cat /shared/.master-ip)
  3. Join cluster: ray start --address=$MASTER_IP:6379
```

### Kubernetes Manifest: Multi-Node Ray Cluster

This manifest deploys a 2-node Ray cluster on GPU instances. Adapt the `nodeSelector`, resource limits, and image to match your environment.

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: ray-techpulse-cluster
  namespace: default
spec:
  completionMode: Indexed
  completions: 2          # Total nodes (1 head + 1 worker)
  parallelism: 2          # All nodes start simultaneously
  backoffLimit: 0
  template:
    metadata:
      labels:
        app: ray-techpulse
    spec:
      restartPolicy: Never
      hostNetwork: true
      dnsPolicy: ClusterFirstWithHostNet
      nodeSelector:
        node.kubernetes.io/instance-type: p4d.24xlarge
      tolerations:
        - key: "nvidia.com/gpu"
          operator: "Exists"
          effect: "NoSchedule"
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            - labelSelector:
                matchLabels:
                  app: ray-techpulse
              topologyKey: kubernetes.io/hostname
      containers:
        - name: ray-node
          image: <YOUR_ECR_REPO>/ray-vllm:latest
          imagePullPolicy: Always
          securityContext:
            privileged: true
            capabilities:
              add:
              - IPC_LOCK
              - SYS_RESOURCE
          command: ["/bin/bash", "-c"]
          args:
            - |
              set -e
              NODE_RANK=${JOB_COMPLETION_INDEX:-0}
              export NODE_RANK
              MY_IP=$(hostname -I | awk '{print $1}')

              echo "=============================================="
              echo "Ray Cluster - TechPulse Media"
              echo "Node Rank: $NODE_RANK"
              echo "Node IP: $MY_IP"
              echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
              echo "=============================================="

              # Fix hostname resolution (required for Gloo backend)
              REAL_HOSTNAME=$(hostname)
              echo "$MY_IP $REAL_HOSTNAME" >> /etc/hosts

              # Master IP discovery via shared filesystem
              MASTER_FILE="/shared/.ray-master-ip"
              if [ $NODE_RANK -eq 0 ]; then
                MASTER_IP=$MY_IP
                rm -f "$MASTER_FILE"
                echo "$MY_IP" > "$MASTER_FILE"
                sync
                echo "Rank 0: Wrote master IP $MASTER_IP to FSx"
              else
                echo "Rank $NODE_RANK: Waiting for master IP..."
                for i in $(seq 1 120); do
                  if [ -f "$MASTER_FILE" ]; then
                    MASTER_IP=$(cat "$MASTER_FILE")
                    if [ -n "$MASTER_IP" ]; then
                      echo "Rank $NODE_RANK: Found master IP $MASTER_IP"
                      break
                    fi
                  fi
                  sleep 2
                done
                if [ -z "$MASTER_IP" ]; then
                  echo "FATAL: Master IP not found after 240s"
                  exit 1
                fi
              fi

              # Start Ray
              GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
              if [ $NODE_RANK -eq 0 ]; then
                ray start --head \
                  --num-gpus=$GPU_COUNT \
                  --port=6379 \
                  --dashboard-host=0.0.0.0 \
                  --temp-dir=/tmp/ray

                # Wait for all workers to join
                python3 -c "
              import ray, time, sys
              ray.init(address='auto')
              target_nodes = 2
              for _ in range(120):
                  nodes = [n for n in ray.nodes() if n['Alive']]
                  gpus = sum(n['Resources'].get('GPU', 0) for n in nodes)
                  if len(nodes) >= target_nodes:
                      print(f'Cluster ready: {len(nodes)} nodes, {int(gpus)} GPUs')
                      ray.shutdown(); sys.exit(0)
                  print(f'Waiting... {len(nodes)}/{target_nodes} nodes')
                  time.sleep(5)
              print('TIMEOUT: Not all nodes joined'); ray.shutdown(); sys.exit(1)
              "

                # HEAD NODE: Run your workload here
                echo "=== Ray cluster ready. Submit workloads. ==="
                # Example: ray job submit -- python my_script.py
                sleep infinity  # Keep head alive (replace with your workload)
              else
                # WORKER NODE: Join the cluster and block
                for i in $(seq 1 60); do
                  python3 -c "
              import socket; s=socket.socket()
              s.settimeout(2); s.connect(('$MASTER_IP',6379)); s.close()
              " 2>/dev/null && break
                  sleep 3
                done
                ray start \
                  --address=$MASTER_IP:6379 \
                  --num-gpus=$GPU_COUNT \
                  --temp-dir=/tmp/ray \
                  --block
              fi

          env:
            - name: HF_HOME
              value: "/shared/hf_cache"
            - name: NCCL_DEBUG
              value: "WARN"
            - name: RAY_worker_register_timeout_seconds
              value: "600"
          resources:
            limits:
              nvidia.com/gpu: 8
              vpc.amazonaws.com/efa: 4
              memory: 384Gi
            requests:
              nvidia.com/gpu: 8
              vpc.amazonaws.com/efa: 4
              memory: 256Gi
              cpu: 48
          volumeMounts:
            - name: shared
              mountPath: /shared
            - name: shm
              mountPath: /dev/shm
      volumes:
        - name: shared
          persistentVolumeClaim:
            claimName: fsx-pvc
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: 128Gi
```

### Key Design Decisions in This Manifest

**`hostNetwork: true`** -- Required for EFA networking. Pods use the host's network namespace, which gives them direct access to EFA devices. Without this, NCCL communication over EFA will fail.

**`podAntiAffinity`** -- Ensures each pod lands on a different physical node. Without this, Kubernetes might schedule both pods on the same node, wasting half your cluster.

**`IPC_LOCK` and `SYS_RESOURCE`** -- Required for RDMA memory registration. EFA pins GPU memory for zero-copy transfers, which requires these capabilities.

**`/dev/shm` as Memory-backed tmpfs** -- NCCL and PyTorch use shared memory for inter-process communication. The default 64MB is too small; 128Gi allows multi-GB tensor transfers.

### Deploying the Cluster

```bash
# Apply the manifest
kubectl apply -f ray-cluster-job.yaml

# Watch pods come up
kubectl get pods -l app=ray-techpulse -w

# Expected output:
# ray-techpulse-cluster-0-xxxxx   1/1   Running   0   30s
# ray-techpulse-cluster-1-xxxxx   1/1   Running   0   30s
```

---

## Method 2: KubeRay RayCluster CRD (Recommended for Serving)

For long-lived inference serving, the KubeRay operator provides a declarative `RayCluster` custom resource. The operator handles Ray head startup, worker registration, and automatic recovery.

### Installing KubeRay

```bash
# Add the KubeRay Helm repo
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update

# Install the operator
helm install kuberay-operator kuberay/kuberay-operator \
  --namespace kuberay-system \
  --create-namespace

# Verify installation
kubectl get pods -n kuberay-system
# kuberay-operator-xxxx   1/1   Running   0   30s
```

### RayCluster Manifest

```yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: techpulse-inference
  namespace: default
spec:
  rayVersion: '2.52.0'
  headGroupSpec:
    rayStartParams:
      dashboard-host: '0.0.0.0'
      num-cpus: '0'       # Head does not run workloads
    template:
      metadata:
        labels:
          app: techpulse-ray
          component: head
      spec:
        containers:
          - name: ray-head
            image: <YOUR_ECR_REPO>/ray-vllm:latest
            resources:
              limits:
                cpu: "8"
                memory: "32Gi"
              requests:
                cpu: "4"
                memory: "16Gi"
            ports:
              - containerPort: 6379
                name: gcs
              - containerPort: 8265
                name: dashboard
              - containerPort: 10001
                name: client
              - containerPort: 8000
                name: serve
            env:
              - name: RAY_DEDUP_LOGS
                value: "0"
              - name: RAY_SERVE_HTTP_HOST
                value: "0.0.0.0"
            volumeMounts:
              - name: shm
                mountPath: /dev/shm
        volumes:
          - name: shm
            emptyDir:
              medium: Memory
              sizeLimit: "16Gi"
  workerGroupSpecs:
    - groupName: gpu-workers
      replicas: 2
      minReplicas: 1
      maxReplicas: 4        # Enables autoscaling
      rayStartParams:
        num-gpus: '8'
      template:
        metadata:
          labels:
            app: techpulse-ray
            component: worker
        spec:
          containers:
            - name: ray-worker
              image: <YOUR_ECR_REPO>/ray-vllm:latest
              resources:
                limits:
                  nvidia.com/gpu: 8
                  vpc.amazonaws.com/efa: 4
                  memory: 384Gi
                requests:
                  nvidia.com/gpu: 8
                  vpc.amazonaws.com/efa: 4
                  memory: 256Gi
                  cpu: 48
              env:
                - name: FI_PROVIDER
                  value: "efa"
                - name: FI_EFA_FORK_SAFE
                  value: "1"
                - name: FI_EFA_USE_DEVICE_RDMA
                  value: "1"
              volumeMounts:
                - name: shm
                  mountPath: /dev/shm
                - name: shared
                  mountPath: /shared
          volumes:
            - name: shm
              emptyDir:
                medium: Memory
                sizeLimit: "128Gi"
            - name: shared
              persistentVolumeClaim:
                claimName: fsx-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: techpulse-ray-head-svc
spec:
  selector:
    app: techpulse-ray
    component: head
  ports:
    - name: dashboard
      port: 8265
      targetPort: 8265
    - name: serve
      port: 8000
      targetPort: 8000
    - name: gcs
      port: 6379
      targetPort: 6379
  type: ClusterIP
```

### Deploying and Verifying

```bash
# Deploy the RayCluster
kubectl apply -f ray-cluster.yaml

# Wait for all pods to be ready
kubectl wait --for=condition=ready pod \
  -l app=techpulse-ray \
  --timeout=300s

# Check cluster status from the head pod
HEAD=$(kubectl get pods -l component=head -o jsonpath='{.items[0].metadata.name}')
kubectl exec $HEAD -- ray status

# Expected output:
# ======== Autoscaler status ========
# Node status
# ---------------------------------------------------------------
# Active:
#  1 node_xxx (head)
#  2 node_xxx (gpu-workers)
# Resources
# ---------------------------------------------------------------
# Usage:
#  0.0/16.0 GPU
#  0.0/104.0 CPU
```

---

## Accessing the Ray Dashboard

The Ray dashboard provides real-time visibility into cluster health, running jobs, actor states, and GPU utilization.

```bash
# Port-forward the dashboard
kubectl port-forward svc/techpulse-ray-head-svc 8265:8265

# Open in browser: http://localhost:8265
```

The dashboard shows:

- **Cluster tab** -- Node count, resource usage, worker status
- **Jobs tab** -- Submitted jobs and their status
- **Actors tab** -- Running actors (vLLM engines, Ray Serve replicas)
- **Metrics tab** -- GPU utilization, memory, and throughput graphs
- **Logs tab** -- Aggregated logs from all nodes

---

## GPU Resource Allocation

Ray tracks GPU resources at the cluster level. When you create a deployment or actor that requests GPUs, Ray checks available resources and schedules it onto a node with sufficient capacity.

### How Ray Assigns GPUs

```
Cluster total: 16 GPUs (2 nodes x 8 GPUs each)

Deployment A: num_gpus=2 (TP=2 for 7B model)
  -> Scheduled on Node 1, GPUs 0-1

Deployment B: num_gpus=4 (TP=4 for 70B model)
  -> Scheduled on Node 1, GPUs 2-5

Deployment C: num_gpus=2 (TP=2 for 7B model)
  -> Scheduled on Node 1, GPUs 6-7

Remaining: 8 GPUs on Node 2 (available for training)
```

### Verifying GPU Visibility

```bash
# From inside the head pod
kubectl exec $HEAD -- python3 -c "
import ray
ray.init(address='auto')
for node in ray.nodes():
    if node['Alive']:
        gpus = node['Resources'].get('GPU', 0)
        name = node['NodeName']
        print(f'{name}: {int(gpus)} GPUs')
ray.shutdown()
"
```

### Monitoring GPU Usage

```bash
# Check nvidia-smi on a specific worker pod
WORKER=$(kubectl get pods -l component=worker \
  -o jsonpath='{.items[0].metadata.name}')
kubectl exec $WORKER -- nvidia-smi

# Continuous monitoring
kubectl exec $WORKER -- watch -n 2 nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv
```

---

## Indexed Job vs. KubeRay: When to Use Which

| Aspect | Indexed Job | KubeRay RayCluster |
|--------|-------------|-------------------|
| **Use case** | Training jobs (start, run, finish) | Long-lived serving clusters |
| **Lifecycle** | Job completes when all pods finish | Cluster stays running until deleted |
| **Autoscaling** | None (fixed parallelism) | Built-in (minReplicas/maxReplicas) |
| **Master discovery** | Shared filesystem file | K8s Service DNS (automatic) |
| **Dependencies** | None (standard K8s APIs) | KubeRay operator CRD |
| **Retry** | `backoffLimit` (fresh cluster) | Operator recreates failed pods |
| **Dashboard** | Must set up manually | Automatic via head pod |
| **Best for** | GRPO training, CPT, SFT | vLLM serving, Ray Serve |

---

## Checkpoint: Verify Your Cluster

Before proceeding to Section 02, confirm:

```bash
# 1. All pods are running
kubectl get pods -l app=techpulse-ray
# All should show Running status

# 2. Ray cluster sees all GPUs
kubectl exec $HEAD -- python3 -c "
import ray; ray.init(address='auto')
gpus = ray.cluster_resources().get('GPU', 0)
print(f'Total GPUs: {int(gpus)}')
assert gpus >= 2, 'Need at least 2 GPUs'
print('PASS: Cluster ready')
ray.shutdown()
"

# 3. Dashboard is accessible
kubectl port-forward svc/techpulse-ray-head-svc 8265:8265 &
curl -s http://localhost:8265/api/cluster_status | python3 -m json.tool
```

---

## Common Issues

### Pods Stuck in Pending

**Symptom:** Pods remain in `Pending` state.

```bash
kubectl describe pod <pod-name> | grep -A 5 Events
```

**Common causes:**
- Insufficient GPU resources. Check: `kubectl describe nodes | grep -A 5 "nvidia.com/gpu"`
- Node taint without matching toleration. Check: `kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.taints}{"\n"}{end}'`
- Anti-affinity preventing scheduling on same node. Ensure you have at least 2 GPU nodes.

### Ray Workers Fail to Join

**Symptom:** Head pod reports fewer nodes than expected.

**Causes and fixes:**
- **Stale master IP file:** If a previous run left `/shared/.ray-master-ip` with an old IP, workers try to connect to a dead address. The manifest above handles this by deleting the file at the start of rank 0.
- **Network policy blocking port 6379:** Ray GCS runs on port 6379. Ensure your network policies allow traffic between pods.
- **Timeout too short:** Increase `RAY_worker_register_timeout_seconds` if nodes take a long time to start (large images, slow pulls).

### NCCL Initialization Timeout

**Symptom:** Training hangs at NCCL initialization.

```
NCCL WARN Timeout waiting for connect from peer
```

**Fixes:**
- Verify `hostNetwork: true` is set (required for EFA)
- Check EFA devices: `kubectl exec <pod> -- ls /dev/infiniband/`
- Set `NCCL_SOCKET_IFNAME=^lo,docker,veth` to exclude virtual interfaces
- Increase timeout: `NCCL_TIMEOUT=7200`

---

Next: [Section 02: vLLM Model Serving](../02-vllm-serving/)
