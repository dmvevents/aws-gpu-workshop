# Section 03: GRPO/RLHF Training with NeMo RL

## Objective

Run GRPO (Group Relative Policy Optimization) alignment training on TechPulse Media's fine-tuned model using NVIDIA NeMo RL. By the end of this section, you will understand the GRPO training loop, have configured a multi-node training run, and observed how reinforcement learning improves model reasoning.

**Estimated time: 25 minutes**

---

## Background: What Is GRPO?

### The Alignment Problem

After supervised fine-tuning (SFT), a model can follow instructions and generate fluent text. But it may still:
- Produce confident but wrong answers
- Miss logical steps in reasoning
- Fail to show its work
- Give different quality answers depending on phrasing

**RLHF (Reinforcement Learning from Human Feedback)** addresses this by training the model to maximize a reward signal that measures response quality. The model learns to generate responses that score highly, effectively learning *how to think* rather than just *what to say*.

### GRPO vs. PPO

Traditional RLHF uses PPO (Proximal Policy Optimization), which requires a separate reward model and a critic model. GRPO simplifies this:

```
PPO (traditional):
  Policy Model  ---generates--->  Response
  Reward Model  ---scores--->     Scalar reward
  Critic Model  ---estimates--->  Value baseline
  Loss = advantage * clipped_ratio - value_loss
  -> 3 models in memory simultaneously

GRPO (simplified):
  Policy Model  ---generates--->  G responses per prompt
  Reward Fn     ---scores--->     G scalar rewards
  Baseline      = mean(rewards within group)
  Advantage     = (reward - baseline) / std(rewards)
  Loss = advantage * clipped_ratio
  -> 1 model + lightweight reward function
```

GRPO generates **multiple responses per prompt** (the "Group" in GRPO), computes rewards for each, and uses the group statistics as the baseline. This eliminates the need for a separate critic model, reducing memory requirements by ~40%.

### The GRPO Training Loop

Each training step follows this cycle:

```
Step 1: GENERATE
  For each prompt in the batch:
    Generate G responses (e.g., G=16)
    using vLLM with temperature sampling
                |
                v
Step 2: SCORE
  For each response:
    Compute reward using a reward function
    (e.g., math_verify checks if the answer is correct)
                |
                v
Step 3: COMPUTE ADVANTAGES
  For each prompt group:
    mean_reward = mean(rewards in group)
    std_reward  = std(rewards in group)
    advantage_i = (reward_i - mean_reward) / std_reward
                |
                v
Step 4: COMPUTE LOG PROBABILITIES
  For each response:
    Compute log P(response | prompt) under current policy
    Compute log P(response | prompt) under reference policy
                |
                v
Step 5: POLICY UPDATE
  Loss = -mean(advantage * min(ratio, clip(ratio, 1-eps, 1+eps)))
       + beta * KL(policy || reference)
  Backpropagate and update model weights
                |
                v
  Repeat from Step 1
```

---

## NeMo RL Architecture

NeMo RL orchestrates the GRPO loop using Ray for distributed coordination. It manages two workloads simultaneously:

### Generation (vLLM)

The generation phase uses vLLM to produce responses. vLLM runs as a Ray actor with GPU resources, handling:
- Batched prompt processing
- Temperature-controlled sampling
- Multiple responses per prompt (controlled by `num_generations_per_prompt`)
- Token-level log probability computation

### Training (DTensor/Megatron)

The training phase uses PyTorch DTensor (or Megatron-Core for large models) for distributed training:
- Full-parameter or LoRA updates
- FSDP (Fully Sharded Data Parallel) for multi-GPU training
- Gradient accumulation for effective large batch sizes
- Mixed-precision (BF16) training

### Colocated vs. Dedicated Generation

NeMo RL supports two modes for sharing GPUs between generation and training:

```
Colocated (generation.colocated.enabled=true):
  +---------------------------+
  |  GPU 0-7 (Same GPUs)     |
  |                           |
  |  Phase 1: vLLM generates  |
  |  (offload training state) |
  |                           |
  |  Phase 2: DTensor trains  |
  |  (offload vLLM KV cache)  |
  +---------------------------+
  Pros: Half the GPU count
  Cons: Cannot overlap generation and training

Dedicated (generation.colocated.enabled=false):
  +--------------+    +--------------+
  |  GPUs 0-7    |    |  GPUs 8-15   |
  |  vLLM only   |    |  Training    |
  |  (generate)  |    |  (DTensor)   |
  +--------------+    +--------------+
  Pros: Generation and training can overlap
  Cons: Requires 2x GPUs
```

**For this workshop**, we use colocated mode since TechPulse Media has 2 nodes (16 GPUs) and wants to use all GPUs for the largest effective batch size.

---

## GRPO Configuration

NeMo RL uses Hydra-based YAML configuration. Here is the structure with key parameters explained:

```yaml
# GRPO algorithm settings
grpo:
  num_prompts_per_step: 32           # Prompts per training step
  num_generations_per_prompt: 16     # Responses generated per prompt (G)
  max_num_steps: 100                 # Total training steps
  normalize_rewards: true            # Normalize rewards within each group
  use_leave_one_out_baseline: true   # LOO baseline (more stable than mean)
  seed: 42

  # Advantage estimator
  adv_estimator:
    name: "grpo"                     # "grpo" or "reinforce_plus_plus"
    normalize_rewards: true

# Loss function
loss_fn:
  reference_policy_kl_penalty: 0.01  # beta: KL divergence penalty
  reference_policy_kl_type: "k3"     # KL approximation method
  ratio_clip_min: 0.2                # PPO-style clipping (epsilon)
  ratio_clip_max: 0.2
  token_level_loss: true             # Per-token loss (more stable)

# Policy (model) configuration
policy:
  model_name: "TechPulse/techpulse-7b-v1"
  tokenizer:
    name: "TechPulse/techpulse-7b-v1"
  train_global_batch_size: 512       # Effective batch = prompts * generations
  train_micro_batch_size: 4          # Per-GPU batch size
  max_total_sequence_length: 512     # Max tokens (prompt + response)
  precision: "bfloat16"
  max_grad_norm: 1.0

  # DTensor (distributed training) config
  dtensor_cfg:
    enabled: true
    tensor_parallel_size: 1          # TP for training (usually 1 for <10B)
    cpu_offload: false
    sequence_parallel: false
    activation_checkpointing: false

    # LoRA for memory-efficient training
    lora_cfg:
      enabled: true                  # Enable LoRA (recommended for GRPO)
      match_all_linear: true         # Apply to all linear layers
      dim: 8                         # LoRA rank (r)
      alpha: 32                      # Scaling factor (alpha/r = 4)
      dropout: 0.0

  # vLLM generation config
  generation:
    backend: "vllm"
    max_new_tokens: 512
    temperature: 1.0                 # High temp for diverse generations
    top_p: 1.0
    vllm_cfg:
      gpu_memory_utilization: 0.6    # Leave room for training
      tensor_parallel_size: 1
      enforce_eager: false
    colocated:
      enabled: true                  # Share GPUs between gen and train

  # Optimizer
  optimizer:
    name: "torch.optim.AdamW"
    kwargs:
      lr: 5.0e-6
      weight_decay: 0.01
      betas: [0.9, 0.999]

# Reward function
env:
  math:
    num_workers: 8
    math_verify_impl: "hf_math_verify"   # Uses math_verify library

# Data
data:
  train:
    dataset_name: "OpenMathInstruct-2"    # Math problems + solutions
  default:
    processor: "math_hf_data_processor"
    env_name: "math"

# Checkpointing
checkpointing:
  enabled: true
  checkpoint_dir: "/shared/grpo-checkpoints"
  save_period: 10                    # Save every 10 steps
  keep_top_k: 5                     # Keep best 5 checkpoints

# Cluster
cluster:
  gpus_per_node: 8
  num_nodes: 2                       # 2x p4d.24xlarge
```

### Key Hyperparameters Explained

**`num_generations_per_prompt` (G=16):** The number of responses generated per prompt. Higher G provides a better estimate of the reward landscape but costs more compute. Values of 8-32 are typical. If all G responses are correct (or all wrong), the prompt provides no learning signal.

**`temperature` (T=1.0):** Controls randomness during generation. Higher temperature produces more diverse responses, which is essential for GRPO to have variance within each group. T=1.0 is standard for training; lower values (0.3-0.7) are used for inference.

**`reference_policy_kl_penalty` (beta=0.01):** Controls how far the policy can drift from the reference model (the starting checkpoint). Higher beta constrains the policy more, preventing reward hacking but slowing learning. Start with 0.01 and increase if the model degenerates.

**`ratio_clip` (epsilon=0.2):** PPO-style clipping bounds. Prevents large policy updates in a single step. The ratio `P_new(response) / P_old(response)` is clipped to [1-epsilon, 1+epsilon] = [0.8, 1.2].

**`gpu_memory_utilization` (0.6):** In colocated mode, vLLM and training share GPUs. Set this lower than inference-only mode (0.90) to leave room for training activations and optimizer state.

---

## Deploying GRPO Training

### Kubernetes Manifest

This manifest deploys a 2-node GRPO training job using the Indexed Job pattern from Section 01:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: techpulse-grpo-training
  namespace: default
spec:
  completionMode: Indexed
  completions: 2
  parallelism: 2
  backoffLimit: 0
  template:
    metadata:
      labels:
        app: techpulse-grpo
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
                  app: techpulse-grpo
              topologyKey: kubernetes.io/hostname
      containers:
        - name: trainer
          image: <YOUR_ECR_REPO>/nemo-rl:latest
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
              echo "TechPulse GRPO Training"
              echo "Node Rank: $NODE_RANK / 2"
              echo "Node IP: $MY_IP"
              echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
              echo "=============================================="

              # Fix hostname resolution for Gloo
              REAL_HOSTNAME=$(hostname)
              echo "$MY_IP $REAL_HOSTNAME" >> /etc/hosts

              # GPU and EFA verification
              nvidia-smi --query-gpu=index,name,memory.total --format=csv
              EFA_COUNT=$(ls /dev/infiniband/uverbs* 2>/dev/null | wc -l)
              echo "EFA devices: $EFA_COUNT"

              # Master IP discovery via shared filesystem
              MASTER_FILE="/shared/.grpo-master-ip"
              if [ $NODE_RANK -eq 0 ]; then
                MASTER_IP=$MY_IP
                rm -f "$MASTER_FILE"
                echo "$MY_IP" > "$MASTER_FILE"
                sync
                echo "Rank 0: Master IP $MASTER_IP"
              else
                echo "Rank $NODE_RANK: Waiting for master IP..."
                for i in $(seq 1 120); do
                  if [ -f "$MASTER_FILE" ]; then
                    MASTER_IP=$(cat "$MASTER_FILE")
                    if [ -n "$MASTER_IP" ]; then
                      echo "Found master IP: $MASTER_IP"
                      break
                    fi
                  fi
                  sleep 2
                done
                if [ -z "$MASTER_IP" ]; then
                  echo "FATAL: Master IP not found"
                  exit 1
                fi
              fi

              # Start Ray cluster
              if [ $NODE_RANK -eq 0 ]; then
                ray start --head --num-gpus=8 --port=6379 --temp-dir=/tmp/ray

                # Wait for all workers
                python3 -c "
              import ray, time, sys
              ray.init(address='auto')
              for _ in range(120):
                  nodes = [n for n in ray.nodes() if n['Alive']]
                  gpus = sum(n['Resources'].get('GPU', 0) for n in nodes)
                  if len(nodes) >= 2 and gpus >= 16:
                      print(f'Cluster ready: {len(nodes)} nodes, {int(gpus)} GPUs')
                      ray.shutdown(); sys.exit(0)
                  print(f'Waiting... {len(nodes)} nodes, {int(gpus)} GPUs')
                  time.sleep(5)
              print('TIMEOUT'); ray.shutdown(); sys.exit(1)
              "

                # Run GRPO training
                cd /workspace/nemo-rl
                python examples/run_grpo.py \
                  cluster.num_nodes=2 \
                  cluster.gpus_per_node=8 \
                  grpo.max_num_steps=100 \
                  grpo.num_prompts_per_step=32 \
                  grpo.num_generations_per_prompt=16 \
                  policy.model_name=TechPulse/techpulse-7b-v1 \
                  policy.tokenizer.name=TechPulse/techpulse-7b-v1 \
                  policy.dtensor_cfg.enabled=true \
                  policy.dtensor_cfg.lora_cfg.enabled=true \
                  policy.dtensor_cfg.lora_cfg.dim=8 \
                  policy.train_global_batch_size=512 \
                  policy.generation.vllm_cfg.gpu_memory_utilization=0.6 \
                  policy.generation.temperature=1.0 \
                  loss_fn.reference_policy_kl_penalty=0.01 \
                  checkpointing.checkpoint_dir=/shared/grpo-checkpoints \
                  checkpointing.save_period=10 \
                  logger.wandb_enabled=false \
                  logger.log_dir=/tmp/grpo-logs

                TRAIN_EXIT=$?
                echo "Training exit code: $TRAIN_EXIT"
                ray stop --force
                exit $TRAIN_EXIT
              else
                # Worker: join Ray cluster and block
                for i in $(seq 1 60); do
                  python3 -c "
              import socket; s=socket.socket()
              s.settimeout(2); s.connect(('$MASTER_IP',6379)); s.close()
              " 2>/dev/null && break
                  sleep 3
                done
                ray start --address=$MASTER_IP:6379 \
                  --num-gpus=8 --temp-dir=/tmp/ray --block
              fi

          env:
            # NCCL settings
            - name: NCCL_DEBUG
              value: "WARN"
            - name: NCCL_TIMEOUT
              value: "7200"
            - name: TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC
              value: "7200"
            # EFA networking
            - name: NCCL_NET_PLUGIN
              value: "ofi"
            - name: FI_PROVIDER
              value: "efa"
            - name: FI_EFA_USE_DEVICE_RDMA
              value: "1"
            - name: FI_EFA_FORK_SAFE
              value: "1"
            - name: RDMAV_FORK_SAFE
              value: "1"
            - name: NCCL_SOCKET_IFNAME
              value: "^lo,docker,veth,eni"
            # HuggingFace cache
            - name: HF_HOME
              value: "/shared/hf_cache"
            - name: HF_DATASETS_CACHE
              value: "/shared/hf_datasets"
            # Ray
            - name: RAY_worker_register_timeout_seconds
              value: "1800"

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

### Launching the Training

```bash
# Deploy
kubectl apply -f grpo-training-job.yaml

# Watch pod startup
kubectl get pods -l app=techpulse-grpo -w

# Stream logs from the head node (rank 0)
HEAD=$(kubectl get pods -l app=techpulse-grpo \
  -o jsonpath='{.items[0].metadata.name}')
kubectl logs -f $HEAD
```

---

## Monitoring Training Progress

### Reading the Training Logs

The training logs show the GRPO loop in action:

```
========================= Step 1/100 =========================
[Generate] 32 prompts x 16 generations = 512 responses (23.4s)
[Reward]   mean=0.31, std=0.46, min=0.0, max=1.0
[Train]    loss=0.042, kl=0.003, grad_norm=1.23
[Time]     total=45.2s (gen=23.4s, reward=1.2s, train=18.6s)

========================= Step 5/100 =========================
[Generate] 32 prompts x 16 generations = 512 responses (22.1s)
[Reward]   mean=0.45, std=0.49, min=0.0, max=1.0
[Train]    loss=0.038, kl=0.008, grad_norm=0.98
[Time]     total=42.8s (gen=22.1s, reward=1.1s, train=17.6s)

========================= Step 10/100 ========================
Saving checkpoint for step 10...
[Generate] 32 prompts x 16 generations = 512 responses (21.8s)
[Reward]   mean=0.58, std=0.49, min=0.0, max=1.0
[Train]    loss=0.031, kl=0.015, grad_norm=0.87
[Time]     total=48.1s (gen=21.8s, reward=1.0s, train=17.3s, ckpt=8.0s)
```

**What to watch for:**

| Metric | Healthy Range | Problem Indicator |
|--------|--------------|-------------------|
| `mean reward` | Increasing over time | Flat or decreasing |
| `kl` | 0.01-0.10 | >0.5 (policy diverging too fast) |
| `loss` | Decreasing trend | Sudden spike or NaN |
| `grad_norm` | <5.0 | >10.0 (training instability) |
| `gen time` | Stable | Increasing (memory pressure) |

### Checking Checkpoints

```bash
# List checkpoints on FSx
kubectl exec $HEAD -- ls -la /shared/grpo-checkpoints/

# Expected:
# step_10/
# step_20/
# step_30/
# ...
```

---

## Reward Functions

The reward function determines what the model optimizes for. NeMo RL supports pluggable reward functions:

### Math Verification (Default)

For math problems, the reward is binary: 1 if the answer is correct, 0 otherwise. The `math_verify` library parses the model's response to extract the final answer and compares it to the ground truth:

```
Prompt:  "What is 15% of 240?"
Response: "To find 15% of 240, I multiply 240 by 0.15.
           240 * 0.15 = 36
           The answer is 36."
Extracted: 36
Expected:  36
Reward:    1.0
```

### Custom Reward Functions

For TechPulse Media's use case (technical Q&A quality), you would define a custom reward function:

```python
# Example custom reward function structure
# (NeMo RL supports custom environments via the env config)

def techpulse_reward(prompt, response, reference_answer):
    """
    Score technical responses on multiple criteria.
    Returns a float in [0, 1].
    """
    score = 0.0

    # Factual accuracy (does it match key facts?)
    if key_facts_present(response, reference_answer):
        score += 0.5

    # Reasoning quality (does it explain why?)
    if contains_explanation(response):
        score += 0.25

    # Conciseness (is it appropriately brief?)
    if len(response.split()) < 200:
        score += 0.15

    # Formatting (uses code blocks for code, etc.)
    if appropriate_formatting(response):
        score += 0.10

    return score
```

Custom reward functions are registered via the `env` configuration in the GRPO config file. See the NeMo RL documentation for the full environment API.

---

## Advanced: Dynamic Sampling

Standard GRPO wastes compute on "trivial" prompts (where all G responses are correct) and "impossible" prompts (where all G responses are wrong). Dynamic sampling filters these out:

```yaml
grpo:
  use_dynamic_sampling: true
  dynamic_sampling_max_gen_batches: 10
```

With dynamic sampling enabled:
- If all G responses for a prompt are correct (100% accuracy): skip it (no learning signal)
- If all G responses are wrong (0% accuracy): skip it (no positive signal)
- Only keep prompts where some responses are correct and some are wrong

This focuses compute on the prompts where the model is at its learning frontier, significantly improving training efficiency.

---

## GRPO Training Timeline (What to Expect)

For a 7B model with LoRA on 2x p4d.24xlarge (16x A100):

| Phase | Duration | Notes |
|-------|----------|-------|
| Ray cluster startup | 2-3 min | Master IP discovery + worker join |
| Model download | 3-15 min | First run only; cached on FSx after |
| vLLM initialization | 1-2 min | Model loading + CUDA graph warmup |
| Training (100 steps) | 60-90 min | ~40-55s per step |
| Checkpoint saves | ~8s each | Every 10 steps |
| **Total** | **~70-110 min** | First run includes download |

Subsequent runs skip the download, reducing total time to ~65-95 minutes.

---

## Checkpoint: Verify Training

```bash
# 1. Training is progressing
kubectl logs $HEAD | grep "Step " | tail -5

# 2. Reward is increasing
kubectl logs $HEAD | grep "Reward" | tail -5

# 3. Checkpoints are being saved
kubectl exec $HEAD -- ls /shared/grpo-checkpoints/

# 4. No NCCL errors
kubectl logs $HEAD | grep -i "nccl.*error" | head -5
# Should return empty (no errors)
```

---

## Common Issues

### NCCL Timeout During Training

**Symptom:** Training hangs for minutes, then:
```
NCCL WARN Timeout waiting for connect from peer
RuntimeError: NCCL communicator was aborted
```

**Causes and fixes:**
- **EFA not configured:** Verify `FI_PROVIDER=efa` and EFA devices are present
- **Timeout too short:** Set `NCCL_TIMEOUT=7200` and `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200`
- **Security group missing self-referencing egress rule:** EFA SRD requires the security group to have an egress rule that references itself. A `0.0.0.0/0` outbound rule is NOT sufficient for SRD traffic.

### OOM During Generation Phase

**Symptom:** `torch.cuda.OutOfMemoryError` during the vLLM generation phase.

**Fix:** Reduce `gpu_memory_utilization` in the vLLM config:
```yaml
policy.generation.vllm_cfg.gpu_memory_utilization: 0.5  # Down from 0.6
```

Or reduce `num_generations_per_prompt` to generate fewer responses per batch.

### Stale Master IP

**Symptom:** Worker pods fail to join the Ray cluster, reporting "connection refused."

**Cause:** A previous training run left a stale IP in `/shared/.grpo-master-ip`. The manifest above handles this by having rank 0 delete the file before writing, but if the job was interrupted before rank 0 started, the stale file may persist.

**Fix:**
```bash
# Delete the stale file from any pod with FSx access
kubectl exec <any-pod-with-fsx> -- rm -f /shared/.grpo-master-ip
```

### KL Divergence Exploding

**Symptom:** `kl` metric grows rapidly above 0.5, and response quality degrades.

**Fix:** Increase the KL penalty:
```yaml
loss_fn.reference_policy_kl_penalty: 0.05  # Up from 0.01
```

Or reduce the learning rate:
```yaml
policy.optimizer.kwargs.lr: 1.0e-6  # Down from 5.0e-6
```

---

Next: [Section 04: Production Deployment Patterns](../04-production/)
