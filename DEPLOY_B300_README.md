# OLMo2-1B + MIRAS Unlimited Context Training

## Target Hardware
- **4× NVIDIA B300 288GB** (1.1TB total VRAM)
- **120 CPU cores**
- Container: `nvcr.io/nvidia/pytorch:25.11-py3`

## Quick Start

### 1. Clone/Copy Project to Brev Machine

```bash
# On Brev machine
cd /workspace
git clone <your-repo> olmo3_miras
# OR scp the project
```

### 2. Set HuggingFace Token

```bash
export HF_TOKEN="your_huggingface_token"
```

### 3. Run Training with Docker

```bash
docker run --rm --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $(pwd):/workspace \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    -w /workspace/olmo3_miras \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    bash -c "pip install einops transformers datasets accelerate --quiet && \
             export PYTHONPATH=/workspace:\$PYTHONPATH && \
             torchrun --nproc_per_node=4 \
                      --master_port=29500 \
                      scripts/train_olmo2_1b_unlimited_context.py \
                      --output_dir ./checkpoints/olmo2_1b_unlimited_64k \
                      --max_seq_length 65536 \
                      --batch_size 8"
```

### 4. Alternative: Run Directly (if PyTorch installed)

```bash
cd /workspace/olmo3_miras
chmod +x deploy_b300.sh
./deploy_b300.sh
```

## Training Configurations

### 64K Context (Default - Recommended)
```bash
--max_seq_length 65536 --batch_size 8
# ~300K tokens/sec, ~3-4 hours for 50K steps
```

### 128K Context (Maximum)
```bash
--max_seq_length 131072 --batch_size 4
# ~250K tokens/sec, ~5-6 hours for 50K steps
```

### 32K Context (Faster)
```bash
--max_seq_length 32768 --batch_size 16
# ~500K tokens/sec, ~2 hours for 50K steps
```

## Expected Performance (4× B300)

| Seq Length | Batch | Tokens/sec | 50K Steps |
|------------|-------|------------|-----------|
| 32K | 16 | ~500K | ~2 hrs |
| 64K | 8 | ~300K | ~3-4 hrs |
| 128K | 4 | ~250K | ~5-6 hrs |

## Monitoring

### Watch Training Progress
```bash
tail -f checkpoints/olmo2_1b_unlimited_64k/training.log
```

### Check GPU Utilization
```bash
watch -n 1 nvidia-smi
```

## Checkpoints

Checkpoints saved every 500 steps to:
- `checkpoints/olmo2_1b_unlimited_64k/checkpoint-500/`
- `checkpoints/olmo2_1b_unlimited_64k/checkpoint-1000/`
- ...

Each checkpoint contains:
- `miras_modules.pt` - MIRAS memory weights
- `optimizer.pt` - Optimizer state
- `scheduler.pt` - LR scheduler state
- `training_state.json` - Step number

## Resume Training

```bash
torchrun --nproc_per_node=4 \
    scripts/train_olmo2_1b_unlimited_context.py \
    --resume_from ./checkpoints/olmo2_1b_unlimited_64k/checkpoint-1000 \
    --output_dir ./checkpoints/olmo2_1b_unlimited_64k
```

## What MIRAS Learns

With 64K context training, MIRAS learns:
1. **Memory compression** - Efficiently store past information
2. **Selective retrieval** - Access relevant memories
3. **Forgetting** - Discard irrelevant information
4. **Momentum tracking** - Detect surprising/important tokens

After training, the model can handle **unlimited context** at inference
by continuously accumulating memory across chunks.
