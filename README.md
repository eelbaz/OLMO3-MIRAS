# OLMo3-MIRAS

Neural Long-Term Memory for OLMo Language Models

Implementation of MIRAS (Memory In Recurrent Attention Systems) neural memory modules integrated with OLMo models, enabling unlimited context length through test-time learning.

## Supported Models

| Model | Status | HuggingFace ID |
|-------|--------|----------------|
| OLMo2-1B | Active | `allenai/OLMo-2-0425-1B` |
| OLMo3-7B | Planned | `allenai/Olmo-3-1025-7B` |
| OLMo3-32B | Planned | `allenai/Olmo-3-1125-32B` |

## Overview

This project implements the Titans-LMM memory architecture from the MIRAS framework, allowing transformers to learn and memorize at test time. The memory module uses gradient-based surprise metrics to dynamically update associative memory during inference.

Key equations from the Titans paper (arXiv:2501.00663):

```
Memory update:  M_t = (1 - alpha_t) * M_{t-1} + S_t
Momentum:       S_t = eta_t * S_{t-1} - theta_t * grad(M; x_t)
Loss:           L(M; k, v) = ||M(k) - v||^2
```

## Features

- Test-time memory learning via gradient descent
- Momentum-based surprise accumulation
- Data-dependent adaptive gates (alpha, eta, theta)
- Chunk-based parallel processing for efficiency
- Persistent memory tokens for short-term context
- Memory-as-Layer (MAL) integration with OLMo2-1B

## Architecture

```
OLMo Base Model (frozen)
    |
    |   Supported: OLMo2-1B (active), OLMo3-7B, OLMo3-32B (planned)
    |
    +-- MIRAS Memory Modules (trainable)
    |       |
    |       +-- Key/Value/Query projections
    |       +-- AssociativeMemoryMLP (2-layer)
    |       +-- Momentum state tracking
    |       +-- Adaptive forget gates
    |
    +-- Persistent Memory (trainable)
    +-- Memory Gates (trainable)
```

## Installation

```bash
git clone https://github.com/eelbaz/olmo3-miras.git
cd olmo3-miras
pip install -e .
```

Dependencies:
- PyTorch 2.0+
- transformers
- einops
- datasets
- accelerate

## Training

Train MIRAS memory modules on Dolma3 data:

```bash
# Single GPU
python scripts/train_olmo2_1b_unlimited_context.py

# With custom batch size
python scripts/train_olmo2_1b_unlimited_context.py --batch_size 128

# Synthetic data validation
python scripts/train_olmo2_1b_unlimited_context.py --synthetic --max_steps_override 10
```

Docker training:

```bash
docker run --rm -d --gpus all --ipc=host \
  -v $(pwd):/workspace/olmo3-miras \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -w /workspace/olmo3-miras \
  --name olmo-training \
  nvcr.io/nvidia/pytorch:25.11-py3 \
  bash -c "pip install einops transformers datasets accelerate zstandard pytz --quiet && \
           python scripts/train_olmo2_1b_unlimited_context.py"
```

## Configuration

Default training configuration (128 context, fastest validation):

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Batch size | 128 |
| Sequence length | 128 |
| Max steps | 50,000 |
| Checkpoint every | 500 |
| Validation every | 500 |

Memory module configuration:

| Parameter | Value |
|-----------|-------|
| Memory hidden size | 256 |
| Memory depth | 2 layers |
| Memory heads | 8 |
| Momentum decay | 0.9 |
| Learning rate (theta) | 0.1 |
| Chunk size | 512 |
| Persistent tokens | 32 |

## Model Parameters

| Component | Parameters |
|-----------|------------|
| OLMo2-1B base (frozen) | 1.28B |
| MIRAS memory modules | 468M |
| Total | 1.75B |
| Trainable | 468M |

## Checkpoints

Checkpoints are saved to `./checkpoints/olmo2_1b_unlimited/` and include:

- `miras_modules.pt` - MIRAS memory weights
- `optimizer.pt` - Optimizer state
- `scheduler.pt` - Learning rate scheduler state
- `rng_state.pt` - RNG states for reproducibility
- `training_state.json` - Step counter and data position

## References

- Titans: Learning to Memorize at Test Time (arXiv:2501.00663)
- MIRAS: A Framework for Designing Deep Learning Architectures (arXiv:2504.13173)
- OLMo: Open Language Model (AllenAI)

## License

Apache 2.0
