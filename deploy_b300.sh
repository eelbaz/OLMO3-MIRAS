#!/bin/bash
# =============================================================================
# OLMo2-1B + MIRAS Unlimited Context Training Deployment Script
# Target: 4× NVIDIA B300 288GB on NVIDIA Brev
# =============================================================================

set -e

echo "=============================================="
echo "OLMo2-1B + MIRAS Unlimited Context Training"
echo "Target: 4× NVIDIA B300 288GB"
echo "=============================================="

# Configuration
export HF_TOKEN="${HF_TOKEN:-your_huggingface_token_here}"
export PYTORCH_ALLOC_CONF="expandable_segments:True"  # PyTorch 2.9+ (was PYTORCH_CUDA_ALLOC_CONF)
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5

# Training parameters (optimized for 4× B300)
MAX_SEQ_LENGTH=65536      # 64K context
BATCH_SIZE_PER_GPU=8      # Per GPU
OUTPUT_DIR="./checkpoints/olmo2_1b_unlimited_64k"

# Check GPUs
echo ""
echo "Checking GPUs..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --quiet einops transformers datasets accelerate

# Set PYTHONPATH
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Start training
echo ""
echo "Starting training..."
echo "  - Sequence length: ${MAX_SEQ_LENGTH}"
echo "  - Batch size per GPU: ${BATCH_SIZE_PER_GPU}"
echo "  - Effective batch: $((BATCH_SIZE_PER_GPU * 4 * 2))"  # 4 GPUs * grad_accum
echo "  - Output: ${OUTPUT_DIR}"
echo ""

# Run with torchrun for multi-GPU
torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    scripts/train_olmo2_1b_unlimited_context.py \
    --output_dir ${OUTPUT_DIR} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --batch_size ${BATCH_SIZE_PER_GPU}

echo ""
echo "Training complete!"
echo "Checkpoints saved to: ${OUTPUT_DIR}"
