#!/usr/bin/env python3
"""
OLMo2-1B + MIRAS Unlimited Context Training Script

Multi-GPU training for MIRAS neural long-term memory on 4× B300 GPUs.
Trains on 64K+ context to enable truly unlimited context at inference.

Usage:
    torchrun --nproc_per_node=4 scripts/train_olmo2_1b_unlimited_context.py

Hardware Requirements:
    - 4× NVIDIA B300 288GB (or similar high-VRAM GPUs)
    - 120+ CPU cores recommended
    - 500GB+ storage for checkpoints
"""

import os
import sys
import json
import time
import math
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.amp import GradScaler, autocast

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset

from olmo3_miras.memory.neural_memory import MIRASMemoryConfig, NeuralLongTermMemory, PersistentMemory


# =============================================================================
# Configuration for Unlimited Context Training
# =============================================================================

# MIRAS config optimized for unlimited context on 4× B300 288GB GPUs
# Following Titans paper equations exactly (arXiv:2501.00663)
#
# Key equations:
#   M_t = (1 - α_t) * M_{t-1} + S_t           # Memory update (Eq 13-14)
#   S_t = η_t * S_{t-1} - θ_t * ∇ℓ(M_{t'}; x_t)  # Momentum (Eq 10)
#   ℓ(M; x) = ||M(k) - v||²                   # Loss (Eq 12)
#
# With 288GB/GPU, we can use larger chunks for efficiency
MIRAS_CONFIG_UNLIMITED = MIRASMemoryConfig(
    hidden_size=2048,           # OLMo2-1B hidden size
    memory_hidden_size=256,     # OPTIMIZED: 256×256×2 = 131K params (uses ~10% of B300 275GB)
    memory_depth=2,             # 2-layer memory MLP (L_M >= 2 per paper)
    num_memory_heads=8,         # OPTIMIZED: 8 heads for better capacity
    use_momentum=True,          # Critical for long context (Eq 10)
    momentum_decay=0.9,         # η_t base value per paper
    learning_rate=0.1,          # θ_t base value per paper
    forget_gate=True,           # Adaptive forgetting (Eq 13-14)
    chunk_size=512,             # OPTIMIZED: larger chunks for B300 efficiency
    num_persistent_tokens=32,   # OPTIMIZED: more persistent memory tokens
    data_dependent_gates=True,  # Data-dependent α, η, θ
    eps=1e-6,
    max_grad_norm=1.0,
    grad_scale=0.1,
)

# Training config for 4× B300 288GB (1.15TB total VRAM)
# With this much memory, we can use larger batches and longer sequences
TRAINING_CONFIG = {
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "warmup_steps": 500,
    "max_steps": 50000,
    "batch_size_per_gpu": 4,        # Per-GPU batch size (conservative for 64K sequences)
    "gradient_accumulation_steps": 4,  # Effective batch = 64 samples
    "max_seq_length": 65536,        # 64K context for unlimited training
    "checkpoint_every": 500,
    "log_every": 1,  # Log every step for visibility
    "eval_every": 500,
    "max_grad_norm": 1.0,
    "bf16": True,
    "gradient_checkpointing": False,  # Disabled - we use chunked forward instead
    "num_workers": 8,               # Per-GPU data workers
}

# Model
BASE_MODEL = "allenai/OLMo-2-0425-1B"

# Dataset - Official Dolma3 OLMo 3 Pretraining Mix
# Source: https://huggingface.co/datasets/allenai/dolma3_mix-6T-1025
# This is the ACTUAL pretraining data used for OLMo 3 7B (6 trillion tokens)
# Has train split, no subset needed, uses 'text' column
# Reference: https://huggingface.co/datasets/allenai/dolma3_pool/blob/main/README.md
DATASET_CONFIG = {
    "name": "allenai/dolma3_mix-6T-1025",  # Official OLMo 3 7B pretraining mix, HAS train split
    "subset": None,  # No subset needed
    "text_column": "text",  # Uses 'text' column
    "streaming": True,
}


# =============================================================================
# Distributed Training Setup
# =============================================================================

def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = 0
        local_rank = 0
        world_size = 1

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    # PyTorch 2.9+ Performance Optimizations for B300 GPUs
    # =====================================================
    # TF32 for faster matmuls (significant speedup on Blackwell/Ampere+)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # cuDNN benchmark mode - finds fastest convolution algorithms
    torch.backends.cudnn.benchmark = True
    # Use high precision for float32 matmuls (faster on modern GPUs)
    torch.set_float32_matmul_precision('high')

    if rank == 0:
        print(f"[PERF] PyTorch {torch.__version__} optimizations enabled:", flush=True)
        print(f"  - TF32 matmuls: {torch.backends.cuda.matmul.allow_tf32}", flush=True)
        print(f"  - cuDNN benchmark: {torch.backends.cudnn.benchmark}", flush=True)
        print(f"  - Float32 matmul precision: high", flush=True)

    return rank, local_rank, world_size


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """Check if this is the main process."""
    return rank == 0


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(output_dir: str, rank: int) -> logging.Logger:
    """Setup logging (only on main process)."""
    logger = logging.getLogger("olmo2_miras_unlimited")
    logger.setLevel(logging.INFO if is_main_process(rank) else logging.WARNING)

    if is_main_process(rank):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

        log_file = Path(output_dir) / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)

    return logger


# =============================================================================
# MIRAS Wrapper for OLMo2
# =============================================================================

class OLMo2MIRASWrapper(nn.Module):
    """
    Wraps OLMo2 with MIRAS memory modules for unlimited context.

    Memory is accumulated across chunks within each sequence,
    enabling the model to access information from arbitrarily long contexts.
    """

    def __init__(
        self,
        base_model: nn.Module,
        miras_config: MIRASMemoryConfig,
        integration_mode: str = "mal",  # Memory as Layer
    ):
        super().__init__()
        self.base_model = base_model
        self.config = miras_config
        self.integration_mode = integration_mode

        # Get model config
        self.num_layers = base_model.config.num_hidden_layers
        self.hidden_size = base_model.config.hidden_size

        # Create MIRAS memory for each layer
        self.memory_modules = nn.ModuleList([
            NeuralLongTermMemory(miras_config)
            for _ in range(self.num_layers)
        ])

        # Persistent memory (learnable)
        self.persistent_memory = PersistentMemory(miras_config)

        # Gate for memory contribution to hidden states
        # Memory module output is already at hidden_size, so no projection needed
        # Gate input: [hidden_mean, mem_output_mean] both at hidden_size
        self.memory_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.Sigmoid()
            )
            for _ in range(self.num_layers)
        ])

        # Freeze base model EXCEPT lm_head and final norm
        # We keep lm_head and norm unfrozen so gradients can flow from loss -> MIRAS
        for name, param in self.base_model.named_parameters():
            # Keep lm_head and final layer norm unfrozen for gradient flow
            if 'lm_head' in name or 'model.norm' in name or 'model.final_layernorm' in name:
                param.requires_grad = True
                print(f"[INFO] Keeping {name} unfrozen for gradient flow")
            else:
                param.requires_grad = False

        # DISABLE gradient checkpointing - it doesn't work well with partially frozen models
        # and causes "None of the inputs have requires_grad=True" warnings
        if hasattr(self.base_model, 'gradient_checkpointing_disable'):
            self.base_model.gradient_checkpointing_disable()

        # Convert MIRAS modules to same dtype as base model (bfloat16)
        # This fixes the "Mismatch dtype between input and weight" warning
        base_dtype = next(self.base_model.parameters()).dtype
        self.memory_modules = self.memory_modules.to(base_dtype)
        self.persistent_memory = self.persistent_memory.to(base_dtype)
        self.memory_gates = self.memory_gates.to(base_dtype)

    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())

    def _chunk_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
        memory_states: Optional[List[torch.Tensor]],
        momentum_states: Optional[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        Forward pass for a single chunk with memory integration.

        Note: memory_states and momentum_states here are the INTERNAL states
        for the NeuralLongTermMemory module (shape: batch, num_params).
        They are NOT the same as context states used for gating.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        dtype = next(self.parameters()).dtype

        # Initialize internal memory states if needed (let memory modules handle it on first call)
        # These are the flattened associative memory weights: shape (batch, num_params)
        if memory_states is None:
            memory_states = [None for _ in range(self.num_layers)]
        if momentum_states is None:
            momentum_states = [None for _ in range(self.num_layers)]

        # Get persistent memory - learnable tokens that condition memory retrieval
        # Shape: (batch, num_persistent_tokens, hidden_size)
        persistent_tokens = self.persistent_memory(batch_size)

        # Hook to capture and modify hidden states
        hidden_states_cache = {}

        def make_hook(layer_idx):
            def hook(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output

                # CRITICAL: Detach hidden from frozen base model and create fresh gradient path
                # This ensures gradients flow through MIRAS even with frozen base model
                hidden_detached = hidden.detach()

                # PERSISTENT MEMORY: Prepend learnable persistent tokens to condition memory
                # This follows Titans paper: [p_1, ..., p_N] || x
                # Persistent tokens learn task-related knowledge during training
                hidden_with_persistent = torch.cat([
                    persistent_tokens,  # (batch, num_persistent, hidden)
                    hidden_detached     # (batch, seq, hidden)
                ], dim=1)

                # Call memory module with persistent-augmented hidden states
                # Pass None to let module initialize internal states if needed
                mem_output_full, new_internal_mem, new_internal_momentum = self.memory_modules[layer_idx](
                    hidden_with_persistent,  # Use persistent-augmented hidden for memory
                    memory_states[layer_idx],  # None on first call, internal state thereafter
                    momentum_states[layer_idx],
                    return_memory_state=True,
                )

                # CRITICAL: Detach memory output to prevent double backward error
                # Memory module learns through internal surprise-based updates, not from main loss
                # The graph from memory's internal torch.autograd.grad() is already consumed
                mem_output_full = mem_output_full.detach()

                # Strip persistent token outputs - only keep sequence outputs
                num_persistent = persistent_tokens.shape[1]
                mem_output = mem_output_full[:, num_persistent:, :]  # (batch, seq, hidden)

                # Update internal memory states for next chunk
                memory_states[layer_idx] = new_internal_mem
                momentum_states[layer_idx] = new_internal_momentum

                # Memory output is already at hidden_size from memory module's output_proj
                # mem_output has shape (batch, seq, hidden_size)
                # CRITICAL FIX: Use per-position memory output, NOT mean pooling
                # Mean pooling loses per-position information (all positions get same memory)

                # Compute gate - use sequence means for gate conditioning
                # Gate shape: (batch, 1, hidden_size) for broadcasting across sequence
                gate_input = torch.cat([
                    hidden_detached.mean(dim=1),  # (batch, hidden_size)
                    mem_output.mean(dim=1)        # (batch, hidden_size)
                ], dim=-1)
                gate = self.memory_gates[layer_idx](gate_input).unsqueeze(1)  # (batch, 1, hidden)

                # Apply gated memory to hidden states PER-POSITION
                # mem_output: (batch, seq, hidden_size) - full per-position memory
                # gate: (batch, 1, hidden_size) - broadcasts across sequence
                # This preserves per-position memory retrieval from Titans paper
                hidden_modified = hidden_detached + gate * mem_output  # (batch, seq, hidden)

                hidden_states_cache[layer_idx] = hidden_modified

                if isinstance(output, tuple):
                    return (hidden_modified,) + output[1:]
                return hidden_modified
            return hook

        # Register hooks
        hooks = []
        for i, layer in enumerate(self.base_model.model.layers):
            hook = layer.register_forward_hook(make_hook(i))
            hooks.append(hook)

        try:
            # Forward through base model WITHOUT labels
            # We compute loss ourselves to ensure gradient flow through MIRAS hooks
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,  # Don't use base model's loss - we compute it ourselves
                output_hidden_states=False,
                return_dict=True,
            )
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()

        # Compute loss ourselves to ensure gradients flow through MIRAS modifications
        # The logits tensor carries the gradient graph from our hook modifications
        logits = outputs.logits
        loss = None
        if labels is not None:
            # Shift for causal LM loss (predict next token)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            # Compute loss - this ensures gradient flow through MIRAS
            loss = loss_fct(shift_logits, shift_labels)

        return loss, memory_states, momentum_states, logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        memory_states: Optional[List[torch.Tensor]] = None,
        momentum_states: Optional[List[torch.Tensor]] = None,
        return_memory_states: bool = True,
    ) -> Dict[str, Any]:
        """
        Forward pass with chunked processing for long sequences.

        CRITICAL MEMORY FIX: Only retain gradients for the LAST chunk.
        Previous chunks build up memory state but their losses are detached
        to prevent graph accumulation that causes OOM.

        This enables:
        1. Processing arbitrarily long sequences
        2. Accumulating memory across chunks
        3. Computing gradients only for the final chunk
        """
        batch_size, total_seq_len = input_ids.shape

        # Use a larger TRAINING chunk size for efficient GPU utilization
        # With 288GB/GPU, we can handle much larger chunks
        # 4096 tokens = ~200MB activations per layer, well within 288GB capacity
        # This reduces overhead from many small forward passes
        training_chunk_size = 4096  # Process 4K tokens per forward pass

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Process in chunks
        total_loss = 0.0
        num_chunks = 0
        all_logits = []

        # Count total chunks for gradient management
        total_chunks = (total_seq_len + training_chunk_size - 1) // training_chunk_size

        for chunk_idx, start_idx in enumerate(range(0, total_seq_len, training_chunk_size)):
            end_idx = min(start_idx + training_chunk_size, total_seq_len)
            is_last_chunk = (chunk_idx == total_chunks - 1)

            chunk_input_ids = input_ids[:, start_idx:end_idx]
            chunk_attention_mask = attention_mask[:, start_idx:end_idx]
            chunk_labels = labels[:, start_idx:end_idx] if labels is not None else None

            # CRITICAL: Only retain gradients for last chunk to prevent OOM
            # Previous chunks still update memory state, but loss is detached
            if is_last_chunk:
                # Last chunk: retain full gradient graph for training
                chunk_loss, memory_states, momentum_states, chunk_logits = self._chunk_forward(
                    chunk_input_ids,
                    chunk_attention_mask,
                    chunk_labels,
                    memory_states,
                    momentum_states,
                )
            else:
                # Earlier chunks: process without gradient to build memory state
                with torch.no_grad():
                    chunk_loss, memory_states, momentum_states, chunk_logits = self._chunk_forward(
                        chunk_input_ids,
                        chunk_attention_mask,
                        chunk_labels,
                        memory_states,
                        momentum_states,
                    )
                # Detach memory states to prevent graph retention
                if memory_states is not None:
                    memory_states = [m.detach() if m is not None else None for m in memory_states]
                if momentum_states is not None:
                    momentum_states = [m.detach() if m is not None else None for m in momentum_states]
                # Clear chunk loss - we don't use it for gradient
                chunk_loss = None
                # Free intermediate tensors from this chunk
                del chunk_logits
                chunk_logits = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if chunk_loss is not None:
                total_loss += chunk_loss
                num_chunks += 1

            # Only store logits from last chunk (with gradients) to save memory
            # For earlier chunks, we only needed to update memory state
            if is_last_chunk and chunk_logits is not None:
                all_logits.append(chunk_logits)

        # Average loss across chunks (should only have 1 chunk with loss now)
        avg_loss = total_loss / max(num_chunks, 1)

        # Logits only from the last chunk (for memory efficiency)
        logits = torch.cat(all_logits, dim=1) if all_logits else None

        result = {
            "loss": avg_loss,
            "logits": logits,
        }

        if return_memory_states:
            result["memory_states"] = memory_states
            result["momentum_states"] = momentum_states

        return result


# =============================================================================
# Data Loading for Long Context
# =============================================================================

class LongContextDataset(torch.utils.data.IterableDataset):
    """
    Dataset that concatenates texts to create long sequences.
    """

    def __init__(
        self,
        tokenizer,
        max_length: int,
        dataset_config: Dict[str, Any],
        hf_token: str,
        split: str = "train",
        num_samples: int = None,  # Limit samples for validation
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_config = dataset_config
        self.hf_token = hf_token
        self.split = split
        self.num_samples = num_samples  # None = unlimited
        self._dataset = None

    def _load_dataset(self):
        """Load official Dolma3 dataset - no fallbacks, only official AllenAI data."""
        dataset_name = self.dataset_config["name"]
        subset = self.dataset_config.get("subset")

        print(f"Loading official AllenAI dataset: {dataset_name}")

        # Load with subset if specified, otherwise load default
        if subset:
            print(f"  Subset: {subset}")
            dataset = load_dataset(
                dataset_name,
                subset,
                streaming=True,
                token=self.hf_token,
                split=self.split,
            )
        else:
            dataset = load_dataset(
                dataset_name,
                streaming=True,
                token=self.hf_token,
                split=self.split,
            )

        print(f"Successfully loaded {dataset_name} (split={self.split})")
        return dataset

    def __iter__(self):
        if self._dataset is None:
            self._dataset = self._load_dataset()

        buffer = []
        buffer_length = 0
        samples_yielded = 0

        # Get the text column name from config
        text_column = self.dataset_config.get("text_column", "text")

        for example in self._dataset:
            # Check sample limit for validation mode
            if self.num_samples is not None and samples_yielded >= self.num_samples:
                break
            # Get text from configurable column (Dolma3 uses 'text')
            text = example.get(text_column, example.get("content", ""))
            if not text:
                continue

            # Tokenize
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)
            buffer_length = len(buffer)

            # Yield full sequences
            while buffer_length >= self.max_length:
                sequence = buffer[:self.max_length]
                buffer = buffer[self.max_length:]
                buffer_length = len(buffer)

                input_ids = torch.tensor(sequence, dtype=torch.long)
                labels = input_ids.clone()
                attention_mask = torch.ones_like(input_ids)

                yield {
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": attention_mask,
                }
                samples_yielded += 1

                # Check sample limit after yielding
                if self.num_samples is not None and samples_yielded >= self.num_samples:
                    return


class SyntheticDataset(torch.utils.data.IterableDataset):
    """Synthetic dataset for fast pipeline validation."""

    def __init__(self, tokenizer, max_length: int, num_samples: int = 100):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_samples = num_samples
        self.vocab_size = tokenizer.vocab_size

    def __iter__(self):
        for i in range(self.num_samples):
            # Generate random tokens (excluding special tokens)
            input_ids = torch.randint(100, self.vocab_size - 100, (self.max_length,))
            labels = input_ids.clone()
            attention_mask = torch.ones_like(input_ids)
            yield {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            }


def create_dataloader(
    tokenizer,
    config: Dict[str, Any],
    dataset_config: Dict[str, Any],
    hf_token: str,
    rank: int,
    world_size: int,
    num_samples: int = None,  # Limit samples for validation
    use_synthetic: bool = False,  # Use synthetic data for fast validation
) -> DataLoader:
    """Create distributed dataloader for long-context training."""

    if use_synthetic:
        # Fast synthetic data for pipeline validation
        dataset = SyntheticDataset(
            tokenizer=tokenizer,
            max_length=config["max_seq_length"],
            num_samples=num_samples or 100,
        )
        num_workers = 0  # No need for workers with synthetic data
    else:
        dataset = LongContextDataset(
            tokenizer=tokenizer,
            max_length=config["max_seq_length"],
            dataset_config=dataset_config,
            hf_token=hf_token,
            num_samples=num_samples,
        )
        num_workers = config["num_workers"]

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size_per_gpu"],
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    output_dir: str,
    rank: int,
):
    """Save checkpoint (only on main process)."""
    if not is_main_process(rank):
        return

    checkpoint_dir = Path(output_dir) / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Get model state (unwrap DDP if needed)
    model_to_save = model.module if hasattr(model, 'module') else model

    # Save MIRAS modules only (base model is frozen)
    miras_state = {
        "memory_modules": model_to_save.memory_modules.state_dict(),
        "persistent_memory": model_to_save.persistent_memory.state_dict(),
        "memory_gates": model_to_save.memory_gates.state_dict(),
    }

    torch.save(miras_state, checkpoint_dir / "miras_modules.pt")
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")

    # Save training state
    with open(checkpoint_dir / "training_state.json", "w") as f:
        json.dump({"step": step}, f)

    print(f"Saved checkpoint at step {step}")


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
) -> int:
    """Load checkpoint and return step number."""
    model_to_load = model.module if hasattr(model, 'module') else model

    miras_state = torch.load(checkpoint_path / "miras_modules.pt")
    model_to_load.memory_modules.load_state_dict(miras_state["memory_modules"])
    model_to_load.persistent_memory.load_state_dict(miras_state["persistent_memory"])
    model_to_load.memory_gates.load_state_dict(miras_state["memory_gates"])

    optimizer.load_state_dict(torch.load(checkpoint_path / "optimizer.pt"))
    scheduler.load_state_dict(torch.load(checkpoint_path / "scheduler.pt"))

    with open(checkpoint_path / "training_state.json") as f:
        state = json.load(f)

    return state["step"]


# =============================================================================
# Training Loop
# =============================================================================

def train(
    model: nn.Module,
    train_data: DataLoader,
    tokenizer,
    config: Dict[str, Any],
    output_dir: str,
    logger: logging.Logger,
    rank: int,
    world_size: int,
    resume_from: Optional[str] = None,
):
    """Main training loop with distributed support."""
    device = torch.device(f"cuda:{rank}")

    # Optimizer (only MIRAS params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # Scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=config["max_steps"],
    )

    # Mixed precision - BFloat16 does NOT need GradScaler (same exponent range as FP32)
    # GradScaler is only needed for Float16 which has smaller dynamic range
    scaler = None  # Disabled for bf16

    # Resume if needed
    start_step = 0
    if resume_from:
        checkpoint_path = Path(resume_from)
        if checkpoint_path.exists():
            start_step = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
            logger.info(f"Resumed from step {start_step}")

    # Training
    model.train()
    global_step = start_step
    running_loss = 0.0
    accumulation_count = 0
    memory_states = None
    momentum_states = None

    logger.info(f"Starting training from step {start_step}")
    logger.info(f"World size: {world_size}")
    logger.info(f"Batch size per GPU: {config['batch_size_per_gpu']}")
    logger.info(f"Gradient accumulation: {config['gradient_accumulation_steps']}")
    logger.info(f"Effective batch size: {config['batch_size_per_gpu'] * world_size * config['gradient_accumulation_steps']}")
    logger.info(f"Sequence length: {config['max_seq_length']}")

    data_iter = iter(train_data)
    start_time = time.time()

    while global_step < config["max_steps"]:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_data)
            batch = next(data_iter)
            # Reset memory at epoch boundary
            memory_states = None
            momentum_states = None

        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        with autocast('cuda', dtype=torch.bfloat16 if config["bf16"] else torch.float32):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                memory_states=memory_states,
                momentum_states=momentum_states,
                return_memory_states=True,
            )

            loss = outputs["loss"] / config["gradient_accumulation_steps"]
            memory_states = outputs.get("memory_states")
            momentum_states = outputs.get("momentum_states")

        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        running_loss += loss.item() * config["gradient_accumulation_steps"]
        accumulation_count += 1

        # Log mini-batch progress (for real-time visibility)
        if is_main_process(rank):
            print(f"  [Accum {accumulation_count}/{config['gradient_accumulation_steps']}] loss: {loss.item() * config['gradient_accumulation_steps']:.4f}", flush=True)

        # Optimizer step
        if accumulation_count >= config["gradient_accumulation_steps"]:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, config["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(trainable_params, config["max_grad_norm"])
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            # Logging
            if global_step % config["log_every"] == 0 and is_main_process(rank):
                elapsed = time.time() - start_time
                tokens_processed = global_step * config["batch_size_per_gpu"] * world_size * config["gradient_accumulation_steps"] * config["max_seq_length"]
                tokens_per_sec = tokens_processed / elapsed
                steps_per_sec = global_step / elapsed
                eta_sec = (config["max_steps"] - global_step) / steps_per_sec if steps_per_sec > 0 else 0
                eta_hours = eta_sec / 3600

                print(
                    f">>> Step {global_step}/{config['max_steps']} | "
                    f"Loss: {running_loss / config['log_every']:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                    f"Tokens/sec: {tokens_per_sec:,.0f} | "
                    f"ETA: {eta_hours:.1f}h",
                    flush=True
                )
                running_loss = 0.0

            # Checkpoint
            if global_step % config["checkpoint_every"] == 0:
                save_checkpoint(model, optimizer, scheduler, global_step, output_dir, rank)

            accumulation_count = 0

    # Final checkpoint
    save_checkpoint(model, optimizer, scheduler, global_step, output_dir, rank)
    logger.info("Training complete!")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="OLMo2-1B + MIRAS Unlimited Context Training")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/olmo2_1b_unlimited",
                       help="Output directory")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Resume from checkpoint")
    parser.add_argument("--max_seq_length", type=int, default=65536,
                       help="Maximum sequence length (default: 64K)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size per GPU")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Limit to N samples for quick validation (default: None = unlimited)")
    parser.add_argument("--max_steps_override", type=int, default=None,
                       help="Override max_steps for validation (default: use config)")
    parser.add_argument("--synthetic", action="store_true",
                       help="Use synthetic data for fast pipeline validation")
    args = parser.parse_args()

    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # Update config
    TRAINING_CONFIG["max_seq_length"] = args.max_seq_length
    TRAINING_CONFIG["batch_size_per_gpu"] = args.batch_size

    # Output dir
    output_dir = Path(args.output_dir)
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)

    # Sync processes
    if world_size > 1:
        dist.barrier()

    # Setup logging
    logger = setup_logging(str(output_dir), rank)

    logger.info("=" * 60)
    logger.info("OLMo2-1B + MIRAS Unlimited Context Training")
    logger.info("=" * 60)
    logger.info(f"Rank: {rank}/{world_size}")
    logger.info(f"Device: {device}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Sequence length: {TRAINING_CONFIG['max_seq_length']}")

    # HF token
    hf_token = os.environ.get("HF_TOKEN", "")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    logger.info(f"Loading base model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16 if TRAINING_CONFIG["bf16"] else torch.float32,
        token=hf_token,
        trust_remote_code=True,
    ).to(device)

    logger.info(f"Base model loaded: {base_model.config.num_hidden_layers} layers")

    # Create MIRAS wrapper
    logger.info("Adding MIRAS memory modules...")
    model = OLMo2MIRASWrapper(
        base_model=base_model,
        miras_config=MIRAS_CONFIG_UNLIMITED,
        integration_mode="mal",
    ).to(device)

    logger.info(f"Total parameters: {model.get_total_params():,}")
    logger.info(f"Trainable parameters: {model.get_trainable_params():,}")

    # torch.compile() for 2x+ speedup on PyTorch 2.x
    # Use 'reduce-overhead' mode for lower latency (good for training)
    # Note: Compilation happens on first forward pass, may take 1-2 minutes
    if is_main_process(rank):
        print("[PERF] Compiling model with torch.compile(mode='reduce-overhead')...", flush=True)
    model = torch.compile(model, mode='reduce-overhead')

    # Wrap with DDP
    # find_unused_parameters=False is faster (no unused param detection overhead)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Create dataloader
    logger.info("Loading training data...")
    if args.synthetic:
        logger.info("VALIDATION MODE: Using SYNTHETIC data for fast pipeline testing")
    if args.num_samples:
        logger.info(f"VALIDATION MODE: Limiting to {args.num_samples} samples")
    train_data = create_dataloader(
        tokenizer=tokenizer,
        config=TRAINING_CONFIG,
        dataset_config=DATASET_CONFIG,
        hf_token=hf_token,
        rank=rank,
        world_size=world_size,
        num_samples=args.num_samples,
        use_synthetic=args.synthetic,
    )

    # Override max_steps for validation if specified
    if args.max_steps_override:
        TRAINING_CONFIG["max_steps"] = args.max_steps_override
        logger.info(f"VALIDATION MODE: max_steps overridden to {args.max_steps_override}")

    # Train
    train(
        model=model,
        train_data=train_data,
        tokenizer=tokenizer,
        config=TRAINING_CONFIG,
        output_dir=str(output_dir),
        logger=logger,
        rank=rank,
        world_size=world_size,
        resume_from=args.resume_from,
    )

    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
