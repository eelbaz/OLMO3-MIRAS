#!/usr/bin/env python3
"""
OLMo3-7B + MIRAS Training Script

Loads pretrained OLMo3-7B, injects MIRAS neural memory modules,
freezes base model, and trains only MIRAS for unlimited context.

Usage:
    python scripts/train_olmo3_7b_miras.py --resume_from checkpoint_dir

Requirements:
    - HuggingFace token with access to allenai/Olmo-3-1025-7B
    - NVIDIA GPU with sufficient memory (tested on GB10 with 119GB unified)
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast

# Add parent directory to path for imports
# When run from /workspace/olmo3_miras/scripts/, we need /workspace in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset

from olmo3_miras.memory.neural_memory import MIRASMemoryConfig, NeuralLongTermMemory, PersistentMemory


# =============================================================================
# Configuration
# =============================================================================

# MIRAS configuration optimized for OLMo3-7B
MIRAS_CONFIG_7B = MIRASMemoryConfig(
    hidden_size=4096,           # Match OLMo3-7B hidden size
    memory_hidden_size=2048,    # 0.5x hidden for efficiency
    memory_depth=2,             # 2-layer memory MLP (paper recommendation)
    num_memory_heads=16,        # Match attention heads
    use_momentum=True,          # Key MIRAS feature
    momentum_decay=0.9,         # Surprise decay rate
    learning_rate=0.1,          # Test-time learning rate
    forget_gate=True,           # Adaptive forgetting
    chunk_size=2048,            # Process in chunks for memory efficiency
    num_persistent_tokens=32,   # Learnable persistent context
    data_dependent_gates=True,  # Dynamic gating
    eps=1e-6,
    max_grad_norm=1.0,
    grad_scale=0.1,
)

# Training configuration
TRAINING_CONFIG = {
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "warmup_steps": 500,
    "max_steps": 50000,         # ~1-2B tokens
    "batch_size": 1,            # Per-GPU batch size (gradient accumulation handles larger effective batch)
    "gradient_accumulation_steps": 8,  # Effective batch size = 8
    "max_seq_length": 2048,     # Reduced for memory efficiency during training
    "checkpoint_every": 500,    # Save every 500 steps
    "log_every": 10,
    "eval_every": 500,
    "max_grad_norm": 1.0,
    "bf16": True,
    "gradient_checkpointing": True,
}

# Model paths
# For testing/validation, use smaller OLMo2-1B first
BASE_MODEL_SMALL = "allenai/OLMo-2-0425-1B"  # OLMo-2 1B for validation (correct name)
# For production training, use OLMo3-7B
BASE_MODEL_FULL = "allenai/OLMo-3-1025-7B"   # OLMo-3 7B for final training

# Default to small model for testing
BASE_MODEL = BASE_MODEL_SMALL

# Dataset
DATASET_CONFIG = {
    "name": "allenai/dolma3_longmino_mix",  # Long-context data
    "fallback": "Salesforce/wikitext",       # Fallback that works with modern datasets lib
    "fallback_config": "wikitext-103-v1",    # Specific config for wikitext
    "streaming": True,
}


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging to file and console."""
    logger = logging.getLogger("olmo3_miras")
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    log_file = Path(output_dir) / "training.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger


# =============================================================================
# MIRAS-Enhanced OLMo3 Model
# =============================================================================

class OLMo3MIRASWrapper(nn.Module):
    """
    Wrapper that adds MIRAS neural memory to a pretrained OLMo model.

    This approach:
    1. Loads the pretrained OLMo model
    2. Injects MIRAS memory modules into each decoder layer
    3. Freezes base model weights
    4. Only trains MIRAS modules
    """

    def __init__(
        self,
        base_model: nn.Module,
        miras_config: MIRASMemoryConfig,
        integration_mode: str = "mal",  # Memory As Layer
    ):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.miras_config = miras_config
        self.integration_mode = integration_mode

        # Get number of layers
        self.num_layers = self.config.num_hidden_layers

        # Create MIRAS memory modules for each layer
        self.neural_memories = nn.ModuleList([
            NeuralLongTermMemory(miras_config)
            for _ in range(self.num_layers)
        ])

        # Persistent memory (shared across layers)
        self.persistent_memory = PersistentMemory(miras_config)

        # Output gating for combining attention and memory outputs
        if integration_mode == "mag":  # Memory As Gate
            self.memory_gates = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(miras_config.hidden_size * 2, miras_config.hidden_size),
                    nn.Sigmoid()
                )
                for _ in range(self.num_layers)
            ])

        # Layer norms for memory outputs
        self.memory_norms = nn.ModuleList([
            nn.RMSNorm(miras_config.hidden_size, eps=1e-5)
            for _ in range(self.num_layers)
        ])

        # Freeze base model
        self._freeze_base_model()

        # Initialize MIRAS weights
        self._init_miras_weights()

        # Convert MIRAS modules to match base model dtype
        self._convert_miras_dtype()

    def _freeze_base_model(self):
        """Freeze all base model parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = False

    def _init_miras_weights(self):
        """Initialize MIRAS module weights."""
        for module in self.modules():
            if isinstance(module, NeuralLongTermMemory):
                for submodule in module.modules():
                    if isinstance(submodule, nn.Linear):
                        nn.init.xavier_uniform_(submodule.weight, gain=0.1)
                        if submodule.bias is not None:
                            nn.init.zeros_(submodule.bias)

    def _convert_miras_dtype(self):
        """Convert MIRAS modules to match base model dtype."""
        # Get dtype from base model (typically bfloat16)
        base_dtype = next(self.base_model.parameters()).dtype
        base_device = next(self.base_model.parameters()).device

        # Convert all MIRAS modules to the same dtype
        for mem in self.neural_memories:
            mem.to(dtype=base_dtype, device=base_device)

        self.persistent_memory.to(dtype=base_dtype, device=base_device)

        for norm in self.memory_norms:
            norm.to(dtype=base_dtype, device=base_device)

        if hasattr(self, 'memory_gates'):
            for gate in self.memory_gates:
                gate.to(dtype=base_dtype, device=base_device)

    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        memory_states: Optional[list] = None,
        momentum_states: Optional[list] = None,
        return_memory_states: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass with MIRAS memory integration.

        The base model's forward is modified through hooks to inject memory.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Initialize memory states if not provided
        if memory_states is None:
            memory_states = [None] * self.num_layers
        if momentum_states is None:
            momentum_states = [None] * self.num_layers

        # Get persistent memory
        persistent_mem = self.persistent_memory(batch_size)

        # Hook to capture hidden states from each layer
        layer_outputs = []
        new_memory_states = []
        new_momentum_states = []

        # Register hooks to capture and modify layer outputs
        hooks = []
        layer_idx = [0]  # Use list to allow modification in closure

        def make_hook(idx):
            def hook(module, input, output):
                # Handle different output formats from different model architectures
                # OLMo2 returns a tuple where first element is hidden_states (batch, seq, hidden)
                # But sometimes it might be just a tensor
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    rest_output = output[1:]
                else:
                    hidden_states = output
                    rest_output = ()

                # Ensure hidden_states has 3 dimensions (batch, seq, hidden)
                if hidden_states.dim() == 2:
                    # Shape is (seq, hidden) - add batch dimension
                    hidden_states = hidden_states.unsqueeze(0)

                # Apply MIRAS memory
                memory_output, new_mem, new_mom = self.neural_memories[idx](
                    hidden_states,
                    memory_state=memory_states[idx],
                    momentum_state=momentum_states[idx],
                    return_memory_state=True
                )

                # Normalize memory output
                memory_output = self.memory_norms[idx](memory_output)

                # Integrate memory based on mode
                if self.integration_mode == "mal":
                    # Memory as Layer: Add memory output to hidden states
                    enhanced_hidden = hidden_states + memory_output
                elif self.integration_mode == "mag":
                    # Memory as Gate: Gate between attention and memory
                    gate_input = torch.cat([hidden_states, memory_output], dim=-1)
                    gate = self.memory_gates[idx](gate_input)
                    enhanced_hidden = gate * hidden_states + (1 - gate) * memory_output
                else:
                    enhanced_hidden = hidden_states

                new_memory_states.append(new_mem)
                new_momentum_states.append(new_mom)

                # Return modified output in same format as input
                if rest_output:
                    return (enhanced_hidden,) + rest_output
                else:
                    return enhanced_hidden
            return hook

        # Register hooks on decoder layers
        for idx, layer in enumerate(self.base_model.model.layers):
            hook = layer.register_forward_hook(make_hook(idx))
            hooks.append(hook)

        try:
            # Forward through base model
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=False,
                return_dict=True,
            )
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()

        result = {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

        if return_memory_states:
            result["memory_states"] = new_memory_states
            result["momentum_states"] = new_momentum_states

        return result


# =============================================================================
# Data Loading
# =============================================================================

def create_dataloader(
    tokenizer,
    config: Dict[str, Any],
    hf_token: str,
) -> torch.utils.data.DataLoader:
    """Create streaming dataloader for long-context training."""

    # Try primary dataset, fall back if needed
    try:
        dataset = load_dataset(
            config["name"],
            streaming=config["streaming"],
            token=hf_token,
            split="train",
        )
        print(f"Loaded dataset: {config['name']}")
    except Exception as e:
        print(f"Failed to load {config['name']}: {e}")
        fallback_name = config['fallback']
        fallback_config = config.get('fallback_config', None)
        print(f"Falling back to {fallback_name}" + (f" ({fallback_config})" if fallback_config else ""))
        dataset = load_dataset(
            fallback_name,
            fallback_config,  # This is the config name like 'wikitext-103-v1'
            streaming=True,
            split="train",
        )

    def tokenize_fn(examples):
        """Tokenize and chunk text."""
        # Get text field (might be 'text' or 'content')
        text_field = "text" if "text" in examples else list(examples.keys())[0]
        texts = examples[text_field]

        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=TRAINING_CONFIG["max_seq_length"],
            padding="max_length",
            return_tensors="pt",
        )

        # Labels are shifted input_ids
        tokenized["labels"] = tokenized["input_ids"].clone()

        return tokenized

    # Map tokenization
    dataset = dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=TRAINING_CONFIG["batch_size"],
        remove_columns=dataset.column_names if hasattr(dataset, 'column_names') else None,
    )

    return dataset


# =============================================================================
# Training Loop
# =============================================================================

def save_checkpoint(
    model: OLMo3MIRASWrapper,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    loss: float,
    output_dir: str,
    memory_states: Optional[list] = None,
    momentum_states: Optional[list] = None,
):
    """Save training checkpoint."""
    checkpoint_dir = Path(output_dir) / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save MIRAS modules only (base model is frozen)
    miras_state_dict = {
        k: v for k, v in model.state_dict().items()
        if "neural_memories" in k or "memory_norms" in k or "persistent_memory" in k
        or "memory_gates" in k
    }

    checkpoint = {
        "step": step,
        "loss": loss,
        "miras_state_dict": miras_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "config": {
            "miras_config": model.miras_config.__dict__,
            "integration_mode": model.integration_mode,
            "base_model": BASE_MODEL,
        },
    }

    torch.save(checkpoint, checkpoint_dir / "checkpoint.pt")

    # Save memory states if provided
    if memory_states is not None:
        memory_checkpoint = {
            "memory_states": [m.cpu() if m is not None else None for m in memory_states],
            "momentum_states": [m.cpu() if m is not None else None for m in momentum_states],
        }
        torch.save(memory_checkpoint, checkpoint_dir / "memory_states.pt")

    # Save config as JSON
    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump(checkpoint["config"], f, indent=2, default=str)

    print(f"Saved checkpoint to {checkpoint_dir}")
    return checkpoint_dir


def load_checkpoint(
    checkpoint_path: str,
    model: OLMo3MIRASWrapper,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> int:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path / "checkpoint.pt", map_location="cpu")

    # Load MIRAS state dict
    model.load_state_dict(checkpoint["miras_state_dict"], strict=False)

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint["step"]


def train(
    model: OLMo3MIRASWrapper,
    train_data,
    tokenizer,
    config: Dict[str, Any],
    output_dir: str,
    logger: logging.Logger,
    resume_from: Optional[str] = None,
):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Optimizer - only train MIRAS parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.95),
    )

    # Scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=config["max_steps"],
    )

    # Mixed precision
    scaler = GradScaler('cuda') if config["bf16"] else None

    # Resume from checkpoint
    start_step = 0
    if resume_from:
        checkpoint_path = Path(resume_from)
        if checkpoint_path.exists():
            start_step = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
            logger.info(f"Resumed from step {start_step}")

    # Training loop
    model.train()
    global_step = start_step
    running_loss = 0.0
    accumulation_count = 0
    memory_states = None
    momentum_states = None

    logger.info(f"Starting training from step {start_step}")
    logger.info(f"Trainable parameters: {model.get_trainable_params():,}")
    logger.info(f"Total parameters: {model.get_total_params():,}")

    data_iter = iter(train_data)
    start_time = time.time()

    while global_step < config["max_steps"]:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_data)
            batch = next(data_iter)
            # Reset memory states at epoch boundary
            memory_states = None
            momentum_states = None

        # Prepare batch
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(device)
        labels = batch["labels"].to(device)

        # Handle batch dimension
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            labels = labels.unsqueeze(0)

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

        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        running_loss += loss.item() * config["gradient_accumulation_steps"]
        accumulation_count += 1

        # Update memory states (detach to prevent gradient accumulation)
        memory_states = [m.detach() if m is not None else None for m in outputs.get("memory_states", [])]
        momentum_states = [m.detach() if m is not None else None for m in outputs.get("momentum_states", [])]

        # Gradient accumulation step
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
            accumulation_count = 0

            # Logging
            if global_step % config["log_every"] == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = (global_step * config["batch_size"] * config["gradient_accumulation_steps"] * config["max_seq_length"]) / elapsed

                logger.info(
                    f"Step {global_step}/{config['max_steps']} | "
                    f"Loss: {running_loss / config['log_every']:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                    f"Tokens/s: {tokens_per_sec:.1f}"
                )
                running_loss = 0.0

            # Checkpoint
            if global_step % config["checkpoint_every"] == 0:
                save_checkpoint(
                    model, optimizer, scheduler, global_step,
                    loss.item(), output_dir, memory_states, momentum_states
                )

    # Final checkpoint
    save_checkpoint(
        model, optimizer, scheduler, global_step,
        loss.item(), output_dir, memory_states, momentum_states
    )

    logger.info("Training complete!")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train OLMo3-7B with MIRAS")
    parser.add_argument("--resume_from", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/olmo3_7b_miras",
                       help="Output directory for checkpoints")
    parser.add_argument("--integration_mode", type=str, default="mal",
                       choices=["mac", "mag", "mal"],
                       help="MIRAS integration mode")
    parser.add_argument("--test_run", action="store_true",
                       help="Run a quick test without full training")
    parser.add_argument("--use_full_model", action="store_true",
                       help="Use OLMo3-7B instead of OLMo2-1B for validation")
    args = parser.parse_args()

    # Select model based on flag
    global BASE_MODEL
    if args.use_full_model:
        BASE_MODEL = BASE_MODEL_FULL
    else:
        BASE_MODEL = BASE_MODEL_SMALL

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(output_dir))

    # Get HuggingFace token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        # Try reading from ~/.bashrc
        bashrc_path = Path.home() / ".bashrc"
        if bashrc_path.exists():
            with open(bashrc_path) as f:
                for line in f:
                    if "HF_TOKEN=" in line:
                        hf_token = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break

    if not hf_token:
        logger.error("HF_TOKEN not found. Set it via environment or ~/.bashrc")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("OLMo3-7B + MIRAS Training")
    logger.info("=" * 60)
    logger.info(f"Base model: {BASE_MODEL}")
    logger.info(f"Integration mode: {args.integration_mode}")
    logger.info(f"Output directory: {output_dir}")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        token=hf_token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    logger.info(f"Loading base model: {BASE_MODEL}")
    logger.info("This may take a few minutes for 7B parameters...")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16 if TRAINING_CONFIG["bf16"] else torch.float32,
        token=hf_token,
        trust_remote_code=True,
        device_map="auto",  # Automatic device placement
    )

    logger.info(f"Base model loaded: {base_model.config.num_hidden_layers} layers, "
                f"{base_model.config.hidden_size} hidden size")

    # Update MIRAS config to match base model
    # Use smaller chunk_size and memory_hidden_size to fit in memory during training
    is_large_model = base_model.config.hidden_size >= 4096
    miras_config = MIRASMemoryConfig(
        hidden_size=base_model.config.hidden_size,
        memory_hidden_size=base_model.config.hidden_size // 4 if is_large_model else base_model.config.hidden_size // 2,
        memory_depth=2,
        num_memory_heads=base_model.config.num_attention_heads // 2 if is_large_model else base_model.config.num_attention_heads,
        use_momentum=True,
        momentum_decay=0.9,
        learning_rate=0.1,
        forget_gate=True,
        chunk_size=64 if is_large_model else 256,  # Smaller chunks for training memory efficiency
        num_persistent_tokens=16 if is_large_model else 32,
        data_dependent_gates=True,
    )

    logger.info(f"MIRAS config for {'large' if is_large_model else 'small'} model:")
    logger.info(f"  memory_hidden_size: {miras_config.memory_hidden_size}")
    logger.info(f"  chunk_size: {miras_config.chunk_size}")

    # Create MIRAS-enhanced model
    logger.info("Adding MIRAS memory modules...")
    model = OLMo3MIRASWrapper(
        base_model,
        miras_config,
        integration_mode=args.integration_mode,
    )

    logger.info(f"Total parameters: {model.get_total_params():,}")
    logger.info(f"Trainable parameters: {model.get_trainable_params():,}")
    logger.info(f"MIRAS parameters: {model.get_trainable_params() / 1e6:.2f}M")

    if args.test_run:
        # Quick test with training steps
        logger.info("Running quick test...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Test forward pass (use shorter seq for memory efficiency)
        test_input = torch.randint(0, tokenizer.vocab_size, (1, 64), device=device)
        test_labels = test_input.clone()

        with torch.no_grad():
            outputs = model(
                input_ids=test_input,
                labels=test_labels,
                return_memory_states=True,
            )

        logger.info(f"Test loss: {outputs['loss'].item():.4f}")
        logger.info(f"Memory states: {len(outputs.get('memory_states', []))} layers")

        # Test training with gradient flow
        logger.info("Testing training step with gradient flow...")
        torch.cuda.empty_cache()  # Free memory from inference test
        model.train()

        # Get only trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=5e-5, weight_decay=0.01)

        losses = []
        for step in range(5):
            optimizer.zero_grad()

            outputs = model(
                input_ids=test_input,
                labels=test_labels,
            )
            loss = outputs['loss']
            loss.backward()

            # Check gradients flowed to MIRAS modules
            if step == 0:
                miras_grads = 0
                for name, param in model.named_parameters():
                    if param.grad is not None and param.grad.abs().sum() > 0:
                        if "neural_memory" in name or "persistent_memory" in name or "memory_norms" in name:
                            miras_grads += 1

                logger.info(f"MIRAS params with gradients: {miras_grads}")

            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            losses.append(loss.item())
            logger.info(f"  Step {step}: Loss = {loss.item():.4f}")

        loss_decreased = losses[-1] < losses[0]
        logger.info(f"Initial loss: {losses[0]:.4f}")
        logger.info(f"Final loss: {losses[-1]:.4f}")
        logger.info(f"Loss decreased: {loss_decreased}")

        if loss_decreased:
            logger.info("Training test passed!")
        else:
            logger.info("WARNING: Loss did not decrease (may need more steps)")

        return

    # Load training data
    logger.info("Loading training data...")
    train_data = create_dataloader(tokenizer, DATASET_CONFIG, hf_token)

    # Train
    train(
        model=model,
        train_data=train_data,
        tokenizer=tokenizer,
        config=TRAINING_CONFIG,
        output_dir=str(output_dir),
        logger=logger,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
