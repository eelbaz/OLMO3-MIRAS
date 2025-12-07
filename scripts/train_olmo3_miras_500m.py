#!/usr/bin/env python3
"""
OLMo3-MIRAS 500M Training Script

Three-stage training pipeline following OLMo3 recipe:
1. Pretraining on Dolma3 pool (web, papers, code)
2. Mid-training on Dolmino (instruction, math, code)
3. Long-context training with MIRAS memory activation

Usage:
    # Stage 1: Pretraining
    torchrun --nproc_per_node=8 train_olmo3_miras_500m.py --stage pretrain

    # Stage 2: Mid-training (from pretrained checkpoint)
    torchrun --nproc_per_node=8 train_olmo3_miras_500m.py --stage midtrain \
        --resume_from ./checkpoints/pretrain/final

    # Stage 3: Long-context
    torchrun --nproc_per_node=8 train_olmo3_miras_500m.py --stage longcontext \
        --resume_from ./checkpoints/midtrain/final
"""

import os
import sys
import math
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Iterator

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset
from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from olmo3_miras.model.olmo3_miras import OLMo3MIRASConfig, OLMo3MIRASForCausalLM
from olmo3_miras.memory.neural_memory import MIRASMemoryConfig
from olmo3_miras.configs.model_configs import (
    get_model_config,
    get_training_config,
    TrainingStage,
    calculate_training_steps,
    calculate_warmup_steps,
    DOLMA3_DATASETS,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Dolma3StreamingDataset(IterableDataset):
    """
    Streaming dataset for Dolma3 with proper tokenization and chunking.
    Handles the massive scale of Dolma3 efficiently.
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        context_length: int,
        seed: int = 42,
        world_size: int = 1,
        rank: int = 0,
        buffer_size: int = 10000,
    ):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.seed = seed
        self.world_size = world_size
        self.rank = rank
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Stream and tokenize documents."""
        try:
            # Try loading the dataset
            dataset = load_dataset(
                self.dataset_name,
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
        except Exception as e:
            logger.warning(f"Could not load {self.dataset_name}: {e}")
            # Fallback to a sample dataset for testing
            logger.info("Using fallback dataset for development...")
            dataset = load_dataset(
                "wikitext",
                "wikitext-103-raw-v1",
                split="train",
                streaming=True,
            )

        # Shard for distributed training
        dataset = dataset.shuffle(seed=self.seed, buffer_size=self.buffer_size)

        token_buffer = []
        for i, example in enumerate(dataset):
            # Skip examples not for this rank
            if i % self.world_size != self.rank:
                continue

            # Get text (handle different column names)
            text = example.get("text", example.get("content", ""))
            if not text or len(text) < 10:
                continue

            # Tokenize
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            token_buffer.extend(tokens)

            # Yield complete sequences
            while len(token_buffer) >= self.context_length + 1:
                input_ids = torch.tensor(token_buffer[:self.context_length])
                labels = torch.tensor(token_buffer[1:self.context_length + 1])
                token_buffer = token_buffer[self.context_length:]

                yield {
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": torch.ones_like(input_ids),
                }


def create_model(config: Dict[str, Any], device: torch.device) -> OLMo3MIRASForCausalLM:
    """Create and initialize the OLMo3-MIRAS model."""
    # Create MIRAS config
    miras_dict = config.pop("miras_config", {})
    miras_config = MIRASMemoryConfig(**miras_dict)

    # Create model config
    model_config = OLMo3MIRASConfig(
        miras_config=miras_config,
        **config
    )

    # Initialize model
    model = OLMo3MIRASForCausalLM(model_config)

    # Enable gradient checkpointing for memory efficiency
    if hasattr(model.model, 'gradient_checkpointing_enable'):
        model.model.gradient_checkpointing_enable()
    else:
        model.model.gradient_checkpointing = True

    return model.to(device)


def create_optimizer(
    model: nn.Module,
    learning_rate: float,
    memory_learning_rate: float,
    weight_decay: float,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.95,
    adam_epsilon: float = 1e-8,
) -> torch.optim.AdamW:
    """Create optimizer with separate learning rates for memory modules."""
    memory_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "neural_memory" in name or "persistent_memory" in name:
            memory_params.append(param)
        else:
            other_params.append(param)

    logger.info(f"Model params: {len(other_params)} regular, {len(memory_params)} memory")

    optimizer_groups = [
        {
            "params": other_params,
            "lr": learning_rate,
            "weight_decay": weight_decay,
        },
        {
            "params": memory_params,
            "lr": memory_learning_rate,
            "weight_decay": weight_decay * 0.1,  # Less decay for memory
        },
    ]

    return torch.optim.AdamW(
        optimizer_groups,
        betas=(adam_beta1, adam_beta2),
        eps=adam_epsilon,
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    loss: float,
    output_dir: Path,
    memory_states: Optional[list] = None,
    momentum_states: Optional[list] = None,
):
    """Save training checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get underlying model if wrapped in DDP
    save_model = model.module if hasattr(model, 'module') else model

    checkpoint = {
        "step": step,
        "loss": loss,
        "model_state_dict": save_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    }

    torch.save(checkpoint, output_dir / "checkpoint.pt")

    # Save config
    save_model.config.save_pretrained(output_dir)

    # Save memory states if available
    if memory_states or momentum_states:
        torch.save({
            "memory_states": memory_states,
            "momentum_states": momentum_states,
        }, output_dir / "memory_states.pt")

    logger.info(f"Saved checkpoint at step {step} to {output_dir}")


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler = None,
) -> int:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path / "checkpoint.pt", map_location="cpu")

    # Get underlying model if wrapped in DDP
    load_model = model.module if hasattr(model, 'module') else model
    load_model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    logger.info(f"Loaded checkpoint from step {checkpoint['step']}")
    return checkpoint["step"]


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    max_grad_norm: float,
    device: torch.device,
    accumulation_steps: int = 1,
    current_step: int = 0,
) -> float:
    """Execute a single training step with mixed precision."""
    model.train()

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss / accumulation_steps

    scaler.scale(loss).backward()

    # Step optimizer on accumulation boundary
    if (current_step + 1) % accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return loss.item() * accumulation_steps


def train(
    stage: TrainingStage,
    model_size: str = "500M",
    output_dir: str = "./checkpoints",
    resume_from: Optional[str] = None,
    local_rank: int = 0,
    world_size: int = 1,
):
    """Main training loop."""
    # Setup distributed if available
    is_distributed = world_size > 1

    if is_distributed:
        dist.init_process_group(backend="nccl")
        local_rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}")
    is_main = local_rank == 0

    # Load configs
    model_config = get_model_config(model_size)
    train_config = get_training_config(stage, model_size)

    if is_main:
        logger.info(f"Starting {stage.value} training for {model_size} model")
        logger.info(f"Training config: {json.dumps(train_config, indent=2, default=str)}")

    # Setup output directory
    output_path = Path(output_dir) / stage.value / model_size
    output_path.mkdir(parents=True, exist_ok=True)

    # Load tokenizer (OLMo3 uses OLMo-2 tokenizer)
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-7B-1124")
    tokenizer.pad_token = tokenizer.eos_token

    # Create model
    model = create_model(model_config.copy(), device)

    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
        logger.info(f"Trainable parameters: {trainable_params:,}")

    # Wrap in DDP if distributed
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Create optimizer
    optimizer = create_optimizer(
        model,
        learning_rate=train_config["learning_rate"],
        memory_learning_rate=train_config["memory_learning_rate"],
        weight_decay=train_config["weight_decay"],
        adam_beta1=train_config.get("adam_beta1", 0.9),
        adam_beta2=train_config.get("adam_beta2", 0.95),
    )

    # Calculate steps
    total_steps = calculate_training_steps(train_config)
    warmup_steps = calculate_warmup_steps(train_config)

    # Create scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Create dataset
    dataset_config = train_config["dataset"]
    dataset = Dolma3StreamingDataset(
        dataset_name=dataset_config["name"],
        tokenizer=tokenizer,
        context_length=train_config["context_length"],
        world_size=world_size,
        rank=local_rank,
    )

    # Calculate batch size from tokens
    tokens_per_batch = train_config["batch_size_tokens"]
    context_length = train_config["context_length"]
    micro_batch_size = max(1, tokens_per_batch // context_length // world_size // 8)
    accumulation_steps = max(1, tokens_per_batch // (context_length * micro_batch_size * world_size))

    if is_main:
        logger.info(f"Micro batch size: {micro_batch_size}")
        logger.info(f"Gradient accumulation steps: {accumulation_steps}")
        logger.info(f"Effective batch size: {micro_batch_size * accumulation_steps * world_size} sequences")
        logger.info(f"Tokens per step: {micro_batch_size * accumulation_steps * world_size * context_length:,}")

    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        num_workers=4,
        pin_memory=True,
    )

    # Load checkpoint if resuming
    start_step = 0
    if resume_from:
        resume_path = Path(resume_from)
        if resume_path.exists():
            start_step = load_checkpoint(resume_path, model, optimizer, scheduler)

    # Training loop
    scaler = GradScaler()
    model.train()
    optimizer.zero_grad()

    running_loss = 0.0
    log_interval = 10
    save_interval = 1000

    if is_main:
        logger.info(f"Starting training from step {start_step}")
        logger.info(f"Total steps: {total_steps:,}")

    step = start_step
    for batch in dataloader:
        if step >= total_steps:
            break

        loss = train_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            scaler=scaler,
            max_grad_norm=train_config["max_grad_norm"],
            device=device,
            accumulation_steps=accumulation_steps,
            current_step=step,
        )

        running_loss += loss

        # Step scheduler
        if (step + 1) % accumulation_steps == 0:
            scheduler.step()

        step += 1

        # Logging
        if is_main and step % log_interval == 0:
            avg_loss = running_loss / log_interval
            lr = optimizer.param_groups[0]["lr"]
            mem_lr = optimizer.param_groups[1]["lr"]
            logger.info(
                f"Step {step}/{total_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {lr:.2e} | "
                f"Mem LR: {mem_lr:.2e}"
            )
            running_loss = 0.0

        # Save checkpoint
        if is_main and step % save_interval == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                loss=loss,
                output_dir=output_path / f"step_{step}",
            )

    # Save final checkpoint
    if is_main:
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            loss=loss,
            output_dir=output_path / "final",
        )
        logger.info(f"Training complete! Final checkpoint saved to {output_path / 'final'}")

    if is_distributed:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Train OLMo3-MIRAS")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["pretrain", "midtrain", "longcontext"],
        required=True,
        help="Training stage"
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="500M",
        choices=["500M", "1B"],
        help="Model size"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="Local rank for distributed training"
    )

    args = parser.parse_args()

    # Get world size from environment
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))

    # Map stage string to enum
    stage_map = {
        "pretrain": TrainingStage.PRETRAIN,
        "midtrain": TrainingStage.MIDTRAIN,
        "longcontext": TrainingStage.LONGCONTEXT,
    }

    train(
        stage=stage_map[args.stage],
        model_size=args.model_size,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        local_rank=local_rank,
        world_size=world_size,
    )


if __name__ == "__main__":
    main()
