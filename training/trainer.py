"""
Training pipeline for OLMo3-MIRAS with support for
long-context training and memory state management.
"""

from __future__ import annotations
import os
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast

from transformers import (
    Trainer,
    TrainingArguments,
    get_cosine_schedule_with_warmup
)

from ..model.olmo3_miras import OLMo3MIRASForCausalLM, OLMo3MIRASConfig


@dataclass
class OLMo3MIRASTrainingArguments(TrainingArguments):
    """Extended training arguments for OLMo3-MIRAS."""
    
    # Memory-specific arguments
    memory_warmup_steps: int = 1000
    memory_learning_rate: float = 1e-4
    reset_memory_every_n_steps: int = 0  # 0 = never reset
    
    # Long-context training
    chunk_training: bool = True
    chunk_size: int = 4096
    gradient_accumulation_across_chunks: bool = True
    
    # Memory state persistence
    save_memory_states: bool = True
    
    # Curriculum learning for context length
    curriculum_learning: bool = True
    min_context_length: int = 2048
    max_context_length: int = 65536
    curriculum_warmup_steps: int = 5000


class ChunkedDataset(Dataset):
    """
    Dataset wrapper that handles chunked long sequences
    for memory-efficient training.
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        chunk_size: int,
        max_seq_length: int,
        overlap: int = 0
    ):
        self.base_dataset = base_dataset
        self.chunk_size = chunk_size
        self.max_seq_length = max_seq_length
        self.overlap = overlap
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.base_dataset[idx]
        input_ids = item["input_ids"]
        
        # Chunk the sequence
        seq_len = len(input_ids)
        chunks = []
        
        start = 0
        while start < seq_len:
            end = min(start + self.chunk_size, seq_len)
            chunks.append(input_ids[start:end])
            start = end - self.overlap if self.overlap > 0 else end
            
        return {
            "input_ids": input_ids[:self.max_seq_length],
            "chunks": chunks,
            "labels": item.get("labels", input_ids)[:self.max_seq_length]
        }


class OLMo3MIRASTrainer(Trainer):
    """
    Custom trainer for OLMo3-MIRAS with memory state management.
    """
    
    def __init__(
        self,
        model: OLMo3MIRASForCausalLM,
        args: OLMo3MIRASTrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        **kwargs
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs
        )
        self.memory_states = None
        self.momentum_states = None
        self.current_context_length = args.min_context_length
        
    def _get_current_context_length(self, step: int) -> int:
        """Get context length for curriculum learning."""
        if not self.args.curriculum_learning:
            return self.args.max_context_length
            
        if step < self.args.curriculum_warmup_steps:
            progress = step / self.args.curriculum_warmup_steps
            length = int(
                self.args.min_context_length + 
                progress * (self.args.max_context_length - self.args.min_context_length)
            )
            # Round to nearest power of 2 for efficiency
            return 2 ** int(math.log2(length))
        return self.args.max_context_length
    
    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        num_items_in_batch: Optional[int] = None
    ) -> torch.Tensor:
        """
        Custom training step with memory state management.
        """
        model.train()
        
        # Update context length for curriculum learning
        self.current_context_length = self._get_current_context_length(self.state.global_step)
        
        # Prepare inputs
        inputs = self._prepare_inputs(inputs)
        
        # Get memory states
        memory_states = self.memory_states
        momentum_states = self.momentum_states
        
        # Reset memory if configured
        if (self.args.reset_memory_every_n_steps > 0 and 
            self.state.global_step % self.args.reset_memory_every_n_steps == 0):
            memory_states = None
            momentum_states = None
        
        # Chunk training for long sequences
        if self.args.chunk_training and inputs["input_ids"].shape[1] > self.args.chunk_size:
            loss = self._chunked_training_step(model, inputs, memory_states, momentum_states)
        else:
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(
                    **inputs,
                    memory_states=memory_states,
                    momentum_states=momentum_states
                )
                loss = outputs.loss

            # Update memory states for next batch (stored in model outputs dict)
            if isinstance(outputs, dict):
                if 'memory_states' in outputs:
                    self.memory_states = [m.detach() if m is not None else None
                                         for m in outputs['memory_states']]
                if 'momentum_states' in outputs:
                    self.momentum_states = [m.detach() if m is not None else None
                                            for m in outputs['momentum_states']]
        
        return loss
    
    def _chunked_training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        memory_states: Optional[List[torch.Tensor]],
        momentum_states: Optional[List[torch.Tensor]]
    ) -> torch.Tensor:
        """
        Training step with chunked processing for long sequences.
        Memory states are carried across chunks.
        """
        input_ids = inputs["input_ids"]
        labels = inputs.get("labels", input_ids)
        attention_mask = inputs.get("attention_mask")
        
        batch_size, seq_len = input_ids.shape
        chunk_size = self.args.chunk_size
        
        total_loss = 0.0
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        
        current_memory = memory_states
        current_momentum = momentum_states
        
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, seq_len)
            
            chunk_inputs = {
                "input_ids": input_ids[:, start:end],
                "labels": labels[:, start:end] if labels is not None else None,
            }
            
            if attention_mask is not None:
                chunk_inputs["attention_mask"] = attention_mask[:, start:end]
                
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(
                    **chunk_inputs,
                    memory_states=current_memory,
                    momentum_states=current_momentum,
                    use_cache=False
                )

            chunk_loss = outputs.loss
            total_loss += chunk_loss

            # Carry memory states to next chunk
            if isinstance(outputs, dict):
                if 'memory_states' in outputs:
                    current_memory = [m.detach() if m is not None else None
                                     for m in outputs['memory_states']]
                if 'momentum_states' in outputs:
                    current_momentum = [m.detach() if m is not None else None
                                       for m in outputs['momentum_states']]
                
            # Gradient accumulation across chunks if enabled
            if self.args.gradient_accumulation_across_chunks:
                (chunk_loss / num_chunks).backward()
            else:
                chunk_loss.backward()
                
        # Store final memory states
        self.memory_states = current_memory
        self.momentum_states = current_momentum
        
        return total_loss / num_chunks
    
    def save_model(self, output_dir: Optional[str] = None, **kwargs):
        """Save model with memory states."""
        super().save_model(output_dir, **kwargs)
        
        if self.args.save_memory_states and output_dir:
            memory_path = os.path.join(output_dir, "memory_states.pt")
            torch.save({
                "memory_states": self.memory_states,
                "momentum_states": self.momentum_states
            }, memory_path)
            
    def load_memory_states(self, checkpoint_dir: str):
        """Load saved memory states."""
        memory_path = os.path.join(checkpoint_dir, "memory_states.pt")
        if os.path.exists(memory_path):
            states = torch.load(memory_path)
            self.memory_states = states.get("memory_states")
            self.momentum_states = states.get("momentum_states")


def create_optimizer_and_scheduler(
    model: OLMo3MIRASForCausalLM,
    args: OLMo3MIRASTrainingArguments,
    num_training_steps: int
):
    """
    Create optimizer with separate learning rates for memory modules.
    """
    # Separate parameters
    memory_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if "neural_memory" in name:
            memory_params.append(param)
        else:
            other_params.append(param)
            
    optimizer_grouped_parameters = [
        {
            "params": other_params,
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay
        },
        {
            "params": memory_params,
            "lr": args.memory_learning_rate,
            "weight_decay": args.weight_decay * 0.1  # Less weight decay for memory
        }
    ]
    
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    return optimizer, scheduler
