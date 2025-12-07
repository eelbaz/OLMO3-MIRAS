"""
Inference utilities for OLMo3-MIRAS with streaming support
and memory state management.
"""

from __future__ import annotations
from typing import Optional, List, Iterator, Dict, Any

import torch
import torch.nn.functional as F
from torch import Tensor

from ..model.olmo3_miras import OLMo3MIRASForCausalLM


class OLMo3MIRASGenerator:
    """
    Text generator for OLMo3-MIRAS with efficient long-context handling.
    """
    
    def __init__(
        self,
        model: OLMo3MIRASForCausalLM,
        tokenizer,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        # Initialize persistent memory states for long conversations
        self.memory_states = None
        self.momentum_states = None
        self.kv_cache = None
        
    def reset_memory(self):
        """Reset all memory states."""
        self.memory_states = None
        self.momentum_states = None
        self.kv_cache = None
        
    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        preserve_memory: bool = True,
        stream: bool = False
    ) -> str | Iterator[str]:
        """
        Generate text continuation.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            repetition_penalty: Penalty for repeating tokens
            preserve_memory: Whether to preserve memory states across calls
            stream: Whether to stream tokens
            
        Returns:
            Generated text or token iterator
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        if stream:
            return self._stream_generate(
                input_ids, max_new_tokens, temperature, top_p, top_k,
                repetition_penalty, preserve_memory
            )
        else:
            return self._batch_generate(
                input_ids, max_new_tokens, temperature, top_p, top_k,
                repetition_penalty, preserve_memory
            )
            
    def _batch_generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        preserve_memory: bool
    ) -> str:
        """Batch generation."""
        generated = input_ids
        past_key_values = self.kv_cache if preserve_memory else None
        memory_states = self.memory_states if preserve_memory else None
        momentum_states = self.momentum_states if preserve_memory else None
        
        for _ in range(max_new_tokens):
            outputs = self.model(
                input_ids=generated if past_key_values is None else generated[:, -1:],
                past_key_values=past_key_values,
                memory_states=memory_states,
                momentum_states=momentum_states,
                use_cache=True
            )
            
            logits = outputs.logits[:, -1]
            past_key_values = outputs.past_key_values
            
            # Update memory states
            if hasattr(outputs, 'memory_states'):
                memory_states = outputs.memory_states
            if hasattr(outputs, 'momentum_states'):
                momentum_states = outputs.momentum_states
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    logits[0, token_id] /= repetition_penalty
                    
            # Sample next token
            next_token = self._sample_token(logits, temperature, top_p, top_k)
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop on EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
                
        # Preserve states if requested
        if preserve_memory:
            self.kv_cache = past_key_values
            self.memory_states = memory_states
            self.momentum_states = momentum_states
            
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)
    
    def _stream_generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        preserve_memory: bool
    ) -> Iterator[str]:
        """Streaming generation."""
        generated = input_ids
        past_key_values = self.kv_cache if preserve_memory else None
        memory_states = self.memory_states if preserve_memory else None
        momentum_states = self.momentum_states if preserve_memory else None
        
        prev_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        
        for _ in range(max_new_tokens):
            outputs = self.model(
                input_ids=generated if past_key_values is None else generated[:, -1:],
                past_key_values=past_key_values,
                memory_states=memory_states,
                momentum_states=momentum_states,
                use_cache=True
            )
            
            logits = outputs.logits[:, -1]
            past_key_values = outputs.past_key_values
            
            if hasattr(outputs, 'memory_states'):
                memory_states = outputs.memory_states
            if hasattr(outputs, 'momentum_states'):
                momentum_states = outputs.momentum_states
            
            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    logits[0, token_id] /= repetition_penalty
                    
            next_token = self._sample_token(logits, temperature, top_p, top_k)
            generated = torch.cat([generated, next_token], dim=1)
            
            # Yield new text
            current_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            new_text = current_text[len(prev_text):]
            if new_text:
                yield new_text
            prev_text = current_text
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
                
        if preserve_memory:
            self.kv_cache = past_key_values
            self.memory_states = memory_states
            self.momentum_states = momentum_states
            
    def _sample_token(
        self,
        logits: Tensor,
        temperature: float,
        top_p: float,
        top_k: int
    ) -> Tensor:
        """Sample next token with temperature, top-p, and top-k."""
        if temperature == 0:
            return logits.argmax(dim=-1, keepdim=True)
            
        logits = logits / temperature
        
        # Top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
            
        # Top-p filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
            
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)


class NeedleInHaystackEvaluator:
    """
    Evaluator for needle-in-haystack tasks to measure
    effective context length.
    """
    
    def __init__(self, model: OLMo3MIRASForCausalLM, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.generator = OLMo3MIRASGenerator(model, tokenizer)
        
    def evaluate(
        self,
        context_lengths: List[int],
        needle_depths: List[float],
        num_samples: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate model on NIAH task across context lengths and depths.
        """
        results = {}
        
        for ctx_len in context_lengths:
            for depth in needle_depths:
                key = f"ctx_{ctx_len}_depth_{depth}"
                correct = 0
                
                for _ in range(num_samples):
                    # Create haystack with needle
                    haystack, needle, answer = self._create_niah_sample(ctx_len, depth)
                    
                    # Generate response
                    prompt = f"{haystack}\n\nQuestion: {needle}\nAnswer:"
                    response = self.generator.generate(
                        prompt,
                        max_new_tokens=50,
                        temperature=0,
                        preserve_memory=False
                    )
                    
                    # Check if answer is correct
                    if answer.lower() in response.lower():
                        correct += 1
                        
                    self.generator.reset_memory()
                    
                results[key] = correct / num_samples
                
        return results
    
    def _create_niah_sample(
        self,
        context_length: int,
        needle_depth: float
    ) -> tuple[str, str, str]:
        """Create a NIAH sample."""
        import random
        import string
        
        # Generate random filler text
        words = [''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 10))) 
                 for _ in range(context_length // 5)]
        
        # Create needle
        secret_number = random.randint(1000, 9999)
        needle = f"The secret number is {secret_number}."
        question = "What is the secret number?"
        
        # Insert needle at specified depth
        insert_pos = int(len(words) * needle_depth)
        words.insert(insert_pos, needle)
        
        haystack = ' '.join(words)
        
        return haystack, question, str(secret_number)
