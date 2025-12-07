# olmo3_miras/model/olmo3_miras.py
"""
OLMo3-MIRAS: Integration of MIRAS neural memory with OLMo3 transformer.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput

from ..memory.neural_memory import (
    MIRASMemoryConfig,
    NeuralLongTermMemory,
    PersistentMemory
)


@dataclass
class OLMo3MIRASCausalLMOutput(ModelOutput):
    """
    Output class for OLMo3-MIRAS causal language model.
    Extends standard output with memory states.
    """
    loss: Optional[Tensor] = None
    logits: Tensor = None
    past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None
    hidden_states: Optional[Tuple[Tensor, ...]] = None
    attentions: Optional[Tuple[Tensor, ...]] = None
    memory_states: Optional[List[Tensor]] = None
    momentum_states: Optional[List[Tensor]] = None


class OLMo3MIRASConfig(PretrainedConfig):
    """Configuration for OLMo3-MIRAS model."""

    model_type = "olmo3_miras"

    def __init__(
        self,
        vocab_size: int = 100352,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        hidden_act: str = "silu",
        max_position_embeddings: int = 65536,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        rope_theta: float = 500000.0,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        miras_config: Optional[Union[MIRASMemoryConfig, dict]] = None,
        integration_mode: str = "mac",
        memory_layers: Optional[List[int]] = None,
        attention_window_size: int = 4096,
        use_sliding_window: bool = True,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.integration_mode = integration_mode
        self.attention_window_size = attention_window_size
        self.use_sliding_window = use_sliding_window

        # Handle MIRAS config
        if miras_config is None:
            self.miras_config = MIRASMemoryConfig(hidden_size=hidden_size)
        elif isinstance(miras_config, dict):
            self.miras_config = MIRASMemoryConfig(**miras_config)
        else:
            self.miras_config = miras_config

        # Ensure miras_config hidden_size matches model hidden_size
        self.miras_config.hidden_size = hidden_size

        # Set memory layers (default: all layers)
        if memory_layers is None:
            self.memory_layers = list(range(num_hidden_layers))
        else:
            self.memory_layers = memory_layers


class OLMo3RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings for OLMo3."""
    
    def __init__(self, dim: int, max_position_embeddings: int = 65536, base: float = 500000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Build cache
        self._set_cos_sin_cache(max_position_embeddings)
        
    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        
    def forward(self, x: Tensor, seq_len: int) -> Tuple[Tensor, Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        return (
            self.cos_cached[:seq_len].to(x.dtype),
            self.sin_cached[:seq_len].to(x.dtype)
        )


def rotate_half(x: Tensor) -> Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, position_ids: Tensor
) -> Tuple[Tensor, Tensor]:
    """Apply rotary position embeddings."""
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class OLMo3MLP(nn.Module):
    """SwiGLU MLP for OLMo3."""
    
    def __init__(self, config: OLMo3MIRASConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        
    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class OLMo3Attention(nn.Module):
    """Grouped Query Attention for OLMo3 with optional sliding window."""
    
    def __init__(self, config: OLMo3MIRASConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.attention_dropout = config.attention_dropout
        
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        
        self.rotary_emb = OLMo3RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta
        )
        
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
        sliding_window: Optional[int] = None
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rotary_emb(value_states, seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        # Handle KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
            
        past_key_value = (key_states, value_states) if use_cache else None
        
        # Expand KV for GQA
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply sliding window mask if needed
        if sliding_window is not None:
            # Create sliding window mask
            kv_seq_len = key_states.shape[2]
            q_seq_len = query_states.shape[2]

            # Create position indices
            q_pos = torch.arange(q_seq_len, device=hidden_states.device).unsqueeze(1)
            k_pos = torch.arange(kv_seq_len, device=hidden_states.device).unsqueeze(0)

            # Sliding window: attend only to nearby positions
            # Adjust for KV cache offset
            distance = k_pos - q_pos + (kv_seq_len - q_seq_len)
            sliding_mask_bool = (distance >= 0) & (distance < sliding_window)
            sliding_mask_bool = sliding_mask_bool.unsqueeze(0).unsqueeze(0)

            # Convert to float mask: 0 for attend, -inf for mask
            sliding_mask = torch.where(
                sliding_mask_bool,
                torch.tensor(0.0, device=hidden_states.device, dtype=attn_weights.dtype),
                torch.tensor(float('-inf'), device=hidden_states.device, dtype=attn_weights.dtype)
            )

            # Apply sliding window mask directly to attention weights
            attn_weights = attn_weights + sliding_mask
            
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, past_key_value


class OLMo3MIRASDecoderLayer(nn.Module):
    """
    OLMo3 decoder layer with integrated MIRAS memory.
    Supports MAC, MAG, and MAL integration modes.
    """
    
    def __init__(self, config: OLMo3MIRASConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        
        # Core attention (short-term memory)
        self.self_attn = OLMo3Attention(config, layer_idx)
        
        # MLP
        self.mlp = OLMo3MLP(config)
        
        # Layer norms
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # MIRAS memory (if enabled for this layer)
        self.has_memory = layer_idx in config.memory_layers
        if self.has_memory:
            self.neural_memory = NeuralLongTermMemory(config.miras_config)
            
            if config.integration_mode == "mag":
                # Gating for MAG mode
                self.memory_gate = nn.Sequential(
                    nn.Linear(config.hidden_size * 2, config.hidden_size),
                    nn.Sigmoid()
                )
            elif config.integration_mode == "mac":
                # Additional norm for memory context in MAC mode
                self.memory_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        memory_state: Optional[Tensor] = None,
        momentum_state: Optional[Tensor] = None,
        persistent_memory: Optional[Tensor] = None,
        use_cache: bool = False
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]], Optional[Tensor], Optional[Tensor]]:
        """
        Forward pass with MIRAS memory integration.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Get memory output if this layer has memory
        memory_output = None
        new_memory_state = memory_state
        new_momentum_state = momentum_state
        
        if self.has_memory and self.config.integration_mode != "mal":
            memory_output, new_memory_state, new_momentum_state = self.neural_memory(
                hidden_states,
                memory_state=memory_state,
                momentum_state=momentum_state,
                return_memory_state=True
            )
        
        if self.config.integration_mode == "mac" and self.has_memory:
            # Memory as Context: Prepend memory and persistent tokens to context
            batch_size = hidden_states.shape[0]
            
            # Concatenate: [persistent_memory, memory_output, hidden_states]
            context_parts = []
            
            if persistent_memory is not None:
                context_parts.append(persistent_memory)
            if memory_output is not None:
                # Use memory output as additional context
                memory_context = self.memory_layernorm(memory_output)
                context_parts.append(memory_context)
            context_parts.append(hidden_states)
            
            extended_hidden = torch.cat(context_parts, dim=1)
            
            # Extend attention mask and position ids
            prefix_len = extended_hidden.shape[1] - hidden_states.shape[1]
            extended_seq = extended_hidden.shape[1]
            if attention_mask is not None:
                # attention_mask has shape (batch, 1, seq, seq) - causal mask
                # We need (batch, 1, extended_seq, extended_seq)

                # Prefix positions can attend to all positions (no causal constraint)
                prefix_to_all = torch.zeros(
                    (batch_size, 1, prefix_len, extended_seq),
                    device=attention_mask.device, dtype=attention_mask.dtype
                )

                # Original positions can attend to prefix (no mask) and self (causal)
                orig_to_prefix = torch.zeros(
                    (batch_size, 1, hidden_states.shape[1], prefix_len),
                    device=attention_mask.device, dtype=attention_mask.dtype
                )
                orig_to_orig = attention_mask  # Keep original causal mask

                # Combine: [prefix rows; original rows with extended keys]
                bottom_rows = torch.cat([orig_to_prefix, orig_to_orig], dim=-1)
                extended_mask = torch.cat([prefix_to_all, bottom_rows], dim=2)
            else:
                extended_mask = None
                
            if position_ids is not None:
                prefix_positions = torch.arange(
                    prefix_len, device=position_ids.device
                ).unsqueeze(0).expand(batch_size, -1)
                extended_positions = torch.cat([prefix_positions, position_ids + prefix_len], dim=1)
            else:
                extended_positions = None
            
            # Attention with extended context
            attn_output, past_key_value = self.self_attn(
                extended_hidden,
                attention_mask=extended_mask,
                position_ids=extended_positions,
                past_key_value=past_key_value,
                use_cache=use_cache,
                sliding_window=self.config.attention_window_size if self.config.use_sliding_window else None
            )
            
            # Extract output for original positions
            attn_output = attn_output[:, prefix_len:]
            
        elif self.config.integration_mode == "mag" and self.has_memory:
            # Memory as Gate: Combine attention and memory with gating
            attn_output, past_key_value = self.self_attn(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
                sliding_window=self.config.attention_window_size if self.config.use_sliding_window else None
            )
            
            # Gate between attention and memory
            gate_input = torch.cat([attn_output, memory_output], dim=-1)
            gate = self.memory_gate(gate_input)
            attn_output = gate * attn_output + (1 - gate) * memory_output
            
        else:
            # MAL mode or no memory: Standard attention
            attn_output, past_key_value = self.self_attn(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
                sliding_window=self.config.attention_window_size if self.config.use_sliding_window else None
            )
        
        hidden_states = residual + attn_output
        
        # MAL mode: Apply memory after attention
        if self.config.integration_mode == "mal" and self.has_memory:
            memory_output, new_memory_state, new_momentum_state = self.neural_memory(
                hidden_states,
                memory_state=memory_state,
                momentum_state=momentum_state,
                return_memory_state=True
            )
            hidden_states = hidden_states + memory_output
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, past_key_value, new_memory_state, new_momentum_state


class OLMo3MIRASModel(PreTrainedModel):
    """
    OLMo3 Transformer with MIRAS Neural Long-Term Memory.
    """
    
    config_class = OLMo3MIRASConfig
    base_model_prefix = "model"
    
    def __init__(self, config: OLMo3MIRASConfig):
        super().__init__(config)
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Persistent memory
        self.persistent_memory = PersistentMemory(config.miras_config)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            OLMo3MIRASDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Gradient checkpointing
        self.gradient_checkpointing = False
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
                
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        memory_states: Optional[List[Tensor]] = None,
        momentum_states: Optional[List[Tensor]] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True
    ):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=device)
            
        # Create causal mask
        # Use a large negative number instead of -inf to avoid NaN when multiplied by 0
        mask_value = torch.finfo(hidden_states.dtype).min
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), mask_value, device=device, dtype=hidden_states.dtype),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        # Combine with attention mask
        # Use torch.where to avoid NaN from 0 * -inf
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = torch.where(
            extended_attention_mask == 1.0,
            torch.tensor(0.0, device=device, dtype=hidden_states.dtype),
            torch.tensor(mask_value, device=device, dtype=hidden_states.dtype)
        )
        combined_mask = causal_mask + extended_attention_mask
        
        # Prepare position ids
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            
        # Get persistent memory
        persistent_mem = self.persistent_memory(batch_size)
        
        # Initialize memory states if not provided
        if memory_states is None:
            memory_states = [None] * len(self.layers)
        if momentum_states is None:
            momentum_states = [None] * len(self.layers)
            
        # Process through layers
        all_hidden_states = () if output_hidden_states else None
        next_cache = [] if use_cache else None
        new_memory_states = []
        new_momentum_states = []
        
        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            past_kv = past_key_values[idx] if past_key_values is not None else None
            
            if self.gradient_checkpointing and self.training:
                hidden_states, present_kv, new_mem, new_mom = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    combined_mask,
                    position_ids,
                    past_kv,
                    memory_states[idx],
                    momentum_states[idx],
                    persistent_mem,
                    use_cache
                )
            else:
                hidden_states, present_kv, new_mem, new_mom = layer(
                    hidden_states,
                    attention_mask=combined_mask,
                    position_ids=position_ids,
                    past_key_value=past_kv,
                    memory_state=memory_states[idx],
                    momentum_state=momentum_states[idx],
                    persistent_memory=persistent_mem,
                    use_cache=use_cache
                )
                
            if use_cache:
                next_cache.append(present_kv)
            new_memory_states.append(new_mem)
            new_momentum_states.append(new_mom)
            
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            
        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_cache,
            "memory_states": new_memory_states,
            "momentum_states": new_momentum_states,
            "hidden_states": all_hidden_states
        }


class OLMo3MIRASForCausalLM(PreTrainedModel, GenerationMixin):
    """
    OLMo3-MIRAS for Causal Language Modeling.
    Inherits from GenerationMixin for text generation capabilities.
    """

    config_class = OLMo3MIRASConfig
    base_model_prefix = "model"
    _supports_cache_class = False  # We handle caching manually with memory states
    
    def __init__(self, config: OLMo3MIRASConfig):
        super().__init__(config)
        self.model = OLMo3MIRASModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie embeddings if configured
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
            
        self._init_weights()
        
    def _init_weights(self):
        """Initialize LM head."""
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=self.config.initializer_range)
        
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        memory_states: Optional[List[Tensor]] = None,
        momentum_states: Optional[List[Tensor]] = None,
        labels: Optional[Tensor] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            memory_states=memory_states,
            momentum_states=momentum_states,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        
        hidden_states = outputs["last_hidden_state"]
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
            
        if not return_dict:
            output = (logits,) + (outputs["past_key_values"],)
            return ((loss,) + output) if loss is not None else output

        return OLMo3MIRASCausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs["past_key_values"],
            hidden_states=outputs.get("hidden_states"),
            memory_states=outputs.get("memory_states"),
            momentum_states=outputs.get("momentum_states")
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids: Tensor,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        memory_states: Optional[List[Tensor]] = None,
        momentum_states: Optional[List[Tensor]] = None,
        attention_mask: Optional[Tensor] = None,
        **kwargs
    ):
        """Prepare inputs for generation."""
        if past_key_values is not None:
            # Only use last token
            input_ids = input_ids[:, -1:]
            
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -1].unsqueeze(-1)
                
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "memory_states": memory_states,
            "momentum_states": momentum_states,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": True
        }

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ):
        """
        Update model kwargs for next generation step.
        Handles memory state persistence during text generation.
        """
        # Update the standard kwargs
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens,
        )

        # Propagate memory states to next step
        if hasattr(outputs, "memory_states") and outputs.memory_states is not None:
            model_kwargs["memory_states"] = outputs.memory_states
        if hasattr(outputs, "momentum_states") and outputs.momentum_states is not None:
            model_kwargs["momentum_states"] = outputs.momentum_states

        return model_kwargs
