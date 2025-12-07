"""
OLMo3-MIRAS Model Configurations.

Pre-configured architectures following OLMo3 design patterns (RMSNorm, RoPE, GQA, SwiGLU)
with MIRAS neural long-term memory for unlimited context length.

Architecture Reference:
- OLMo3-7B: 32 layers, 4096 hidden, 32 heads, 8 KV heads
- OLMo3-32B: 64 layers, 5120 hidden, 40 heads, 8 KV heads

Scaling for 0.5B/1B models follows Chinchilla-optimal ratios.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, List


class TrainingStage(Enum):
    """Three-stage training following OLMo3 recipe."""
    PRETRAIN = "pretrain"       # Stage 1: Large-scale pretraining on Dolma3
    MIDTRAIN = "midtrain"       # Stage 2: High-quality instruction/math/code
    LONGCONTEXT = "longcontext" # Stage 3: Extended context with MIRAS


# =============================================================================
# DOLMA3 DATASET CONFIGURATIONS
# =============================================================================

DOLMA3_DATASETS = {
    # Stage 1: Pretraining data (ODC-BY licensed)
    "pretrain": {
        "name": "allenai/dolma3_pool",
        "description": "Full Dolma3 pool: 9T+ tokens from web, papers, code",
        "columns": ["id", "text", "metadata"],
        "streaming": True,  # Required for large dataset
    },

    # Stage 2: Mid-training (Dolmino mix)
    # Composition: 20% code, 28% web, 19% math, 14% QA, 8% thinking, 6% instruction, 5% PDFs
    "midtrain": {
        "name": "allenai/dolma3_dolmino_mix",
        "fallback": "allenai/tulu-3-sft-mixture",  # If dolmino not available
        "description": "100B tokens of high-quality instruction/math/code data",
        "columns": ["text"],
        "streaming": True,
    },

    # Stage 3: Long-context training (Longmino mix)
    # Composition: 66% midtraining data, 34% PDFs with extended context
    "longcontext": {
        "name": "allenai/dolma3_longmino_mix",
        "fallback": "emozilla/pg19-test",  # Long documents fallback
        "description": "50-100B tokens of long-context documents",
        "columns": ["text"],
        "streaming": True,
    },
}


# =============================================================================
# MODEL ARCHITECTURE CONFIGURATIONS
# =============================================================================

OLMO3_MIRAS_500M_CONFIG = {
    # Model architecture (0.5B parameters)
    # Following OLMo3 design: hidden/heads ratio, GQA, SwiGLU
    "vocab_size": 100352,           # OLMo3 tokenizer vocabulary
    "hidden_size": 1024,            # Model dimension
    "intermediate_size": 2816,      # SwiGLU: ~2.67x hidden for efficiency
    "num_hidden_layers": 24,        # Depth
    "num_attention_heads": 16,      # Query heads
    "num_key_value_heads": 4,       # GQA with 4:1 ratio (like OLMo3)
    "hidden_act": "silu",           # SiLU activation (SwiGLU)
    "max_position_embeddings": 65536,  # 64K context (MIRAS extends beyond)
    "initializer_range": 0.02,
    "rms_norm_eps": 1e-5,           # RMSNorm epsilon
    "use_cache": True,
    "tie_word_embeddings": False,   # Separate input/output embeddings
    "rope_theta": 500000.0,         # RoPE base frequency (OLMo3 uses 500K)
    "attention_dropout": 0.0,       # No attention dropout
    "attention_bias": False,        # No bias in attention
    "mlp_bias": False,              # No bias in MLP

    # MIRAS Memory Configuration
    "miras_config": {
        "hidden_size": 1024,
        "memory_hidden_size": 512,       # Memory dimension (0.5x hidden)
        "memory_depth": 2,               # Memory MLP depth
        "num_memory_heads": 8,           # Memory attention heads
        "use_momentum": True,            # Momentum-based surprise (MIRAS key feature)
        "momentum_decay": 0.9,           # Momentum decay rate
        "learning_rate": 0.1,            # Test-time learning rate
        "forget_gate": True,             # Data-dependent forgetting
        "chunk_size": 512,               # Chunk size for memory updates
        "num_persistent_tokens": 16,     # Learnable persistent memory
        "data_dependent_gates": True,    # Dynamic gating
    },

    # Integration settings
    "integration_mode": "mac",       # Memory as Context (best for long-range)
    "memory_layers": None,           # None = all layers have memory
    "attention_window_size": 4096,   # Sliding window for local attention
    "use_sliding_window": True,      # Enable sliding window
}


OLMO3_MIRAS_1B_CONFIG = {
    # Model architecture (1B parameters)
    "vocab_size": 100352,
    "hidden_size": 1536,
    "intermediate_size": 4096,       # ~2.67x hidden
    "num_hidden_layers": 28,
    "num_attention_heads": 24,
    "num_key_value_heads": 6,        # GQA 4:1
    "hidden_act": "silu",
    "max_position_embeddings": 65536,
    "initializer_range": 0.02,
    "rms_norm_eps": 1e-5,
    "use_cache": True,
    "tie_word_embeddings": False,
    "rope_theta": 500000.0,
    "attention_dropout": 0.0,
    "attention_bias": False,
    "mlp_bias": False,

    # MIRAS Memory Configuration
    "miras_config": {
        "hidden_size": 1536,
        "memory_hidden_size": 768,
        "memory_depth": 2,
        "num_memory_heads": 12,
        "use_momentum": True,
        "momentum_decay": 0.9,
        "learning_rate": 0.1,
        "forget_gate": True,
        "chunk_size": 512,
        "num_persistent_tokens": 16,
        "data_dependent_gates": True,
    },

    "integration_mode": "mac",
    "memory_layers": None,
    "attention_window_size": 4096,
    "use_sliding_window": True,
}


# =============================================================================
# TRAINING CONFIGURATIONS (3-Stage Recipe)
# =============================================================================

TRAINING_CONFIGS = {
    TrainingStage.PRETRAIN: {
        "500M": {
            # Following Chinchilla: ~10B tokens for 500M model
            # Scale up with compute budget
            "total_tokens": 10_000_000_000,  # 10B tokens (adjust based on budget)
            "batch_size_tokens": 2_097_152,  # 2M tokens per batch
            "learning_rate": 3e-4,
            "min_learning_rate": 3e-5,       # 10x decay
            "warmup_tokens": 200_000_000,    # 200M warmup
            "weight_decay": 0.1,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_epsilon": 1e-8,
            "max_grad_norm": 1.0,
            "lr_scheduler": "cosine",
            "bf16": True,
            "context_length": 4096,          # Start with shorter context
            "gradient_checkpointing": True,

            # MIRAS-specific
            "memory_learning_rate": 1e-4,    # Separate LR for memory params
            "memory_warmup_steps": 1000,
            "chunk_training": True,
            "chunk_size": 2048,
        },
        "1B": {
            "total_tokens": 25_000_000_000,  # 25B tokens
            "batch_size_tokens": 4_194_304,  # 4M tokens per batch
            "learning_rate": 2e-4,
            "min_learning_rate": 2e-5,
            "warmup_tokens": 500_000_000,
            "weight_decay": 0.1,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_epsilon": 1e-8,
            "max_grad_norm": 1.0,
            "lr_scheduler": "cosine",
            "bf16": True,
            "context_length": 4096,
            "gradient_checkpointing": True,
            "memory_learning_rate": 1e-4,
            "memory_warmup_steps": 2000,
            "chunk_training": True,
            "chunk_size": 2048,
        },
    },

    TrainingStage.MIDTRAIN: {
        "500M": {
            # High-quality data stage
            "total_tokens": 1_000_000_000,   # 1B tokens
            "batch_size_tokens": 1_048_576,  # 1M tokens per batch
            "learning_rate": 1e-4,           # Lower LR for fine-tuning
            "min_learning_rate": 1e-5,
            "warmup_tokens": 50_000_000,
            "weight_decay": 0.05,            # Less regularization
            "max_grad_norm": 1.0,
            "bf16": True,
            "context_length": 8192,          # Longer context
            "gradient_checkpointing": True,
            "memory_learning_rate": 5e-5,
            "chunk_training": True,
            "chunk_size": 4096,
        },
        "1B": {
            "total_tokens": 2_000_000_000,   # 2B tokens
            "batch_size_tokens": 2_097_152,
            "learning_rate": 8e-5,
            "min_learning_rate": 8e-6,
            "warmup_tokens": 100_000_000,
            "weight_decay": 0.05,
            "max_grad_norm": 1.0,
            "bf16": True,
            "context_length": 8192,
            "gradient_checkpointing": True,
            "memory_learning_rate": 4e-5,
            "chunk_training": True,
            "chunk_size": 4096,
        },
    },

    TrainingStage.LONGCONTEXT: {
        "500M": {
            # Long-context training - MIRAS shines here
            "total_tokens": 500_000_000,     # 500M tokens of long documents
            "batch_size_tokens": 524_288,    # 512K tokens per batch
            "learning_rate": 5e-5,
            "min_learning_rate": 5e-6,
            "warmup_tokens": 25_000_000,
            "weight_decay": 0.01,
            "max_grad_norm": 0.5,            # Tighter clipping for stability
            "bf16": True,
            "context_length": 32768,         # 32K context
            "max_context_length": 65536,     # Up to 64K
            "gradient_checkpointing": True,

            # MIRAS-specific for long context
            "memory_learning_rate": 2e-5,
            "chunk_training": True,
            "chunk_size": 4096,
            "curriculum_learning": True,     # Gradually increase context
            "curriculum_warmup_steps": 2000,
            "min_context_length": 8192,
            "save_memory_states": True,
        },
        "1B": {
            "total_tokens": 1_000_000_000,
            "batch_size_tokens": 1_048_576,
            "learning_rate": 4e-5,
            "min_learning_rate": 4e-6,
            "warmup_tokens": 50_000_000,
            "weight_decay": 0.01,
            "max_grad_norm": 0.5,
            "bf16": True,
            "context_length": 32768,
            "max_context_length": 65536,
            "gradient_checkpointing": True,
            "memory_learning_rate": 1.5e-5,
            "chunk_training": True,
            "chunk_size": 4096,
            "curriculum_learning": True,
            "curriculum_warmup_steps": 3000,
            "min_context_length": 8192,
            "save_memory_states": True,
        },
    },
}


def get_model_config(size: str = "500M") -> Dict[str, Any]:
    """Get model configuration by size.

    Args:
        size: Model size ("500M" or "1B")

    Returns:
        Model configuration dictionary
    """
    configs = {
        "500M": OLMO3_MIRAS_500M_CONFIG,
        "1B": OLMO3_MIRAS_1B_CONFIG,
    }
    if size not in configs:
        raise ValueError(f"Unknown model size: {size}. Choose from {list(configs.keys())}")
    return configs[size].copy()


def get_training_config(
    stage: TrainingStage,
    size: str = "500M"
) -> Dict[str, Any]:
    """Get training configuration for a specific stage and model size.

    Args:
        stage: Training stage (PRETRAIN, MIDTRAIN, LONGCONTEXT)
        size: Model size ("500M" or "1B")

    Returns:
        Training configuration dictionary
    """
    if stage not in TRAINING_CONFIGS:
        raise ValueError(f"Unknown stage: {stage}")
    if size not in TRAINING_CONFIGS[stage]:
        raise ValueError(f"Unknown size {size} for stage {stage}")

    config = TRAINING_CONFIGS[stage][size].copy()
    config["dataset"] = DOLMA3_DATASETS[stage.value].copy()
    return config


def get_dataset_config(stage: TrainingStage) -> Dict[str, Any]:
    """Get dataset configuration for a training stage.

    Args:
        stage: Training stage

    Returns:
        Dataset configuration dictionary
    """
    return DOLMA3_DATASETS[stage.value].copy()


def calculate_training_steps(config: Dict[str, Any]) -> int:
    """Calculate total training steps from token-based config."""
    total_tokens = config.get("total_tokens", 10_000_000_000)
    batch_tokens = config.get("batch_size_tokens", 2_097_152)
    return total_tokens // batch_tokens


def calculate_warmup_steps(config: Dict[str, Any]) -> int:
    """Calculate warmup steps from token-based config."""
    warmup_tokens = config.get("warmup_tokens", 200_000_000)
    batch_tokens = config.get("batch_size_tokens", 2_097_152)
    return warmup_tokens // batch_tokens


# Parameter count estimation
def estimate_parameters(config: Dict[str, Any]) -> int:
    """Estimate total parameter count for a model config."""
    vocab = config["vocab_size"]
    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    layers = config["num_hidden_layers"]
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    head_dim = hidden // heads

    # Embeddings
    embed_params = vocab * hidden

    # Per layer
    # Attention: Q, K, V, O projections
    q_params = hidden * hidden
    k_params = kv_heads * head_dim * hidden
    v_params = kv_heads * head_dim * hidden
    o_params = hidden * hidden
    attn_params = q_params + k_params + v_params + o_params

    # MLP: gate, up, down (SwiGLU)
    mlp_params = 3 * hidden * intermediate

    # Layer norms
    norm_params = 2 * hidden

    layer_params = attn_params + mlp_params + norm_params

    # Total transformer
    transformer_params = layers * layer_params + hidden  # final norm

    # LM head (separate from embeddings)
    lm_head_params = hidden * vocab

    # MIRAS memory (rough estimate)
    miras_config = config.get("miras_config", {})
    mem_hidden = miras_config.get("memory_hidden_size", hidden // 2)
    mem_depth = miras_config.get("memory_depth", 2)
    num_mem_heads = miras_config.get("num_memory_heads", 8)
    persistent = miras_config.get("num_persistent_tokens", 16)

    # Memory per layer: projections + memory MLP + persistent
    mem_proj = 3 * hidden * mem_hidden  # Q, K, V projections
    mem_mlp = mem_depth * (mem_hidden * mem_hidden * 4)  # Memory network
    mem_persistent = persistent * hidden
    memory_per_layer = mem_proj + mem_mlp + mem_persistent
    memory_params = layers * memory_per_layer

    total = embed_params + transformer_params + lm_head_params + memory_params
    return total


if __name__ == "__main__":
    # Print configuration summary
    print("=" * 60)
    print("OLMo3-MIRAS Model Configurations")
    print("=" * 60)

    for size in ["500M", "1B"]:
        config = get_model_config(size)
        params = estimate_parameters(config)
        print(f"\n{size} Model:")
        print(f"  Estimated parameters: {params:,} ({params/1e9:.2f}B)")
        print(f"  Hidden size: {config['hidden_size']}")
        print(f"  Layers: {config['num_hidden_layers']}")
        print(f"  Heads: {config['num_attention_heads']} Q / {config['num_key_value_heads']} KV")
        print(f"  Max context: {config['max_position_embeddings']:,}")
        print(f"  Integration mode: {config['integration_mode']}")

    print("\n" + "=" * 60)
    print("Training Stages (Dolma3 Dataset)")
    print("=" * 60)

    for stage in TrainingStage:
        dataset = get_dataset_config(stage)
        print(f"\n{stage.value.upper()}:")
        print(f"  Dataset: {dataset['name']}")
        print(f"  Description: {dataset['description']}")

        for size in ["500M"]:
            train_config = get_training_config(stage, size)
            steps = calculate_training_steps(train_config)
            print(f"  {size}: {train_config['total_tokens']/1e9:.1f}B tokens, ~{steps:,} steps")
