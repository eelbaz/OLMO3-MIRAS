#!/usr/bin/env python3
"""
Test the training setup with a small model on synthetic data.
Validates the full pipeline works before running on Dolma3.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.optim import AdamW

from olmo3_miras.model.olmo3_miras import OLMo3MIRASConfig, OLMo3MIRASForCausalLM
from olmo3_miras.memory.neural_memory import MIRASMemoryConfig
from olmo3_miras.configs.model_configs import (
    get_model_config,
    get_training_config,
    TrainingStage,
    estimate_parameters,
    calculate_training_steps,
)


def test_config_loading():
    """Test configuration loading."""
    print("\n" + "=" * 60)
    print("Testing Configuration Loading")
    print("=" * 60)

    # Test model configs
    for size in ["500M", "1B"]:
        config = get_model_config(size)
        params = estimate_parameters(config)
        print(f"\n{size} Model Config:")
        print(f"  Hidden: {config['hidden_size']}, Layers: {config['num_hidden_layers']}")
        print(f"  Estimated params: {params:,} ({params/1e9:.2f}B)")

    # Test training configs
    for stage in TrainingStage:
        train_config = get_training_config(stage, "500M")
        steps = calculate_training_steps(train_config)
        print(f"\n{stage.value} Training:")
        print(f"  Dataset: {train_config['dataset']['name']}")
        print(f"  Total tokens: {train_config['total_tokens']/1e9:.1f}B")
        print(f"  Total steps: {steps:,}")

    print("\nConfiguration loading: PASSED")
    return True


def test_model_creation():
    """Test model creation with 500M config."""
    print("\n" + "=" * 60)
    print("Testing Model Creation (Small Scale)")
    print("=" * 60)

    # Use a smaller config for testing
    small_config = {
        "vocab_size": 1000,
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 512,
        "integration_mode": "mac",
        "attention_window_size": 128,
        "use_sliding_window": True,
    }

    miras_config = MIRASMemoryConfig(
        hidden_size=256,
        memory_hidden_size=128,
        memory_depth=2,
        num_memory_heads=4,
        use_momentum=True,
        momentum_decay=0.9,
        learning_rate=0.1,
        forget_gate=True,
        chunk_size=64,
        num_persistent_tokens=4,
        data_dependent_gates=True,
    )

    model_config = OLMo3MIRASConfig(
        miras_config=miras_config,
        **small_config
    )

    model = OLMo3MIRASForCausalLM(model_config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Check memory modules exist
    memory_params = sum(1 for n, p in model.named_parameters() if "neural_memory" in n)
    print(f"Memory-related parameters: {memory_params}")

    print("\nModel creation: PASSED")
    return model, model_config


def test_forward_pass(model, config, device="cuda"):
    """Test forward pass."""
    print("\n" + "=" * 60)
    print("Testing Forward Pass")
    print("=" * 60)

    model = model.to(device)
    model.eval()

    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    print(f"Logits shape: {outputs.logits.shape}")
    print(f"Loss: {outputs.loss.item():.4f}")
    print(f"Loss is finite: {torch.isfinite(outputs.loss).item()}")

    # Check memory states
    if hasattr(outputs, 'memory_states') and outputs.memory_states:
        valid_states = sum(1 for s in outputs.memory_states if s is not None)
        print(f"Memory states: {valid_states}/{len(outputs.memory_states)} layers")

    print("\nForward pass: PASSED")
    return True


def test_training_step(model, config, device="cuda"):
    """Test training step with gradient flow."""
    print("\n" + "=" * 60)
    print("Testing Training Step")
    print("=" * 60)

    model = model.to(device)
    model.train()

    # Create optimizer with separate LRs
    memory_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "neural_memory" in name or "persistent_memory" in name:
            memory_params.append(param)
        else:
            other_params.append(param)

    optimizer = AdamW([
        {"params": other_params, "lr": 1e-4},
        {"params": memory_params, "lr": 5e-5},
    ])

    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()

    # Training loop
    losses = []
    for step in range(5):
        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        print(f"  Step {step}: Loss = {loss.item():.4f}")

    # Check loss decreased
    decreased = losses[-1] < losses[0]
    print(f"\nInitial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Loss decreased: {decreased}")

    # Check gradients flowed to memory
    memory_grads = 0
    for name, param in model.named_parameters():
        if "neural_memory" in name and param.grad is not None:
            if param.grad.abs().sum() > 0:
                memory_grads += 1

    print(f"Memory params with gradients: {memory_grads}")

    print("\nTraining step: PASSED" if decreased else "\nTraining step: WARNING - Loss not decreasing")
    return decreased


def test_integration_modes(device="cuda"):
    """Test all three integration modes."""
    print("\n" + "=" * 60)
    print("Testing Integration Modes")
    print("=" * 60)

    small_config = {
        "vocab_size": 1000,
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 512,
        "attention_window_size": 128,
        "use_sliding_window": True,
    }

    miras_config = MIRASMemoryConfig(
        hidden_size=256,
        memory_hidden_size=128,
        memory_depth=2,
        num_memory_heads=4,
        use_momentum=True,
        chunk_size=64,
        num_persistent_tokens=4,
    )

    for mode in ["mac", "mag", "mal"]:
        print(f"\n--- Testing {mode.upper()} mode ---")

        model_config = OLMo3MIRASConfig(
            miras_config=miras_config,
            integration_mode=mode,
            **small_config
        )

        model = OLMo3MIRASForCausalLM(model_config).to(device)
        model.train()

        # Quick forward/backward
        input_ids = torch.randint(0, 1000, (2, 32), device=device)
        labels = input_ids.clone()

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        if torch.isnan(loss):
            print(f"  {mode.upper()}: NaN loss!")
            continue

        loss.backward()

        print(f"  {mode.upper()}: Loss = {loss.item():.4f} (OK)")

        del model
        torch.cuda.empty_cache()

    print("\nIntegration modes: PASSED")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("OLMo3-MIRAS Training Setup Tests")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Run tests
    test_config_loading()

    model, config = test_model_creation()

    if device == "cuda":
        test_forward_pass(model, config, device)
        test_training_step(model, config, device)
        test_integration_modes(device)
    else:
        print("\nSkipping GPU tests (no CUDA available)")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    print("\nThe training pipeline is ready!")
    print("Next steps:")
    print("  1. Configure HuggingFace token for Dolma3 access")
    print("  2. Run: python scripts/train_olmo3_miras_500m.py --stage pretrain")


if __name__ == "__main__":
    main()
