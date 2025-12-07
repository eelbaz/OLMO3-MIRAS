"""
Example training script for OLMo3-MIRAS.
"""

import torch
from transformers import AutoTokenizer
from datasets import load_dataset

from olmo3_miras.model.olmo3_miras import (
    OLMo3MIRASForCausalLM,
    OLMo3MIRASConfig
)
from olmo3_miras.memory.neural_memory import MIRASMemoryConfig
from olmo3_miras.training.trainer import (
    OLMo3MIRASTrainer,
    OLMo3MIRASTrainingArguments,
    ChunkedDataset
)


def main():
    # Configuration
    miras_config = MIRASMemoryConfig(
        hidden_size=4096,
        memory_hidden_size=2048,
        memory_depth=2,
        num_memory_heads=8,
        use_momentum=True,
        momentum_decay=0.9,
        learning_rate=0.1,
        forget_gate=True,
        chunk_size=512,
        num_persistent_tokens=16,
        data_dependent_gates=True
    )
    
    model_config = OLMo3MIRASConfig(
        vocab_size=100352,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=65536,
        miras_config=miras_config,
        integration_mode="mac",  # Memory as Context
        attention_window_size=4096,
        use_sliding_window=True
    )
    
    # Initialize model
    model = OLMo3MIRASForCausalLM(model_config)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-7B")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    dataset = load_dataset("allenai/dolma-v1_7-sample", split="train")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=model_config.max_position_embeddings,
            padding="max_length"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Wrap with chunked dataset for long-context training
    train_dataset = ChunkedDataset(
        tokenized_dataset,
        chunk_size=4096,
        max_seq_length=65536,
        overlap=512
    )
    
    # Training arguments
    training_args = OLMo3MIRASTrainingArguments(
        output_dir="./olmo3_miras_output",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        learning_rate=6e-4,
        memory_learning_rate=1e-4,
        warmup_steps=2000,
        weight_decay=0.1,
        logging_steps=10,
        save_steps=1000,
        bf16=True,
        chunk_training=True,
        chunk_size=4096,
        gradient_accumulation_across_chunks=True,
        curriculum_learning=True,
        min_context_length=2048,
        max_context_length=65536,
        curriculum_warmup_steps=5000,
        save_memory_states=True
    )
    
    # Initialize trainer
    trainer = OLMo3MIRASTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )
    
    # Train
    trainer.train()
    
    # Save final model
    trainer.save_model("./olmo3_miras_final")


if __name__ == "__main__":
    main()
