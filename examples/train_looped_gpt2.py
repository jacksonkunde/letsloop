import os
import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer
from datasets import load_dataset
from letsloop.models.looped_gpt2_model import LoopedGPT2ModelLMHead
from letsloop.config import LoopedGPT2Config

# Setup configuration
def get_config():
    return LoopedGPT2Config(
        vocab_size=50257,  # GPT-2's vocab size
        max_position_embeddings=1024,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        positional_embed_type="NoPE",  # No positional embeddings
        stopping_criteria="fixed_n",  # Iterate for fixed iterations
        max_iterations=5,  # Maximum iterations
    )

# Initialize model and tokenizer
def get_model_and_tokenizer(config):
    model = LoopedGPT2ModelLMHead(config)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have a pad token; use eos instead
    return model, tokenizer

# Load data
def load_data(tokenizer):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")  # Example dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized_datasets

# Training Loop
def main():
    config = get_config()
    model, tokenizer = get_model_and_tokenizer(config)
    datasets = load_data(tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",  # Directory for saving model checkpoints
        eval_strategy="epoch",  # Evaluate at each epoch
        learning_rate=5e-5,  # Learning rate
        per_device_train_batch_size=4,  # Batch size per device
        per_device_eval_batch_size=4,
        num_train_epochs=3,  # Total epochs
        weight_decay=0.01,  # Weight decay for regularization
        save_strategy="epoch",  # Save model at each epoch
        save_total_limit=2,  # Keep last 2 checkpoints
        logging_dir="./logs",  # Directory for logs
        logging_steps=50,
        report_to="none",  # Disable integration with tools like Wandb
        fp16=torch.cuda.is_available(),  # Use mixed precision if CUDA is available
    )

    # Data collator for padding and masking
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # GPT-2 uses causal language modeling (not masked)
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

    # Save the final model
    trainer.save_model("./final_model")
    tokenizer.save_pretrained("./final_model")

if __name__ == "__main__":
    main()
