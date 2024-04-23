"""
Language Model Fine-Tuning and Interaction Script

!!! EXPERIMENTAL !!!

This script is designed to fine-tune a language model using LoRA adapters for enhanced adaptability and efficiency.
It includes functionality for loading custom training data from a JSON file, setting up training configurations,
training the model, and saving the trained model with a timestamp for version tracking.

The script further provides an interactive session allowing users to input arbitrary prompts to test the model's
response capabilities in real-time.

Dependencies:
    - torch: For tensor operations and model training.
    - transformers: Provides access to model architectures and pre-trained models.
    - datasets: Facilitates easy loading and manipulation of datasets.
    - unsloth: Custom library for managing LoRA adapter integration.

Usage:
    - Ensure all dependencies are installed using pip install torch transformers datasets
    - Configure paths and model names as needed.
    - Run the script to train the model and then interact with it in real-time.

(c) 2024 Jan Miller (@miller_itsec) affiliated with OPSWAT, Inc. All rights reserved.
"""
import json
import torch

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer
from unsloth import FastLanguageModel
import datetime

# Install the Unsloth library for Ampere and Hopper architecture from GitHub
# pip install "unsloth[colab_ampere] @ git+https://github.com/unslothai/unsloth.git" -q
# Apply the following for older GPUs (V100, Tesla T4, RTX 20xx, etc.)
# pip install "unsloth[colab] @ git+https://github.com/unslothai/unsloth.git" -q

from config import *

# TODO: change this to match the training data file generated using GENERATE_TRAINING_DATA (see config.py)
TRAINING_DATA_FILE_PATH = 'training_data_20240422174817.json'


# Function to load and set up the model with LoRA adapters
def load_model_with_lora(base_model):
    model = FastLanguageModel.from_pretrained(base_model)
    return FastLanguageModel.get_peft_model(
        model,
        r=16,  # Rank for LoRA adapters
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


# Function to load JSON data as a dataset
def load_json_as_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return Dataset.from_dict(data)


# Function to set up training with the necessary parameters
def setup_training(model, tokenizer, dataset):
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,  # Adjusted epochs for better learning
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,  # Increase effective batch size
        optim="paged_adamw_32bit",
        save_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.005,  # Adjusted weight decay
        fp16=True,  # Enable mixed precision for faster training
        max_grad_norm=1.0,
        warmup_ratio=0.06,  # Adjusted warmup phase
        lr_scheduler_type="linear",  # Changed to linear decay
        logging_dir='logs',
        evaluation_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )
    return trainer


# Interactive session allowing arbitrary user prompts
def interact_with_model(model, tokenizer):
    print("\nPress Ctrl+C to exit.")
    while True:
        try:
            prompt = input("Enter your prompt: ")
            if prompt.lower() == 'quit':
                print("Exiting...")
                break

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=50)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("Generated Response:", response)
        except KeyboardInterrupt:
            print("\nInterrupt received, stopping...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    dataset = load_json_as_dataset(TRAINING_DATA_FILE_PATH)  # Load and prepare your dataset
    model = load_model_with_lora(MODEL_PATH)  # Load the model with LoRA adapters

    # Record memory before training
    start_gpu_memory = torch.cuda.memory_reserved()

    # Train the model
    trainer = setup_training(model, tokenizer, dataset)
    trainer_stats = trainer.train()

    # Save the trained model with a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    model.save_pretrained(f"lora_model_{timestamp}")
    print(f"Training completed. Model saved to lora_model_{timestamp}")

    # Enable caching and set the model to evaluation mode
    model.config.use_cache = True
    model.eval()

    # Record and print memory usage
    end_gpu_memory = torch.cuda.memory_reserved()
    used_memory = (end_gpu_memory - start_gpu_memory) / 1024 ** 3
    print(f"Peak memory usage: {used_memory:.3f} GB")

    # Run interactive session
    interact_with_model(model, tokenizer)
