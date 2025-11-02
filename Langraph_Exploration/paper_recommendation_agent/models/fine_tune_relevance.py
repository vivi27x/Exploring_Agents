import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import json
import numpy as np
import sys 
import os
# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
print(f"Added to Python path: {project_root}")

from utils.helpers import load_config

def load_training_data():
    """Load training data from JSON file"""
    with open("data/training_data/training_samples.json", "r") as f:
        data = json.load(f)
    return data

def prepare_dataset():
    """Prepare dataset for training"""
    data = load_training_data()
    
    texts = []
    labels = []
    
    for sample in data:
        text = f"Interests: {sample['user_interests']} Paper: {sample['paper_title']} {sample['paper_abstract'][:400]}"
        texts.append(text)
        labels.append(sample['relevance_score'])
    
    dataset = Dataset.from_dict({
        'text': texts,
        'label': labels
    })
    
    return dataset.train_test_split(test_size=0.2, seed=42)

def fine_tune_model():
    """Fine-tune the relevance model using LoRA"""
    config = load_config()
    
    # Load model and tokenizer
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=1,
        problem_type="regression"
    )
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query_proj", "value_proj", "key_proj", "dense"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare dataset
    dataset = prepare_dataset()
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config['paths']['models_dir'],
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir='./logs',
        logging_steps=10,
    )
    
    # Compute metrics
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.flatten()
        
        mse = np.mean((predictions - labels) ** 2)
        mae = np.mean(np.abs(predictions - labels))
        
        return {"mse": mse, "mae": mae}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(config['paths']['models_dir'])
    
    print("Fine-tuning completed!")

if __name__ == "__main__":
    fine_tune_model()