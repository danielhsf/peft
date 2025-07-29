#!/usr/bin/env python3
"""
Script de exemplo para treinamento com PEFT
Demonstra como usar LoRA para fine-tuning eficiente
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import json

def load_model_and_tokenizer(model_name="microsoft/DialoGPT-medium"):
    """Carrega modelo e tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Adiciona padding token se necessário
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def setup_lora_config():
    """Configura LoRA"""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # Rank da decomposição
        lora_alpha=32,  # Alpha para scaling
        lora_dropout=0.1,  # Dropout
        target_modules=["q_proj", "v_proj"]  # Módulos para aplicar LoRA
    )

def prepare_dataset(tokenizer, texts, max_length=512):
    """Prepara dataset para treinamento"""
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
    
    dataset = Dataset.from_dict({"text": texts})
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

def train_model(model, tokenizer, dataset, output_dir="./models/peft_model"):
    """Treina o modelo com PEFT"""
    # Aplica LoRA
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    
    # Configura treinamento
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Treina
    trainer.train()
    
    # Salva modelo
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer

def main():
    """Função principal"""
    print("🚀 Iniciando treinamento com PEFT...")
    
    # Carrega modelo e tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Exemplo de dados (substitua pelos seus dados)
    sample_texts = [
        "Olá, como você está?",
        "O que você gosta de fazer?",
        "Conte-me uma história interessante.",
        "Qual é sua opinião sobre tecnologia?",
    ]
    
    # Prepara dataset
    dataset = prepare_dataset(tokenizer, sample_texts)
    
    # Treina modelo
    trained_model, trained_tokenizer = train_model(model, tokenizer, dataset)
    
    print("✅ Treinamento concluído!")
    print(f"Modelo salvo em: ./models/peft_model")

if __name__ == "__main__":
    main() 