#!/usr/bin/env python3
"""
Script para avaliação de modelos PEFT
Avalia performance e compara com modelo base
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
import time
import psutil
import numpy as np
from typing import List, Dict, Any

def load_peft_model(base_model_name: str, peft_model_path: str):
    """Carrega modelo PEFT"""
    print(f"📥 Carregando modelo PEFT de: {peft_model_path}")
    
    # Carrega configuração PEFT
    config = PeftConfig.from_pretrained(peft_model_path)
    
    # Carrega modelo base
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Carrega adaptadores PEFT
    model = PeftModel.from_pretrained(base_model, peft_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    return model, tokenizer

def measure_memory_usage():
    """Mede uso de memória"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,
        'vms_mb': memory_info.vms / 1024 / 1024
    }

def generate_text(model, tokenizer, prompt: str, max_length: int = 100):
    """Gera texto com o modelo"""
    inputs = tokenizer(prompt, return_tensors="pt")
    
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generation_time = time.time() - start_time
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text, generation_time

def evaluate_model(model, tokenizer, test_prompts: List[str]):
    """Avalia modelo com prompts de teste"""
    results = []
    
    print("🔍 Avaliando modelo...")
    
    for i, prompt in enumerate(test_prompts):
        print(f"  Teste {i+1}/{len(test_prompts)}: {prompt[:50]}...")
        
        # Mede memória antes
        memory_before = measure_memory_usage()
        
        # Gera texto
        generated_text, generation_time = generate_text(model, tokenizer, prompt)
        
        # Mede memória depois
        memory_after = measure_memory_usage()
        
        # Calcula métricas
        input_tokens = len(tokenizer.encode(prompt))
        output_tokens = len(tokenizer.encode(generated_text)) - input_tokens
        tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0
        
        result = {
            'prompt': prompt,
            'generated_text': generated_text,
            'generation_time': generation_time,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'tokens_per_second': tokens_per_second,
            'memory_before_mb': memory_before['rss_mb'],
            'memory_after_mb': memory_after['rss_mb'],
            'memory_increase_mb': memory_after['rss_mb'] - memory_before['rss_mb']
        }
        
        results.append(result)
    
    return results

def print_evaluation_summary(results: List[Dict[str, Any]]):
    """Imprime resumo da avaliação"""
    print("\n" + "="*60)
    print("📊 RESUMO DA AVALIAÇÃO")
    print("="*60)
    
    # Estatísticas de tempo
    generation_times = [r['generation_time'] for r in results]
    tokens_per_second = [r['tokens_per_second'] for r in results]
    
    print(f"⏱️  Tempo médio de geração: {np.mean(generation_times):.3f}s")
    print(f"⚡ Tokens por segundo: {np.mean(tokens_per_second):.2f}")
    
    # Estatísticas de memória
    memory_increases = [r['memory_increase_mb'] for r in results]
    print(f"💾 Aumento médio de memória: {np.mean(memory_increases):.2f} MB")
    
    # Estatísticas de tokens
    total_input_tokens = sum(r['input_tokens'] for r in results)
    total_output_tokens = sum(r['output_tokens'] for r in results)
    print(f"📝 Total de tokens de entrada: {total_input_tokens}")
    print(f"📝 Total de tokens de saída: {total_output_tokens}")
    
    print("\n📋 Exemplos de geração:")
    for i, result in enumerate(results[:3]):  # Mostra apenas os 3 primeiros
        print(f"\n  Exemplo {i+1}:")
        print(f"    Prompt: {result['prompt'][:100]}...")
        print(f"    Geração: {result['generated_text'][:200]}...")

def main():
    """Função principal"""
    # Configurações
    base_model_name = "microsoft/DialoGPT-medium"
    peft_model_path = "./models/peft_model"
    
    # Prompts de teste
    test_prompts = [
        "Olá, como você está?",
        "Conte-me uma história sobre tecnologia.",
        "Qual é sua opinião sobre inteligência artificial?",
        "Explique o que é machine learning.",
        "Dê-me algumas dicas para programar em Python."
    ]
    
    try:
        # Carrega modelo
        model, tokenizer = load_peft_model(base_model_name, peft_model_path)
        
        # Avalia modelo
        results = evaluate_model(model, tokenizer, test_prompts)
        
        # Imprime resultados
        print_evaluation_summary(results)
        
    except FileNotFoundError:
        print("❌ Modelo PEFT não encontrado!")
        print("Execute primeiro o script de treinamento: python scripts/train_peft.py")
    except Exception as e:
        print(f"❌ Erro durante avaliação: {e}")

if __name__ == "__main__":
    main() 