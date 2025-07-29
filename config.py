"""
Configurações centralizadas para o projeto PEFT
"""

import os
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class PEFTConfig:
    """Configurações para PEFT"""
    # Modelo base
    base_model_name: str = "microsoft/DialoGPT-medium"
    
    # Configurações LoRA
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    
    # Configurações de treinamento
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    
    # Configurações de geração
    max_length: int = 512
    temperature: float = 0.7
    do_sample: bool = True
    
    # Caminhos
    output_dir: str = "./models/peft_model"
    data_dir: str = "./data"
    scripts_dir: str = "./scripts"
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]

@dataclass
class ModelConfig:
    """Configurações para diferentes modelos"""
    
    # Modelos pequenos (para testes)
    small_models = {
        "dialoGPT": "microsoft/DialoGPT-small",
        "gpt2": "gpt2",
        "distilgpt2": "distilgpt2"
    }
    
    # Modelos médios (para desenvolvimento)
    medium_models = {
        "dialoGPT": "microsoft/DialoGPT-medium",
        "gpt2_medium": "gpt2-medium",
        "bloom": "bigscience/bloom-560m"
    }
    
    # Modelos grandes (para produção)
    large_models = {
        "gpt2_large": "gpt2-large",
        "bloom_1b": "bigscience/bloom-1b1",
        "llama": "meta-llama/Llama-2-7b-hf"  # Requer acesso
    }

# Configurações padrão
DEFAULT_CONFIG = PEFTConfig()

# Configurações para diferentes cenários
DEVELOPMENT_CONFIG = PEFTConfig(
    base_model_name="microsoft/DialoGPT-small",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    max_length=256
)

PRODUCTION_CONFIG = PEFTConfig(
    base_model_name="microsoft/DialoGPT-medium",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-4
)

# Configurações de ambiente
ENV_VARS = {
    "CUDA_VISIBLE_DEVICES": "0",  # GPU a usar
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_CACHE": "./cache",
    "HF_HOME": "./cache"
}

def setup_environment():
    """Configura variáveis de ambiente"""
    for key, value in ENV_VARS.items():
        os.environ[key] = value

def get_config(env: str = "default") -> PEFTConfig:
    """Retorna configuração baseada no ambiente"""
    configs = {
        "default": DEFAULT_CONFIG,
        "development": DEVELOPMENT_CONFIG,
        "production": PRODUCTION_CONFIG
    }
    return configs.get(env, DEFAULT_CONFIG) 