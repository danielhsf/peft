# Makefile para projeto PEFT
# Comandos úteis para desenvolvimento

.PHONY: help install setup train evaluate clean test

# Variáveis
PYTHON = python3
PIP = pip3
JUPYTER = jupyter

help: ## Mostra esta ajuda
	@echo "Comandos disponíveis:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Instala dependências
	$(PIP) install -r requirements.txt

setup: ## Configura estrutura do projeto
	mkdir -p notebooks data models scripts cache
	@echo "✅ Estrutura do projeto configurada"

train: ## Treina modelo PEFT
	$(PYTHON) scripts/train_peft.py

evaluate: ## Avalia modelo treinado
	$(PYTHON) scripts/evaluate_peft.py

test: ## Executa testes básicos
	$(PYTHON) -c "import torch; import transformers; import peft; print('✅ Todas as dependências estão funcionando')"

jupyter: ## Inicia Jupyter Lab
	$(JUPYTER) lab --ip=0.0.0.0 --port=8888 --no-browser

notebook: ## Inicia Jupyter Notebook
	$(JUPYTER) notebook --ip=0.0.0.0 --port=8888 --no-browser

clean: ## Limpa arquivos temporários
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	@echo "🧹 Arquivos temporários removidos"

clean-models: ## Remove modelos salvos
	rm -rf models/*
	@echo "🗑️ Modelos removidos"

clean-cache: ## Remove cache
	rm -rf cache/*
	@echo "🗑️ Cache removido"

format: ## Formata código Python
	$(PIP) install black isort
	black scripts/ config.py
	isort scripts/ config.py

lint: ## Executa linter
	$(PIP) install flake8
	flake8 scripts/ config.py

check: format lint test ## Executa todas as verificações

dev-setup: install setup ## Configuração completa para desenvolvimento

# Comandos específicos para diferentes modelos
train-small: ## Treina com modelo pequeno
	$(PYTHON) -c "from config import DEVELOPMENT_CONFIG; print('Configuração para modelo pequeno:', DEVELOPMENT_CONFIG.base_model_name)"

train-medium: ## Treina com modelo médio
	$(PYTHON) -c "from config import DEFAULT_CONFIG; print('Configuração para modelo médio:', DEFAULT_CONFIG.base_model_name)"

train-large: ## Treina com modelo grande
	$(PYTHON) -c "from config import PRODUCTION_CONFIG; print('Configuração para modelo grande:', PRODUCTION_CONFIG.base_model_name)"

# Comandos de monitoramento
gpu-info: ## Mostra informações da GPU
	nvidia-smi

memory-info: ## Mostra uso de memória
	free -h

disk-info: ## Mostra uso de disco
	df -h

# Comandos de backup
backup: ## Faz backup dos modelos
	tar -czf backup_models_$(shell date +%Y%m%d_%H%M%S).tar.gz models/
	@echo "💾 Backup criado"

restore: ## Restaura backup (especificar arquivo com BACKUP_FILE=arquivo.tar.gz)
	@if [ -z "$(BACKUP_FILE)" ]; then echo "❌ Especifique BACKUP_FILE=arquivo.tar.gz"; exit 1; fi
	tar -xzf $(BACKUP_FILE)
	@echo "✅ Backup restaurado" 