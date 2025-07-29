# PEFT - Parameter-Efficient Fine-Tuning

## Definição

PEFT (Parameter-Efficient Fine-Tuning ou <i>Ajuste Fino Eficiente em Parâmetros</i>) é uma biblioteca para adaptar eficientemente Large Language Models (LLMs) pré-treinados a diversas aplicações específicas (downstream tasks) sem a necessidade de realizar o ajuste fino de todos os parâmetros do modelo, pois isso tem um alto custo. Os métodos PEFT realizam o ajuste fino apenas de um pequeno número de parâmetros (adicionais ou não) do modelo — diminuindo significativamente os custos computacionais e de armazenamento — enquanto proporcionam um desempenho comparável ao de um modelo totalmente ajustado (fully fine-tuned). Isso torna mais acessível treinar e armazenar grandes modelos de linguagem (LLMs) em hardware de consumidor.

O PEFT está integrado às bibliotecas Transformers, Diffusers e Accelerate para oferecer uma forma mais rápida e fácil de carregar, treinar e usar grandes modelos para inferência.

## 📁 Estrutura do Projeto

```
peft/
├── notebooks/                    # Tutoriais e experimentos
│   ├── peft_basics_tutorial.ipynb
│   ├── PEFT_Quicktour.ipynb
│   ├── Quantized_Low_Rank_Adaptation_(QLoRA).ipynb
│   └── M2M100_com_Peft.ipynb
├── data/                        # Dados de treinamento
├── models/                      # Modelos salvos
├── scripts/                     # Scripts Python
├── requirements.txt
└── README.md
```

## 🚀 Como Usar

### Instalação

```bash
pip install -r requirements.txt
```

### Executando os Tutoriais

1. **PEFT Basics**: `peft_basics_tutorial.ipynb`
   - Introdução aos conceitos básicos do PEFT
   - LoRA (Low-Rank Adaptation)
   - Prefix Tuning

2. **PEFT Quick Tour**: `PEFT_Quicktour.ipynb`
   - Visão geral rápida das funcionalidades
   - Exemplos práticos

3. **QLoRA**: `Quantized_Low_Rank_Adaptation_(QLoRA).ipynb`
   - Fine-tuning quantizado
   - Otimização de memória

4. **M2M100 com PEFT**: `M2M100_com_Peft.ipynb`
   - Aplicação em modelos multilingues

## 🔧 Métodos PEFT Suportados

- **LoRA** (Low-Rank Adaptation)
- **Prefix Tuning**
- **P-Tuning**
- **Prompt Tuning**
- **AdaLoRA**
- **QLoRA** (Quantized LoRA)

## 📊 Monitoramento e Avaliação

- Use `accelerate` para treinamento distribuído
- Monitore métricas com `wandb` ou `tensorboard`
- Avalie com `sacrebleu` para tarefas de tradução

## 💡 Próximos Passos Recomendados

1. Experimente diferentes métodos PEFT
2. Compare performance vs. custo computacional
3. Aplique em seus próprios dados
4. Explore quantização para otimização de memória