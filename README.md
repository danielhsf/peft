# PEFT

## Definição

PEFT (Parameter-Efficient Fine-Tuning ou <i>Ajuste Fino Eficiente em Parâmetros</i>) é uma biblioteca para adaptar eficientemente Large Language Models (LLMs) pré-treinados a diversas aplicações específicas (downstream tasks) sem a necessidade de realizar o ajuste fino de todos os parâmetros do modelo, pois isso tem um alto custo. Os métodos PEFT realizam o ajuste fino apenas de um pequeno número de parâmetros (adicionais ou não) do modelo — diminuindo significativamente os custos computacionais e de armazenamento — enquanto proporcionam um desempenho comparável ao de um modelo totalmente ajustado (fully fine-tuned). Isso torna mais acessível treinar e armazenar grandes modelos de linguagem (LLMs) em hardware de consumidor.

O PEFT está integrado às bibliotecas Transformers, Diffusers e Accelerate para oferecer uma forma mais rápida e fácil de carregar, treinar e usar grandes modelos para inferência.