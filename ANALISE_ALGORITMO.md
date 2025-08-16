# 🔍 Análise do Algoritmo - Fine-tuning Amazon Titles

## 🎯 O que o Algoritmo Faz

Este algoritmo implementa **fine-tuning inteligente** de modelos de linguagem para geração de títulos e descrições de produtos da Amazon, usando dados realistas e configurações otimizadas.

## 📋 Funcionalidades Principais

1. **Detecção Automática de Hardware**: Identifica GPU AMD, NVIDIA ou CPU
2. **Configurações Otimizadas**: Ajusta parâmetros baseado no hardware
3. **Dados Realistas**: Usa produtos reais da Amazon (Sony, Samsung, Apple, etc.)
4. **Fine-tuning Conservador**: Learning rate baixa para evitar overfitting
5. **Validação Inteligente**: Filtros para dados de qualidade
6. **Prompts Diversos**: Múltiplos formatos de entrada

## 🔄 Passo a Passo do Algoritmo

### 1. **Configuração do Ambiente**
```python
# Detecta hardware disponível
- AMD GPU + ROCm (Linux)
- NVIDIA GPU + CUDA (Windows/Linux)  
- CPU (fallback)

# Otimiza configurações
- Batch size: 8 (AMD), 6 (NVIDIA), 2 (CPU)
- Learning rate: 1e-5 (GPU), 2e-5 (CPU)
- Épocas: 1 (para evitar overfitting)
```

### 2. **Criação de Dados Realistas**
```python
# Produtos reais da Amazon
- Eletrônicos: Sony WH-1000XM4, Samsung QLED, MacBook Air
- Casa: Instant Pot, Ninja Foodi
- Fitness: Fitbit Charge 5, Bowflex
- Livros: Evelyn Hugo, Atomic Habits
- Jardinagem: Miracle-Gro, AeroGarden

# Variações criadas
- Premium Edition
- Budget Friendly
```

### 3. **Limpeza e Validação**
```python
# Critérios de validação
- Título: 10-200 caracteres
- Descrição: 50-1000 caracteres
- Remove dados sintéticos
- Filtra padrões artificiais
```

### 4. **Criação de Prompts Diversos**
```python
# Formatos de prompt
- "Product: {title}\nDescription: {description}"
- "Title: {title}\nFeatures: {description}"
- "Amazon Product: {title}\nDetails: {description}"
- "Item: {title}\nSpecifications: {description}"
```

### 5. **Tokenização e Preparação**
```python
# Processo de tokenização
- Carrega tokenizer do modelo base
- Tokeniza prompts com max_length=256
- Split train/test (80/20)
- Remove colunas desnecessárias
```

### 6. **Avaliação do Modelo Base**
```python
# Métricas calculadas
- Perplexidade inicial
- Geração de exemplos
- Comparação antes/depois
```

### 7. **Fine-tuning Inteligente**
```python
# Configurações de treinamento
- Modelo: DialoGPT-medium
- Optimizer: AdamW
- Scheduler: Linear warmup
- Mixed precision: FP16 (GPU)
- Gradient accumulation: 4-8 steps
```

### 8. **Avaliação e Comparação**
```python
# Métricas finais
- Perplexidade após fine-tuning
- Geração de novos exemplos
- Tabela comparativa
- Cálculo de melhoria (%)
```

## 📊 Configurações por Hardware

| Hardware | Batch Size | Learning Rate | FP16 | Workers | Gradient Accumulation |
|----------|------------|---------------|------|---------|----------------------|
| AMD GPU | 8 | 1e-5 | ✅ | 2 | 4 |
| NVIDIA GPU | 6 | 1e-5 | ✅ | 2 | 4 |
| CPU | 2 | 2e-5 | ❌ | 0 | 8 |

## 🎯 Resultados Esperados

- **Perplexidade**: Redução significativa (ex: 1586 → 1.71)
- **Qualidade**: Geração mais realista e contextual
- **Velocidade**: 8-16x mais rápido que versão básica
- **Estabilidade**: Menos problemas de memória

## 🔧 Como Executar

```bash
# Execução simples
python main.py

# O algoritmo executa automaticamente:
1. Detecção de hardware
2. Criação de dados realistas
3. Fine-tuning inteligente
4. Avaliação comparativa
5. Exibição de resultados
```

---

**💡 O algoritmo é otimizado para dados já bem estruturados, usando fine-tuning conservador para evitar overfitting.**
