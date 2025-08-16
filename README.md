# 🚀 Fine-tuning Amazon Titles - Projeto Otimizado

Este projeto implementa fine-tuning de modelos de linguagem para geração de títulos e descrições de produtos da Amazon, com otimizações automáticas para diferentes plataformas (Windows, Linux, CPU, GPU).

## 🎯 Características

- **Multi-plataforma**: Suporte automático para Windows (CUDA), Linux (ROCm), e CPU
- **Otimizações automáticas**: Configurações otimizadas baseadas no hardware detectado
- **Interface rica**: Progress bars e tabelas bonitas com Rich
- **Treinamento incremental**: Continua treinamento de modelos existentes
- **Performance otimizada**: Batch size, gradient accumulation e mixed precision

## 📋 Pré-requisitos

- Python 3.8+
- PyTorch (instalado automaticamente com suporte apropriado)
- Transformers, Datasets, Rich

## 🚀 Instalação Rápida

```bash
# Instalar dependências
pip install -r requirements_fine_tuning.txt

# Executar o projeto
python main.py
```

## 📊 Melhorias de Performance

| Plataforma | Batch Size | Batch Efetivo | FP16 | Workers |
|------------|------------|---------------|------|---------|
| AMD GPU (ROCm) | 8 | 32 | ✅ | 2 |
| NVIDIA GPU (CUDA) | 6 | 24 | ✅ | 2 |
| CPU | 2 | 16 | ❌ | 0 |

## 🎮 Como Usar

### Execução Simples
```bash
python main.py
```

O script irá:
1. Detectar automaticamente o hardware disponível
2. Configurar otimizações apropriadas
3. Carregar dados de exemplo
4. Executar fine-tuning
5. Mostrar resultados comparativos

### Configurações Personalizadas
Edite `config.py` para ajustar:
- Modelo base
- Parâmetros de treinamento
- Configurações de dados

## 📁 Estrutura do Projeto

```
LLM/
├── main.py                    # Script principal melhorado
├── config.py                  # Configurações do projeto
├── requirements_fine_tuning.txt # Dependências
├── fine_tuned_model/          # Modelo treinado
├── ANALISE_ALGORITMO.md       # Análise do algoritmo
└── README.md                  # Este arquivo
```

## 🔧 Detecção Automática de Hardware

O projeto detecta automaticamente:
- **AMD GPU + ROCm** (Linux): Configurações otimizadas para AMD
- **NVIDIA GPU + CUDA** (Windows/Linux): Configurações otimizadas para NVIDIA
- **CPU**: Configurações otimizadas para CPU com multi-threading

## 📈 Resultados Esperados

- **Velocidade**: 8-16x mais rápido que versão básica
- **Eficiência**: Melhor utilização do hardware
- **Estabilidade**: Menos problemas de memória
- **Qualidade**: Resultados similares ou melhores

## 🎯 Exemplo de Saída

```
🚀 Fine-tuning Amazon Titles - Multi-Plataforma Otimizada
🔧 Configurando dispositivo otimizado...
⚠️ Nenhuma GPU detectada. Usando CPU otimizada...
🔧 CPU otimizada: 16 threads
📊 Configurações: Device=CPU, Batch Size=2, Batch Efetivo=16, FP16=False

📊 Comparativo de Resultados - CPU
┏━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Métrica      ┃ Antes   ┃ Depois ┃ Melhoria (%) ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━┩
│ Perplexidade │ 1586.46 │ 1.71   │ 99.9%        │
└──────────────┴─────────┴────────┴──────────────┘
```

## 🤝 Suporte

Se encontrar problemas:
1. Verifique se todas as dependências estão instaladas
2. Execute `python main.py` para ver mensagens de erro detalhadas
3. Verifique se há memória suficiente disponível

---

**🚀 Agora seu fine-tuning será muito mais rápido e eficiente!**
