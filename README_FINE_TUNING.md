# 🚀 Fine-tuning Amazon Titles - Tech Challenge

Projeto completo de Fine-tuning de Foundation Models usando o dataset Amazon Titles (~2 milhões de registros).

## 🎯 Objetivo do Projeto

Implementar um pipeline completo de fine-tuning para modelos de linguagem usando dados de produtos da Amazon, demonstrando:

- **Preparação de dados** em larga escala
- **Fine-tuning otimizado** com Unsloth
- **Avaliação de performance** antes e depois
- **Análise exploratória** completa dos dados

## 📊 Dataset

**Amazon Titles Dataset**
- **Arquivo**: `trn.json`
- **Tamanho**: ~2 milhões de registros
- **Estrutura**: Títulos e descrições de produtos
- **Formato**: JSON

### Estrutura dos Dados
```json
{
  "title": "Nome do Produto",
  "description": "Descrição detalhada do produto"
}
```

## 🏗️ Arquitetura do Projeto

```
fine_tuning_project/
├── fine_tuning_project.py    # Pipeline principal
├── data_analysis.py          # Análise exploratória
├── config.py                 # Configurações
├── requirements_fine_tuning.txt
├── README_FINE_TUNING.md
└── outputs/
    ├── fine_tuned_model/     # Modelo treinado
    ├── logs/                 # Logs de treinamento
    ├── plots/                # Visualizações
    └── reports/              # Relatórios
```

## 🚀 Como Executar

### 1. Preparação do Ambiente

```bash
# Instalar dependências
pip install -r requirements_fine_tuning.txt

# Para GPU (recomendado)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Análise Exploratória

```bash
# Executar análise dos dados
python data_analysis.py
```

**Saídas**:
- `data_analysis_plots.png` - Visualizações
- `wordcloud_titles.png` - Wordcloud dos títulos
- `wordcloud_descriptions.png` - Wordcloud das descrições
- `data_analysis_report.json` - Relatório completo

### 3. Fine-tuning

```bash
# Executar pipeline completo
python fine_tuning_project.py
```

**Saídas**:
- `fine_tuned_model/` - Modelo treinado
- `test_results.json` - Resultados dos testes
- `fine_tuning.log` - Logs do processo

## 🔧 Configurações

### Modelo Base
- **Modelo**: `microsoft/DialoGPT-medium`
- **Razão**: Boa performance para geração de texto
- **Alternativas**: `gpt2`, `EleutherAI/gpt-neo-125M`

### Parâmetros de Treinamento
```python
MODEL_CONFIG = {
    "max_length": 512,
    "batch_size": 4,
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "warmup_steps": 500
}
```

### Otimizações
- **Quantização 4-bit**: Economia de memória
- **Gradient Accumulation**: Simula batches maiores
- **Mixed Precision**: Acelera treinamento

## 📈 Fluxo de Trabalho

### 1. Preparação de Dados
- ✅ Carregamento do JSON
- ✅ Limpeza (NAs, HTML, caracteres especiais)
- ✅ Filtros de qualidade
- ✅ Criação de prompts

### 2. Análise Exploratória
- ✅ Estatísticas básicas
- ✅ Distribuição de comprimentos
- ✅ Análise de palavras frequentes
- ✅ Relatório de qualidade

### 3. Fine-tuning
- ✅ Carregamento otimizado com Unsloth
- ✅ Tokenização eficiente
- ✅ Treinamento com early stopping
- ✅ Salvamento de checkpoints

### 4. Avaliação
- ✅ Teste do modelo base
- ✅ Teste do modelo fine-tunado
- ✅ Comparação de resultados

## 🎯 Conceitos Técnicos

### Fine-tuning vs RAG

| Aspecto | Fine-tuning | RAG |
|---------|-------------|-----|
| **Processo** | Retreinar modelo | Busca + Geração |
| **Dados** | Estáticos | Dinâmicos |
| **Performance** | Superior | Boa |
| **Complexidade** | Alta | Baixa |
| **Custo** | Alto | Baixo |

### Vantagens do Fine-tuning
- ✅ **Melhor qualidade** de geração
- ✅ **Consistência** com dados específicos
- ✅ **Performance** otimizada
- ✅ **Menor latência** na inferência

### Vantagens do RAG
- ✅ **Atualização fácil** dos dados
- ✅ **Implementação simples**
- ✅ **Menor custo** computacional
- ✅ **Flexibilidade** de dados

## 📊 Métricas de Avaliação

### Antes do Fine-tuning
- Loss inicial
- Perplexidade
- Qualidade das respostas

### Depois do Fine-tuning
- Loss final
- Melhoria na perplexidade
- Qualidade das respostas específicas

## 🔍 Análise de Dados

### Estatísticas Esperadas
- **Total de registros**: ~2M
- **Títulos únicos**: ~80-90%
- **Descrições únicas**: ~70-80%
- **Comprimento médio título**: 50-100 caracteres
- **Comprimento médio descrição**: 200-500 caracteres

### Limpeza Aplicada
- ✅ Remoção de HTML tags
- ✅ Normalização de espaços
- ✅ Filtro por comprimento mínimo
- ✅ Remoção de duplicatas

## 🛠️ Dicas Práticas

### Performance
- **Começar com 100k registros** para teste
- **Aumentar gradualmente** se necessário
- **Monitorar uso de memória**
- **Usar GPU** quando disponível

### Qualidade dos Dados
- **Manter dados em inglês** para melhor performance
- **Não reduzir abaixo de 100k** registros
- **Validar limpeza** antes do treinamento
- **Testar diferentes modelos** se necessário

### Treinamento
- **Curva de erro pode oscilar** (normal)
- **Early stopping** para evitar overfitting
- **Checkpoints regulares** para recuperação
- **Monitoramento de métricas** em tempo real

## 📝 Entregáveis

### ✅ Código Completo
- Pipeline de fine-tuning
- Análise exploratória
- Configurações otimizadas

### ✅ Documentação
- README detalhado
- Comentários no código
- Relatórios de análise

### ✅ Resultados
- Modelo fine-tunado
- Métricas de avaliação
- Comparações antes/depois

### ✅ Vídeo de Demonstração
- Explicação do processo
- Demonstração dos resultados
- Análise dos dados

## 🚀 Próximos Passos

### Melhorias Possíveis
- [ ] Testar diferentes modelos base
- [ ] Implementar RAG como comparação
- [ ] Otimizar hiperparâmetros
- [ ] Adicionar mais métricas de avaliação
- [ ] Interface web para demonstração

### Expansões
- [ ] Dataset maior (todos os 2M registros)
- [ ] Modelos maiores (7B+ parâmetros)
- [ ] Fine-tuning LoRA/QLoRA
- [ ] Integração com APIs

## 📚 Referências

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Amazon Product Data](https://jmcauley.ucsd.edu/data/amazon/)
- [Fine-tuning Best Practices](https://huggingface.co/docs/transformers/training)

---

**"The best model is the one that works for your specific use case."** 🤖

*Projeto desenvolvido para Tech Challenge - Fine-tuning de Foundation Models* 