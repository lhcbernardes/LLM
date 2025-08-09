# Configurações para o projeto de Fine-tuning Amazon Titles

# Configurações do Modelo
MODEL_CONFIG = {
    "base_model": "microsoft/DialoGPT-medium",  # Modelo base
    "max_length": 512,  # Comprimento máximo das sequências
    "batch_size": 4,  # Tamanho do batch
    "learning_rate": 2e-5,  # Taxa de aprendizado
    "num_epochs": 3,  # Número de épocas
    "warmup_steps": 500,  # Passos de aquecimento
    "weight_decay": 0.01,  # Decaimento de peso
    "save_steps": 1000,  # Salvar a cada X passos
    "eval_steps": 500,  # Avaliar a cada X passos
    "logging_steps": 100,  # Log a cada X passos
}

# Configurações de Dados
DATA_CONFIG = {
    "input_file": "trn.json",  # Arquivo de entrada
    "sample_size": 100000,  # Tamanho da amostra (None para usar todos)
    "train_split": 0.9,  # Proporção para treino
    "val_split": 0.1,  # Proporção para validação
    "min_title_length": 10,  # Comprimento mínimo do título
    "min_description_length": 20,  # Comprimento mínimo da descrição
}

# Configurações de Geração
GENERATION_CONFIG = {
    "max_new_tokens": 100,  # Máximo de tokens gerados
    "temperature": 0.7,  # Temperatura para geração
    "top_p": 0.9,  # Top-p sampling
    "do_sample": True,  # Usar sampling
    "repetition_penalty": 1.1,  # Penalidade de repetição
}

# Configurações de Saída
OUTPUT_CONFIG = {
    "model_output_dir": "./fine_tuned_model",  # Diretório do modelo
    "logs_dir": "./logs",  # Diretório de logs
    "results_file": "test_results.json",  # Arquivo de resultados
    "checkpoint_dir": "./checkpoints",  # Diretório de checkpoints
}

# Configurações de Hardware
HARDWARE_CONFIG = {
    "device": "auto",  # Dispositivo (auto, cuda, cpu)
    "load_in_4bit": True,  # Quantização 4-bit para economia de memória
    "gradient_accumulation_steps": 4,  # Acumulação de gradientes
    "fp16": True,  # Usar precisão mista
}

# Configurações de Limpeza de Dados
CLEANING_CONFIG = {
    "remove_html": True,  # Remover tags HTML
    "remove_special_chars": True,  # Remover caracteres especiais
    "normalize_whitespace": True,  # Normalizar espaços em branco
    "lowercase": False,  # Converter para minúsculas
    "remove_duplicates": True,  # Remover duplicatas
}

# Configurações de Logging
LOGGING_CONFIG = {
    "level": "INFO",  # Nível de log
    "format": "%(asctime)s - %(levelname)s - %(message)s",
    "file": "fine_tuning.log",  # Arquivo de log
}

# Configurações de Teste
TEST_CONFIG = {
    "num_test_samples": 5,  # Número de amostras para teste
    "test_prompts": [
        "Title: Wireless Bluetooth Headphones\nDescription:",
        "Title: Organic Coffee Beans\nDescription:",
        "Title: Smartphone Case\nDescription:",
        "Title: Yoga Mat\nDescription:",
        "Title: Kitchen Knife Set\nDescription:"
    ]
}

# Configurações de Validação
VALIDATION_CONFIG = {
    "metrics": ["loss", "perplexity"],  # Métricas para monitorar
    "early_stopping_patience": 3,  # Paciência para early stopping
    "save_best_only": True,  # Salvar apenas o melhor modelo
} 