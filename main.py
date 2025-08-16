#!/usr/bin/env python3
"""
🎯 Fine-tuning Amazon Titles - Versão Melhorada e Realista
Tech Challenge - Foundation Models
Foco: Dados reais e fine-tuning inteligente
"""

import os
import gc
import json
import math
import platform
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import Dataset
import accelerate
import re

# Console para saída bonita
console = Console()

# ------------------- Configuração Multi-Plataforma -------------------

def setup_optimal_device():
    """Configura o melhor dispositivo disponível."""
    console.log("[cyan]🔧 Configurando dispositivo otimizado...[/cyan]")
    
    if platform.system() == "Linux" and hasattr(torch.version, 'hip') and torch.version.hip is not None:
        console.log(f"[green]✅ ROCm detectado: {torch.version.hip}[/green]")
        device = torch.device("cuda")
        console.log(f"[green]✅ GPU AMD disponível: {torch.cuda.get_device_name()}[/green]")
        return device, "AMD_ROCm"
    elif torch.cuda.is_available():
        console.log("[green]✅ CUDA detectado[/green]")
        device = torch.device("cuda")
        console.log(f"[green]✅ GPU NVIDIA disponível: {torch.cuda.get_device_name()}[/green]")
        return device, "NVIDIA_CUDA"
    else:
        console.log("[yellow]⚠️ Nenhuma GPU detectada. Usando CPU otimizada...[/yellow]")
        device = torch.device("cpu")
        return device, "CPU"

def optimize_memory_settings(device_type):
    """Otimiza configurações de memória."""
    if device_type in ["AMD_ROCm", "NVIDIA_CUDA"]:
        torch.backends.cudnn.benchmark = True
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        console.log("[cyan]🔧 Configurações de memória GPU otimizadas aplicadas[/cyan]")
    else:
        torch.set_num_threads(os.cpu_count())
        console.log(f"[cyan]🔧 CPU otimizada: {os.cpu_count()} threads[/cyan]")

def get_optimal_config(device_type):
    """Retorna configurações otimizadas."""
    if device_type == "AMD_ROCm":
        return {
            "model_name": "microsoft/DialoGPT-medium",
            "max_length": 256,
            "batch_size": 8,
            "learning_rate": 1e-5,  # Taxa menor para dados reais
            "num_epochs": 1,  # Menos épocas para evitar overfitting
            "warmup_steps": 50,
            "weight_decay": 0.01,
            "gradient_accumulation_steps": 4,
            "fp16": True,
            "dataloader_num_workers": 2
        }
    elif device_type == "NVIDIA_CUDA":
        return {
            "model_name": "microsoft/DialoGPT-medium",
            "max_length": 256,
            "batch_size": 6,
            "learning_rate": 1e-5,
            "num_epochs": 1,
            "warmup_steps": 50,
            "weight_decay": 0.01,
            "gradient_accumulation_steps": 4,
            "fp16": True,
            "dataloader_num_workers": 2
        }
    else:  # CPU
        return {
            "model_name": "microsoft/DialoGPT-medium",
            "max_length": 256,
            "batch_size": 2,
            "learning_rate": 2e-5,
            "num_epochs": 1,
            "warmup_steps": 20,
            "weight_decay": 0.01,
            "gradient_accumulation_steps": 8,
            "fp16": False,
            "dataloader_num_workers": 0
        }

# ------------------- Funções de Dados Melhoradas -------------------

def create_realistic_amazon_data():
    """Cria dados mais realistas baseados em produtos reais da Amazon."""
    console.log("[cyan]📝 Criando dados realistas da Amazon...[/cyan]")
    
    realistic_data = [
        # Eletrônicos
        {
            "title": "Sony WH-1000XM4 Wireless Noise Canceling Headphones",
            "description": "Industry-leading noise canceling with Dual Noise Sensor technology. Next-level music with Edge-AI, co-developed with Sony Music Studios Tokyo. Up to 30-hour battery life with quick charge (10 min charge for 5 hours of playback). Touch controls with Speak-to-Chat technology."
        },
        {
            "title": "Samsung 65-inch Class QLED 4K Q60T Series",
            "description": "Quantum HDR delivers enhanced contrast and brightness. Dual LED backlighting creates deeper blacks and brighter whites. Quantum Processor 4K transforms everything you watch into 4K resolution. Game Mode optimizes your TV for the best gaming experience."
        },
        {
            "title": "Apple MacBook Air M1 Chip 13-inch",
            "description": "Apple-designed M1 chip for a giant leap in CPU, GPU, and machine learning performance. Get more done with up to 18 hours of battery life. 8-core CPU delivers up to 3.5x faster performance than the previous generation. All-day battery life."
        },
        
        # Casa e Cozinha
        {
            "title": "Instant Pot Duo 7-in-1 Electric Pressure Cooker",
            "description": "7-in-1 functionality: pressure cooker, slow cooker, rice cooker, steamer, sauté pan, yogurt maker, and warmer. 14 one-touch smart programs for hands-free cooking. Advanced microprocessor monitors pressure, temperature, and timing."
        },
        {
            "title": "Ninja Foodi 9-in-1 Deluxe XL Cooker",
            "description": "9-in-1 functionality: pressure cooker, air fryer, slow cooker, steamer, sauté pan, yogurt maker, warmer, and dehydrator. 6.5L capacity perfect for family meals. TenderCrisp technology delivers crispy results without deep frying."
        },
        
        # Fitness e Saúde
        {
            "title": "Fitbit Charge 5 Advanced Fitness Tracker",
            "description": "Advanced health metrics including heart rate variability, SpO2, and skin temperature. Built-in GPS for pace and distance without your phone. 7-day battery life. Stress management score and guided breathing sessions."
        },
        {
            "title": "Bowflex SelectTech 552 Adjustable Dumbbells",
            "description": "Adjust from 5 to 52.5 pounds in 2.5-pound increments up to the first 25 pounds. Space-efficient design replaces 15 sets of weights. Durable molded plates with metal handle. Easy-to-use selection dial."
        },
        
        # Livros e Mídia
        {
            "title": "The Seven Husbands of Evelyn Hugo by Taylor Jenkins Reid",
            "description": "Aging and reclusive Hollywood movie icon Evelyn Hugo is finally ready to tell the truth about her glamorous and scandalous life. When she chooses unknown magazine reporter Monique Grant for the job, no one is more astounded than Monique herself."
        },
        {
            "title": "Atomic Habits by James Clear",
            "description": "No matter your goals, Atomic Habits offers a proven framework for improving every day. James Clear, one of the world's leading experts on habit formation, reveals practical strategies that will teach you exactly how to form good habits."
        },
        
        # Jardinagem
        {
            "title": "Miracle-Gro Water Soluble All Purpose Plant Food",
            "description": "Instantly feeds to grow bigger, more beautiful plants versus unfed plants. Feed every 1-2 weeks. Use with the Miracle-Gro Garden Feeder or any watering can. Safe for all plants, guaranteed not to burn when used as directed."
        },
        {
            "title": "AeroGarden Harvest with Gourmet Herb Seed Pod Kit",
            "description": "Grow up to 6 plants, 5 inches tall, indoors year-round. Includes 6-pod seed kit with Genovese Basil, Curly Parsley, Dill, Thyme, Thai Basil, and Mint. LED grow lights automatically turn on and off. No soil, no mess, no weeds."
        }
    ]
    
    # Expandir dados com variações realistas
    expanded_data = []
    for item in realistic_data:
        expanded_data.append(item)
        
        # Criar variações realistas
        variations = [
            {
                "title": f"{item['title']} - Premium Edition",
                "description": f"{item['description']} Premium version with enhanced features and extended warranty."
            },
            {
                "title": f"{item['title']} - Budget Friendly",
                "description": f"Affordable version of {item['title'].split(' -')[0]}. {item['description']}"
            }
        ]
        expanded_data.extend(variations)
    
    console.log(f"[green]✅ Dados realistas criados: {len(expanded_data)} produtos[/green]")
    return expanded_data

def clean_and_prepare_data(data):
    """Limpa e prepara dados de forma mais inteligente."""
    console.log("[cyan]🧹 Limpando e preparando dados...[/cyan]")
    
    # Filtrar dados válidos
    valid_data = []
    for item in data:
        title = item.get('title', '').strip()
        description = item.get('description', '').strip()
        
        # Critérios mais rigorosos
        if (len(title) > 10 and len(title) < 200 and
            len(description) > 50 and len(description) < 1000 and
            not re.search(r'sample data|test purposes|version \d+', title.lower()) and
            not re.search(r'sample data|test purposes|version \d+', description.lower())):
            valid_data.append(item)
    
    console.log(f"[green]✅ Dados válidos: {len(valid_data)} de {len(data)}[/green]")
    return valid_data

def create_diverse_prompts(data):
    """Cria prompts mais diversos e realistas."""
    console.log("[cyan]📝 Criando prompts diversos...[/cyan]")
    
    prompts = []
    
    for item in data:
        title = item['title']
        description = item['description']
        
        # Diferentes formatos de prompt
        prompt_formats = [
            f"Product: {title}\nDescription: {description}",
            f"Title: {title}\nFeatures: {description}",
            f"Amazon Product: {title}\nDetails: {description}",
            f"Item: {title}\nSpecifications: {description}"
        ]
        
        prompts.extend(prompt_formats)
    
    console.log(f"[green]✅ Prompts criados: {len(prompts)}[/green]")
    return prompts

# ------------------- Funções Auxiliares -------------------

def check_existing_model(model_path="./fine_tuned_model"):
    """Verifica se já existe um modelo fine-tunado válido."""
    if os.path.exists(model_path):
        required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
        existing_files = os.listdir(model_path)
        if all(f in existing_files for f in required_files):
            return True
    return False

def evaluate_model(model, tokenizer, prompts, max_new_tokens=50):
    """Gera respostas do modelo para uma lista de prompts."""
    responses = []
    model.eval()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Gerando respostas...", total=len(prompts))
        
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=100)
            if model.device.type != "cpu":
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            responses.append(text)
            progress.update(task, advance=1)
    
    return responses

def compute_perplexity(model, tokenizer, eval_dataset, device_type):
    """Calcula perplexidade real."""
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    eval_batch_size = 4 if device_type != "CPU" else 1
    num_workers = 2 if device_type != "CPU" else 0
    
    args = TrainingArguments(
        output_dir="./tmp_eval",
        per_device_eval_batch_size=eval_batch_size,
        report_to=None,
        dataloader_pin_memory=(device_type != "CPU"),
        dataloader_num_workers=num_workers
    )
    trainer = Trainer(model=model, args=args, eval_dataset=eval_dataset, data_collator=data_collator)
    eval_results = trainer.evaluate()
    return math.exp(eval_results["eval_loss"])

def load_or_train_model(model_name, tokenizer, train_dataset, eval_dataset, config, device_type, force_retrain=False):
    """Carrega modelo existente e continua treinando, ou inicia novo se não existir."""
    
    model_path = "./fine_tuned_model"
    
    if check_existing_model(model_path) and not force_retrain:
        console.log(f"[green]Carregando modelo existente[/green] de {model_path} para treinamento incremental...")
        base_model_name = model_path
    else:
        console.log("[yellow]Iniciando do modelo base...[/yellow]")
        base_model_name = model_name
    
    model_kwargs = {
        "torch_dtype": torch.float16 if device_type != "CPU" else torch.float32,
        "device_map": "auto" if device_type != "CPU" else None,
        "low_cpu_mem_usage": device_type != "CPU"
    }
    
    model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    
    if device_type == "CPU":
        model = model.cpu()
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=model_path,
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        warmup_steps=config["warmup_steps"],
        weight_decay=config["weight_decay"],
        logging_steps=5,
        save_steps=20,
        save_strategy="steps",
        remove_unused_columns=False,
        report_to=None,
        fp16=config.get("fp16", False),
        dataloader_pin_memory=(device_type != "CPU"),
        dataloader_num_workers=config.get("dataloader_num_workers", 0),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        max_grad_norm=1.0,
        dataloader_prefetch_factor=2 if device_type != "CPU" else None,
        eval_steps=20,
        save_total_limit=3,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )

    console.log(f"[cyan]🚀 Iniciando fine-tuning inteligente para {device_type}...[/cyan]")
    trainer.train()

    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    return model, False

# ------------------- Função Principal -------------------

def main():
    console.rule("[bold blue]🚀 Fine-tuning Amazon Titles - Versão Melhorada e Realista[/bold blue]")

    # Configurar dispositivo
    device, device_type = setup_optimal_device()
    optimize_memory_settings(device_type)

    # Configurações otimizadas
    config = get_optimal_config(device_type)

    console.log(f"[cyan]📊 Configurações:[/cyan] Device={device_type}, "
                f"Batch Size={config['batch_size']}, "
                f"Learning Rate={config['learning_rate']}, "
                f"Epochs={config['num_epochs']}")

    # 1. Criar dados realistas
    realistic_data = create_realistic_amazon_data()
    
    # 2. Limpar e preparar dados
    clean_data = clean_and_prepare_data(realistic_data)
    
    # 3. Criar prompts diversos
    prompts = create_diverse_prompts(clean_data)

    # 4. Criar dataset tokenizado
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=False, max_length=config["max_length"])

    dataset = Dataset.from_dict({"text": prompts})
    dataset_split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset_split["train"].map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    eval_dataset = dataset_split["test"].map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    # 5. Teste antes do fine-tuning
    if not check_existing_model():
        console.log("[yellow]Avaliando modelo base...[/yellow]")
        model_kwargs = {
            "torch_dtype": torch.float16 if device_type != "CPU" else torch.float32,
            "device_map": "auto" if device_type != "CPU" else None,
            "low_cpu_mem_usage": device_type != "CPU"
        }
        
        base_model = AutoModelForCausalLM.from_pretrained(config["model_name"], **model_kwargs)
        if device_type == "CPU":
            base_model = base_model.cpu()
            
        base_perplexity = compute_perplexity(base_model, tokenizer, eval_dataset, device_type)
        base_responses = evaluate_model(base_model, tokenizer, prompts[:2])
    else:
        base_perplexity = None
        base_responses = None

    # 6. Fine-tuning inteligente
    fine_tuned_model, _ = load_or_train_model(
        config["model_name"], tokenizer, train_dataset, eval_dataset, config, device_type
    )

    # Limpeza de memória
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 7. Teste depois do fine-tuning
    tuned_perplexity = compute_perplexity(fine_tuned_model, tokenizer, eval_dataset, device_type)
    tuned_responses = evaluate_model(fine_tuned_model, tokenizer, prompts[:2])

    # 8. Tabela comparativa
    table = Table(title=f"📊 Comparativo de Resultados - {device_type}", show_header=True, header_style="bold magenta")
    table.add_column("Métrica", justify="left")
    table.add_column("Antes", style="red")
    table.add_column("Depois", style="green")
    table.add_column("Melhoria (%)", style="cyan")

    if base_perplexity:
        improvement = ((base_perplexity - tuned_perplexity) / base_perplexity) * 100
        table.add_row(
            "Perplexidade",
            f"{base_perplexity:.2f}",
            f"{tuned_perplexity:.2f}",
            f"{improvement:.1f}%"
        )
    else:
        table.add_row(
            "Perplexidade",
            "-",
            f"{tuned_perplexity:.2f}",
            "-"
        )

    console.print(table)

    # 9. Exibir exemplos
    console.rule("[bold green]Exemplos de Respostas - Dados Realistas[/bold green]")
    for before, after in zip(base_responses or tuned_responses, tuned_responses):
        if before:
            console.print(f"[red]Base:[/red] {before}")
        console.print(f"[green]Tuned:[/green] {after}")
        console.print("─" * 50)

    console.rule(f"[bold blue]✅ Fine-tuning inteligente concluído![/bold blue]")

if __name__ == "__main__":
    main()
