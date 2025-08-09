# Importações otimizadas: Foco em eficiência, datasets para streaming
import json
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
from sklearn.model_selection import train_test_split
import re
from tqdm import tqdm
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AmazonFineTuning:
    """
    Classe para Fine-tuning de modelos de linguagem com dataset Amazon Titles
    """
    
    def __init__(self, model_name="microsoft/DialoGPT-medium", max_length=512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def load_data(self, file_path="trn.json"):
        """
        Carrega e prepara os dados do dataset Amazon Titles
        """
        logger.info("Carregando dataset Amazon Titles...")
        
        try:
            # Carregar dados JSON
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Converter para DataFrame
            df = pd.DataFrame(data)
            logger.info(f"Dataset carregado: {len(df)} registros")
            
            # Limpeza básica
            df = self._clean_data(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {e}")
            return None
    
    def _clean_data(self, df):
        """
        Limpa os dados: remove NAs, HTML, caracteres especiais
        """
        logger.info("Iniciando limpeza dos dados...")
        
        # Remover linhas com valores nulos
        initial_count = len(df)
        df = df.dropna()
        logger.info(f"Removidas {initial_count - len(df)} linhas com valores nulos")
        
        # Limpar HTML tags
        df['title'] = df['title'].apply(self._remove_html)
        df['description'] = df['description'].apply(self._remove_html)
        
        # Remover caracteres especiais excessivos
        df['title'] = df['title'].apply(self._clean_text)
        df['description'] = df['description'].apply(self._clean_text)
        
        # Filtrar por comprimento mínimo
        df = df[df['title'].str.len() > 10]
        df = df[df['description'].str.len() > 20]
        
        logger.info(f"Dataset limpo: {len(df)} registros")
        return df
    
    def _remove_html(self, text):
        """Remove tags HTML"""
        if pd.isna(text):
            return ""
        return re.sub(r'<[^>]+>', '', str(text))
    
    def _clean_text(self, text):
        """Limpa texto removendo caracteres especiais excessivos"""
        if pd.isna(text):
            return ""
        # Remove múltiplos espaços
        text = re.sub(r'\s+', ' ', str(text))
        # Remove caracteres especiais excessivos
        text = re.sub(r'[^\w\s\-.,!?]', '', text)
        return text.strip()
    
    def prepare_prompts(self, df, sample_size=None):
        """
        Prepara prompts para fine-tuning
        """
        logger.info("Preparando prompts para fine-tuning...")
        
        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
            logger.info(f"Usando amostra de {len(df)} registros")
        
        # Criar prompts no formato: "Title: {title}\nDescription: {description}"
        prompts = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Criando prompts"):
            prompt = f"Title: {row['title']}\nDescription: {row['description']}"
            prompts.append(prompt)
        
        return prompts
    
    def load_model_and_tokenizer(self):
        """
        Carrega modelo e tokenizer usando transformers padrão
        """
        logger.info(f"Carregando modelo: {self.model_name}")
        
        try:
            # Carregamento padrão do transformers
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16
            )
            
            # Configurar padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Mover modelo para GPU se disponível
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(device)
            
            # Ativar modo de avaliação
            self.model.eval()
            
            logger.info("Modelo e tokenizer carregados com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise
    
    def create_dataset(self, prompts):
        """
        Cria dataset para treinamento
        """
        logger.info("Criando dataset para treinamento...")
        
        # Tokenizar prompts
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
        
        # Criar dataset
        dataset_dict = {"text": prompts}
        dataset = Dataset.from_dict(dataset_dict)
        
        # Dividir em train/validation
        train_val = dataset.train_test_split(test_size=0.1, seed=42)
        
        # Tokenizar
        tokenized_train = train_val["train"].map(tokenize_function, batched=True)
        tokenized_val = train_val["test"].map(tokenize_function, batched=True)
        
        return tokenized_train, tokenized_val
    
    def train_model(self, train_dataset, val_dataset, output_dir="./fine_tuned_model"):
        """
        Executa fine-tuning do modelo
        """
        logger.info("Iniciando fine-tuning...")
        
        # Configurar argumentos de treinamento
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            load_best_model_at_end=True,
            save_total_limit=3,
            report_to=None,  # Desabilitar wandb
        )
        
        # Criar trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Executar treinamento
        logger.info("Iniciando treinamento...")
        self.trainer.train()
        
        # Salvar modelo
        self.trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Modelo salvo em: {output_dir}")
    
    def test_model(self, test_prompts, model_path="./fine_tuned_model"):
        """
        Testa o modelo fine-tunado
        """
        logger.info("Testando modelo fine-tunado...")
        
        # Carregar modelo treinado se especificado
        if model_path and model_path != "./fine_tuned_model":
            try:
                self.model = AutoModelForCausalLM.from_pretrained(model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = self.model.to(device)
                self.model.eval()
            except:
                logger.warning("Modelo treinado não encontrado, usando modelo base")
        
        results = []
        for prompt in test_prompts[:5]:  # Testar apenas 5 exemplos
            # Gerar resposta
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
            
            # Mover para o mesmo dispositivo do modelo
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append({
                "prompt": prompt,
                "generated": generated_text
            })
        
        return results

def main():
    """
    Função principal para executar o fine-tuning
    """
    logger.info("=== INICIANDO PROJETO DE FINE-TUNING AMAZON TITLES ===")
    
    # Inicializar classe
    fine_tuner = AmazonFineTuning()
    
    # 1. Carregar e preparar dados
    df = fine_tuner.load_data()
    if df is None:
        logger.error("Falha ao carregar dados. Encerrando.")
        return
    
    # 2. Preparar prompts (usar amostra para teste)
    prompts = fine_tuner.prepare_prompts(df, sample_size=100000)  # 100k registros
    
    # 3. Carregar modelo e tokenizer
    fine_tuner.load_model_and_tokenizer()
    
    # 4. Criar dataset
    train_dataset, val_dataset = fine_tuner.create_dataset(prompts)
    
    # 5. Executar fine-tuning
    fine_tuner.train_model(train_dataset, val_dataset)
    
    # 6. Testar modelo
    test_results = fine_tuner.test_model(prompts[:10])
    
    # 7. Salvar resultados
    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    logger.info("=== FINE-TUNING CONCLUÍDO ===")
    logger.info("Resultados salvos em: test_results.json")

if __name__ == "__main__":
    main() 