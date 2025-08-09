# Análise Exploratória dos Dados - Amazon Titles Dataset
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from wordcloud import WordCloud
import logging
from config import DATA_CONFIG, CLEANING_CONFIG

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAnalyzer:
    """
    Classe para análise exploratória dos dados Amazon Titles
    """
    
    def __init__(self, file_path="trn.json"):
        self.file_path = file_path
        self.df = None
        self.analysis_results = {}
    
    def load_data(self):
        """Carrega os dados do arquivo JSON"""
        logger.info(f"Carregando dados de: {self.file_path}")
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.df = pd.DataFrame(data)
            logger.info(f"Dados carregados: {len(self.df)} registros")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {e}")
            return False
    
    def basic_statistics(self):
        """Estatísticas básicas do dataset"""
        logger.info("Calculando estatísticas básicas...")
        
        stats = {
            "total_records": len(self.df),
            "columns": list(self.df.columns),
            "missing_values": self.df.isnull().sum().to_dict(),
            "data_types": self.df.dtypes.to_dict()
        }
        
        # Estatísticas de texto
        if 'title' in self.df.columns:
            stats["title_stats"] = {
                "mean_length": self.df['title'].str.len().mean(),
                "median_length": self.df['title'].str.len().median(),
                "min_length": self.df['title'].str.len().min(),
                "max_length": self.df['title'].str.len().max(),
                "unique_titles": self.df['title'].nunique()
            }
        
        if 'description' in self.df.columns:
            stats["description_stats"] = {
                "mean_length": self.df['description'].str.len().mean(),
                "median_length": self.df['description'].str.len().median(),
                "min_length": self.df['description'].str.len().min(),
                "max_length": self.df['description'].str.len().max(),
                "unique_descriptions": self.df['description'].nunique()
            }
        
        self.analysis_results["basic_stats"] = stats
        return stats
    
    def text_analysis(self):
        """Análise detalhada dos textos"""
        logger.info("Realizando análise de texto...")
        
        text_analysis = {}
        
        # Análise de títulos
        if 'title' in self.df.columns:
            title_analysis = self._analyze_text_column(self.df['title'], "title")
            text_analysis["title"] = title_analysis
        
        # Análise de descrições
        if 'description' in self.df.columns:
            desc_analysis = self._analyze_text_column(self.df['description'], "description")
            text_analysis["description"] = desc_analysis
        
        self.analysis_results["text_analysis"] = text_analysis
        return text_analysis
    
    def _analyze_text_column(self, series, column_name):
        """Análise detalhada de uma coluna de texto"""
        analysis = {
            "length_distribution": {
                "mean": series.str.len().mean(),
                "median": series.str.len().median(),
                "std": series.str.len().std(),
                "percentiles": series.str.len().quantile([0.25, 0.5, 0.75, 0.9, 0.95]).to_dict()
            },
            "word_count": {
                "mean": series.str.split().str.len().mean(),
                "median": series.str.split().str.len().median(),
                "max": series.str.split().str.len().max()
            },
            "unique_words": len(set(' '.join(series.dropna()).split())),
            "empty_texts": series.isna().sum() + (series == "").sum()
        }
        
        return analysis
    
    def create_visualizations(self):
        """Cria visualizações dos dados"""
        logger.info("Criando visualizações...")
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Distribuição de comprimento dos títulos
        if 'title' in self.df.columns:
            title_lengths = self.df['title'].str.len()
            axes[0, 0].hist(title_lengths, bins=50, alpha=0.7, color='skyblue')
            axes[0, 0].set_title('Distribuição do Comprimento dos Títulos')
            axes[0, 0].set_xlabel('Comprimento')
            axes[0, 0].set_ylabel('Frequência')
        
        # 2. Distribuição de comprimento das descrições
        if 'description' in self.df.columns:
            desc_lengths = self.df['description'].str.len()
            axes[0, 1].hist(desc_lengths, bins=50, alpha=0.7, color='lightcoral')
            axes[0, 1].set_title('Distribuição do Comprimento das Descrições')
            axes[0, 1].set_xlabel('Comprimento')
            axes[0, 1].set_ylabel('Frequência')
        
        # 3. Palavras mais frequentes nos títulos
        if 'title' in self.df.columns:
            title_words = ' '.join(self.df['title'].dropna()).lower()
            word_freq = Counter(title_words.split())
            top_words = dict(word_freq.most_common(10))
            
            axes[1, 0].bar(top_words.keys(), top_words.values(), color='lightgreen')
            axes[1, 0].set_title('Top 10 Palavras nos Títulos')
            axes[1, 0].set_xlabel('Palavras')
            axes[1, 0].set_ylabel('Frequência')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Relação entre comprimento de título e descrição
        if 'title' in self.df.columns and 'description' in self.df.columns:
            axes[1, 1].scatter(
                self.df['title'].str.len(), 
                self.df['description'].str.len(), 
                alpha=0.5, 
                color='purple'
            )
            axes[1, 1].set_title('Relação: Comprimento Título vs Descrição')
            axes[1, 1].set_xlabel('Comprimento do Título')
            axes[1, 1].set_ylabel('Comprimento da Descrição')
        
        plt.tight_layout()
        plt.savefig('data_analysis_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Visualizações salvas em: data_analysis_plots.png")
    
    def generate_wordcloud(self):
        """Gera wordcloud dos textos"""
        logger.info("Gerando wordcloud...")
        
        # Wordcloud dos títulos
        if 'title' in self.df.columns:
            title_text = ' '.join(self.df['title'].dropna())
            wordcloud_title = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                max_words=100
            ).generate(title_text)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud_title, interpolation='bilinear')
            plt.axis('off')
            plt.title('Wordcloud - Títulos')
            plt.savefig('wordcloud_titles.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Wordcloud das descrições
        if 'description' in self.df.columns:
            desc_text = ' '.join(self.df['description'].dropna())
            wordcloud_desc = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                max_words=100
            ).generate(desc_text)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud_desc, interpolation='bilinear')
            plt.axis('off')
            plt.title('Wordcloud - Descrições')
            plt.savefig('wordcloud_descriptions.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        logger.info("Wordclouds salvos")
    
    def data_quality_report(self):
        """Gera relatório de qualidade dos dados"""
        logger.info("Gerando relatório de qualidade...")
        
        quality_report = {
            "missing_data": self.df.isnull().sum().to_dict(),
            "duplicate_records": self.df.duplicated().sum(),
            "empty_texts": {
                "title": (self.df['title'] == "").sum() if 'title' in self.df.columns else 0,
                "description": (self.df['description'] == "").sum() if 'description' in self.df.columns else 0
            },
            "text_length_issues": {
                "very_short_titles": (self.df['title'].str.len() < 5).sum() if 'title' in self.df.columns else 0,
                "very_long_titles": (self.df['title'].str.len() > 200).sum() if 'title' in self.df.columns else 0,
                "very_short_descriptions": (self.df['description'].str.len() < 10).sum() if 'description' in self.df.columns else 0,
                "very_long_descriptions": (self.df['description'].str.len() > 1000).sum() if 'description' in self.df.columns else 0
            }
        }
        
        self.analysis_results["quality_report"] = quality_report
        return quality_report
    
    def save_analysis_report(self, filename="data_analysis_report.json"):
        """Salva relatório completo da análise"""
        logger.info(f"Salvando relatório em: {filename}")
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
        
        logger.info("Relatório salvo com sucesso")
    
    def print_summary(self):
        """Imprime resumo da análise"""
        print("\n" + "="*50)
        print("RELATÓRIO DE ANÁLISE - AMAZON TITLES DATASET")
        print("="*50)
        
        if "basic_stats" in self.analysis_results:
            stats = self.analysis_results["basic_stats"]
            print(f"\n📊 ESTATÍSTICAS BÁSICAS:")
            print(f"   Total de registros: {stats['total_records']:,}")
            print(f"   Colunas: {', '.join(stats['columns'])}")
            
            if 'title' in stats:
                title_stats = stats['title_stats']
                print(f"\n📝 ANÁLISE DOS TÍTULOS:")
                print(f"   Comprimento médio: {title_stats['mean_length']:.1f} caracteres")
                print(f"   Comprimento mediano: {title_stats['median_length']:.1f} caracteres")
                print(f"   Títulos únicos: {title_stats['unique_titles']:,}")
            
            if 'description' in stats:
                desc_stats = stats['description_stats']
                print(f"\n📄 ANÁLISE DAS DESCRIÇÕES:")
                print(f"   Comprimento médio: {desc_stats['mean_length']:.1f} caracteres")
                print(f"   Comprimento mediano: {desc_stats['median_length']:.1f} caracteres")
                print(f"   Descrições únicas: {desc_stats['unique_descriptions']:,}")
        
        if "quality_report" in self.analysis_results:
            quality = self.analysis_results["quality_report"]
            print(f"\n🔍 QUALIDADE DOS DADOS:")
            print(f"   Registros duplicados: {quality['duplicate_records']:,}")
            print(f"   Títulos vazios: {quality['empty_texts']['title']:,}")
            print(f"   Descrições vazias: {quality['empty_texts']['description']:,}")
        
        print("\n" + "="*50)

def main():
    """Função principal para executar análise"""
    logger.info("=== INICIANDO ANÁLISE EXPLORATÓRIA ===")
    
    # Inicializar analisador
    analyzer = DataAnalyzer()
    
    # Carregar dados
    if not analyzer.load_data():
        logger.error("Falha ao carregar dados. Encerrando.")
        return
    
    # Executar análises
    analyzer.basic_statistics()
    analyzer.text_analysis()
    analyzer.data_quality_report()
    
    # Criar visualizações
    try:
        analyzer.create_visualizations()
        analyzer.generate_wordcloud()
    except Exception as e:
        logger.warning(f"Erro ao criar visualizações: {e}")
    
    # Salvar relatório
    analyzer.save_analysis_report()
    
    # Imprimir resumo
    analyzer.print_summary()
    
    logger.info("=== ANÁLISE CONCLUÍDA ===")

if __name__ == "__main__":
    main() 