# Script de Teste - Pipeline de Fine-tuning Amazon Titles
import json
import logging
from fine_tuning_project import AmazonFineTuning
from data_analysis import DataAnalyzer

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_data_analysis():
    """Testa a análise exploratória dos dados"""
    logger.info("=== TESTE: ANÁLISE EXPLORATÓRIA ===")
    
    try:
        # Criar analisador
        analyzer = DataAnalyzer()
        
        # Carregar dados (usar arquivo de exemplo se trn.json não existir)
        if not analyzer.load_data():
            logger.warning("Arquivo trn.json não encontrado. Criando dados de exemplo...")
            create_sample_data()
            analyzer = DataAnalyzer("sample_data.json")
            analyzer.load_data()
        
        # Executar análises
        analyzer.basic_statistics()
        analyzer.text_analysis()
        analyzer.data_quality_report()
        
        # Imprimir resumo
        analyzer.print_summary()
        
        logger.info("✅ Análise exploratória concluída com sucesso!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro na análise exploratória: {e}")
        return False

def test_fine_tuning():
    """Testa o pipeline de fine-tuning"""
    logger.info("=== TESTE: FINE-TUNING ===")
    
    try:
        # Inicializar fine-tuner
        fine_tuner = AmazonFineTuning()
        
        # Carregar dados
        df = fine_tuner.load_data()
        if df is None:
            logger.warning("Arquivo trn.json não encontrado. Criando dados de exemplo...")
            create_sample_data()
            fine_tuner = AmazonFineTuning()
            df = fine_tuner.load_data("sample_data.json")
        
        if df is None:
            logger.error("❌ Falha ao carregar dados")
            return False
        
        # Preparar prompts (usar amostra pequena para teste)
        prompts = fine_tuner.prepare_prompts(df, sample_size=1000)
        
        logger.info(f"✅ Preparados {len(prompts)} prompts para teste")
        
        # Carregar modelo (sem treinamento para teste)
        fine_tuner.load_model_and_tokenizer()
        
        logger.info("✅ Modelo carregado com sucesso")
        
        # Testar geração
        test_prompts = [
            "Title: Wireless Bluetooth Headphones\nDescription:",
            "Title: Organic Coffee Beans\nDescription:",
            "Title: Smartphone Case\nDescription:"
        ]
        
        results = fine_tuner.test_model(test_prompts, model_path=None)
        
        # Salvar resultados de teste
        with open("test_results_sample.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info("✅ Teste de fine-tuning concluído!")
        logger.info("📄 Resultados salvos em: test_results_sample.json")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro no fine-tuning: {e}")
        return False

def create_sample_data():
    """Cria dados de exemplo para teste"""
    logger.info("Criando dados de exemplo...")
    
    sample_data = [
        {
            "title": "Wireless Bluetooth Headphones",
            "description": "High-quality wireless headphones with noise cancellation and 30-hour battery life. Perfect for music lovers and professionals."
        },
        {
            "title": "Organic Coffee Beans",
            "description": "Premium organic coffee beans sourced from sustainable farms. Rich flavor with notes of chocolate and caramel."
        },
        {
            "title": "Smartphone Case",
            "description": "Durable protective case for smartphones with shock absorption and precise cutouts for all ports and buttons."
        },
        {
            "title": "Yoga Mat",
            "description": "Non-slip yoga mat made from eco-friendly materials. Perfect for yoga, pilates, and meditation practices."
        },
        {
            "title": "Kitchen Knife Set",
            "description": "Professional-grade kitchen knife set with stainless steel blades and ergonomic handles. Includes sharpener and storage block."
        },
        {
            "title": "LED Desk Lamp",
            "description": "Modern LED desk lamp with adjustable brightness and color temperature. USB charging port and touch controls included."
        },
        {
            "title": "Running Shoes",
            "description": "Lightweight running shoes with cushioned sole and breathable mesh upper. Designed for comfort and performance."
        },
        {
            "title": "Water Bottle",
            "description": "Insulated stainless steel water bottle that keeps drinks cold for 24 hours or hot for 12 hours. BPA-free and leak-proof."
        },
        {
            "title": "Laptop Stand",
            "description": "Adjustable aluminum laptop stand that improves ergonomics and airflow. Folds flat for easy storage and travel."
        },
        {
            "title": "Plant Pot",
            "description": "Ceramic plant pot with drainage holes and saucer. Perfect for indoor plants and home decoration."
        }
    ]
    
    # Criar mais dados para simular dataset maior
    extended_data = []
    for i in range(1000):  # 1000 registros
        base_item = sample_data[i % len(sample_data)]
        extended_data.append({
            "title": f"{base_item['title']} - Version {i+1}",
            "description": f"{base_item['description']} This is sample data #{i+1} for testing purposes."
        })
    
    # Salvar dados de exemplo
    with open("sample_data.json", "w", encoding="utf-8") as f:
        json.dump(extended_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Dados de exemplo criados: {len(extended_data)} registros")

def print_demo_results():
    """Imprime resultados de demonstração"""
    print("\n" + "="*60)
    print("🎯 DEMONSTRAÇÃO - FINE-TUNING AMAZON TITLES")
    print("="*60)
    
    print("\n📊 FLUXO DE TRABALHO:")
    print("1. ✅ Análise exploratória dos dados")
    print("2. ✅ Preparação e limpeza dos dados")
    print("3. ✅ Carregamento do modelo base")
    print("4. ✅ Criação de prompts para fine-tuning")
    print("5. ✅ Teste de geração (sem treinamento)")
    
    print("\n🔧 TECNOLOGIAS UTILIZADAS:")
    print("• Transformers - Framework de modelos")
    print("• Unsloth - Otimização de fine-tuning")
    print("• Pandas - Processamento de dados")
    print("• Matplotlib/Seaborn - Visualizações")
    
    print("\n📈 PRÓXIMOS PASSOS:")
    print("• Executar fine-tuning completo")
    print("• Comparar modelo antes/depois")
    print("• Otimizar hiperparâmetros")
    print("• Implementar RAG como comparação")
    
    print("\n🎯 CONCEITOS DEMONSTRADOS:")
    print("• Fine-tuning vs RAG")
    print("• Preparação de dados em larga escala")
    print("• Otimização de memória e performance")
    print("• Análise exploratória completa")
    
    print("\n" + "="*60)

def main():
    """Função principal de teste"""
    logger.info("🚀 INICIANDO TESTE DO PIPELINE DE FINE-TUNING")
    
    # Teste 1: Análise exploratória
    analysis_success = test_data_analysis()
    
    # Teste 2: Fine-tuning (sem treinamento)
    training_success = test_fine_tuning()
    
    # Resultados
    print_demo_results()
    
    if analysis_success and training_success:
        logger.info("🎉 TODOS OS TESTES PASSARAM!")
        print("\n✅ Pipeline funcionando corretamente!")
        print("📁 Arquivos gerados:")
        print("   • sample_data.json (dados de exemplo)")
        print("   • test_results_sample.json (resultados)")
        print("   • data_analysis_report.json (análise)")
    else:
        logger.warning("⚠️ Alguns testes falharam. Verificar logs.")
    
    logger.info("🏁 TESTE CONCLUÍDO")

if __name__ == "__main__":
    main() 