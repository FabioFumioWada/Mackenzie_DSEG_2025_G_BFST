# -*- coding: utf-8 -*-
"""
================================================================================
Pipeline Completo de Machine Learning para Previsão de Attrition - TechCorp Brasil
================================================================================

Este script é o orquestrador central do projeto e executa todas as etapas
em uma sequência lógica e reproduzível.

Fluxo de Execução:
1.  **Análise Exploratória Inicial (EDA):** Executa as funções dos scripts
    originais para gerar relatórios e visualizações iniciais, validando a
    compreensão básica dos dados.
2.  **Processamento e Limpeza de Dados:** Utiliza o novo módulo 'data_processing'
    para tratar valores faltantes e outliers.
3.  **Engenharia de Features:** Desenvolve novas variáveis para enriquecer o
    modelo, utilizando o módulo 'lib_news_feature'.
4.  **Modelagem Avançada:** Treina, otimiza com Optuna e avalia múltiplos
    algoritmos de ponta para selecionar o melhor modelo, usando o módulo
    'lib_modelagem_v2'.
5.  **Avaliação e Interpretabilidade:** Analisa o modelo campeão em
    profundidade, verificando métricas de negócio e gerando insights de
    interpretabilidade com SHAP, através do módulo 'lib_aval_interpret_v2'.
6.  **Salvamento do Artefato:** O modelo final treinado é salvo em um
    arquivo '.pkl' para que possa ser usado em produção.

Para executar o pipeline completo, basta rodar este arquivo.
"""

# --- Importação dos Módulos do Projeto ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl as pxl
import joblib
import warnings
import os
import datetime # Módulo para obter a data e hora

# Módulos customizados (pasta 'library')
import library.dataset_library as lib_ds #biblioteca resposnável pela geração do dataset sintético
import library.ae_1_1_analise_completa_variaveis as lib_analise_completa #biblioteca responsável pela análise completa do dataset
import library.ae_1_2_ident_padroes_correlacoes as lib_ident_padroes #biblioteca responsável pela identificação de padrões e correlações do dataset
import library.ae_1_3_visualizacoes_criativas_informativas as lib_visualizacoes #biblioteca responsável pelas visualizações criativas e informativas do dataset
import library.fe_2_1_novas_features as lib_news_feature # Feature Engineering
import library.fe_2_3_impacto_novas_features as lib_news_feature_impacto # Processamento de Dados
import library.mo_3_X_modelagem as lib_modelagem # Modelagem
import library.ai_4_X_avaliacao_interpretacao as lib_aval_interpret #Avaliação e interpretação


# Ignora avisos para uma saída mais limpa
warnings.filterwarnings('ignore')

def run_full_pipeline():
    """
    Função principal que executa o pipeline de ponta a ponta.
    """
    print("=========================================================")
    print("=== INICIANDO PIPELINE DE PREVISÃO DE ATTRITION v2.0 ===")
    print("=========================================================")
    # Adiciona o print com a data e hora atuais
    print(f"Início da execução: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # --- Configuração de Diretórios ---
    # Cria uma pasta 'results' para salvar os artefatos, se não existir
    results_path = 'results'
    os.makedirs(results_path, exist_ok=True)
    
    # --- ETAPA 0: GERAÇÃO DE DADOS ---
    print("\n[ETAPA 0/6] Gerando o dataset de funcionários...")
    # Adiciona o print com a data e hora atuais
    print(f"Início da execução: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    df_original = lib_ds.fn_cria_dataset()
    print(f"Dataset gerado com sucesso. Dimensões: {df_original.shape}")

    # --- ETAPA 1: ANÁLISE EXPLORATÓRIA INICIAL (Scripts Originais) ---
    print("\n[ETAPA 1/6] Executando Análise Exploratória de Dados (EDA)...")
    # Adiciona o print com a data e hora atuais
    print(f"Início da execução: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    path_eda = os.path.join(results_path, '1_analise_exploratoria')
    os.makedirs(path_eda, exist_ok=True)
    
    # Gerando relatório estatístico completo
    df_analise = lib_analise_completa.fn_gerar_analise_completa_dataframe(df_original.copy())
    df_analise.to_excel(os.path.join(path_eda, 'analise_completa_variaveis.xlsx'), index=False)
    print("  - Relatório estatístico completo salvo.")

    # Gerando gráficos iniciais
    lib_visualizacoes.fn_gerar_graficos_attrition_atual(df_original.copy(), path_eda)
    lib_visualizacoes.fn_gerar_matriz_correlacao(df_original.copy(), path_eda)
    print("  - Gráficos de análise exploratória salvos.")

    # --- ETAPA 2: PROCESSAMENTO E LIMPEZA DE DADOS ---
    print("\n[ETAPA 2/6] Processando e limpando os dados...")
    # Adiciona o print com a data e hora atuais
    print(f"Início da execução: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    df_processed = lib_analise_completa.fn_tratar_dados_faltantes(df_original.copy())
    df_processed = lib_analise_completa.fn_tratar_outliers(df_processed)
    print("Limpeza de dados concluída.")

    # --- ETAPA 3: ENGENHARIA DE FEATURES ---
    print("\n[ETAPA 3/6] Criando novas features para o modelo...")
    # Adiciona o print com a data e hora atuais
    print(f"Início da execução: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    df_featured = lib_news_feature.fn_criar_novas_features(df_processed.copy())
    print("Engenharia de features concluída.")

    # --- ETAPA 4: MODELAGEM AVANÇADA ---
    print("\n[ETAPA 4/6] Executando o pipeline de modelagem avançada...")
    # Adiciona o print com a data e hora atuais
    print(f"Início da execução: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    modelo_final, X_train, X_test, y_train, y_test = lib_modelagem.fn_executar_modelagem_avancada(df_featured.copy())
    print("Modelagem avançada concluída.")
    
    # --- ETAPA 5: AVALIAÇÃO E INTERPRETABILIDADE ---
    print("\n[ETAPA 5/6] Executando a avaliação aprofundada do modelo...")
    # Adiciona o print com a data e hora atuais
    print(f"Início da execução: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lib_aval_interpret.fn_run_lib_aval_interpret_pipeline_avancada(modelo_final, X_test, y_test, df_featured)
    print("Avaliação detalhada e análise SHAP concluídas.")

    # --- ETAPA 6: SALVAMENTO DO MODELO FINAL ---
    print("\n[ETAPA 6/6] Salvando o artefato do modelo final...")
    # Adiciona o print com a data e hora atuais
    print(f"Início da execução: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    model_filename = os.path.join(results_path, 'modelo_attrition_final_v2.pkl')
    joblib.dump(modelo_final, model_filename)
    print(f"Modelo final salvo com sucesso como '{model_filename}'")
    
    print("\n=========================================================")
    print("====== PIPELINE CONCLUÍDO COM SUCESSO =====")
    print("=========================================================")
    # Adiciona o print com a data e hora atuais
    print(f"Início da execução: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Este bloco garante que a função `run_full_pipeline` seja executada
# apenas quando o script `main.py` é rodado diretamente.
if __name__ == "__main__":
    run_full_pipeline()