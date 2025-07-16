import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc,
    f1_score
)

def fn_plot_confusion_matrix(y_true, y_pred, model_name):
    """
    Plota uma Matriz de Confusão estilizada.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Attrition', 'Yes Attrition'], 
                yticklabels=['No Attrition', 'Yes Attrition'])
    plt.title(f'Matriz de Confusão - {model_name}', fontsize=16)
    plt.ylabel('Verdadeiro', fontsize=12)
    plt.xlabel('Predito', fontsize=12)
    plt.show()

def fn_detailed_error_analysis(model, X_test, y_test, original_df):
    """
    Identifica e retorna um DataFrame com os erros de classificação (Falsos Positivos e Falsos Negativos).
    """
    print("\n--- Análise Detalhada de Erros ---")
    predictions = model.predict(X_test)
    
    # Alinha os índices do DataFrame original com os dados de teste
    test_indices = X_test.index
    original_test_df = original_df.loc[test_indices].copy()
    
    original_test_df['prediction'] = predictions
    original_test_df['true_attrition'] = y_test
    
    # Filtra apenas os erros
    errors_df = original_test_df[original_test_df['prediction'] != original_test_df['true_attrition']]
    
    fp_count = len(errors_df[(errors_df['true_attrition'] == 0) & (errors_df['prediction'] == 1)])
    fn_count = len(errors_df[(errors_df['true_attrition'] == 1) & (errors_df['prediction'] == 0)])
    
    print(f"Total de erros no conjunto de teste: {len(errors_df)}")
    print(f"Falsos Positivos (FP): {fp_count}")
    print(f"Falsos Negativos (FN): {fn_count}  <-- Erro mais crítico para o negócio!")
    
    return errors_df

def fn_fairness_analysis(model, X_test, y_test, original_df, sensitive_feature):
    """
    Analisa o desempenho do modelo em subgrupos de uma feature sensível (ex: Gênero).
    """
    print(f"\n--- Análise de Viés e Fairness para a feature '{sensitive_feature}' ---")
    
    test_indices = X_test.index
    original_test_df = original_df.loc[test_indices].copy()
    
    unique_groups = original_test_df[sensitive_feature].unique()
    
    for group in unique_groups:
        print(f"\nAnalisando grupo: {group}")
        
        # Filtra os dados para o subgrupo específico
        group_indices = original_test_df[original_test_df[sensitive_feature] == group].index
        X_test_group = X_test.loc[group_indices]
        y_test_group = y_test.loc[group_indices]
        
        if len(y_test_group) == 0:
            print("Nenhuma amostra encontrada para este grupo no conjunto de teste.")
            continue
            
        # Faz predições para o subgrupo
        y_pred_group = model.predict(X_test_group)
        
        print(f"Relatório de Classificação para '{group}':")
        print(classification_report(y_test_group, y_pred_group, target_names=['No Attrition', 'Yes Attrition']))

def fn_find_optimal_threshold(model, X_test, y_test):
    """
    Encontra e recomenda um threshold ótimo baseado na curva Precision-Recall.
    """
    print("\n--- Análise de Threshold Ótimo ---")
    
    # Obtém as probabilidades da classe positiva (Attrition = 1)
    y_probas = model.predict_proba(X_test)[:, 1]
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_probas)
    
    # Plota a curva Precision-Recall
    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Encontra o threshold que maximiza o F1-Score
    f1_scores = (2 * recall * precision) / (recall + precision)
    # Ignora o último valor que pode ser NaN
    f1_scores = f1_scores[:-1]
    thresholds = thresholds[:len(f1_scores)]

    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx]
    best_f1_score = f1_scores[best_f1_idx]
    
    print(f"Threshold padrão: 0.5")
    print(f"Threshold que maximiza o F1-Score: {best_threshold:.4f} (F1-Score: {best_f1_score:.4f})")
    print("Recomendação: Usar este threshold para balancear Precision e Recall.")
    
    return best_threshold

def fn_run_evaluation_pipeline(model, X_test, y_test, original_df, X_train, y_train):
    """
    Orquestra a execução de todas as análises de avaliação.
    """
    print("\n======================================================")
    print("INICIANDO A FASE DE AVALIAÇÃO E INTERPRETAÇÃO")
    print("======================================================")
    
    # 1. Métricas e Matriz de Confusão
    y_pred = model.predict(X_test)
    print("\n--- Relatório de Classificação Final (com threshold padrão 0.5) ---")
    print(classification_report(y_test, y_pred, target_names=['No Attrition', 'Yes Attrition']))
    fn_plot_confusion_matrix(y_test, y_pred, model.__class__.__name__)
    
    # 2. Análise de Erro Detalhada
    errors_df = fn_detailed_error_analysis(model, X_test, y_test, original_df)
    # print("\nPrimeiros 5 erros (Falsos Positivos ou Negativos):")
    # print(errors_df.head())
    
    # 3. Análise de Viés e Fairness
    # Analisando para as features 'Gender' e 'AgeGroup'
    fn_fairness_analysis(model, X_test, y_test, original_df, 'Gender')
    fn_fairness_analysis(model, X_test, y_test, original_df, 'AgeGroup')
    
    # 4. Recomendação de Threshold Ótimo
    optimal_threshold = fn_find_optimal_threshold(model, X_test, y_test)
    
    # Avaliação com o novo threshold
    y_pred_optimal = (model.predict_proba(X_test)[:, 1] >= optimal_threshold).astype(int)
    print(f"\n--- Relatório de Classificação com Threshold Ótimo ({optimal_threshold:.2f}) ---")
    print(classification_report(y_test, y_pred_optimal, target_names=['No Attrition', 'Yes Attrition']))
    fn_plot_confusion_matrix(y_test, y_pred_optimal, f"{model.__class__.__name__} (Optimal Threshold)")

    print("\n======================================================")
    print("AVALIAÇÃO E INTERPRETAÇÃO CONCLUÍDAS")
    print("======================================================")