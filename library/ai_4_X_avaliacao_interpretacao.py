import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    average_precision_score
)

def fn_plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Attrition', 'Yes Attrition'], 
                yticklabels=['No Attrition', 'Yes Attrition'])
    plt.title(f'Matriz de Confusão - {model_name}', fontsize=16)
    plt.ylabel('Verdadeiro', fontsize=12)
    plt.xlabel('Predito', fontsize=12)
    plt.show()

def fn_interpretability_with_shap(model, X_test, model_type='tree'):
    """
    Gera gráficos de interpretabilidade usando a biblioteca SHAP.
    """
    print("\n--- Análise de Interpretabilidade com SHAP ---")
    
    # O explainer do SHAP precisa do modelo treinado e dos dados
    # O modelo real está dentro do pipeline
    classifier = model.named_steps['classifier']
    
    if model_type == 'tree':
        explainer = shap.TreeExplainer(classifier)
    else: # Para modelos lineares, etc.
        explainer = shap.KernelExplainer(classifier.predict_proba, X_test.sample(100)) # Usa uma amostra para performance
        
    shap_values = explainer.shap_values(X_test)
    
    # Para modelos de classificação, shap_values pode ser uma lista (um array por classe)
    # Vamos usar os valores para a classe positiva (Attrition = 1)
    shap_values_for_plot = shap_values[1] if isinstance(shap_values, list) else shap_values

    print("Gerando SHAP Summary Plot...")
    shap.summary_plot(shap_values_for_plot, X_test, plot_type="bar", max_display=20, show=False)
    plt.title("Importância das Features (Baseado em SHAP)")
    plt.tight_layout()
    plt.savefig('shap_summary_bar.png')
    plt.show()

    print("Gerando SHAP Beeswarm Plot...")
    shap.summary_plot(shap_values_for_plot, X_test, max_display=20, show=False)
    plt.title("Impacto das Features nas Predições (SHAP)")
    plt.tight_layout()
    plt.savefig('shap_summary_beeswarm.png')
    plt.show()


def fn_run_evaluation_pipeline_avancada(model, X_test, y_test, original_df):
    """
    Orquestra a execução de todas as análises de avaliação avançada.
    """
    print("\n======================================================")
    print("INICIANDO A FASE DE AVALIAÇÃO E INTERPRETAÇÃO AVANÇADA")
    print("======================================================")
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 1. Métricas Avançadas
    print("\n--- Relatório de Métricas Avançadas ---")
    print(classification_report(y_test, y_pred, target_names=['No Attrition', 'Yes Attrition']))
    
    pr_auc = average_precision_score(y_test, y_proba)
    f2_score = fbeta_score(y_test, y_pred, beta=2)
    mcc = matthews_corrcoef(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    print(f"Precision-Recall AUC: {pr_auc:.4f}  <-- Métrica chave para desbalanceamento")
    print(f"F2-Score (prioriza Recall): {f2_score:.4f}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")

    # 2. Matriz de Confusão
    fn_plot_confusion_matrix(y_test, y_pred, "Modelo Final")
    
    # 3. Análise de Viés e Fairness (mantida da versão anterior)
    # fn_fairness_analysis(model, X_test, y_test, original_df, 'Gender')
    
    # 4. Interpretabilidade com SHAP
    fn_interpretability_with_shap(model, X_test, model_type='tree')

    print("\n======================================================")
    print("AVALIAÇÃO AVANÇADA CONCLUÍDA")
    print("======================================================")
