#old"

import pandas as pd
import numpy as np

# Pré-processamento e Modelagem
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
import xgboost as xgb

# Métricas
from sklearn.metrics import classification_report, roc_auc_score, f1_score, recall_score, precision_score

# Função principal para executar a modelagem
def fn_executar_modelagem(df: pd.DataFrame):
    print("Iniciando a análise de Modelagem...")
    
    # --- 1. Preparação dos Dados ---
    # Remover colunas não numéricas que não serão usadas diretamente no modelo
    df_model = df.select_dtypes(include=np.number).copy()
    df_model.drop(columns=['Attrition'], errors='ignore', inplace=True) # Garante que a feature original não está em X
    
    X = df_model
    y = df['Attrition'].map({'Yes': 1, 'No': 0})

    # Divisão Estratificada em Treino e Teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Tamanho do conjunto de treino: {X_train.shape}")
    print(f"Tamanho do conjunto de teste: {X_test.shape}")

    # --- 2. Implementação dos Algoritmos com Pipeline e SMOTE ---
    
    # Define a estratégia de validação cruzada
    v_cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Criação dos pipelines para cada modelo
    # Cada pipeline irá: 1. Padronizar os dados, 2. Aplicar SMOTE, 3. Treinar o classificador
    
    # Pipeline para Regressão Logística
    v_pipe_lr = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', LogisticRegression(random_state=42, solver='liblinear'))
    ])

    # Pipeline para Random Forest
    v_pipe_rf = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Pipeline para XGBoost
    pipe_xgb = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
    ])

    # Pipeline para SVC
    pipe_svc = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', SVC(probability=True, random_state=42)) # probability=True é necessário para ROC-AUC e Voting
    ])

    models = {
        "Logistic Regression": v_pipe_lr,
        "Random Forest": v_pipe_rf,
        "XGBoost": pipe_xgb,
        "SVC": pipe_svc
    }

    v_results = {}
    # Avaliação inicial com validação cruzada
    print("\n--- Avaliação Inicial dos Modelos (com Validação Cruzada) ---")
    for name, model in models.items():
        # Usamos 'recall' como a principal métrica para avaliar
        scores = cross_val_score(model, X_train, y_train, cv=v_cv_strategy, scoring='recall_weighted', n_jobs=-1)
        v_results[name] = scores
        print(f"{name}: Recall Médio = {np.mean(scores):.4f} (DP = {np.std(scores):.4f})")

    # --- 3. Otimização de Hiperparâmetros (Exemplo com Random Forest) ---
    print("\n--- Otimização de Hiperparâmetros para Random Forest ---")

    param_dist_rf = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [10, 20, 30, None],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__max_features': ['sqrt', 'log2']
    }

    random_search = RandomizedSearchCV(
        estimator=v_pipe_rf,
        param_distributions=param_dist_rf,
        n_iter=20,  # Número de combinações a testar
        cv=v_cv_strategy,
        scoring='recall_weighted',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    random_search.fit(X_train, y_train)
    print(f"Melhores parâmetros encontrados: {random_search.best_params_}")
    print(f"Melhor Recall (CV): {random_search.best_score_:.4f}")

    # Melhor modelo encontrado pela otimização
    v_best_rf = random_search.best_estimator_

    # --- 4. Análise de Ensemble Methods (Voting Classifier) ---
    print("\n--- Análise de Ensemble com VotingClassifier ---")
    
    # Criamos um ensemble com os melhores modelos
    # Usaremos o Logistic Regression (rápido) e o Random Forest otimizado
    v_voting_clf = VotingClassifier(
        estimators=[
            ('lr', v_pipe_lr), # Usando o pipeline original da Regressão Logística
            ('rf', v_best_rf)  # Usando o melhor Random Forest encontrado
        ],
        voting='soft' # 'soft' usa as probabilidades e geralmente é melhor
    )

    # Avalia o VotingClassifier com validação cruzada
    voting_scores = cross_val_score(v_voting_clf, X_train, y_train, cv=v_cv_strategy, scoring='recall_weighted', n_jobs=-1)
    print(f"Voting Classifier: Recall Médio = {np.mean(voting_scores):.4f} (DP = {np.std(voting_scores):.4f})")

    # --- 5. Avaliação Final no Conjunto de Teste ---
    print("\n--- Avaliação Final do Melhor Modelo no Conjunto de Teste ---")
    
    # Treinando o melhor modelo (Random Forest otimizado) com todos os dados de treino
    v_best_rf.fit(X_train, y_train)
    y_pred_test = v_best_rf.predict(X_test)
    y_proba_test = v_best_rf.predict_proba(X_test)[:, 1]

    print("Relatório de Classificação no Teste:")
    print(classification_report(y_test, y_pred_test))
    
    print(f"ROC-AUC Score no Teste: {roc_auc_score(y_test, y_proba_test):.4f}")
    
    return v_best_rf, v_results

# Para executar este script, você precisaria de um 'main.py' que:
# 1. Gera o dataset (ds_lib.fn_cria_dataset)
# 2. Faz a engenharia de features (feature_engineering.fn_criar_features)
# 3. Passa o dataframe resultante para fn_executar_modelagem(df)