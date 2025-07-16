import pandas as pd
import numpy as np
import warnings

# Pré-processamento e Modelagem
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier

# Modelos Avançados
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.linear_model import LogisticRegression

# Otimização com Optuna
import optuna

# Métricas
from sklearn.metrics import recall_score, f1_score

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

def optimize_model(X_train, y_train, model_pipeline, param_space_func, n_trials=30):
    """
    Função genérica para otimização de hiperparâmetros usando Optuna.
    """
    def objective(trial):
        params = param_space_func(trial)
        model_pipeline.set_params(**params)
        
        # Validação cruzada para avaliação
        cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in cv_strategy.split(X_train, y_train):
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model_pipeline.fit(X_train_fold, y_train_fold)
            preds = model_pipeline.predict(X_val_fold)
            scores.append(f1_score(y_val_fold, preds, average='weighted'))
            
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Melhor F1-Score (CV): {study.best_value:.4f}")
    print(f"Melhores parâmetros: {study.best_params}")
    return study.best_params

def fn_executar_modelagem_avancada(df: pd.DataFrame):
    """
    Executa um pipeline de modelagem avançado com algoritmos de ponta e otimização com Optuna.
    """
    print("\n======================================================")
    print("INICIANDO A FASE DE MODELAGEM AVANÇADA")
    print("======================================================")
    
    # 1. Preparação dos Dados
    print("\n--- Etapa 1: Preparação dos Dados ---")
    target = 'Attrition'
    y = df[target].map({'Yes': 1, 'No': 0})
    features = df.drop(columns=[target])
    X = pd.get_dummies(features, drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Dataset dividido. Treino: {X_train.shape[0]} amostras, Teste: {X_test.shape[0]} amostras.")

    # 2. Definição dos Modelos e Espaços de Parâmetros para Optuna
    
    # LightGBM
    def lgb_param_space(trial):
        return {
            'classifier__n_estimators': trial.suggest_int('classifier__n_estimators', 100, 1000),
            'classifier__learning_rate': trial.suggest_float('classifier__learning_rate', 0.01, 0.3),
            'classifier__num_leaves': trial.suggest_int('classifier__num_leaves', 20, 300),
        }
    pipe_lgb = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', lgb.LGBMClassifier(random_state=42))
    ])

    # BalancedRandomForest
    def brf_param_space(trial):
        return {
            'classifier__n_estimators': trial.suggest_int('classifier__n_estimators', 100, 500),
            'classifier__max_depth': trial.suggest_int('classifier__max_depth', 10, 50),
            'classifier__min_samples_leaf': trial.suggest_int('classifier__min_samples_leaf', 1, 10),
        }
    pipe_brf = ImbPipeline([
        ('scaler', StandardScaler()),
        # Não precisa de SMOTE, o modelo já é balanceado
        ('classifier', BalancedRandomForestClassifier(random_state=42))
    ])
    
    # 3. Otimização e Seleção do Melhor Modelo
    print("\n--- Etapa 2: Otimização de Hiperparâmetros com Optuna ---")
    
    print("\nOtimizando LightGBM...")
    best_params_lgb = optimize_model(X_train, y_train, pipe_lgb, lgb_param_space, n_trials=25)
    
    print("\nOtimizando BalancedRandomForest...")
    best_params_brf = optimize_model(X_train, y_train, pipe_brf, brf_param_space, n_trials=25)

    # 4. Treinamento do Modelo Final
    print("\n--- Etapa 3: Treinamento do Modelo Final ---")
    
    # Vamos eleger o BalancedRandomForest como nosso campeão (geralmente robusto)
    # e treiná-lo com os melhores parâmetros encontrados.
    final_model = ImbPipeline([
        ('scaler', StandardScaler()),
        ('classifier', BalancedRandomForestClassifier(random_state=42, **best_params_brf))
    ])
    
    final_model.fit(X_train, y_train)
    print("Modelo final (BalancedRandomForest) treinado com sucesso.")
    
    print("\n======================================================")
    print("MODELAGEM AVANÇADA CONCLUÍDA")
    print("======================================================")
    
    return final_model, X_train, X_test, y_train, y_test