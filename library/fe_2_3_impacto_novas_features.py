import pandas as pd

def fn_analisar_impacto_preliminar(df_com_features: pd.DataFrame):

    # Garante que Attrition seja numérica
    df_com_features['Attrition'] = df_com_features['Attrition'].map({'Yes': 1, 'No': 0})
    
    # Seleciona apenas novas colunas numéricas (exemplo)
    v_novas_features_num = [
        'IncomePerYearOfWork', 'TenureToAgeRatio', 'OverallSatisfactionScore',
        'WorkLifeImbalance', 'IsLongCommute', 'IsUnderpaid', 
        'JobRoleFrequency', 'Age^2', 'Age YearsAtCompany', 'YearsAtCompany^2'
    ]
    
    # Calcula a correlação
    v_correlacoes = df_com_features[v_novas_features_num + ['Attrition']].corr()
    
    print("--- Correlação das Novas Features com Attrition ---")
    print(v_correlacoes['Attrition'].sort_values(ascending=False))