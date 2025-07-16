import pandas as pd
import numpy as np
import library.dataset_library as ds_lib #biblioteca resposnável pela geração do dataset sintético

print("===============================================================================")
print("Etapa 1.2: Análise Exploratória - Identificação de Padrões e Corelações      ==")
print("===============================================================================")


def fn_ident_padroes_correlacoes(df: pd.DataFrame) -> pd.DataFrame:

    # Convertendo a feature 'Attrition' para numérica
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    df_encoded = pd.get_dummies(df, columns=['Department', 'Gender', 'MaritalStatus', 'Education', 'EducationField', 'JobRole', 'OverTime', 'BusinessTravel'], dtype=int)

    return df_encoded
    