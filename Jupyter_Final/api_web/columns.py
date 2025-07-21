import pandas as pd
#Carregue o seu dataframe com todas as features, antes da divisão de treino/teste
df_featured = ... 
X = pd.get_dummies(df_featured.drop(columns=['Attrition']))
with open('model_columns.txt', 'w') as f:
    for col in X.columns:
        f.write(col + '\n')