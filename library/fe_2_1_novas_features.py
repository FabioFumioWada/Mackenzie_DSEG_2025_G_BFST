import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

def fn_criar_novas_features(df: pd.DataFrame) -> pd.DataFrame:
    
    # 1. Renda por Ano de Trabalho
    df['IncomePerYearOfWork'] = df['MonthlyIncome'] / (df['YearsAtCompany'] + 1)

    # 2. Proporção de Tempo de Casa pela Idade
    df['TenureToAgeRatio'] = df['YearsAtCompany'] / df['Age']

    # 3. Score Geral de Satisfação
    df['OverallSatisfactionScore'] = (df['JobSatisfaction'] + df['EnvironmentSatisfaction'] + df['RelationshipSatisfaction']) / 3

    # 4. Desequilíbrio Vida-Trabalho
    df['WorkLifeImbalance'] = ((df['OverTime'] == 'Yes') & (df['WorkLifeBalance'] == 1)).astype(int)

    # 5. Idade
    df['AgeGroup'] = pd.cut(df['Age'], bins=[17, 30, 45, 65], labels=['Young_Adult', 'Mid_Career', 'Senior'])

    # 6. Faixa Salarial
    df['IncomeBracket'] = pd.qcut(df['MonthlyIncome'], q=3, labels=['Low', 'Medium', 'High'])
    
    # 7. Tempo de Casa
    df['TenureGroup'] = pd.cut(df['YearsAtCompany'], bins=[-1, 2, 7, 41], labels=['Newcomer', 'Experienced', 'Veteran'])

    # 8. Deslocamento
    df['IsLongCommute'] = (df['DistanceFromHome'] > df['DistanceFromHome'].median()).astype(int)

    # 9.Remuneração abaixo da média do JobLevel
    avg_income_by_level = df.groupby('JobLevel')['MonthlyIncome'].transform('mean')
    df['IsUnderpaid'] = (df['MonthlyIncome'] < avg_income_by_level).astype(int)
    
    # 10. Tempo no Cargo
    df['JobRoleFrequency'] = df.groupby('JobRole')['JobRole'].transform('count')
    
    # 11. Departamento e Cargo
    df['Dept_JobLevel_Interaction'] = df['Department'].astype(str) + '_' + df['JobLevel'].astype(str)

    # 12. Features Polinomiais (de grau 2)
    v_poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    v_poly_features = v_poly.fit_transform(df[['Age', 'YearsAtCompany']])
    
    # Cria um DataFrame com os nomes das novas colunas
    df_poly = pd.DataFrame(v_poly_features, columns=v_poly.get_feature_names_out(['Age', 'YearsAtCompany']))
    
    # Concatena as novas features no dataframe principal
    df = pd.concat([df.reset_index(drop=True), df_poly.reset_index(drop=True)], axis=1)

    print(f"{len(df.columns) - 35} novas features criadas com sucesso.") 
    return df
