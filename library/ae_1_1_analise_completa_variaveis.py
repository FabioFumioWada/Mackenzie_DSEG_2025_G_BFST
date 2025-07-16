import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import library.dataset_library as ds_lib

print("===============================================================================")
print("Etapa 1: Análise Exploratória                                                 =")
print("===============================================================================")

print("===============================================================================")
print("Etapa 1.1: Análise Exploratória - Análise estatística completa das variáveis ==")
print("===============================================================================")


#Function responsável por gerar uma análise básica do dataset de funcionários.
def fn_gerar_analise_basica_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Análise Básica ===")
    if df is None or df.empty:
        print("Aviso: O DataFrame de entrada está vazio ou é None. Não é possível gerar análise básica.")
        return pd.DataFrame() # Retorna um DataFrame vazio

    # Criar um DataFrame a partir das informações de dtypes e non-null counts
    df_info = pd.DataFrame({
        'Column': df.columns,
        'Non-Null Count': df.count().values,
        'Dtype': df.dtypes.values
    })
    return df_info
#Function responsável por gerar uma análise estatística descritiva do dataset de funcionários.
def fn_gerar_analise_estatistica_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Análise Estatística Descritiva ===")
    if df is None or df.empty:
        print("Aviso: O DataFrame de entrada está vazio ou é None. Não é possível gerar análise estatística.")
        return pd.DataFrame() # Retorna um DataFrame vazio para evitar AttributeError
    return df.describe().T

#Function responsável por gerar uma análise completa do dataset de funcionários.
def fn_gerar_analise_completa_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Análise Completa ===")
    v_lista_resultados = []

    for v_coluna in df.columns:
        serie = df[v_coluna]
        df_info_coluna = {
            'Coluna': v_coluna,
            'Tipo_Campo': str(serie.dtype),
            
            'Total_Linhas': len(serie),
            'Nao_Nulos': serie.count(),
            'Percentual_Nao_Nulos': (serie.count() / len(serie)) * 100
        }

        if pd.api.types.is_numeric_dtype(serie):
            df_info_coluna['Minimo'] = serie.min()
            df_info_coluna['Maximo'] = serie.max()
            df_info_coluna['Media'] = serie.mean()
            df_info_coluna['Mediana'] = serie.median()
            df_info_coluna['Desvio_Padrao'] = serie.std()
            df_info_coluna['Num_Valores_Unicos'] = serie.nunique()
            df_info_coluna['Moda'] = serie.mode().tolist() if not serie.mode().empty else None
            # Para v_colunas numéricas, a contagem de valores pode ser muito extensa
            # então não incluímos ela diretamente na tabela sumarizada.
            #df_info_coluna['Contagem_Valores_Categoricos'] = None # Não aplicável/muito extenso
        elif pd.api.types.is_object_dtype(serie) or pd.api.types.is_string_dtype(serie) or pd.api.types.is_categorical_dtype(serie) or pd.api.types.is_bool_dtype(serie):
            df_info_coluna['Minimo'] = None # Não aplicável
            df_info_coluna['Maximo'] = None # Não aplicável
            df_info_coluna['Media'] = None # Não aplicável
            df_info_coluna['Mediana'] = None # Não aplicável
            df_info_coluna['Desvio_Padrao'] = None # Não aplicável
            df_info_coluna['Num_Valores_Unicos'] = serie.nunique()
            df_info_coluna['Moda'] = serie.mode().tolist() if not serie.mode().empty else None
            # Contagem de valores para campos categóricos
            df_info_coluna['Contagem_Valores_Categoricos'] = serie.value_counts().to_dict()
        else: # Para outros tipos como datetime
            df_info_coluna['Minimo'] = serie.min() if hasattr(serie, 'min') else None
            df_info_coluna['Maximo'] = serie.max() if hasattr(serie, 'max') else None
            df_info_coluna['Media'] = None
            df_info_coluna['Mediana'] = None
            df_info_coluna['Desvio_Padrao'] = None
            df_info_coluna['Num_Valores_Unicos'] = serie.nunique()
            df_info_coluna['Moda'] = serie.mode().tolist() if not serie.mode().empty else None
            df_info_coluna['Contagem_Valores_Categoricos'] = None

        v_lista_resultados.append(df_info_coluna)

    # Cria o DataFrame final a partir da lista de dicionários
    df_analise = pd.DataFrame(v_lista_resultados)

    # Reordena as v_colunas para melhor visualização
    v_ordem_colunas = [
        'Coluna', 'Tipo_Campo', 'Total_Linhas', 'Nao_Nulos', 'Percentual_Nao_Nulos',
        'Minimo', 'Maximo', 'Media', 'Mediana', 'Desvio_Padrao',
        'Num_Valores_Unicos', 'Moda', 'Contagem_Valores_Categoricos'
    ]

    # Garante que apenas v_colunas existentes e na ordem desejada sejam incluídas
    v_colunas_finais = [col for col in v_ordem_colunas if col in df_analise.columns]
    return df_analise[v_colunas_finais]

def fn_gerar_analise_attrition_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== DISTRIBUIÇÃO DA VARIÁVEL ALVO (Attrition) ===")

    if 'Attrition' not in df.columns:
        print("Erro: A coluna 'Attrition' não foi encontrada no DataFrame.")
        return pd.DataFrame() # Retorna um DataFrame vazio se a coluna não existir

    attrition_counts = df['Attrition'].value_counts()
    print(attrition_counts)

    # Calcula a taxa de Attrition
    total_employees = len(df)
    attrition_yes = attrition_counts.get('Yes', 0) # Usa .get para evitar KeyError se 'Yes' não existir
    
    if total_employees > 0:
        attrition_rate = (attrition_yes / total_employees) * 100
    else:
        attrition_rate = 0.0

    print(f"\nTaxa de Attrition: {attrition_rate:.2f}%")

    df_result = attrition_counts.reset_index()
    df_result.columns = ['Attrition_Value', 'Count']

    return df_result

#Valores Ausentes
def fn_gerar_analise_valores_ausentes(df: pd.DataFrame, v_local_gravacao: str) -> pd.DataFrame:
    print("\n=== Análise de Valores Ausentes ===")

    v_missing_values = df.isnull().sum()

    #v_output_filename = 'missing_values_report.txt'

    with open(v_local_gravacao, 'w') as f:
        f.write("=== ANÁLISE DE VALORES AUSENTES ===\n")
        if v_missing_values.sum() == 0:
            f.write("Não há valores ausentes no dataset!\n")
        else:
            f.write(v_missing_values[v_missing_values > 0].to_string() + "\n")

    print(f"Relatório de valores ausentes salvo em {v_local_gravacao}")


#Variáveis Numéricas
def fn_gerar_analise_variaveis_numericas(df: pd.DataFrame, v_local_gravacao: str) -> None: # Alterado para None, pois a função não retorna um DataFrame
    print("\n=== Análise de Variáveis Numéricas ===")

    v_numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    with open(v_local_gravacao, 'w') as f:
        f.write("=== ANÁLISE DE VARIÁVEIS NUMÉRICAS ===\n\n") # Título corrigido e linha extra
        if len(v_numeric_cols) == 0:
            f.write("Não há variáveis numéricas no dataset!\n")
        else:
            # --- CORREÇÃO AQUI ---
            # Converte a lista de colunas em uma única string, com cada nome separado por ", "
            string_de_colunas = ", ".join(v_numeric_cols)
            f.write("Colunas numéricas encontradas:\n")
            f.write(string_de_colunas)
            f.write("\n") # Adiciona uma nova linha no final para formatação

    print(f"Relatório de variáveis numéricas salvo em {v_local_gravacao}")

# Dados Faltantes
def fn_tratar_dados_faltantes(df: pd.DataFrame) -> pd.DataFrame:
    print("Iniciando análise de dados faltantes...")
    for v_col in df.columns:
        if df[v_col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[v_col]):
                v_mediana = df[v_col].median()
                df[v_col].fillna(v_mediana, inplace=True)
                print(f"Coluna numérica '{v_col}' preenchida com a mediana ({v_mediana}).")
            elif pd.api.types.is_object_dtype(df[v_col]) or pd.api.types.is_categorical_dtype(df[v_col]):
                v_moda = df[v_col].mode()[0]
                df[v_col].fillna(v_moda, inplace=True)
                print(f"Coluna categórica '{v_col}' preenchida com a moda ('{v_moda}').")
    print("Tratamento de dados faltantes concluído.")
    return df
#Outliers
def fn_tratar_outliers(df: pd.DataFrame) -> pd.DataFrame:
    print("Iniciando análise de outliers")
    v_numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    for v_col in v_numeric_cols:
        Q1 = df[v_col].quantile(0.25)
        Q3 = df[v_col].quantile(0.75)
        IQR = Q3 - Q1
        
        v_limite_inferior = Q1 - 1.5 * IQR
        v_limite_superior = Q3 + 1.5 * IQR
        
        v_outliers_count = ((df[v_col] < v_limite_inferior) | (df[v_col] > v_limite_superior)).sum()
        
        if v_outliers_count > 0:
            print(f"Encontrados {v_outliers_count} outliers na coluna '{v_col}'. Realizando o capping...")
            # Capping dos outliers
            df[v_col] = df[v_col].clip(lower=v_limite_inferior, upper=v_limite_superior)
        else:
            print(f"Nenhum outlier detectado na coluna '{v_col}'.")
            
    print("Tratamento de outliers finalizado.")
    return df