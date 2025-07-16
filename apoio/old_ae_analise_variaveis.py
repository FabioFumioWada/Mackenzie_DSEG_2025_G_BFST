import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import library.dataset_library as ds_lib

print("=============================================================================")
print("Etapa 1: Análise Exploratória - Análise estatística completa das variáveis ==")
print("=============================================================================")


#Function responsável por gerar uma análise completa do dataset de funcionários.
def fn_gerar_analise_completa_dataframe(df: pd.DataFrame) -> pd.DataFrame:
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
