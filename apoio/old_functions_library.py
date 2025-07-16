import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import library.dataset_library as ds_lib

print("=========================================================================")
print("Etapa: Criação das functions  de Análise de Dados =======================")
print("=========================================================================")


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



#Analisando o cenário atual

# Analisando variáveis que fazem correlação com a variável target 'Attrition'
def fn_gerar_graficos_cenario_atual(df: pd.DataFrame, v_local_gravacao: str) -> None:
    # Garante que o diretório de gravação exista
    import os
    if not os.path.exists(v_local_gravacao):
        os.makedirs(v_local_gravacao)
        print(f"Diretório '{v_local_gravacao}' criado.")

    ############################################################################################
    # 1. Obter as contagens (e estatísticas)
    ############################################################################################
    v_overtime_cenario_atual = df['OverTime'].value_counts()
    v_jobasatisfaction_cenario_atual = df['JobSatisfaction'].value_counts()
    v_age_cenario_atual = df['Age'].value_counts() # Embora não seja usada diretamente para plot.pie ou bar aqui, a linha original a inclui
    v_yearsatcompany_cenario_atual = df['YearsAtCompany'].value_counts()
    v_monthlyincome_cenario_atual = df['MonthlyIncome'].value_counts()
    v_distancefromhome_cenario_atual = df['DistanceFromHome'].value_counts()
    v_bussinesstravel_cenario_atual = df['BusinessTravel'].value_counts()
    v_maritalstatus_cenario_atual = df['MaritalStatus'].value_counts()

    ############################################################################################
    # 2. Gerar os gráficos individualmente
    ############################################################################################

    # --- Gráfico de Overtime ---
    print("\033[1m===================================.\033[0m")
    print("\033[1mAnálise do cenário atual (Overtime).\033[0m")
    print("\033[1m===================================.\033[0m")
    plt.figure(figsize=(6, 6)) 
    v_overtime_cenario_atual.plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title('Overtime (Cenário Atual)')
    plt.ylabel('')
    plt.savefig(os.path.join(v_local_gravacao, 'overtime_cenario_atual.png'), bbox_inches='tight', dpi=300)
    plt.close() 

    # --- Gráfico de Idade ---
    print("\033[1m==============================.\033[0m")
    print("\033[1mAnálise do cenário atual (Age).\033[0m")
    print("\033[1m==============================.\033[0m")
    v_age_min = df['Age'].min()
    v_age_max = df['Age'].max()
    v_age_mean = df['Age'].mean()

    v_age_estatisticas = pd.DataFrame({
        'Metric': ['Min', 'Mean', 'Max'],
        'Age': [v_age_min, v_age_mean, v_age_max]
    })
    
    plt.figure(figsize=(7, 5)) 
    plt.bar(v_age_estatisticas['Metric'], v_age_estatisticas['Age'], color=['skyblue', 'lightgreen', 'salmon'])

    for index, value in enumerate(v_age_estatisticas['Age']):
        plt.text(index, value + 0.5, f'{value:.2f}', ha='center', va='bottom')

    plt.title('Minimum, Average and Maximum Age of Employees')
    plt.xlabel('Metric') 
    plt.ylabel('Age')
    plt.ylim(0, v_age_max * 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(v_local_gravacao, 'age_cenario_atual.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # --- Gráfico de Satisfação no Trabalho ---
    print("\033[1m==========================================.\033[0m")
    print("\033[1mAnálise do cenário atual (JobSatisfaction).\033[0m")
    print("\033[1m==========================================.\033[0m")
    v_jobasatisfaction_labels = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}
    # Mapeia os índices para os rótulos de forma segura, tratando possíveis chaves ausentes
    v_jobasatisfaction_labels_grafico = [v_jobasatisfaction_labels.get(i, str(i)) 
                                         for i in v_jobasatisfaction_cenario_atual.index]
    
    plt.figure(figsize=(6, 6)) 
    v_jobasatisfaction_cenario_atual.plot.pie(autopct='%1.1f%%', startangle=90, 
                                                 labels=v_jobasatisfaction_labels_grafico)
    plt.title('Job Satisfaction (Cenário Atual)') 
    plt.ylabel('')
    plt.savefig(os.path.join(v_local_gravacao, 'jobasatisfaction_cenario_atual.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # --- Gráfico de Anos na Empresa ---
    print("\033[1m=========================================.\033[0m")
    print("\033[1mAnálise do cenário atual (YearsAtCompany).\033[0m")
    print("\033[1m=========================================.\033[0m")
    v_yearscompany_min = df['YearsAtCompany'].min()
    v_yearscompany_max = df['YearsAtCompany'].max()
    v_yearscompany_mean = df['YearsAtCompany'].mean()

    v_yearscompany_estatisticas = pd.DataFrame({
        'Metric': ['Min', 'Mean', 'Max'],
        'Years': [v_yearscompany_min, v_yearscompany_mean, v_yearscompany_max]
    })

    plt.figure(figsize=(7, 5)) 
    plt.bar(v_yearscompany_estatisticas['Metric'], v_yearscompany_estatisticas['Years'], color=['skyblue', 'lightgreen', 'salmon'])

    for index, value in enumerate(v_yearscompany_estatisticas['Years']):
        plt.text(index, value + 0.5, f'{value:.2f}', ha='center', va='bottom')

    plt.title('Minimum, Average and Maximum Years at Company')
    plt.xlabel('Metric')
    plt.ylabel('Years') 
    plt.ylim(0, v_yearscompany_max * 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(v_local_gravacao, 'yearscompany_cenario_atual.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # --- Gráfico de Renda Mensal ---
    print("\033[1m========================================.\033[0m")
    print("\033[1mAnálise do cenário atual (MonthlyIncome).\033[0m")
    print("\033[1m========================================.\033[0m")
    v_monthlyincome_min = df['MonthlyIncome'].min()
    v_monthlyincome_max = df['MonthlyIncome'].max()
    v_monthlyincome_mean = df['MonthlyIncome'].mean()

    v_monthlyincome_estatisticas = pd.DataFrame({
        'Metric': ['Min', 'Mean', 'Max'],
        'Income': [v_monthlyincome_min, v_monthlyincome_mean, v_monthlyincome_max]
    })

    plt.figure(figsize=(8, 5)) 
    plt.barh(v_monthlyincome_estatisticas['Metric'],
             v_monthlyincome_estatisticas['Income'],
             color=['skyblue', 'lightgreen', 'salmon'])

    for index, value in enumerate(v_monthlyincome_estatisticas['Income']):
        plt.text(value + (v_monthlyincome_max * 0.02),
                 index,
                 f'{value:.2f}',
                 ha='left',
                 va='center')

    plt.title('Minimum, Average and Maximum of Monthly Income') 
    plt.xlabel('Income')
    plt.ylabel('Metric')
    plt.xlim(0, v_monthlyincome_max * 1.1)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(v_local_gravacao, 'monthlyincome_cenario_atual.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # --- Gráfico de Distância de Casa ---
    print("\033[1m===========================================.\033[0m")
    print("\033[1mAnálise do cenário atual (DistanceFromHome).\033[0m")
    print("\033[1m===========================================.\033[0m")
    v_distancefromhome_min = df['DistanceFromHome'].min()
    v_distancefromhome_max = df['DistanceFromHome'].max()
    v_distancefromhome_mean = df['DistanceFromHome'].mean()

    v_distancefromhome_estatisticas = pd.DataFrame({
        'Metric': ['Min', 'Mean', 'Max'],
        'Distance': [v_distancefromhome_min, v_distancefromhome_mean, v_distancefromhome_max]
    })

    plt.figure(figsize=(7, 5)) 
    plt.bar(v_distancefromhome_estatisticas['Metric'], v_distancefromhome_estatisticas['Distance'], color=['skyblue', 'lightgreen', 'salmon'])

    for index, value in enumerate(v_distancefromhome_estatisticas['Distance']):
        plt.text(index, value + 0.5, f'{value:.2f}', ha='center', va='bottom')

    plt.title('Minimum, Average and Maximum Distance from Home') 
    plt.xlabel('Metric')
    plt.ylabel('Distance')
    plt.ylim(0, v_distancefromhome_max * 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(v_local_gravacao, 'distancefromhome_cenario_atual.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # --- Gráfico de Viagens de Negócios ---
    print("\033[1m==========================================.\033[0m")
    print("\033[1mAnálise do cenário atual (BussinessTravel).\033[0m")
    print("\033[1m==========================================.\033[0m")
    plt.figure(figsize=(6, 6)) 
    v_bussinesstravel_cenario_atual.plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title('Business Travel (Cenário Atual)') 
    plt.ylabel('')
    plt.savefig(os.path.join(v_local_gravacao, 'bussinesstravel_cenario_atual.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # --- Gráfico de Estado Civil ---
    print("\033[1m========================================.\033[0m")
    print("\033[1mAnálise do cenário atual (MaritalStatus).\033[0m")
    print("\033[1m========================================.\033[0m")
    plt.figure(figsize=(6, 6)) 
    v_maritalstatus_cenario_atual.plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title('Marital Status (Cenário Atual)') 
    plt.ylabel('')
    plt.savefig(os.path.join(v_local_gravacao, 'maritalstatus_cenario_atual.png'), bbox_inches='tight', dpi=300)
    plt.close()

    return df 