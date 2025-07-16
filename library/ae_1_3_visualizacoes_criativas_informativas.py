import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
#import library.dataset_library as ds_lib

print("=========++====================================================================")
print("Etapa 1.3: Análise Exploratória - Visualizações criativas e informativas     ==")
print("===========++==================================================================")

#Cenário Atual

# Analisando a variável 'Attrition'
def fn_gerar_graficos_attrition_atual(df: pd.DataFrame, v_local_gravacao: str) -> None:
    print("\n=== Analisando a variável Attrition ===")

    #Caso o diretório não exista, o mesmo será criado
    if not os.path.exists(v_local_gravacao):
        os.makedirs(v_local_gravacao)
        print(f"Diretório '{v_local_gravacao}' criado.")
    
    v_attrition_counts = df['Attrition'].value_counts()

    # Definido o tamanho da figura
    v_fig, v_ax = plt.subplots(1, 2, figsize=(12, 5))

    print("\033[1m=============================================================.\033[0m")
    print("\033[1mAnálise do cenário atual (Attrition) - Distribuição/Proporção.\033[0m")
    print("\033[1m=============================================================.\033[0m")
    v_attrition_counts.plot(kind='bar', ax=v_ax[0], color=['#2ecc71', '#e74c3c'])
    v_ax[0].set_title('Distribuição de Attrition')
    v_ax[0].set_ylabel('Quantidade')
    v_ax[0].set_xticklabels(['No', 'Yes'], rotation=0)
        #v_fig.savefig(os.path.join(v_local_gravacao, 'attrition_cenario_atual_barra.png'), bbox_inches='tight', dpi=300)
        #plt.close(v_fig)

    v_attrition_counts.plot(kind='pie', ax=v_ax[1], autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
    v_ax[1].set_title('Proporção de Attrition')
    v_ax[1].set_ylabel('')
    v_fig.savefig(os.path.join(v_local_gravacao, 'attrition_cenario_atual_distribuicao_proporcao.png'), bbox_inches='tight', dpi=300)
    plt.close(v_fig)



# Analisando variáveis que fazem correlação com a variável target 'Attrition'
def fn_gerar_graficos_correlacao_attrition_atual(df: pd.DataFrame, v_local_gravacao: str) -> None:
    print("\n=== Analisando as variáveis que fazem correlação com a variável target Attrition ===")

    #Caso o diretório não exista, o mesmo será criado
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

# Matriz de correlação (Análise de variáveis numéricas)
# Existe um mesmo bloco de análise no arquivo "ae_1_1_analise_completa_variaveis.py"
def fn_gerar_matriz_correlacao(df: pd.DataFrame, v_local_gravacao: str) -> None:
    print("\n=== Analisando Matriz de Correlacao ===")
    
    v_numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    #print(f"\n=== VARIÁVEIS NUMÉRICAS ({len(numeric_cols)}) ===")
    #print(numeric_cols.tolist())

    # Matriz de correlação
    #plt.figure(figsize=(20, 16))
    v_correlation_matrix = df[v_numeric_cols].corr()
    v_mask = np.triu(np.ones_like(v_correlation_matrix, dtype=bool))
    sns.heatmap(v_correlation_matrix, mask=v_mask, annot=True, fmt='.2f', cmap='coolwarm',
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Matriz de Correlação das Variáveis Numéricas', fontsize=16)
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(v_local_gravacao, 'matriz_correlacao.png'), bbox_inches='tight', dpi=300)
    plt.close()
