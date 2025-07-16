import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl as pxl

import library.dataset_library as ds_lib #biblioteca resposnável pela geração do dataset sintético
import library.ae_1_1_analise_completa_variaveis as fn_analise_completa #biblioteca responsável pela análise completa do dataset
import library.ae_1_2_ident_padroes_correlacoes as fn_ident_padroes #biblioteca responsável pela identificação de padrões e correlações do dataset
import library.ae_1_3_visualizacoes_criativas_informativas as fn_visualizacoes #biblioteca responsável pelas visualizações criativas e informativas do dataset

#Variáveis
v_local_resultados = 'G:/Meu Drive/[MBA]/[Data Science Experience]/[Trabalho Final]/3_Git/Mackenzie_DSEG_2025_G_BFST/results/'

#gerar o dataset de empregados
df_dataset_employess = ds_lib.fn_cria_dataset()
    #-->df_dataset_employess.to_excel('dataset_employess.xlsx', index=False)

###############################################################
## 1.1 Análise e Estrutura Completa das Variáveis do Dataset ##
###############################################################

#gerar a Análise e Estrutura Completa das Variáveis do Dataset
#análise completa
df_dataset_employess_comp_properties = fn_analise_completa.fn_gerar_analise_completa_dataframe(df_dataset_employess)
df_dataset_employess_comp_properties.to_excel(v_local_resultados+'/1_1_analise_variaveis/'+'dataset_employess_completa_properties.xlsx', index=False)

#análise básica
df_dataset_employess_basica_properties = fn_analise_completa.fn_gerar_analise_basica_dataframe(df_dataset_employess)
df_dataset_employess_basica_properties.to_excel(v_local_resultados+'/1_1_analise_variaveis/'+'dataset_employess_basica_properties.xlsx', index=False)

#análse estatística descritiva
df_dataset_employess_estatistica_properties = fn_analise_completa.fn_gerar_analise_estatistica_dataframe(df_dataset_employess)
df_dataset_employess_estatistica_properties.to_excel(v_local_resultados+'/1_1_analise_variaveis/'+'dataset_employess_estatistica_properties.xlsx', index=False)

#análise attriton taxa
df_dataset_employess_attriton_properties = fn_analise_completa.fn_gerar_analise_attrition_dataframe(df_dataset_employess)
df_dataset_employess_attriton_properties.to_excel(v_local_resultados+'/1_1_analise_variaveis/'+'dataset_employess_attrition_taxa.xlsx', index=False)

#análise valores_ausentes
fn_analise_completa.fn_gerar_analise_valores_ausentes(df_dataset_employess, v_local_resultados+'/1_1_analise_variaveis/'+'dataset_employess_valores_ausentes.txt')

#análise variaveis_numericas
fn_analise_completa.fn_gerar_analise_variaveis_numericas(df_dataset_employess, v_local_resultados+'/1_1_analise_variaveis/'+'dataset_employess_variaveis_numericas.txt')


###############################################################
## 1.3 Visualizações Criativas e Informativas do Dataset     ##
###############################################################

#gerar gráficos de análise do dataset   
fn_visualizacoes.fn_gerar_graficos_attrition_atual(df_dataset_employess, v_local_resultados+'1_3_graficos_cenario_atual/')  
fn_visualizacoes.fn_gerar_graficos_correlacao_attrition_atual(df_dataset_employess, v_local_resultados+'1_3_graficos_cenario_atual/')
fn_visualizacoes.fn_gerar_matriz_correlacao(df_dataset_employess, v_local_resultados+'1_3_graficos_cenario_atual/') 


###############################################################
#### Alteração na apresentação de valores do dataframe     ####
###############################################################

###############################################################
## 1.2 Identificação de Padrões e Correlações do Dataset     ##
###############################################################
#gerar a identificação de padrões correlações
df_dataset_employess_padronizado = fn_ident_padroes.fn_ident_padroes_correlacoes(df_dataset_employess)
    #-->df_dataset_employess_padronizado.to_excel(v_local_resultados+'dataset_employess_padronizado.xlsx', index=False)


df_dataset_employess.info()    