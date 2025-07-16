import pandas as pd
import numpy as np

print("=========================================================================")
print("Etapa: Geração do dataset de Empregados =================================")
print("=========================================================================")


def fn_cria_dataset():
    # Gera Datset Sintético de Empregados

    # Gerando a amostra aleatória (gerador de números aleatórios com um valor específico), garantindo que os resultados "aleatórios" gerados posteriormente
    # sejam reproduzíveis. Isso é essencial para testes, depuração e experimentos científicos onde a consistência dos resultados é importante.
    np.random.seed(42)

    # Definindo o número de amostras através do parâmetro n_samples do numpy
    n_samples = 1000000

    v_employes_data = {
        #Variáveis Demográficas
        'Age': np.random.randint(18, 65, n_samples),
        'Gender': np.random.choice(['Female', 'Male'], n_samples),
        'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
        'Education': np.random.choice(['Bellow College', 'College', 'Bachelor','Master','Doctor'], n_samples),
        'EducationField': np.random.choice(['Information Technology', 'Other', 'Engineering', 'Marketing', 'Computer Science', 'Human Resources'], n_samples),
        #Variáveis Profissionais
        'Department': np.random.choice(['Sales', 'Research & Development', 'Human Resources'], n_samples),
        'JobRole': np.random.choice(['Sales Executive', 'Software Developer', 'Project Manager',
                                    'Financial Analyst', 'Director', 'Manager',
                                    'Consultant', 'Analyst', 'Human Resources'], n_samples),
            #    'JobRole': np.random.choice(['Sales Executive', 'Research Scientist', 'Laboratory Technician',
            #                        'Manufacturing Director', 'Healthcare Representative', 'Manager',
            #                        'Sales Representative', 'Research Director', 'Human Resources'], n_samples),
        'JobLevel': np.random.randint(1, 6, n_samples),
        'JobInvolvement': np.random.randint(1, 5, n_samples),
        'YearsAtCompany': np.random.randint(0, 40, n_samples),
        #Variáveis de Compensação
        'MonthlyIncome': np.random.randint(1000, 20000, n_samples),
        'PercentSalaryHike': np.random.randint(11, 26, n_samples),
        'StockOptionLevel': np.random.randint(0, 4, n_samples),
        #Variáveis de Satisfação
        'JobSatisfaction': np.random.randint(1, 5, n_samples),
        'EnvironmentSatisfaction': np.random.randint(1, 5, n_samples),
        'RelationshipSatisfaction': np.random.randint(1, 5, n_samples),
        #Variáveis de Trabalho Atual (WorkLife)
        'OverTime': np.random.choice(['Yes', 'No'], n_samples, p=[0.28, 0.72]),
        'WorkLifeBalance': np.random.randint(1, 5, n_samples),
        'BusinessTravel': np.random.choice(['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'], n_samples),
        'DistanceFromHome': np.random.randint(1, 30, n_samples),
        #Variáveis de Performance
        'PerformanceRating': np.random.choice([3, 4], n_samples, p=[0.84, 0.16]),
        'TrainingTimesLastYear': np.random.randint(0, 7, n_samples),
        #Variáveis de Target
        'Attrition': np.random.choice(['Yes', 'No'], n_samples, p=[0.16, 0.84])
    }

    # gerando o dataframe através dos dados dos funcionários
    df_dataset_employess = pd.DataFrame(v_employes_data)
    return df_dataset_employess


