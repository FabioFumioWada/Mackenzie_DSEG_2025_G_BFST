# app.py

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# 1. Inicializa a aplicação FastAPI
app = FastAPI(
    title="API de Previsão de Attrition - TechCorpBrasil",
    description="Uma API para prever a probabilidade de um funcionário deixar a empresa, baseada no modelo treinado no MBA em Engenharia de Dados.",
    version="1.0.0"
)

# 2. Carrega o pipeline do modelo treinado ao iniciar a API
#    Isso garante que o modelo não seja recarregado a cada requisição, otimizando o desempenho.
try:
    pipeline = joblib.load('attrition_model.pkl')
    print("Modelo carregado com sucesso.")
except FileNotFoundError:
    print("Erro: Arquivo 'attrition_model.pkl' não encontrado.")
    pipeline = None
except Exception as e:
    print(f"Ocorreu um erro ao carregar o modelo: {e}")
    pipeline = None


# 3. Define a estrutura de dados para a entrada da API usando Pydantic
#    Isso garante validação automática e documentação dos dados de entrada.
#    As colunas devem ser exatamente as mesmas que o modelo espera.
class EmployeeData(BaseModel):
    Age: int
    BusinessTravel: str
    Department: str
    DistanceFromHome: int
    Education: int
    EducationField: str
    EnvironmentSatisfaction: int
    Gender: str
    JobInvolvement: int
    JobLevel: int
    JobSatisfaction: int
    MaritalStatus: str
    MonthlyIncome: int
    NumCompaniesWorked: int
    OverTime: str
    PerformanceRating: int
    RelationshipSatisfaction: int
    StockOptionLevel: int
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int

# 4. Cria um endpoint raiz para verificar se a API está online
@app.get("/")
def read_root():
    return {"status": "API online", "message": "Bem-vindo à API de Previsão de Attrition!"}

# 5. Cria o endpoint de previsão
@app.post("/predict")
def predict_attrition(data: EmployeeData):
    """
    Prevê o Attrition de um funcionário com base nos dados fornecidos.
    
    - **Entrada**: JSON com os dados do funcionário.
    - **Saída**: JSON com a previsão de Attrition ('Sim' ou 'Não').
    """
    if pipeline is None:
        return {"error": "Modelo não está carregado. Verifique os logs do servidor."}

    try:
        # Converte os dados de entrada Pydantic para um DataFrame do Pandas
        input_df = pd.DataFrame([data.dict()])

        # Realiza a previsão usando o pipeline carregado
        prediction_numeric = pipeline.predict(input_df)

        # Mapeia a saída numérica para o resultado categórico esperado
        prediction_text = 'Sim' if prediction_numeric[0] == 1 else 'Não'

        # Retorna o resultado da previsão
        return {"attrition_predito": prediction_text}
    
    except Exception as e:
        return {"error": "Ocorreu um erro durante a previsão.", "details": str(e)}


# Permite executar a API localmente para testes com 'python app.py'
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)