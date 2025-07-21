# **Sistema Preditivo de Attrition \- TechCorp**

## **1\. Visão Geral do Projeto**

Este projeto apresenta uma solução completa de Machine Learning para prever a rotatividade de funcionários (attrition) na empresa fictícia TechCorp. Diante de um aumento de 35% na taxa de attrition e custos anuais de R$ 45 milhões, foi desenvolvido um sistema para identificar proativamente funcionários com alto risco de saída, permitindo que o departamento de RH tome ações preventivas e estratégicas.

A solução abrange desde a análise exploratória dos dados e engenharia de features até o treinamento, otimização e interpretação de múltiplos modelos de classificação. O resultado final é um pipeline de Machine Learning robusto, salvo como attrition\_model.pkl, e uma API REST construída com Flask para servir o modelo em um ambiente de produção.

## **2\. Estrutura do Projeto**

.  
├── api/  
│   └── api.py                  \# Aplicação Flask para servir o modelo  
├── notebooks/  
│   └── codigo\_analise\_attrition.py \# Script Python/Notebook para análise e treinamento  
├── attrition\_model.pkl         \# Artefato do modelo treinado (gerado pelo notebook)  
├── README.md                   \# Documentação do projeto (este arquivo)  
└── requirements.txt            \# Dependências Python do projeto

## **3\. Principais Funcionalidades**

* **Análise Exploratória de Dados (EDA):** Investigação completa dos dados para identificar padrões e correlações com a rotatividade.  
* **Engenharia de Features:** Criação de 11 novas variáveis para enriquecer o modelo com insights de negócio (ex: PromotionStagnation, WorkLifeImbalance).  
* **Modelagem Preditiva:** Treinamento e avaliação de múltiplos algoritmos (Regressão Logística, Random Forest, XGBoost) com tratamento para dados desbalanceados (SMOTE).  
* **Otimização de Hiperparâmetros:** Uso de RandomizedSearchCV para encontrar a melhor configuração de parâmetros para os modelos.  
* **Interpretabilidade:** Análise com **SHAP** para entender quais fatores mais influenciam as previsões do modelo.  
* **API REST com Flask:** Uma aplicação web que:  
  * Carrega o modelo treinado (.pkl).  
  * Permite o upload de arquivos .csv para predição em lote.  
  * Oferece um endpoint (/predict) para predições em tempo real.  
  * Simula um dashboard de monitoramento e um sistema de alertas.

## **4\. Como Utilizar**

### **Pré-requisitos**

* Python 3.8 ou superior  
* Git

### **Parte 1: Treinamento do Modelo**

Siga estes passos para analisar os dados e gerar o arquivo attrition\_model.pkl.

1. **Clone o repositório:**  
   git clone \<url-do-seu-repositorio\>  
   cd \<nome-do-repositorio\>

2. **Crie e ative um ambiente virtual (recomendado):**  
   python \-m venv venv  
   \# No Windows:  
   venv\\Scripts\\activate  
   \# No macOS/Linux:  
   source venv/bin/activate

3. **Instale as dependências:**  
   pip install \-r requirements.txt

4. Execute o script de treinamento:  
   Navegue até a pasta notebooks/ e execute o script Python. Se estiver usando um Jupyter Notebook, abra-o e execute todas as células.  
   cd notebooks  
   python codigo\_analise\_attrition.py

   Ao final da execução, o arquivo attrition\_model.pkl será gerado na raiz do projeto.

### 

## **5\. Stack de Tecnologias**

* **Linguagem:** Python 3  
* **Análise de Dados:** Pandas, NumPy  
* **Visualização:** Matplotlib, Seaborn  
* **Machine Learning:** Scikit-learn, XGBoost, Imbalanced-learn (imblearn)  
* **Interpretabilidade:** SHAP  
* **API Web:** Flask  
* **Serialização do Modelo:** Joblib