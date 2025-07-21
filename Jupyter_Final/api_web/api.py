import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

# Inicializa a aplicação Flask
app = Flask(__name__)
# Habilita o CORS para permitir que a sua aplicação web comunique com esta API
CORS(app)

# Carrega o modelo treinado (certifique-se de que este ficheiro está na mesma pasta)
try:
    model = joblib.load('modelo_attrition_final_v2.pkl')
    print("Modelo carregado com sucesso.")
except FileNotFoundError:
    print("Erro: Ficheiro 'modelo_attrition_final_v2.pkl' não encontrado.")
    model = None

# Carrega as colunas usadas durante o treino para garantir consistência
try:
    with open('model_columns.txt', 'r') as f:
        model_columns = f.read().splitlines()
    print("Colunas do modelo carregadas.")
except FileNotFoundError:
    print("Aviso: Ficheiro 'model_columns.txt' não encontrado. A API pode falhar se os dados de entrada não corresponderem.")
    model_columns = [] # Defina um fallback se necessário


@app.route('/')
def home():
    return "API do Modelo Preditivo de Attrition está a funcionar."

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modelo não foi carregado.'}), 500

    # Obtém os dados JSON enviados no pedido
    data = request.get_json()
    
    # Converte os dados para um DataFrame do Pandas
    df_input = pd.DataFrame([data])

    # ==================================================================
    # Pré-processamento dos dados de entrada
    # Esta parte deve espelhar a engenharia de features feita no treino.
    # Para simplificar, assumimos que as features mais importantes são passadas diretamente.
    # Numa implementação completa, a função fn_criar_features seria chamada aqui.
    # ==================================================================
    
    # Garante que os dados de entrada têm as mesmas colunas que o modelo espera
    df_processed = pd.get_dummies(df_input)
    
    # Alinha as colunas com as do modelo treinado
    missing_cols = set(model_columns) - set(df_processed.columns)
    for c in missing_cols:
        df_processed[c] = 0
    df_processed = df_processed[model_columns]

    # Faz a predição
    try:
        prediction_proba = model.predict_proba(df_processed)[:, 1]
        risk_score = int(prediction_proba[0] * 100)

        # Retorna o resultado como JSON
        return jsonify({'risk_score': risk_score})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    # Executa a aplicação (para testes locais)
    app.run(debug=True, port=5000)
