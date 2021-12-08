import pickle
import sklearn.linear_model._logistic
import pandas as pd
from flask import Flask, jsonify, request

# carrega modelo
model = pickle.load(open('model.pkl', 'rb'))

# app
app = Flask(__name__)

# rotas
@app.route('/', methods=['POST'])
def predict():
    # obtém dados
    data = request.get_json(force=True)

    # converte dados em dataframe
    data.update((x, [y]) for x, y in data.items())  #se quiser aplicar para vários passageiros ao mesmo tempo
    data_df = pd.DataFrame.from_dict(data)

    # previsões
    result = model.predict(data_df)

    # retorna ao browser
    output = {'results': int(result[0])}  #transf em json

    # retorna os dados
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)