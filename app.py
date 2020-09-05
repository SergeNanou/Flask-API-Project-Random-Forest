# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

model = pickle.load(open('clf.pkl','rb'))
app = Flask(__name__)

@app.route("/")
def recep():
        return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text=output)


if __name__ == "__main__":
    app.run(debug=True)