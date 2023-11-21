import numpy as np
import pandas as pd
from flask import Flask,render_template,request
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods =['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0],2)
    output = np.round_(output)

    return render_template('index.html', prediction_text = "Red Wine quality is : {}".format(output))

if __name__ == '__main__':
    app.run(debug = True)