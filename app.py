import pickle
from flask import Flask, Request,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

application = Flask(__name__)
app=application
## laoding the model
scaler = pickle.load(open('scaling.pkl','rb'))
regmodel = pickle.load(open('regmodel.pkl','rb'))


@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output = regmodel.predict(new_data)
    print(output)
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = regmodel.predict(final_input)
    return render_template("home.html",prediction_text = "The house price is aprox {}".format(output))

if __name__=="__main__":
    app.run()