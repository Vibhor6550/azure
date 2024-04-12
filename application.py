import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

# import regressor model and standard scaler pickle file

regressor_model = pickle.load(open('models/regression.pkl','rb'))
scaler_model = pickle.load(open('models/scalermodel.pkl','rb'))

# route for home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))

        new_data = scaler_model.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes]])
        result = regressor_model.predict(new_data)

        return render_template('home.html', result=result)
    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")
