import pickle
from flask import Flask, render_template, request
import numpy as np
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Load trained model
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))

# Load fitted scaler
standardscalerpickle = pickle.load(open('models/scaler.pkl', 'rb'))


# Home page
@app.route('/')
def index():
    return render_template('index.html')


# Prediction page
@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoints():

    if request.method == 'POST':

        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        # Scale the input data
        new_data_scaled = standardscalerpickle.transform(
            [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
        )

        # Make prediction
        result = ridge_model.predict(new_data_scaled)

        return render_template('home.html', result=result[0])

    else:
        return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)