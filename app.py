import pickle
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler

# Create Flask application
application = Flask(__name__)
app = application

# Load trained ML model
ridge_model = pickle.load(
    open('models/ridge.pkl', 'rb')
)

# Load fitted StandardScaler
standardscalerpickle = pickle.load(
    open('models/scaler.pkl', 'rb')
)


# ---------------- HOME PAGE ----------------
@app.route('/')
def index():
    return render_template('index.html')


# ---------------- PREDICTION PAGE ----------------
@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoints():

    if request.method == 'POST':

        # Get values from HTML form
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        # Put all inputs into one list
        input_data = [[
            Temperature,
            RH,
            Ws,
            Rain,
            FFMC,
            DMC,
            ISI,
            Classes,
            Region
        ]]

        # Scale input using the fitted scaler
        new_data_scaled = standardscalerpickle.transform(input_data)

        # Make prediction
        result = ridge_model.predict(new_data_scaled)

        # Send prediction to HTML page
        return render_template(
            'home.html',
            result=result[0]
        )

    # If page is opened normally
    return render_template('home.html')


# ---------------- RUN APPLICATION ----------------
if __name__ == '__main__':
    app.run(debug=True)