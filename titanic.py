from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load Model
try:
    model = load_model('titanic_model.h5')
    scaler = joblib.load('titanic_scaler.pkl')
except:
    print("Run create_titanic_model.py first!")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ""
    if request.method == 'POST':
        try:
            # Get inputs
            pclass = int(request.form['Pclass'])
            sex = int(request.form['Sex']) # 0 or 1
            age = float(request.form['Age'])
            sibsp = int(request.form['SibSp'])
            parch = int(request.form['Parch'])
            fare = float(request.form['Fare'])

            # Prepare array
            features = np.array([[pclass, sex, age, sibsp, parch, fare]])
            features_scaled = scaler.transform(features)

            # Predict
            prediction = model.predict(features_scaled)
            probability = prediction[0][0]

            if probability > 0.5:
                res = "SURVIVED"
            else:
                res = "DID NOT SURVIVE"
            
            prediction_text = f"Prediction: {res} (Chance: {probability*100:.1f}%)"

        except Exception as e:
            prediction_text = f"Error: {e}"

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)