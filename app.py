from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained ML model
model = pickle.load(open("predictive_maintenance_xgb.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    data = pd.read_csv(file)

    # Ensure column names match your model training
    features = data[['temperature', 'vibration', 'pressure', 'speed']]

    prediction = model.predict(features)
    prob = model.predict_proba(features)[:, 1].mean()

    result = "Machine Failure Detected ⚠️" if prob > 0.5 else "Machine Healthy ✅"

    return jsonify({
        'prediction': result,
        'probability': float(prob)
    })

if __name__ == "__main__":
    app.run(debug=True)
