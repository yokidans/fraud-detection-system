from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('models/best_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = preprocess_input(data)
    prediction = model.predict_proba([features])[0][1]
    return jsonify({'fraud_probability': float(prediction)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)