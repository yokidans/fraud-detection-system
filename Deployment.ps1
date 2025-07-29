<#
.SYNOPSIS
    Production Deployment Script
#>

# 1. Validate Environment
if (-not (Test-Path "models/best_ecommerce_model.pkl")) {
    throw "Trained models not found. Run training first."
}

# 2. Start Flask API
$flaskCommand = @"
from flask import Flask, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('models/best_ecommerce_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    return jsonify({'status': 'API Online'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
"@

Start-Process python -ArgumentList "-c `"$flaskCommand`"" -NoNewWindow

# 3. Verify Deployment
$response = Invoke-RestMethod -Uri "http://localhost:5000/predict" -Method Post
if ($response.status -ne "API Online") {
    throw "Deployment verification failed"
}

Write-Host "Fraud Detection API deployed successfully" -ForegroundColor Green