from flask import Flask, request, jsonify, render_template
from src.models.evaluate import FraudPredictor
from src.config import MODEL_PATH, FeatureNames, THRESHOLD_CONFIG, FALLBACK_VALUES
import logging
from typing import Dict, Any

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

predictor = FraudPredictor(MODEL_PATH)

@app.route('/dashboard')
def dashboard():
    """Render the evaluation dashboard"""
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle transaction evaluation requests with proper JSON serialization"""
    # Check content type
    if not request.is_json:
        return jsonify({
            'error': 'Content-Type must be application/json',
            'status': 415
        }), 415
        
    try:
        data = request.get_json()  # Use get_json() instead of .json for better error handling
        
        # Validate required fields
        required = [
            'amount',
            'ip_address',
            'device_id',
            'user_id',
            'timestamp'
        ]
        missing = [field for field in required if field not in data]
        if missing:
            return jsonify({
                'error': f'Missing required fields: {missing}',
                'status': 400
            }), 400
        
        # Set default values for optional fields
        defaults = {
            'user_age': float(FALLBACK_VALUES.get('user_age', 30)),
            'account_age_days': float(FALLBACK_VALUES.get('account_age_days', 365)),
            'transaction_count_7d': float(FALLBACK_VALUES.get('transaction_count_7d', 10)),
            'is_foreign_ip': float(FALLBACK_VALUES.get('is_foreign_ip', 0)),
            'device_change_flag': float(FALLBACK_VALUES.get('device_change_flag', 0)),
            'country_risk_score': float(FALLBACK_VALUES.get('country_risk_score', 0.5)),
            'time_since_last_transaction': float(FALLBACK_VALUES.get('time_since_last_transaction', 1440)),
            'purchase_frequency_24h': float(FALLBACK_VALUES.get('purchase_frequency_24h', 1)),
            'billing_shipping_mismatch': float(FALLBACK_VALUES.get('billing_shipping_mismatch', 0)),
            'user_avg_transaction': float(FALLBACK_VALUES.get('user_avg_transaction', 100))
        }
        
        transaction = {**defaults, **data}
        
        # Evaluate transaction
        result = predictor.predict_with_dynamic_threshold(transaction)
        
        return jsonify({
            'status': 200,
            'result': result.to_dict()
        })
        
    except Exception as e:
        logger.error(f"API Error: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'status': 500
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)