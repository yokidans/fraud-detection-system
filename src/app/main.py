from flask import Flask, render_template, request, jsonify
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import joblib
import numpy as np
import logging
from src.models.explainability import ModelExplainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
flask_app = Flask(__name__)

# Initialize Dash app within Flask
dash_app = dash.Dash(
    __name__,
    server=flask_app,
    url_base_pathname='/dashboard/'
)

class FraudDetectionApp:
    def __init__(self, model_path, preprocessor_path):
        try:
            self.model = joblib.load(model_path)
            self.preprocessor = joblib.load(preprocessor_path)
            self.feature_names = self._get_feature_names()
            self.explainer = ModelExplainer(self.model, self.preprocessor, self.feature_names)
        except Exception as e:
            logger.error(f"Failed to initialize app: {str(e)}")
            raise
            
    def _get_feature_names(self):
        """Extract feature names from preprocessor"""
        try:
            # This should be implemented based on your preprocessor structure
            # Example for ColumnTransformer with OneHotEncoder
            numeric_features = self.preprocessor.transformers_[0][2]
            ohe = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
            categorical_features = ohe.get_feature_names_out(
                self.preprocessor.transformers_[1][2])
            return list(numeric_features) + list(categorical_features)
        except Exception as e:
            logger.warning(f"Could not extract feature names: {str(e)}")
            return []
            
    def predict(self, input_data):
        """Make prediction using the model"""
        try:
            processed_data = self.preprocessor.transform(input_data)
            prediction = self.model.predict(processed_data)
            probability = self.model.predict_proba(processed_data)[:, 1]
            return prediction, probability
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
            
    def create_dash_layout(self):
        """Create layout for Dash dashboard"""
        return html.Div([
            html.H1('Fraud Detection Dashboard'),
            dcc.Tabs([
                dcc.Tab(label='Real-time Monitoring', children=[
                    dcc.Graph(id='live-update-graph'),
                    dcc.Interval(
                        id='interval-component',
                        interval=60*1000,  # in milliseconds
                        n_intervals=0
                    )
                ]),
                dcc.Tab(label='Model Analysis', children=[
                    dcc.Graph(id='feature-importance-graph'),
                    html.Div(id='shap-summary-graph')
                ]),
                dcc.Tab(label='Transaction Explorer', children=[
                    dcc.Dropdown(
                        id='transaction-id-dropdown',
                        options=[],  # Will be updated
                        value=None
                    ),
                    html.Div(id='transaction-details')
                ])
            ])
        ])
        
    def setup_dash_callbacks(self):
        """Setup Dash callbacks"""
        @dash_app.callback(
            Output('live-update-graph', 'figure'),
            Input('interval-component', 'n_intervals'))
        def update_graph_live(n):
            # This would be replaced with actual data fetching logic
            df = pd.DataFrame({
                'time': pd.date_range(start='2023-01-01', periods=100, freq='H'),
                'fraud_prob': np.random.rand(100)
            })
            fig = px.line(df, x='time', y='fraud_prob', title='Real-time Fraud Probability')
            return fig
            
        @dash_app.callback(
            Output('feature-importance-graph', 'figure'),
            Input('feature-importance-graph', 'id'))
        def update_feature_importance(_):
            # Get feature importances from model
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
                features = self.feature_names[:len(importance)]
                df = pd.DataFrame({'Feature': features, 'Importance': importance})
                df = df.sort_values('Importance', ascending=False).head(20)
                fig = px.bar(df, x='Importance', y='Feature', orientation='h',
                            title='Top 20 Important Features')
                return fig
            return {}
            
# Initialize the app
try:
    app = FraudDetectionApp(
        model_path='models/best_model.pkl',
        preprocessor_path='models/preprocessor.pkl')
        
    # Set up Dash
    dash_app.layout = app.create_dash_layout()
    app.setup_dash_callbacks()
except Exception as e:
    logger.error(f"Failed to initialize application: {str(e)}")

# Flask routes
@flask_app.route('/')
def home():
    return render_template('index.html')

@flask_app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        prediction, probability = app.predict(input_df)
        return jsonify({
            'prediction': int(prediction[0]),
            'probability': float(probability[0]),
            'status': 'success'
        })
    except Exception as e:
        logger.error(f"API prediction error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    flask_app.run(debug=True, host='0.0.0.0')