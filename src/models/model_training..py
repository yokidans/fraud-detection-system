# model_training.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.calibration import CalibratedClassifierCV
import joblib
import optuna
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import shap
import warnings
warnings.filterwarnings('ignore')

# Configure environment
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")
pd.set_option('display.max_columns', 50)

# %%
# ======================
# 1. DATA PREPARATION
# ======================
class DataLoader:
    def __init__(self, ecommerce_path, creditcard_path):
        self.ecommerce_path = ecommerce_path
        self.creditcard_path = creditcard_path
        
    def load_and_split(self, dataset='ecommerce'):
        """Load dataset with automated train-test splitting"""
        if dataset == 'ecommerce':
            df = pd.read_csv(self.ecommerce_path)
            X = df.drop(columns=['class'])
            y = df['class']
        else:
            df = pd.read_csv(self.creditcard_path)
            X = df.drop(columns=['Class'])
            y = df['Class']
            
        return train_test_split(
            X, y, 
            test_size=0.2, 
            stratify=y,
            random_state=42
        )

# %%
# ======================
# 2. MODEL DEVELOPMENT
# ======================
class FraudModelTrainer:
    def __init__(self):
        self.models = {
            'logistic': self._build_logistic_regression(),
            'xgboost': self._build_xgboost()
        }
        self.best_model = None
        self.scaler = StandardScaler()
        
    def _build_logistic_regression(self):
        """Calibrated logistic regression with class weights"""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', CalibratedClassifierCV(
                LogisticRegression(
                    class_weight='balanced',
                    max_iter=1000,
                    random_state=42
                ),
                method='isotonic'
            ))
        ])
        
    def _build_xgboost(self):
        """Optimized XGBoost with custom objective"""
        return XGBClassifier(
            scale_pos_weight=30,  # Approx 1/(fraud_ratio)
            objective='binary:logistic',
            eval_metric='aucpr',
            random_state=42,
            n_jobs=-1
        )
        
    def _optimize_xgboost(self, X, y):
        """Hyperparameter tuning with Optuna"""
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
            }
            
            model = XGBClassifier(**params, scale_pos_weight=30, random_state=42)
            return np.mean(cross_val_score(model, X, y, cv=5, scoring='average_precision'))
            
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        return study.best_params
    
    def train(self, X_train, y_train, optimize=False):
        """Train all models with optional hyperparameter tuning"""
        if optimize:
            best_params = self._optimize_xgboost(X_train, y_train)
            self.models['xgboost'].set_params(**best_params)
            
        for name, model in self.models.items():
            print(f"Training {name} model...")
            model.fit(X_train, y_train)
            
        return self
    
    def evaluate(self, X_test, y_test):
        """Comprehensive model evaluation"""
        results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            results[name] = {
                'auprc': average_precision_score(y_test, y_proba),
                'f1': f1_score(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            # Generate SHAP explanations for best model
            if name == 'xgboost' and hasattr(model, 'feature_importances_'):
                self._generate_shap(model, X_test)
                
        return results
    
    def _generate_shap(self, model, X_test, sample_size=1000):
        """SHAP analysis for model interpretability"""
        explainer = shap.Explainer(model)
        shap_values = explainer(X_test.sample(sample_size))
        
        plt.figure(figsize=(12, 6))
        shap.summary_plot(shap_values, X_test.sample(sample_size))
        plt.savefig('reports/figures/shap_summary.png')
        plt.close()
        
    def select_best_model(self, results):
        """Select model based on AUPRC and business constraints"""
        best_name = max(results, key=lambda x: results[x]['auprc'])
        self.best_model = self.models[best_name]
        print(f"Selected best model: {best_name} with AUPRC: {results[best_name]['auprc']:.4f}")
        return self.best_model
    
    def save_model(self, path='models/best_model.pkl'):
        """Save the best model for production"""
        joblib.dump(self.best_model, path)
        print(f"Model saved to {path}")

# %%
# ======================
# 3. EXECUTION PIPELINE
# ======================
def run_training_pipeline(dataset='ecommerce'):
    # 1. Load data
    loader = DataLoader(
        'data/processed/ecommerce_features.csv',
        'data/processed/creditcard_features.csv'
    )
    X_train, X_test, y_train, y_test = loader.load_and_split(dataset)
    
    # 2. Train models
    trainer = FraudModelTrainer()
    trainer.train(X_train, y_train, optimize=True)
    
    # 3. Evaluate
    results = trainer.evaluate(X_test, y_test)
    
    # 4. Select and save best model
    trainer.select_best_model(results)
    trainer.save_model()
    
    return results

# Execute for both datasets
ecommerce_results = run_training_pipeline('ecommerce')
creditcard_results = run_training_pipeline('creditcard')