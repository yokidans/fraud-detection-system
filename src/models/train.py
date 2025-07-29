"""
Enhanced Fraud Detection Training Script with:
- Probability threshold adjustment
- SHAP value analysis
- Improved model configuration
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import IsolationForest
from src.config import FEATURES, TARGET, COST_MATRIX, MODEL_PATH, DATA_PATHS, PREPROCESSOR_CONFIG
from src.data.preprocessing import DataPreprocessor
import joblib
import shap
import matplotlib.pyplot as plt
import logging
import inspect
from datetime import datetime 
from typing import List, Tuple, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CompatibleDataPreprocessor:
    """Wrapper class to ensure compatibility with different DataPreprocessor versions"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.preprocessor = None
        
        try:
            self.preprocessor = DataPreprocessor(config=config)
        except TypeError as e:
            if "unexpected keyword argument 'config'" in str(e):
                self.preprocessor = DataPreprocessor()
                if hasattr(self.preprocessor, 'config'):
                    self.preprocessor.config = config
            else:
                raise
        
        if not hasattr(self.preprocessor, 'preprocess_data'):
            raise AttributeError("DataPreprocessor is missing required 'preprocess_data' method")

    def preprocess_data(self, df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        try:
            if hasattr(self.preprocessor, '_validate_data'):
                return self.preprocessor.preprocess_data(df, target_col)
            
            logger.warning("Using DataPreprocessor without validation - adding basic checks")
            
            required_cols = set(self.config['feature_config'].get('required_columns', []) + [target_col])
            missing = required_cols - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            
            return self.preprocessor.preprocess_data(df, target_col)
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Data preprocessing error: {str(e)}") from e

class FraudDetectionModel:
    """Enhanced fraud detection model class with threshold tuning and SHAP"""
    def __init__(self, model_type: str = 'xgb'):
        self.model_type = model_type
        self.model = None
        self.optimal_threshold = 0.5  # Default threshold
        self.feature_names = None
        
    def train(self, X_train, y_train, feature_names=None):
        # Handle duplicate features
        if feature_names:
            # Remove duplicates while preserving order
            seen = set()
            self.feature_names = []
            for f in feature_names:
                if f not in seen:
                    seen.add(f)
                    self.feature_names.append(f)
            
            # Verify we didn't lose any features
            if len(self.feature_names) != X_train.shape[1]:
                raise ValueError(f"Feature count mismatch after deduplication: {len(self.feature_names)} vs {X_train.shape[1]}")
        
        if self.model_type == 'xgb':
            self.model = xgb.XGBClassifier(
                scale_pos_weight=COST_MATRIX['false_negative']/COST_MATRIX['false_positive'],
                eval_metric='logloss',
                enable_categorical=False,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                n_estimators=200,
                random_state=42
            )
        elif self.model_type == 'brf':
            self.model = BalancedRandomForestClassifier(
                n_estimators=100,
                sampling_strategy='auto',
                replacement=True,
                random_state=42
            )
        elif self.model_type == 'iso':
            self.model = IsolationForest(
                n_estimators=100,
                contamination='auto',
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Store feature names if provided
        if feature_names:
            self.feature_names = feature_names
            
        # Now train the model
        self.model.fit(X_train, y_train)
        
        # Add feature importance check and validation
        train_pred = self.model.predict(X_train)
        if np.mean(train_pred == y_train) > 0.99:
            logger.warning("Model may be overfitting - accuracy too high on training data")
            
        # Verify feature uniqueness
        if feature_names and len(feature_names) != len(set(feature_names)):
            raise ValueError("Feature names must be unique")
        
        # Find optimal threshold if using XGBoost
        if self.model_type == 'xgb':
            self._find_optimal_threshold(X_train, y_train)
        
    def _find_optimal_threshold(self, X, y):
        """Find optimal probability threshold using precision-recall tradeoff"""
        y_proba = self.model.predict_proba(X)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y, y_proba)
        
        # Find threshold that maximizes F1 score
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
        ix = np.argmax(f1_scores)
        self.optimal_threshold = thresholds[ix]
        
        logger.info(f"Optimal probability threshold: {self.optimal_threshold:.3f}")
        
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        if self.model_type == 'xgb':
            # Use optimal threshold for XGBoost
            y_proba = self.model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= self.optimal_threshold).astype(int)
        else:
            # Default prediction for other models
            y_pred = self.model.predict(X_test)
        
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        if self.model_type == 'xgb':
            self._plot_shap_values(X_test)
        
    def _plot_shap_values(self, X):
        """Generate SHAP plots to explain model predictions"""
        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
            
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X, feature_names=self.feature_names, plot_type="bar")
            plt.title("Feature Importance (SHAP Values)")
            plt.tight_layout()
            plt.savefig("shap_feature_importance.png")
            plt.close()
            
            logger.info("Saved SHAP feature importance plot to shap_feature_importance.png")
            
        except Exception as e:
            logger.warning(f"Could not generate SHAP plots: {str(e)}")
        
    def save_model(self, path: str):
        """Save model with all required components"""
        if not hasattr(self, 'model'):
            raise AttributeError("Model has not been trained yet")
        
        # Get feature count from the model
        feature_count = getattr(self.model, 'n_features_in_', len(self.feature_names))
        
        # Create save dictionary with fallbacks
        save_dict = {
            'model': self.model,
            'feature_names': self.feature_names,
            'exact_feature_count': feature_count,
            'threshold': self.optimal_threshold,
            'model_type': self.model_type,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add feature types if available
        try:
            if hasattr(self.model, 'get_booster'):
                feature_types = self.model.get_booster().feature_types
                if feature_types is not None:
                    save_dict['feature_dtypes'] = dict(zip(self.feature_names, feature_types))
        except Exception as e:
            logger.warning(f"Could not save feature types: {str(e)}")
        
        # Add SHAP explainer if available
        if hasattr(self, 'explainer'):
            save_dict['explainer'] = self.explainer
        
        try:
            joblib.dump(save_dict, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise
        
    def load_model(self, path: str):
        loaded = joblib.load(path)
        self.model = loaded['model']
        self.optimal_threshold = loaded.get('threshold', 0.5)
        self.feature_names = loaded.get('feature_names')

def initialize_preprocessor() -> Any:
    try:
        logger.info("Initializing data preprocessor with compatibility layer")
        preprocessor = CompatibleDataPreprocessor(PREPROCESSOR_CONFIG)
        logger.info("Preprocessor initialized successfully with compatibility layer")
        return preprocessor
        
    except Exception as e:
        logger.error(f"Preprocessor initialization failed: {str(e)}", exc_info=True)
        raise RuntimeError(f"Could not initialize preprocessor: {str(e)}")

def main():
    try:
        logger.info("Starting fraud detection training pipeline")
        preprocessor = initialize_preprocessor()
        
        logger.info(f"Loading raw data from {DATA_PATHS['raw_data']}")
        try:
            df = pd.read_csv(DATA_PATHS['raw_data'])
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
        except Exception as e:
            logger.error(f"Failed to load raw data: {str(e)}", exc_info=True)
            raise
        
        logger.info("Preprocessing data...")
        try:
            X, y, feature_names = preprocessor.preprocess_data(df, TARGET)
            logger.info(f"Data preprocessing completed. Features shape: {X.shape}")
        except Exception as e:
            logger.error(f"Data preprocessing failed: {str(e)}", exc_info=True)
            raise
        
        logger.info("Splitting data...")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                stratify=y,
                random_state=42
            )
            logger.info(f"Data split into train: {X_train.shape}, test: {X_test.shape}")
        except Exception as e:
            logger.error(f"Data splitting failed: {str(e)}", exc_info=True)
            raise
        
        logger.info("Initializing and training model...")
        try:
            model = FraudDetectionModel(model_type='xgb')
            model.train(X_train, y_train, feature_names)
            model.evaluate(X_test, y_test)
            
            # Save model with threshold and feature names
            model.save_model(MODEL_PATH)
            logger.info(f"Model saved to {MODEL_PATH}")
            
        except Exception as e:
            logger.error(f"Model training/evaluation failed: {str(e)}", exc_info=True)
            raise
        
        logger.info("Training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()