"""
Enhanced Fraud Prediction System with Robust Error Handling - Final
"""

import pandas as pd
import numpy as np
import joblib
import logging
import shap
import json
import time
import os
from datetime import datetime
from typing import List, Dict, Union, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from src.config import THRESHOLD_CONFIG, FALLBACK_VALUES, FeatureNames, MODEL_CONFIG

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Container for comprehensive prediction results"""
    prediction: int
    probability: float
    threshold_used: float
    features: Dict[str, float]
    shap_values: Optional[Dict[str, Any]] = None
    warnings: List[str] = None
    prediction_time: Optional[float] = None
    model_version: Optional[str] = None
    behavioral_anomalies: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return asdict(self)

class FraudPredictor:
    def __init__(self, model_path: str):
        """
        Initialize the fraud predictor
        
        Args:
            model_path: Path to the trained model file
        """
        self.model_path = model_path
        self.feature_names = FeatureNames.get_ordered_features()
        self.exact_feature_count = len(self.feature_names)  # Moved this line up
        
        # Debug output
        print("Feature count:", self.exact_feature_count)
        print("Feature names:")
        for i, feat in enumerate(self.feature_names, 1):
            print(f"{i}. {feat}")
        
        # Validate feature count immediately
        if self.exact_feature_count != 26:
            logger.error(f"Critical: Model requires 26 features but got {self.exact_feature_count}")
            raise ValueError(f"Feature count mismatch. Expected 26, got {self.exact_feature_count}")
        
        self.explainer = None
        self.model_version = "unknown"
        self.optimal_threshold = 0.5
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Load and validate the trained model"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
                
            model_data = joblib.load(self.model_path)
            
            if 'model' not in model_data:
                raise ValueError("Model file is missing the 'model' key")
                
            self.model = model_data['model']
            self.model_version = model_data.get('version', MODEL_CONFIG.version)
            self.optimal_threshold = model_data.get('threshold', THRESHOLD_CONFIG.default)
            
            if hasattr(self.model, 'predict_proba'):
                try:
                    self.explainer = shap.TreeExplainer(self.model)
                except Exception as e:
                    logger.warning(f"SHAP explainer initialization failed: {str(e)}")
                    
            logger.info(f"Loaded model version {self.model_version} expecting {self.exact_feature_count} features")
            
        except Exception as e:
            logger.critical(f"Model initialization failed: {str(e)}")
            raise RuntimeError(f"Could not initialize model: {str(e)}")

    def _validate_feature_matrix(self, X: np.ndarray) -> None:
        """Validate feature matrix matches model expectations"""
        if X.shape[1] != self.exact_feature_count:
            raise ValueError(
                f"Feature dimension mismatch. Expected {self.exact_feature_count}, got {X.shape[1]}. "
                f"Required features in order: {self.feature_names}"
            )

    def _process_core_features(self, aligned: pd.DataFrame, transaction: pd.DataFrame, warnings: List[str]) -> None:
        """
        Process core features from raw transaction data
        """
        core_features = {
            FeatureNames.AMOUNT,
            FeatureNames.IP_ADDRESS,
            FeatureNames.DEVICE_ID,
            FeatureNames.USER_ID,
            FeatureNames.USER_AGE,
            FeatureNames.ACCOUNT_AGE,
            FeatureNames.TRANSACTION_COUNT,
            FeatureNames.IS_FOREIGN_IP,
            FeatureNames.DEVICE_CHANGE_FLAG
        }
        
        for feat in core_features:
            try:
                if feat in transaction.columns:
                    # Handle special cases
                    if feat == FeatureNames.IP_ADDRESS:
                        aligned[feat] = float(hash(str(transaction[feat].iloc[0])) % (10**8))
                    elif feat == FeatureNames.DEVICE_ID:
                        aligned[feat] = float(hash(str(transaction[feat].iloc[0])) % (10**8))
                    elif feat == FeatureNames.USER_ID:
                        aligned[feat] = float(hash(str(transaction[feat].iloc[0])) % (10**8))
                    else:
                        aligned[feat] = float(transaction[feat].iloc[0])
                else:
                    aligned[feat] = float(FALLBACK_VALUES.get(feat, 0))
            except Exception as e:
                warnings.append(f"Error processing {feat}: {str(e)}")
                aligned[feat] = float(FALLBACK_VALUES.get(feat, 0))

    def _generate_time_features(self, df: pd.DataFrame, timestamp: Any, warnings: List[str]) -> None:
        """Generate all time-based features from timestamp"""
        try:
            ts = pd.to_datetime(timestamp)
            
            # Basic time features
            df[FeatureNames.HOUR_OF_DAY] = float(ts.hour)
            df[FeatureNames.DAY_OF_WEEK] = float(ts.dayofweek)
            df[FeatureNames.IS_WEEKEND] = float(ts.dayofweek >= 5)
            
            # Cyclical features
            df[FeatureNames.HOUR_SIN] = float(np.sin(2 * np.pi * ts.hour/24))
            df[FeatureNames.HOUR_COS] = float(np.cos(2 * np.pi * ts.hour/24))
            df[FeatureNames.DAY_SIN] = float(np.sin(2 * np.pi * ts.dayofweek/7))
            df[FeatureNames.DAY_COS] = float(np.cos(2 * np.pi * ts.dayofweek/7))
            
            # Time categories
            df[FeatureNames.TRANSACTION_HOUR_CATEGORY] = float(ts.hour // 6)
            
            # Time since last transaction (simplified)
            if FeatureNames.TIME_SINCE_LAST in df.columns:
                df[FeatureNames.TIME_SINCE_LAST] = float(FALLBACK_VALUES.get(FeatureNames.TIME_SINCE_LAST, 1440))
                
        except Exception as e:
            warnings.append(f"Time feature generation failed: {str(e)}")
            logger.warning(f"Could not generate time features: {str(e)}")

    def _generate_derived_features(self, df: pd.DataFrame, warnings: List[str]):
        """Generate only the expected derived features"""
        try:
            # Only generate features that are in our expected feature_names
            expected_features = set(self.feature_names)
            
            # Amount transformations
            if 'amount' in df.columns:
                if 'transaction_amount_log' in expected_features:
                    df['transaction_amount_log'] = float(np.log1p(df['amount'].iloc[0]))
                
                # Only create amount_to_avg_ratio if it's in our expected features
                if 'amount_to_avg_ratio' in expected_features:
                    avg = float(df.get('user_avg_transaction', pd.Series([1])).iloc[0]) 
                    df['amount_to_avg_ratio'] = float(df['amount'].iloc[0] / max(1, avg))
            
            # User behavior features
            if 'user_tenure_days' in expected_features:
                df['user_tenure_days'] = float(df.get('account_age_days', pd.Series([365])).iloc[0])
            
            if 'avg_amount_7d' in expected_features:
                df['avg_amount_7d'] = float(df.get('amount', pd.Series([0])).iloc[0] * 0.9)
                
            if 'ip_risk_category' in expected_features:
                df['ip_risk_category'] = float(0 if df.get('is_foreign_ip', pd.Series([0])).iloc[0] else 1)
                
        except Exception as e:
            warnings.append(f"Derived feature generation failed: {str(e)}")

    def _align_features(self, transaction: Union[Dict, pd.DataFrame]) -> Tuple[pd.DataFrame, List[str]]:
        """Complete feature alignment ensuring exactly 26 features"""
        warnings = []
        try:
            if isinstance(transaction, dict):
                transaction = pd.DataFrame([transaction])
            
            # Initialize with exactly the expected features
            aligned = pd.DataFrame(columns=self.feature_names, index=[0])
            for feat in self.feature_names:
                aligned[feat] = float(FALLBACK_VALUES.get(feat, 0))
            
            # Process features
            self._process_core_features(aligned, transaction, warnings)
            
            if 'timestamp' in transaction.columns:
                self._generate_time_features(aligned, transaction['timestamp'].iloc[0], warnings)
            
            self._generate_derived_features(aligned, warnings)
            
            # Verify we have exactly the expected columns
            extra_cols = set(aligned.columns) - set(self.feature_names)
            if extra_cols:
                aligned = aligned[self.feature_names]  # Drop any extra columns
                warnings.append(f"Removed extra columns: {extra_cols}")
            
            return aligned, warnings
        except Exception as e:
            logger.error(f"Feature alignment failed: {str(e)}")
            raise ValueError(f"Feature processing error: {str(e)}")

    def _get_dynamic_threshold(self, transaction: pd.DataFrame) -> float:
        """Calculate dynamic threshold based on transaction characteristics"""
        try:
            threshold = THRESHOLD_CONFIG.default
            transaction_dict = transaction.iloc[0].to_dict()
            
            rules = [
                THRESHOLD_CONFIG.high_amount_rule,
                THRESHOLD_CONFIG.trusted_user_rule,
                THRESHOLD_CONFIG.new_user_rule,
                THRESHOLD_CONFIG.high_risk_country_rule
            ]
            
            for rule in rules:
                if eval(rule.condition, {'transaction_dict': transaction_dict}):
                    threshold = rule.threshold
            
            return float(threshold)
            
        except Exception as e:
            logger.warning(f"Threshold calculation failed, using default: {str(e)}")
            return self.optimal_threshold

    def predict_with_dynamic_threshold(self, transaction: Union[Dict, pd.DataFrame]) -> PredictionResult:
        """Enhanced prediction with complete feature validation"""
        start_time = time.time()
        warnings = []
        
        try:
            # Process and validate features
            transaction_df, feature_warnings = self._align_features(transaction)
            warnings.extend(feature_warnings)
            
            # Prepare feature matrix in exact training order
            X_pred = transaction_df[self.feature_names].values.astype('float32')
            self._validate_feature_matrix(X_pred)
            
            # Make prediction
            proba = float(self.model.predict_proba(X_pred)[0, 1])
            threshold = self._get_dynamic_threshold(transaction_df)
            
            # Generate explanations
            shap_values = None
            if self.explainer:
                try:
                    shap_vals = self.explainer.shap_values(X_pred)
                    # Handle both binary and single-class cases
                    if isinstance(shap_vals, list):
                        # Binary classification - use index 1 for fraud class
                        if len(shap_vals) > 1:
                            shap_vals = shap_vals[1]
                        else:
                            shap_vals = shap_vals[0]
                    shap_values = {
                        'values': [float(x) for x in shap_vals[0]],
                        'feature_names': self.feature_names,
                        'base_value': float(self.explainer.expected_value[1] 
                                    if isinstance(self.explainer.expected_value, list)
                                    else self.explainer.expected_value)
                    }
                except Exception as e:
                    logger.warning(f"SHAP explanation failed: {str(e)}")
            
            return PredictionResult(
                prediction=int(proba >= threshold),
                probability=proba,
                threshold_used=threshold,
                features=self._prepare_feature_output(transaction_df),
                shap_values=shap_values,
                warnings=warnings,
                prediction_time=float(time.time() - start_time),
                model_version=self.model_version,
                behavioral_anomalies=self._detect_behavioral_anomalies(transaction_df.iloc[0].to_dict())
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            raise ValueError(f"Prediction error: {str(e)}")

    def _prepare_feature_output(self, df: pd.DataFrame) -> Dict[str, float]:
        """Convert features to JSON-serializable format"""
        return {
            feat: float(df[feat].iloc[0])
            for feat in FeatureNames.ENGINEERED_FEATURES
            if feat in df.columns
        }

    def _detect_behavioral_anomalies(self, features: Dict) -> List[str]:
        """Detect behavioral anomalies in transaction"""
        anomalies = []
        
        # Velocity anomalies
        if features.get(FeatureNames.TRANSACTION_COUNT, 0) > 15:
            anomalies.append("High transaction frequency (>15 in 7 days)")
            
        # Amount anomalies
        if features.get(FeatureNames.AMOUNT, 0) > 5000 and features.get(FeatureNames.USER_AVG_TRANSACTION, 0) < 100:
            anomalies.append("Large amount compared to user's typical transactions")
            
        # Time anomalies
        if features.get(FeatureNames.HOUR_OF_DAY, 12) in range(1, 5) and features.get(FeatureNames.AMOUNT, 0) > 1000:
            anomalies.append("Unusually large transaction during early morning hours")
            
        return anomalies

    def predict(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simplified prediction interface"""
        result = self.predict_with_dynamic_threshold(transaction_data)
        return result.to_dict()