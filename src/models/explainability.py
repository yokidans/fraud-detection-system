import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from sklearn.inspection import permutation_importance
from sklearn.base import is_classifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelExplainer:
    def __init__(self, model, preprocessor, feature_names):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        self.explainer = None
        
    def _get_transformed_feature_names(self):
        """Get feature names after preprocessing"""
        try:
            # Handle OneHotEncoded features
            if hasattr(self.preprocessor, 'named_transformers_'):
                ohe = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
                cat_features = ohe.get_feature_names_out(self.preprocessor.feature_names_in_[
                    self.preprocessor.transformers_[1][2]])
                    
                num_features = self.preprocessor.transformers_[0][2]
                return list(num_features) + list(cat_features)
            return self.feature_names
        except Exception as e:
            logger.warning(f"Could not get transformed feature names: {str(e)}")
            return self.feature_names
            
    def shap_analysis(self, X, sample_size=100):
        """Perform SHAP analysis on the model"""
        try:
            # Sample data if too large
            if len(X) > sample_size:
                X_sample = X[np.random.choice(X.shape[0], sample_size, replace=False)]
            else:
                X_sample = X
                
            # Create explainer based on model type
            if hasattr(self.model, 'predict_proba'):
                explainer = shap.Explainer(self.model.predict_proba, X_sample)
                shap_values = explainer(X_sample)
                return shap_values[..., 1]  # Get SHAP values for positive class
            else:
                explainer = shap.Explainer(self.model, X_sample)
                return explainer(X_sample)
                
        except Exception as e:
            logger.error(f"Error in SHAP analysis: {str(e)}")
            raise
            
    def plot_shap_summary(self, shap_values, feature_names=None):
        """Plot SHAP summary plot"""
        try:
            if feature_names is None:
                feature_names = self._get_transformed_feature_names()
                
            plt.figure()
            shap.summary_plot(shap_values, feature_names=feature_names)
            plt.tight_layout()
            return plt
        except Exception as e:
            logger.error(f"Error plotting SHAP summary: {str(e)}")
            raise
            
    def lime_explanation(self, X, instance_idx, num_features=5):
        """Generate LIME explanation for a specific instance"""
        try:
            feature_names = self._get_transformed_feature_names()
            
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X, 
                feature_names=feature_names,
                class_names=['Not Fraud', 'Fraud'],
                mode='classification' if is_classifier(self.model) else 'regression',
                discretize_continuous=True)
                
            exp = explainer.explain_instance(
                X[instance_idx], 
                self.model.predict_proba,
                num_features=num_features)
                
            return exp
        except Exception as e:
            logger.error(f"Error generating LIME explanation: {str(e)}")
            raise
            
    def permutation_importance(self, X, y, n_repeats=10, random_state=None):
        """Calculate permutation importance"""
        try:
            result = permutation_importance(
                self.model, X, y,
                n_repeats=n_repeats,
                random_state=random_state,
                n_jobs=-1)
                
            return result
        except Exception as e:
            logger.error(f"Error calculating permutation importance: {str(e)}")
            raise
            
    def plot_feature_importance(self, importance, feature_names=None):
        """Plot feature importance"""
        try:
            if feature_names is None:
                feature_names = self._get_transformed_feature_names()
                
            if len(importance) != len(feature_names):
                logger.warning("Feature importance length doesn't match feature names")
                return None
                
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            plt.barh(importance_df['Feature'][:20], importance_df['Importance'][:20])
            plt.xlabel('Importance')
            plt.title('Feature Importance')
            plt.tight_layout()
            return plt
        except Exception as e:
            logger.error(f"Error plotting feature importance: {str(e)}")
            raise