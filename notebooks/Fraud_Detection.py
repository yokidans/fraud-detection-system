# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocess import FraudDataPreprocessor
from src.train import FraudDetectionModel
from src.explainability import ModelExplainer
from src.config import MODEL_CONFIG, DATA_PATHS, FEATURES
from src.utils import load_data, save_data
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Load and Explore Data
logger.info("Loading data...")
df = load_data(DATA_PATHS['raw_data'])

# Initial exploration
print(f"Data shape: {df.shape}")
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nClass distribution:")
print(df[FEATURES['target']].value_counts(normalize=True))

# 2. Data Preprocessing
logger.info("Preprocessing data...")
preprocessor = FraudDataPreprocessor(
    numeric_features=FEATURES['numeric_features'],
    categorical_features=FEATURES['categorical_features']
)

X = df.drop(FEATURES['target'], axis=1)
y = df[FEATURES['target']]

X_processed = preprocessor.fit_transform(X)
feature_names = preprocessor.get_feature_names()

# Save processed data
processed_df = pd.DataFrame.sparse.from_spmatrix(X_processed, columns=feature_names)
processed_df[FEATURES['target']] = y.values
save_data(processed_df, DATA_PATHS['processed_data'])

# 3. Model Training
logger.info("Training models...")
fraud_model = FraudDetectionModel(MODEL_CONFIG)

# Split data
X_train, X_test, y_train, y_test = fraud_model.train_test_split(
    X_processed, y, test_size=0.2
)

# Train ensemble model
best_model = fraud_model.train_ensemble(X_train, y_train, optimize=True)

# Evaluate on test set
test_metrics = fraud_model.evaluate_model(best_model, X_test, y_test)
logger.info(f"Test ROC AUC: {test_metrics['roc_auc']:.4f}")

# 4. Model Explainability
logger.info("Generating explanations...")
explainer = ModelExplainer(best_model, preprocessor, feature_names)

# SHAP analysis
shap_values = explainer.shap_analysis(X_train)
explainer.plot_shap_summary(shap_values)

# LIME explanation for a specific instance
lime_exp = explainer.lime_explanation(X_train, instance_idx=0)
lime_exp.show_in_notebook()

# Permutation importance
perm_importance = explainer.permutation_importance(X_test, y_test)
explainer.plot_feature_importance(perm_importance.importances_mean)

# 5. Save Model
logger.info("Saving model...")
fraud_model.save_model(best_model, "models/fraud_detection_model.pkl")