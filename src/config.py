"""
Enterprise Fraud Detection Configuration Module - Final
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, validator, Field
import os
import logging
from datetime import datetime

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========== Type Definitions ==========
class ThresholdRule(BaseModel):
    condition: str
    threshold: float
    description: str

class ThresholdConfig(BaseModel):
    default: float = Field(0.5, gt=0, lt=1)
    high_amount_rule: ThresholdRule = ThresholdRule(
        condition="transaction_dict.get('amount', 0) > 1000",
        threshold=0.3,
        description="High amount transactions (>$1000)"
    )
    trusted_user_rule: ThresholdRule = ThresholdRule(
        condition="transaction_dict.get('user_transaction_count', 0) > 50",
        threshold=0.6,
        description="Trusted users with >50 transactions"
    )
    new_user_rule: ThresholdRule = ThresholdRule(
        condition="transaction_dict.get('user_transaction_count', 0) <= 3",
        threshold=0.4,
        description="New users with ≤3 transactions"
    )
    high_risk_country_rule: ThresholdRule = ThresholdRule(
        condition="transaction_dict.get('country_risk_score', 0) > 0.7",
        threshold=0.35,
        description="High risk country IP addresses"
    )

class DataPaths(BaseModel):
    fraud_data: Path
    ip_mapping: Path
    creditcard_data: Path
    processed_data: Path
    processed_target: Path
    model_explanations: Path


class FeatureNames:
    """
    Complete feature specification for fraud detection model with exactly 26 features.
    
    This class defines all input features, engineered features, and their exact ordering
    that must match the model training configuration. The ordering is critical for
    proper model predictions.
    
    Attributes:
        REQUIRED_INPUTS: Raw input fields that must be provided for each transaction
        AMOUNT-USER_AGE: Core feature definitions with consistent naming
        ENGINEERED_FEATURES: List of features derived from raw inputs
        FEATURE_ORDER: Exact feature ordering matching model training
        TARGET: The prediction target variable
    """
    
    # Required input fields from transaction data (10 features)
    REQUIRED_INPUTS = [
        'amount',               # Transaction amount in currency
        'ip_address',           # IP address of transaction origin
        'device_id',            # Unique device identifier
        'user_id',              # Unique user identifier  
        'timestamp',            # Exact time of transaction
        'user_age',             # Age of user in years
        'account_age_days',    # Days since account creation
        'transaction_count_7d', # User's transaction count in last 7 days
        'is_foreign_ip',       # Boolean if IP is outside user's home country
        'device_change_flag'    # Boolean if device changed from previous transaction
    ]
    
    # Core transaction features (10) - Constants for consistent referencing
    AMOUNT = 'amount'
    IP_ADDRESS = 'ip_address'
    DEVICE_ID = 'device_id'
    USER_ID = 'user_id'
    TIMESTAMP = 'timestamp'
    USER_AGE = 'user_age'
    ACCOUNT_AGE = 'account_age_days'
    TRANSACTION_COUNT = 'transaction_count_7d'
    IS_FOREIGN_IP = 'is_foreign_ip'
    DEVICE_CHANGE_FLAG = 'device_change_flag'
    
    # Engineered features (16) - Derived from raw inputs
    ENGINEERED_FEATURES = [
        'hour_of_day',                  # Extracted hour from timestamp (0-23)
        'day_of_week',                  # Extracted day of week (0-6)
        'is_weekend',                   # Boolean for weekend transactions
        'time_since_last_transaction',   # Seconds since user's last transaction
        'purchase_frequency_24h',       # User's transactions in last 24 hours
        'country_risk_score',           # Risk score based on IP country
        'billing_shipping_mismatch',    # Boolean for address mismatch
        'user_avg_transaction',         # User's historical average amount
        'transaction_amount_log',       # Log-transformed amount
        'hour_sin',                     # Cyclical encoding of hour
        'hour_cos',                     # Cyclical encoding of hour  
        'day_sin',                      # Cyclical encoding of day
        'day_cos',                      # Cyclical encoding of day
        'user_tenure_days',             # Days since user first transaction
        'avg_amount_7d',                # User's average amount last 7 days
        'transaction_hour_category',     # Categorized hour (morning/afternoon/evening/night)
        'ip_risk_category'             # Risk category from IP intelligence
    ]
    
    # Feature aliases for consistent referencing
    TRANSACTION_AMOUNT_LOG = 'transaction_amount_log'
    HOUR_OF_DAY = 'hour_of_day'
    DAY_OF_WEEK = 'day_of_week'
    HOUR_SIN = 'hour_sin'
    HOUR_COS = 'hour_cos'
    DAY_SIN = 'day_sin'
    DAY_COS = 'day_cos'
    TIME_SINCE_LAST = 'time_since_last_transaction'
    PURCHASE_FREQ = 'purchase_frequency_24h'
    COUNTRY_RISK = 'country_risk_score'
    BILLING_MISMATCH = 'billing_shipping_mismatch'
    USER_AVG_TRANSACTION = 'user_avg_transaction'
    USER_TENURE_DAYS = 'user_tenure_days'
    AVG_AMOUNT_7D = 'avg_amount_7d'
    TRANSACTION_HOUR_CATEGORY = 'transaction_hour_category'
    IP_RISK_CATEGORY = 'ip_risk_category'
    IS_WEEKEND = 'is_weekend'
    
    # Must match EXACTLY the order used during model training (26 features)
    FEATURE_ORDER = [
        # Amount features (2)
        AMOUNT, 
        TRANSACTION_AMOUNT_LOG,
        
        # Time features (6)
        HOUR_OF_DAY, 
        DAY_OF_WEEK, 
        HOUR_SIN, 
        HOUR_COS, 
        DAY_SIN, 
        DAY_COS,
        
        # Transaction patterns (2)
        TIME_SINCE_LAST, 
        PURCHASE_FREQ,
        
        # Risk indicators (2)
        COUNTRY_RISK, 
        IS_FOREIGN_IP,
        
        # Device/user patterns (2)
        DEVICE_CHANGE_FLAG, 
        BILLING_MISMATCH,
        
        # User history (4)
        TRANSACTION_COUNT, 
        USER_AVG_TRANSACTION,
        USER_TENURE_DAYS, 
        AVG_AMOUNT_7D,
        
        # Categorical features (4)
        TRANSACTION_HOUR_CATEGORY, 
        ACCOUNT_AGE,
        IP_RISK_CATEGORY, 
        IS_WEEKEND,
        
        # Identifiers (3)
        IP_ADDRESS, 
        DEVICE_ID, 
        USER_ID,
        
        # Additional feature to reach 26
        USER_AGE  # Added as the 26th feature
    ]
    
    # Target variable for model training
    TARGET = 'is_fraud'  # Binary flag indicating fraudulent transaction
    
    @classmethod
    def get_ordered_features(cls) -> List[str]:
        """
        Returns exactly 26 features in PREDICTION ORDER required by the model.
        
        Returns:
            List[str]: Ordered list of 26 feature names
            
        Raises:
            ValueError: If feature count doesn't match expected 26 features
        """
        if len(cls.FEATURE_ORDER) != 26:
            raise ValueError(
                f"Feature order must contain exactly 26 features, got {len(cls.FEATURE_ORDER)}. "
                "Model will not produce correct results without exact feature matching."
            )
        return cls.FEATURE_ORDER

class ModelConfig(BaseModel):
    path: Path
    version: str
    required_features: List[str]
    expected_auc: float = Field(..., gt=0.5, lt=1)

# ========== Configuration ==========
BASE_DIR = Path(__file__).resolve().parent.parent
ENV = os.getenv('ENVIRONMENT', 'development')

def get_data_paths(env: str) -> DataPaths:
    """Environment-aware path configuration"""
    base = {
        'fraud_data': BASE_DIR / 'data/raw/Fraud_Data.csv',
        'ip_mapping': BASE_DIR / 'data/raw/IpAddress_to_Country.csv',
        'creditcard_data': BASE_DIR / 'data/raw/creditcard.csv',
        'processed_data': BASE_DIR / 'data/processed/features.parquet',
        'processed_target': BASE_DIR / 'data/processed/target.parquet',
        'model_explanations': BASE_DIR / 'models/explanations'
    }
    
    if env == 'production':
        base.update({
            'fraud_data': Path('/mnt/data/Fraud_Data.csv'),
            'ip_mapping': Path('/mnt/data/IpAddress_to_Country.csv'),
            'creditcard_data': Path('/mnt/data/creditcard.csv')
        })
    
    return DataPaths(**base)

DATA_PATHS = get_data_paths(ENV)

# Create required directories
for path in DATA_PATHS.dict().values():
    if path.suffix:  # It's a file
        path.parent.mkdir(parents=True, exist_ok=True)
    else:  # It's a directory
        path.mkdir(parents=True, exist_ok=True)

# Threshold configuration
THRESHOLD_CONFIG = ThresholdConfig()

# Model configuration
MODEL_CONFIG = ModelConfig(
    path=BASE_DIR / 'models/fraud_detection_model.pkl',
    version='3.1.0',
    required_features=FeatureNames.get_ordered_features(),
    expected_auc=0.92
)

# Fallback values
FALLBACK_VALUES = {
    FeatureNames.AMOUNT: 0.0,
    FeatureNames.TIME_SINCE_LAST: 1440.0,
    FeatureNames.PURCHASE_FREQ: 1.0,
    FeatureNames.IS_FOREIGN_IP: 0.0,
    FeatureNames.DEVICE_CHANGE_FLAG: 0.0,
    FeatureNames.TRANSACTION_COUNT: 10.0,
    FeatureNames.USER_AVG_TRANSACTION: 100.0,
    FeatureNames.COUNTRY_RISK: 0.5,
    FeatureNames.ACCOUNT_AGE: 365.0,
    FeatureNames.USER_AGE: 30.0
}

# Feature and target configuration
FEATURES = FeatureNames.get_ordered_features()
TARGET = FeatureNames.TARGET
MODEL_PATH = MODEL_CONFIG.path

def validate_configuration() -> None:
    """Validate all configuration components"""
    errors = []
    
    # Validate file paths
    required_files = [
        DATA_PATHS.fraud_data,
        DATA_PATHS.ip_mapping,
        DATA_PATHS.creditcard_data
    ]
    
    for file in required_files:
        if not file.exists():
            errors.append(f"Required file not found: {file}")
    
    # Validate feature count
    if len(FeatureNames.get_ordered_features()) != 26:
        errors.append(f"Feature count must be 26, got {len(FeatureNames.get_ordered_features())}")
    
    # Validate threshold logic
    if THRESHOLD_CONFIG.default >= THRESHOLD_CONFIG.trusted_user_rule.threshold:
        errors.append("Default threshold should be lower than trusted user threshold")
    
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"• {e}" for e in errors)
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info("✅ Configuration validation passed")

# Run validation when module is imported
if __name__ != '__main__':
    try:
        validate_configuration()
    except Exception as e:
        if ENV == 'development':
            logger.warning(f"Running in development mode with validation errors: {str(e)}")
        else:
            raise

__all__ = [
    'ThresholdConfig',
    'DataPaths',
    'FeatureNames',
    'ModelConfig',
    'THRESHOLD_CONFIG',
    'DATA_PATHS',
    'MODEL_CONFIG',
    'FALLBACK_VALUES',
    'FEATURES',
    'TARGET',
    'MODEL_PATH'
]