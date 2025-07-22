MODEL_CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'cv_folds': 5
}

DATA_PATHS = {
    'raw_data': 'data/raw/transactions.csv',
    'processed_data': 'data/processed/transactions_processed.csv'
}

FEATURES = {
    'target': 'is_fraud',
    'numeric_features': ['amount', 'transaction_hour', 'days_since_last_transaction'],
    'categorical_features': ['merchant_category', 'payment_method', 'country']
}