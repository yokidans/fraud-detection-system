"""
Enhanced Data Processing Pipeline with:
- Robust data loading
- Comprehensive validation
- Flexible output formats
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from typing import Tuple, Dict, Any
from src.data.preprocessing import DataPreprocessor
from src.config import PREPROCESSOR_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load data with robust parsing"""
    try:
        logger.info(f"Loading data from {file_path}")
        
        df = pd.read_csv(
            file_path,
            parse_dates=['signup_time', 'purchase_time'],
            date_format='%Y-%m-%d %H:%M:%S',
            dtype={
                'user_id': 'str',
                'purchase_value': 'float64',
                'device_id': 'str',
                'ip_address': 'str',
                'class': 'int8'
            }
        )
        
        if df.empty:
            raise ValueError("Loaded empty dataframe")
            
        return df, {
            'time_range': (df['purchase_time'].min(), df['purchase_time'].max()),
            'user_count': df['user_id'].nunique()
        }
        
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}", exc_info=True)
        raise

def validate_data(df: pd.DataFrame) -> None:
    """Validate data structure and quality"""
    required_columns = {
        'user_id': ['object', 'str'],
        'signup_time': ['datetime64[ns]'],
        'purchase_time': ['datetime64[ns]'],
        'purchase_value': ['float64', 'int64'],
        'class': ['int8', 'int16', 'int32', 'int64']
    }
    
    logger.info("Validating data...")
    
    # Check columns
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # Check dtypes
    for col, allowed in required_columns.items():
        actual = str(df[col].dtype)
        if not any(t in actual for t in allowed):
            raise ValueError(f"Column '{col}' has invalid type: {actual}")
    
    logger.info("✅ Data validation passed")

def save_results(X: np.ndarray, y: np.ndarray, features: list, output_dir: str) -> None:
    """Save processed data with format fallback"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame(X, columns=features)
        df['target'] = y
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Try multiple formats
        try:
            path = f"{output_dir}/processed_{timestamp}.parquet"
            df.to_parquet(path, index=False, engine='pyarrow')
            logger.info(f"Saved Parquet to {path}")
        except Exception as parquet_error:
            logger.warning(f"Parquet save failed: {str(parquet_error)}")
            path = f"{output_dir}/processed_{timestamp}.csv.gz"
            df.to_csv(path, index=False, compression='gzip')
            logger.info(f"Saved compressed CSV to {path}")
            
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")
        raise

def main():
    try:
        # Load and validate
        df, meta = load_data('data/raw/Fraud_Data.csv')
        validate_data(df)
        
        # Process
        logger.info("Processing data...")
        preprocessor = DataPreprocessor(config=PREPROCESSOR_CONFIG)
        X, y, features = preprocessor.preprocess_data(df, target_col='class')
        
        # Save
        save_results(X, y, features, 'data/processed')
        logger.info("✅ Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()