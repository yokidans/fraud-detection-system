import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import FeatureHasher
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import logging
from scipy.sparse import csr_matrix, save_npz
from typing import List, Tuple, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleImputerWithNames(SimpleImputer):
    """Enhanced SimpleImputer that preserves feature names"""
    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        if input_features is None:
            return [f"feature_{i}" for i in range(self.n_features_in_)]
        return input_features

class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for fraud detection.
    Handles numerical, categorical, and high-cardinality features,
    datetime conversion, IP address processing, and class imbalance.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the preprocessor with configuration.
        
        Args:
            config (dict): Configuration dictionary containing:
                - missing_values_strategy: Strategy for handling missing values
                - sampling_strategy: Strategy for handling class imbalance
                - random_state: Random seed for reproducibility
        """
        self.config = config
        self.preprocessor = None
        self.categorical_features: List[str] = []
        self.numerical_features: List[str] = []
        self.high_cardinality_features: List[str] = []
        self.feature_names: List[str] = []
        
    def _convert_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert datetime columns to numeric timestamps"""
        datetime_cols = ['signup_time', 'purchase_time']
        for col in datetime_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col]).astype('int64') // 10**9
                except Exception as e:
                    logger.warning(f"Failed to convert {col} to timestamp: {str(e)}")
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
        
    def _preprocess_ip(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess IP address to reduce cardinality with robust type handling"""
        if 'ip_address' in df.columns:
            try:
                # Ensure IP address is string type
                df['ip_address'] = df['ip_address'].astype(str)
                # Extract first octet safely
                df['ip_prefix'] = df['ip_address'].str.split('.').str[0]
                # Convert to numeric if possible, otherwise categorize
                try:
                    df['ip_prefix'] = pd.to_numeric(df['ip_prefix'])
                except ValueError:
                    df['ip_prefix'] = df['ip_prefix'].astype('category')
                df = df.drop('ip_address', axis=1)
            except Exception as e:
                logger.warning(f"IP address preprocessing failed: {str(e)}")
                df = df.drop('ip_address', axis=1)
        return df
        
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with column-specific strategies"""
        try:
            numeric_cols = df.select_dtypes(include=np.number).columns
            non_numeric_cols = df.select_dtypes(exclude=np.number).columns
            
            if self.config.get('missing_values_strategy') == 'drop':
                return df.dropna()
            else:
                # Use our enhanced imputer that preserves feature names
                num_imputer = SimpleImputerWithNames(strategy='median')
                df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
                
                if len(non_numeric_cols) > 0:
                    cat_imputer = SimpleImputerWithNames(strategy='most_frequent', fill_value='missing')
                    df[non_numeric_cols] = cat_imputer.fit_transform(df[non_numeric_cols])
                
                return df
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            raise
            
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows while preserving the first occurrence"""
        return df.drop_duplicates(keep='first')
        
    def _correct_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure correct data types for all columns"""
        type_map = {
            'user_id': 'str',
            'purchase_value': 'float32',
            'age': 'int32',
            'class': 'int32',
            'device_id': 'str',
            'ip_address': 'str'
        }
        for col, dtype in type_map.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except Exception as e:
                    logger.warning(f"Failed to convert {col} to {dtype}: {str(e)}")
                    if col not in ['user_id', 'device_id']:
                        df = df.drop(col, axis=1)
        return df
        
    def _get_feature_names(self, column_transformer: ColumnTransformer) -> List[str]:
        """Get feature names from ColumnTransformer with comprehensive handling"""
        feature_names = []
        
        for name, transformer, features in column_transformer.transformers_:
            if transformer == 'drop':
                continue
                
            if name == 'hash':
                # Special handling for FeatureHasher
                names = [f'hash_{i}' for i in range(transformer.named_steps['hasher'].n_features)]
            elif hasattr(transformer, 'get_feature_names_out'):
                # Modern sklearn versions (>=1.0)
                names = transformer.get_feature_names_out(features)
            elif hasattr(transformer, 'get_feature_names'):
                # Older sklearn versions
                names = transformer.get_feature_names(features)
            elif hasattr(transformer, 'named_steps'):
                # Handle pipeline components
                names = features
                for step_name, step in transformer.named_steps.items():
                    if hasattr(step, 'get_feature_names_out'):
                        names = step.get_feature_names_out(names)
                    elif hasattr(step, 'get_feature_names'):
                        names = step.get_feature_names(names)
            else:
                # Fallback to original feature names
                names = features
                
            feature_names.extend(names)
            
        return feature_names
        
    def _identify_feature_types(self, X: pd.DataFrame):
        """Identify feature types including high-cardinality features"""
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        
        # Identify high-cardinality features (>100 unique values)
        self.high_cardinality_features = []
        regular_categorical = []
        
        for col in self.categorical_features:
            if X[col].nunique() > 100:
                self.high_cardinality_features.append(col)
            else:
                regular_categorical.append(col)
        
        self.categorical_features = regular_categorical
        
        logger.info(f"Numerical features: {self.numerical_features}")
        logger.info(f"Regular categorical features: {self.categorical_features}")
        logger.info(f"High-cardinality features: {self.high_cardinality_features}")
        
    def preprocess_data(self, df: pd.DataFrame, target_col: str) -> Tuple[Union[np.ndarray, csr_matrix], np.ndarray, List[str]]:
        """
        Main preprocessing method that handles the entire pipeline.
        
        Args:
            df: Input DataFrame containing raw data
            target_col: Name of the target column
            
        Returns:
            tuple: (processed_features, target, feature_names)
        """
        try:
            logger.info("Starting data preprocessing...")
            
            # Initial cleaning and type conversion
            df = self._correct_data_types(df)
            df = self._convert_datetime(df)
            df = self._preprocess_ip(df)
            df = self._handle_missing_values(df)
            df = self._remove_duplicates(df)
            
            # Separate features and target
            X = df.drop(columns=[target_col])
            y = df[target_col].values
            
            # Identify feature types
            self._identify_feature_types(X)
            
            # Create preprocessing pipelines with our enhanced imputer
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputerWithNames(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            # Regular categorical features (low cardinality)
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputerWithNames(strategy='most_frequent', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=True))
            ])
            
            # High-cardinality features - using FeatureHasher
            hasher = FeatureHasher(n_features=50, input_type='string')
            high_card_transformer = Pipeline(steps=[
                ('imputer', SimpleImputerWithNames(strategy='constant', fill_value='missing')),
                ('hasher', hasher)
            ])
            
            # Create the complete ColumnTransformer
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, self.numerical_features),
                    ('cat', categorical_transformer, self.categorical_features),
                    ('hash', high_card_transformer, self.high_cardinality_features)
                ],
                remainder='drop'  # Drop any columns not explicitly transformed
            )
            
            # Fit and transform the data
            X_processed = self.preprocessor.fit_transform(X)
            self.feature_names = self._get_feature_names(self.preprocessor)
            
            # Handle class imbalance
            if self.config.get('sampling_strategy') == 'SMOTE':
                smote = SMOTE(random_state=self.config.get('random_state', 42))
                X_processed, y = smote.fit_resample(X_processed, y)
            elif self.config.get('sampling_strategy') == 'undersample':
                rus = RandomUnderSampler(random_state=self.config.get('random_state', 42))
                X_processed, y = rus.fit_resample(X_processed, y)
            
            logger.info(f"Processed data shape: {X_processed.shape}")
            logger.info(f"Number of features: {len(self.feature_names)}")
            logger.info("Preprocessing completed successfully")
            
            return X_processed, y, self.feature_names
            
        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}", exc_info=True)
            raise RuntimeError(f"Preprocessing failed: {str(e)}") from e