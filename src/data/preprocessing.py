import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
import logging
from typing import List, Tuple, Dict, Union, Set
from collections import OrderedDict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts time-based features with improved feature naming"""
    def __init__(self, time_config: Dict):
        self.time_config = time_config
        self.feature_names_ = OrderedDict([
            ('time_since_signup', 'time_since_signup'),
            ('purchase_hour', 'purchase_hour'),
            ('purchase_day_of_week', 'purchase_day_of_week'),
            ('is_weekend', 'is_weekend'),
            ('time_since_last_purchase', 'time_since_last_purchase'),
            ('hourly_purchase_pattern', 'hourly_purchase_pattern')
        ])
        
        # Validate required config fields
        required_fields = ['signup_time_col', 'purchase_time_col', 'time_format', 'id_column']
        for field in required_fields:
            if field not in time_config:
                raise ValueError(f"Missing required field in time_config: {field}")
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        df = X.copy()
        # Convert timestamp columns with error handling
        for col in [self.time_config['signup_time_col'], self.time_config['purchase_time_col']]:
            try:
                df[col] = pd.to_datetime(df[col], format=self.time_config['time_format'])
            except Exception as e:
                raise ValueError(f"Failed to parse time column '{col}': {str(e)}")
        
        # Create time features with null checks
        df['time_since_signup'] = (df[self.time_config['purchase_time_col']] - 
                                 df[self.time_config['signup_time_col']]).dt.total_seconds()
        
        df['purchase_hour'] = df[self.time_config['purchase_time_col']].dt.hour
        df['purchase_day_of_week'] = df[self.time_config['purchase_time_col']].dt.dayofweek
        df['is_weekend'] = df['purchase_day_of_week'].isin([5, 6]).astype(int)
        
        # Time since last purchase with group validation
        df = df.sort_values([self.time_config['id_column'], self.time_config['purchase_time_col']])
        time_diff = df.groupby(self.time_config['id_column'])[self.time_config['purchase_time_col']].diff()
        df['time_since_last_purchase'] = df.groupby(self.time_config['id_column'])[self.time_config['purchase_time_col']].diff().dt.total_seconds().fillna(0)
        
        # Hourly purchase pattern with temporary column cleanup
        df['_temp_hour'] = df[self.time_config['purchase_time_col']].dt.hour
        df['hourly_purchase_pattern'] = df.groupby(
            [self.time_config['id_column'], '_temp_hour']
        )[self.time_config['purchase_time_col']].transform('count')
        df = df.drop(columns=['_temp_hour'])
        
        return df

    def get_feature_names_out(self, input_features=None):
        return list(self.feature_names_.values())

class UserBehaviorTransformer(BaseEstimator, TransformerMixin):
    """Enhanced user behavior feature transformer with duplicate prevention"""
    def __init__(self, behavior_config: Dict):
        self.behavior_config = behavior_config
        self.feature_names_ = OrderedDict()
        self.feature_names_['device_change_flag'] = 'device_change_flag'
        
        if 'ip_column' in behavior_config:
            self.feature_names_['ip_change_flag'] = 'ip_change_flag'
        
        # Validate required fields
        required_fields = ['id_column', 'device_column', 'purchase_time_col']
        for field in required_fields:
            if field not in behavior_config:
                raise ValueError(f"Missing required field in behavior_config: {field}")
        
        # Validate and register window features
        if 'window_sizes' in behavior_config:
            for window in behavior_config['window_sizes']:
                if not isinstance(window, (int, float)) or window <= 0:
                    logger.warning(f"Invalid window size {window} - must be positive number")
                    continue
                
                window = int(window)  # Ensure integer hours
                freq_feature = f'purchase_freq_{window}h'
                sum_feature = f'value_sum_{window}h'
                
                if freq_feature in self.feature_names_:
                    logger.warning(f"Duplicate window feature {freq_feature} - skipping")
                else:
                    self.feature_names_[freq_feature] = freq_feature
                    
                    if 'value_column' in behavior_config:
                        if sum_feature in self.feature_names_:
                            logger.warning(f"Duplicate window feature {sum_feature} - skipping")
                        else:
                            self.feature_names_[sum_feature] = sum_feature
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        df = X.copy()
        time_col = self.behavior_config['purchase_time_col']
        id_col = self.behavior_config['id_column']
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        
        # Sort by user and time
        df = df.sort_values([id_col, time_col])
        
        # Window features with proper validation
        if 'window_sizes' in self.behavior_config:
            for window in self.behavior_config['window_sizes']:
                if not isinstance(window, int) or window <= 0:
                    logger.warning(f"Invalid window size {window} - skipping")
                    continue
                    
                try:
                    # Create window features
                    window_str = f'{window}h'
                    
                    # Group by user and create rolling windows
                    grouped = df.groupby(id_col, group_keys=False)
                    
                    # Purchase frequency
                    df[f'purchase_freq_{window}h'] = (grouped[time_col]
                        .rolling(window_str, closed='left')
                        .count()
                        .reset_index(level=0, drop=True))
                    
                    # Value sum if configured
                    if 'value_column' in self.behavior_config:
                        df[f'value_sum_{window}h'] = (grouped[self.behavior_config['value_column']]
                            .rolling(window_str, closed='left')
                            .sum()
                            .reset_index(level=0, drop=True))
                            
                except Exception as e:
                    logger.error(f"Failed to create {window}h window features: {str(e)}")
                    # Fill with zeros if creation fails
                    df[f'purchase_freq_{window}h'] = 0
                    if 'value_column' in self.behavior_config:
                        df[f'value_sum_{window}h'] = 0
        
        return df

    def get_feature_names_out(self, input_features=None):
        return list(self.feature_names_.values())

class DataPreprocessor:
    """Complete preprocessing pipeline with duplicate prevention"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.preprocessor = None
        self.feature_names = []
        self._sample_df = None
        self._feature_registry = set()  # Track all features to prevent duplicates
    
    def _validate_data(self, df: pd.DataFrame, target_col: str) -> None:
        """Validate input data meets requirements"""
        required_cols = set(self.config['feature_config'].get('required_columns', []))
        if not required_cols:
            # Build required columns from config if not explicitly specified
            time_config = self.config['feature_config']['time_features']
            behavior_config = self.config['feature_config']['user_history']
            required_cols = {
                time_config['signup_time_col'],
                time_config['purchase_time_col'],
                behavior_config['id_column'],
                behavior_config['device_column'],
                target_col
            }
            if 'ip_mapping' in self.config['feature_config']:
                ip_config = self.config['feature_config']['ip_mapping']
                if 'ip_column' in ip_config:
                    required_cols.add(ip_config['ip_column'])
        
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
            
        if df[target_col].nunique() != 2:
            logger.warning("Target column may not be properly encoded as binary")
    
    def _create_feature_engineering_pipeline(self) -> Pipeline:
        """Create pipeline for feature engineering steps with duplicate checks"""
        time_extractor = TimeFeatureExtractor(self.config['feature_config']['time_features'])
        behavior_processor = UserBehaviorTransformer(self.config['feature_config']['user_history'])
        
        return Pipeline([
            ('time_extractor', time_extractor),
            ('behavior_processor', behavior_processor)
        ])
    
    def _create_preprocessing_pipeline(self) -> Pipeline:
        """Create the complete preprocessing pipeline with feature validation"""
        feature_engineering = self._create_feature_engineering_pipeline()
        
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        numeric_features = make_column_selector(dtype_include=np.number)
        
        # Get categorical features that exist in the data
        cat_config = self.config['feature_config'].get('categorical_features', {})
        available_categorical = []
        if 'columns' in cat_config:
            available_categorical = [col for col in cat_config['columns'] 
                                  if col in self._sample_df.columns]
        
        transformers = [
            ('num', numeric_transformer, numeric_features)
        ]
        
        if available_categorical:
            transformers.append(('cat', categorical_transformer, available_categorical))
        
        preprocessing = ColumnTransformer(transformers, remainder='drop')
        
        return Pipeline([
            ('feature_engineering', feature_engineering),
            ('preprocessing', preprocessing)
        ])
    
    def _get_unique_feature_names(self) -> List[str]:
        """Generate unique feature names with validation"""
        unique_features = []
        
        # Get engineered features
        for step in ['time_extractor', 'behavior_processor']:
            transformer = self.preprocessor.named_steps['feature_engineering'].named_steps[step]
            for name in transformer.get_feature_names_out():
                if name not in self._feature_registry:
                    self._feature_registry.add(name)
                    unique_features.append(name)
        
        # Get preprocessed features
        preprocessor = self.preprocessor.named_steps['preprocessing']
        for name, trans, cols in preprocessor.transformers_:
            if hasattr(trans, 'get_feature_names_out'):
                features = trans.get_feature_names_out()
                unique_features.extend(f for f in features if f not in self._feature_registry)
        
        return unique_features[:self.preprocessor.transform(self._sample_df).shape[1]]
        
    def preprocess_data(self, df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Main preprocessing method with enhanced error handling and feature tracking"""
        try:
            logger.info("Starting data preprocessing pipeline...")
            self._validate_data(df, target_col)
            
            # Store a sample for column checking
            self._sample_df = df.copy()
            self._feature_registry = set()  # Reset feature registry
            
            # Convert data types with validation
            time_config = self.config['feature_config']['time_features']
            df[time_config['purchase_time_col']] = pd.to_datetime(
                df[time_config['purchase_time_col']],
                errors='coerce'
            )
            df[time_config['signup_time_col']] = pd.to_datetime(
                df[time_config['signup_time_col']],
                errors='coerce'
            )
            
            # Handle IP mapping if configured
            if 'ip_mapping' in self.config['feature_config']:
                ip_config = self.config['feature_config']['ip_mapping']
                if 'country_column' in ip_config and ip_config['country_column'] not in df.columns:
                    logger.warning(f"Country column {ip_config['country_column']} not found - skipping")
            
            # Create and fit the preprocessing pipeline
            self.preprocessor = self._create_preprocessing_pipeline()
            X = df.drop(columns=[target_col])
            y = df[target_col].values
            
            X_processed = self.preprocessor.fit_transform(X, y)
            
            # Get unique feature names with duplicate prevention
            self.feature_names = self._get_unique_feature_names()
            
            # Validate feature count matches processed data
            if len(self.feature_names) != X_processed.shape[1]:
                raise ValueError(
                    f"Feature count mismatch: {len(self.feature_names)} names vs {X_processed.shape[1]} features"
                )
            
            # Handle class imbalance if configured
            if self.config.get('sampling_strategy') == 'SMOTE':
                smote = SMOTE(
                    sampling_strategy=self.config.get('sampling_ratio', 0.3),
                    random_state=self.config.get('random_state', 42)
                )
                X_processed, y = smote.fit_resample(X_processed, y)
            
            logger.info(f"Successfully processed data. Shape: {X_processed.shape}")
            logger.info(f"Feature names: {self.feature_names}")
            return X_processed, y, self.feature_names
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Data preprocessing error: {str(e)}") from e