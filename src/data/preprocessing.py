import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import FeatureHasher
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import logging
from scipy.sparse import csr_matrix, save_npz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.preprocessor = None
        self.categorical_features = []
        self.numerical_features = []
        self.high_cardinality_features = []
        self.feature_names = []
        
    def _convert_datetime(self, df):
        """Convert datetime columns to numeric timestamps"""
        datetime_cols = ['signup_time', 'purchase_time']
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col]).astype('int64') // 10**9
        return df
        
    def _preprocess_ip(self, df):
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
        
    def _handle_missing_values(self, df):
        """Handle missing values with column-specific strategies"""
        try:
            numeric_cols = df.select_dtypes(include=np.number).columns
            non_numeric_cols = df.select_dtypes(exclude=np.number).columns
            
            if self.config['missing_values_strategy'] == 'drop':
                return df.dropna()
            else:
                num_imputer = SimpleImputer(strategy='median')
                df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
                
                if len(non_numeric_cols) > 0:
                    cat_imputer = SimpleImputer(strategy='most_frequent', fill_value='missing')
                    df[non_numeric_cols] = cat_imputer.fit_transform(df[non_numeric_cols])
                
                return df
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            raise
            
    def _remove_duplicates(self, df):
        return df.drop_duplicates()
        
    def _correct_data_types(self, df):
        type_map = {
            'user_id': 'str',
            'purchase_value': 'float32',
            'age': 'int32',
            'class': 'int32',
            'device_id': 'str',
            'ip_address': 'str'  # Ensure IP address is treated as string
        }
        for col, dtype in type_map.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except Exception as e:
                    logger.warning(f"Failed to convert {col} to {dtype}: {str(e)}")
                    # If conversion fails, drop the column if it's not critical
                    if col not in ['user_id', 'device_id']:
                        df = df.drop(col, axis=1)
        return df
        
    def _get_feature_names(self, column_transformer):
        """Get feature names from ColumnTransformer"""
        feature_names = []
        for name, transformer, features in column_transformer.transformers_:
            if transformer == 'drop':
                continue
            if name == 'hash':
                # Special handling for FeatureHasher
                names = [f'hash_{i}' for i in range(transformer.named_steps['hasher'].n_features)]
            elif hasattr(transformer, 'get_feature_names_out'):
                names = transformer.get_feature_names_out(features)
            elif hasattr(transformer, 'get_feature_names'):
                # Fallback for older sklearn v .,ersions
                names = transformer.get_feature_names(features)
            else:
                names = features
            feature_names.extend(names)
        return feature_names
        
    def _identify_feature_types(self, X):
        """Identify feature types including high-cardinality features"""
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        
        # Identify high-cardinality features (assuming >100 unique values)
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
        
    def preprocess_data(self, df, target_col):
        try:
            logger.info("Starting data preprocessing...")
            
            # Initial cleaning
            df = self._correct_data_types(df)  # Convert types first
            df = self._convert_datetime(df)
            df = self._preprocess_ip(df)
            df = self._handle_missing_values(df)
            df = self._remove_duplicates(df)
            
            # Separate features and target
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # Identify feature types
            self._identify_feature_types(X)
            
            # Create preprocessing pipelines
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            # Regular categorical features (low cardinality)
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
            ])
            
            # High-cardinality features - using FeatureHasher
            hasher = FeatureHasher(n_features=50, input_type='string')
            high_card_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('hasher', hasher)
            ])
            
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, self.numerical_features),
                    ('cat', categorical_transformer, self.categorical_features),
                    ('hash', high_card_transformer, self.high_cardinality_features)
                ])
            
            # Fit and transform the data
            X_processed = self.preprocessor.fit_transform(X)
            self.feature_names = self._get_feature_names(self.preprocessor)
            
            # Handle class imbalance
            if self.config.get('sampling_strategy') == 'SMOTE':
                smote = SMOTE(random_state=self.config['random_state'])
                X_processed, y = smote.fit_resample(X_processed, y)
            elif self.config.get('sampling_strategy') == 'undersample':
                rus = RandomUnderSampler(random_state=self.config['random_state'])
                X_processed, y = rus.fit_resample(X_processed, y)
            
            logger.info(f"Processed data shape: {X_processed.shape}")
            logger.info(f"Feature names: {self.feature_names[:10]}...")  # Log first 10 feature names
            logger.info("Preprocessing completed successfully")
            
            return X_processed, y, self.feature_names
            
        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}")
            raise