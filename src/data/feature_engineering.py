"""
Enhanced Feature Engineering with Complete Feature Support
"""


import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from .geolocation import IPRiskEvaluator
from src.config import FEATURES, TARGET, FeatureNames  # Ensure FeatureNames is imported

class FeatureEngineer:
    def __init__(self):
        self.required_inputs = FeatureNames.REQUIRED_INPUTS
        self.expected_features = FeatureNames.ALL_FEATURES
        self.user_history = {}
        self.ip_mapper = IPRiskEvaluator()
        
    def enrich_transaction(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive feature engineering with validation"""
        # Validate input
        missing = [f for f in self.required_inputs if f not in transaction_data]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        
        enriched = {}
        
        # Basic features
        enriched['amount'] = float(transaction_data['amount'])
        enriched['user_id'] = str(transaction_data['user_id'])
        enriched['device_id'] = str(transaction_data['device_id'])
        
        # Time-based features
        try:
            timestamp = pd.to_datetime(transaction_data['timestamp'])
            enriched['hour_of_day'] = timestamp.hour
            enriched['day_of_week'] = timestamp.dayofweek
            enriched['is_weekend'] = int(timestamp.dayofweek >= 5)
        except Exception as e:
            raise ValueError(f"Timestamp processing failed: {str(e)}")
        
        # IP features
        enriched['is_foreign_ip'] = int(transaction_data.get('is_foreign_ip', 0))
        enriched['country_risk_score'] = float(transaction_data.get('country_risk_score', 0))
        
        # Device features
        enriched['device_change_flag'] = int(transaction_data.get('device_change_flag', 0))
        
        # Transaction patterns
        enriched['time_since_last_transaction'] = float(
            transaction_data.get('time_since_last_transaction', 1440)
        )
        enriched['purchase_frequency_24h'] = float(
            transaction_data.get('purchase_frequency_24h', 1)
        )
        enriched['user_transaction_count'] = float(
            transaction_data.get('user_transaction_count', 1)
        )
        enriched['user_avg_transaction'] = float(
            transaction_data.get('user_avg_transaction', 
                               transaction_data['amount'])
        )
        
        # Billing/shipping
        enriched['billing_shipping_mismatch'] = int(
            transaction_data.get('billing_shipping_mismatch', 0)
        )
        
        # Ensure all expected features are present
        for feat in self.expected_features:
            if feat not in enriched:
                enriched[feat] = 0.0  # Default fallback
                
        return enriched
            
    def _ensure_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Ensure all required features exist with fallback values"""
        result = {}
        for feature in FEATURES:
            if feature in df.columns:
                result[feature] = df[feature].iloc[0]
            else:
                # Provide sensible defaults for missing features
                if feature == 'billing_shipping_mismatch':
                    result[feature] = 0  # Default to no mismatch
                elif feature == 'user_avg_transaction':
                    user_id = df.get('user_id', [None])[0]
                    result[feature] = self.user_history.get(user_id, {}).get('total_spent', 0) / max(
                        1, self.user_history.get(user_id, {}).get('transaction_count', 1)
                    )
                elif feature in ['hour_of_day', 'day_of_week']:
                    if 'timestamp' in df.columns:
                        dt = pd.to_datetime(df['timestamp'].iloc[0])
                        result['hour_of_day'] = dt.hour
                        result['day_of_week'] = dt.dayofweek
                        result['is_weekend'] = int(dt.dayofweek in [5, 6])
                    else:
                        result[feature] = 0
                else:
                    result[feature] = 0  # Default fallback
        return result
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic data cleaning"""
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['amount'] = df['amount'].astype(float)
        if 'is_fraud' in df.columns:
            df['is_fraud'] = df['is_fraud'].astype(int)
        return df
    
    def _extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features"""
        if 'timestamp' in df.columns:
            df['hour_of_day'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        return df
    
    def _calculate_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate transaction velocity features"""
        if 'user_id' in df.columns and 'timestamp' in df.columns:
            # Convert to datetime if not already
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by user and time
            df = df.sort_values(['user_id', 'timestamp'])
            
            # Time since last transaction
            df['time_since_last_transaction'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds().div(60).fillna(1440)
            
            # For single transactions, we can't calculate rolling frequency
            if len(df) > 1:
                # Purchase frequency in last 24 hours
                df['purchase_frequency_24h'] = df.groupby('user_id')['timestamp'].transform(
                    lambda x: x.rolling('24H', closed='left').count()
                )
            else:
                # For single transaction, use user history or default
                user_id = df['user_id'].iloc[0] if 'user_id' in df.columns else None
                if user_id and user_id in self.user_history:
                    df['purchase_frequency_24h'] = self.user_history[user_id].get('recent_frequency', 1)
                else:
                    df['purchase_frequency_24h'] = 1
        else:
            # Fill with defaults if required columns are missing
            df['time_since_last_transaction'] = 1440  # 24 hours in minutes
            df['purchase_frequency_24h'] = 1
        
        return df
    
    def _add_geolocation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add geolocation-based features"""
        if 'ip_address' in df.columns:
            # Use the evaluate method which includes is_foreign_ip
            ip_features = df['ip_address'].apply(
                lambda ip: self.ip_mapper.evaluate(ip)
            )
            
            # Convert the dict results to columns
            ip_df = pd.json_normalize(ip_features)
            df = pd.concat([df, ip_df], axis=1)
        
        return df
    
    def _add_device_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add device-related features"""
        if 'user_id' in df.columns and 'device_id' in df.columns:
            df['device_change_flag'] = (df.groupby('user_id')['device_id'].shift() != df['device_id']).astype(int)
            df['device_change_flag'] = df['device_change_flag'].fillna(0)
        
        if 'billing_address' in df.columns and 'shipping_address' in df.columns:
            df['billing_shipping_mismatch'] = (df['billing_address'] != df['shipping_address']).astype(int)
        elif 'billing_shipping_mismatch' not in df.columns:
            df['billing_shipping_mismatch'] = 0
        
        return df
    
    def _add_user_history_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features based on user history"""
        if 'user_id' in df.columns:
            # Update user history
            for _, row in df.iterrows():
                user_id = row['user_id']
                if user_id not in self.user_history:
                    self.user_history[user_id] = {
                        'transaction_count': 0,
                        'total_spent': 0,
                        'last_transaction_time': None,
                        'recent_frequency': 1
                    }
                
                self.user_history[user_id]['transaction_count'] += 1
                self.user_history[user_id]['total_spent'] += row.get('amount', 0)
                if 'timestamp' in row:
                    self.user_history[user_id]['last_transaction_time'] = row['timestamp']
            
            # Add features
            df['user_transaction_count'] = df['user_id'].map(
                lambda x: self.user_history.get(x, {}).get('transaction_count', 0)
            )
            df['user_avg_transaction'] = df['user_id'].map(
                lambda x: self.user_history.get(x, {}).get('total_spent', 0) / 
                        max(1, self.user_history.get(x, {}).get('transaction_count', 1))
            )
        
        return df