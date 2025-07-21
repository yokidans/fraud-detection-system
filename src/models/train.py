import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                           classification_report, confusion_matrix)
import joblib
import logging
import optuna
from optuna.samplers import TPESampler
from sklearn.base import BaseEstimator, TransformerMixin
from copy import deepcopy
from sklearn.pipeline import Pipeline
from scipy.sparse import issparse, csr_matrix
import time
from tqdm import tqdm
import warnings
from functools import partial

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)

class FraudDetectionModel:
    def __init__(self, config):
        """Initialize with configuration including performance tuning params"""
        self.config = {
            'random_state': 42,
            'n_jobs': -1,  # Use all cores
            'early_stopping_rounds': 20,
            'optuna_timeout': 3600,  # 1 hour timeout for optimization
            'optuna_n_trials': 30,  # Reduced from 50
            'cv_n_splits': 3,  # Reduced from 5
            'progress_bar': True,
            **config  # Allow override of defaults
        }
        
        # Configure models with sensible defaults
        self.models = {
            'random_forest': RandomForestClassifier(
                random_state=self.config['random_state'],
                n_jobs=self.config['n_jobs'],
                class_weight='balanced',
                verbose=1
            ),
            'xgboost': XGBClassifier(
                random_state=self.config['random_state'],
                eval_metric='logloss',
                use_label_encoder=False,
                n_jobs=self.config['n_jobs'],
                verbosity=1
            ),
            'lightgbm': LGBMClassifier(
                random_state=self.config['random_state'],
                n_jobs=self.config['n_jobs'],
                verbose=1
            ),
            'logistic': LogisticRegression(
                random_state=self.config['random_state'],
                max_iter=1000,
                class_weight='balanced',
                n_jobs=self.config['n_jobs']
            ),
            'gradient_boosting': GradientBoostingClassifier(
                random_state=self.config['random_state'],
                verbose=1
            )
        }
        self.best_model = None
        self.best_score = 0
        self.feature_importances = None
        self.training_metadata = {}

    def _log_data_stats(self, X, y):
        """Log detailed dataset statistics"""
        logger.info(f"\n{'='*50}\nDataset Characteristics\n{'='*50}")
        logger.info(f"Shape: {X.shape}")
        
        if issparse(X):
            logger.info(f"Sparse matrix with {X.nnz} non-zero elements")
            density = X.nnz / (X.shape[0] * X.shape[1])
            logger.info(f"Density: {density:.2%}")
        else:
            logger.info("Dense matrix")
            
        if isinstance(y, (pd.Series, pd.DataFrame)):
            class_dist = y.value_counts(normalize=True)
        else:
            class_dist = pd.Series(y).value_counts(normalize=True)
            
        logger.info(f"\nClass Distribution:\n{class_dist.to_string()}")
        logger.info(f"Imbalance Ratio: {class_dist.iloc[0]/class_dist.iloc[1]:.1f}:1")

    def check_data_quality(self, X, y):
        """Enhanced data quality checks with more detailed logging"""
        self._log_data_stats(X, y)
        
        # Handle sparse matrices
        if issparse(X):
            if not isinstance(X, csr_matrix):
                X = csr_matrix(X)
            if np.isnan(X.data).any():
                nan_count = np.isnan(X.data).sum()
                raise ValueError(f"Sparse matrix contains {nan_count} NaN values")
        else:
            if isinstance(X, pd.DataFrame):
                non_numeric_cols = X.select_dtypes(exclude=['int64', 'float64']).columns
                if len(non_numeric_cols) > 0:
                    raise ValueError(f"Non-numeric columns: {non_numeric_cols.tolist()}")
            
            nan_count = np.isnan(X).sum()
            if nan_count > 0:
                raise ValueError(f"Dense matrix contains {nan_count} NaN values")
        
        # Check target variable
        if isinstance(y, (pd.Series, pd.DataFrame)):
            if not pd.api.types.is_numeric_dtype(y):
                raise ValueError("Target must be numeric")
        elif issparse(y):
            raise ValueError("Target cannot be sparse")
            
        logger.info("✅ Data quality check passed")

    def train_test_split(self, X, y, test_size=0.2):
        """Stratified split with data size logging"""
        X, y = self._ensure_numpy(X, y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.config['random_state'],
            stratify=y
        )
        logger.info(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
        return X_train, X_test, y_train, y_test

    def _ensure_numpy(self, X, y):
        """Conversion with memory optimization"""
        # Convert X
        if issparse(X):
            if not isinstance(X, csr_matrix):
                X = csr_matrix(X)
        elif hasattr(X, 'values'):
            X = X.values
        elif isinstance(X, (list, tuple)):
            X = np.array(X, dtype=np.float32)  # Save memory
            
        # Convert y
        if issparse(y):
            y = y.toarray().flatten()
        elif hasattr(y, 'values'):
            y = y.values.astype(np.int8).flatten()  # Save memory for binary classification
        elif isinstance(y, (list, tuple)):
            y = np.array(y, dtype=np.int8).flatten()
            
        return X, y

    def evaluate_model(self, model, X_test, y_test):
        """Enhanced evaluation with timing and more metrics"""
        start_time = time.time()
        try:
            X_test, y_test = self._ensure_numpy(X_test, y_test)
            
            # Prediction with timing
            predict_start = time.time()
            if issparse(X_test):
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
            predict_time = time.time() - predict_start
            
            # Calculate metrics
            metrics = {
                'roc_auc': roc_auc_score(y_test, y_proba),
                'pr_auc': average_precision_score(y_test, y_proba),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'prediction_time_sec': predict_time,
                'evaluation_time_sec': time.time() - start_time
            }
            
            # Log important metrics
            logger.info(f"\n{'='*50}\nEvaluation Results\n{'='*50}")
            logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
            logger.info(f"PR AUC: {metrics['pr_auc']:.4f}")
            logger.info(f"Prediction time: {predict_time:.2f}s")
            
            return metrics
        except Exception as e:
            logger.error(f"Evaluation error: {str(e)}", exc_info=True)
            raise

    def cross_validate(self, model, X, y):
        """Optimized CV with progress tracking"""
        start_time = time.time()
        try:
            X, y = self._ensure_numpy(X, y)
            self.check_data_quality(X, y)
            
            cv = StratifiedKFold(
                n_splits=self.config['cv_n_splits'],
                shuffle=True,
                random_state=self.config['random_state']
            )
            
            scores = []
            fit_times = []
            predict_times = []
            
            iterator = cv.split(X, y)
            if self.config['progress_bar']:
                iterator = tqdm(iterator, total=self.config['cv_n_splits'], desc='CV Progress')
            
            for fold, (train_idx, val_idx) in enumerate(iterator, 1):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train with timing
                fit_start = time.time()
                model.fit(X_train, y_train)
                fit_times.append(time.time() - fit_start)
                
                # Predict with timing
                predict_start = time.time()
                y_proba = model.predict_proba(X_val)[:, 1]
                predict_times.append(time.time() - predict_start)
                
                score = roc_auc_score(y_val, y_proba)
                scores.append(score)
                
                if self.config['progress_bar']:
                    iterator.set_postfix({'Fold ROC AUC': score})
            
            cv_metrics = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'fit_times': fit_times,
                'predict_times': predict_times,
                'total_time': time.time() - start_time
            }
            
            logger.info(f"\nCV Results - ROC AUC: {cv_metrics['mean_score']:.4f} ± {cv_metrics['std_score']:.4f}")
            logger.info(f"Avg fit time: {np.mean(fit_times):.2f}s")
            logger.info(f"Avg predict time: {np.mean(predict_times):.2f}s")
            logger.info(f"Total CV time: {cv_metrics['total_time']:.2f}s")
            
            return cv_metrics['mean_score'], cv_metrics['std_score']
            
        except Exception as e:
            logger.error(f"CV error: {str(e)}", exc_info=True)
            raise

    def optimize_hyperparameters(self, model_name, X, y):
        """Optimized hyperparameter tuning with early stopping and parallelization"""
        def objective(trial):
            try:
                params = {}
                
                # Common params
                params['n_jobs'] = self.config['n_jobs']
                
                if model_name == 'random_forest':
                    params.update({
                        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                        'max_depth': trial.suggest_int('max_depth', 5, 30, step=5),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.8]),
                        'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
                    })
                elif model_name == 'xgboost':
                    params.update({
                        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1),
                        'gamma': trial.suggest_float('gamma', 0, 2),
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
                    })
                elif model_name == 'lightgbm':
                    params.update({
                        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                        'max_depth': trial.suggest_int('max_depth', 3, 12),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'num_leaves': trial.suggest_int('num_leaves', 15, 100, step=5),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1),
                        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1)
                    })
                
                model = deepcopy(self.models[model_name]).set_params(**params)
                score, _ = self.cross_validate(model, X, y)
                return score
            except Exception as e:
                logger.warning(f"Trial failed: {str(e)}")
                raise optuna.TrialPruned()
        
        try:
            logger.info(f"\n{'='*50}\nStarting {model_name} hyperparameter optimization\n{'='*50}")
            
            # Create study with progress callback
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=self.config['random_state']),
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
            )
            
            # Add progress bar for Optuna trials
            progress_callback = self._get_optuna_progress_callback()
            
            study.optimize(
                objective,
                n_trials=self.config['optuna_n_trials'],
                timeout=self.config['optuna_timeout'],
                n_jobs=min(4, self.config['n_jobs']),  # Don't use all cores to avoid thrashing
                callbacks=[progress_callback],
                gc_after_trial=True
            )
            
            logger.info(f"\nBest trial for {model_name}:")
            logger.info(f"Value: {study.best_value:.4f}")
            logger.info(f"Params: {study.best_params}")
            
            return study.best_params
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}", exc_info=True)
            raise

    def _get_optuna_progress_callback(self):
        """Create a progress tracking callback for Optuna"""
        last_log_time = time.time()
        
        def callback(study, trial):
            nonlocal last_log_time
            current_time = time.time()
            
            # Log every 30 seconds or on trial completion
            if current_time - last_log_time > 30 or trial.state == optuna.trial.TrialState.COMPLETE:
                logger.info(
                    f"Trial {trial.number}/{self.config['optuna_n_trials']} - "
                    f"Current best: {study.best_value:.4f} - "
                    f"Last value: {trial.value if trial.value else 'N/A'}"
                )
                last_log_time = current_time
                
        return callback

    def train(self, X, y, model_name='xgboost', optimize=False):
        """Optimized training with comprehensive logging"""
        start_time = time.time()
        try:
            logger.info(f"\n{'='*50}\nTraining {model_name} model\n{'='*50}")
            X, y = self._ensure_numpy(X, y)
            self.check_data_quality(X, y)
            
            # Hyperparameter optimization
            best_params = {}
            if optimize:
                try:
                    best_params = self.optimize_hyperparameters(model_name, X, y)
                    self.models[model_name].set_params(**best_params)
                    logger.info(f"Optimized parameters:\n{best_params}")
                except Exception as e:
                    logger.warning(f"Optimization failed, using defaults: {str(e)}")
                    optimize = False
            
            model = deepcopy(self.models[model_name])
            
            # Cross-validation
            cv_start = time.time()
            mean_score, std_score = self.cross_validate(model, X, y)
            logger.info(f"CV completed in {time.time() - cv_start:.2f}s")
            
            # Final training
            train_start = time.time()
            model.fit(X, y)
            train_time = time.time() - train_start
            logger.info(f"Final training completed in {train_time:.2f}s")
            
            # Store metadata
            self.training_metadata[model_name] = {
                'params': model.get_params(),
                'cv_score': mean_score,
                'cv_std': std_score,
                'train_time': train_time,
                'optimized': optimize,
                'best_params': best_params if optimize else None
            }
            
            # Feature importances
            if hasattr(model, 'feature_importances_'):
                self.feature_importances = model.feature_importances_
            
            logger.info(f"\n{'='*50}\nTraining completed for {model_name}\n{'='*50}")
            logger.info(f"Total training time: {time.time() - start_time:.2f}s")
            
            return model
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise

    def train_ensemble(self, X, y, optimize=False):
        """Optimized ensemble training with parallel model training"""
        start_time = time.time()
        try:
            logger.info(f"\n{'='*50}\nStarting Ensemble Training\n{'='*50}")
            X, y = self._ensure_numpy(X, y)
            self.check_data_quality(X, y)
            
            trained_models = {}
            model_scores = {}
            
            # Train models in order of expected performance/complexity
            model_order = ['xgboost', 'lightgbm', 'random_forest', 'logistic']
            
            for model_name in model_order:
                try:
                    logger.info(f"\n{'='*30}\nTraining {model_name}\n{'='*30}")
                    model = self.train(X, y, model_name, optimize)
                    trained_models[model_name] = model
                    
                    # Quick evaluation
                    y_proba = model.predict_proba(X)[:, 1]
                    score = roc_auc_score(y, y_proba)
                    model_scores[model_name] = score
                    logger.info(f"{model_name} training ROC AUC: {score:.4f}")
                    
                    if score > self.best_score:
                        self.best_score = score
                        self.best_model = model
                except Exception as e:
                    logger.error(f"Skipping {model_name}: {str(e)}")
                    continue
            
            if not trained_models:
                raise ValueError("No models trained successfully")
            
            # Create voting classifier
            logger.info("\nCreating voting classifier...")
            voting_clf = VotingClassifier(
                estimators=list(trained_models.items()),
                voting='soft',
                n_jobs=self.config['n_jobs']
            )
            
            # Train voting classifier
            voting_start = time.time()
            voting_clf.fit(X, y)
            logger.info(f"Voting classifier trained in {time.time() - voting_start:.2f}s")
            
            # Evaluate
            y_proba = voting_clf.predict_proba(X)[:, 1]
            voting_score = roc_auc_score(y, y_proba)
            logger.info(f"Voting classifier ROC AUC: {voting_score:.4f}")
            
            if voting_score > self.best_score:
                self.best_score = voting_score
                self.best_model = voting_clf
            
            logger.info(f"\n{'='*50}\nEnsemble Training Complete\n{'='*50}")
            logger.info(f"Best model: {type(self.best_model).__name__}")
            logger.info(f"Best score: {self.best_score:.4f}")
            logger.info(f"Total training time: {time.time() - start_time:.2f}s")
            
            return self.best_model
            
        except Exception as e:
            logger.error(f"Ensemble training failed: {str(e)}", exc_info=True)
            raise

    def save_model(self, model, filepath):
        """Save model with metadata"""
        try:
            save_data = {
                'model': model,
                'metadata': {
                    'timestamp': pd.Timestamp.now(),
                    'config': self.config,
                    'feature_importances': self.feature_importances,
                    'training_metadata': self.training_metadata,
                    'best_score': self.best_score
                }
            }
            joblib.dump(save_data, filepath)
            logger.info(f"Model and metadata saved to {filepath}")
        except Exception as e:
            logger.error(f"Save failed: {str(e)}", exc_info=True)
            raise

    def load_model(self, filepath):
        """Load model with metadata"""
        try:
            data = joblib.load(filepath)
            model = data['model']
            self.feature_importances = data['metadata'].get('feature_importances')
            self.best_score = data['metadata'].get('best_score')
            logger.info(f"Model loaded from {filepath}")
            logger.info(f"Originally trained on {data['metadata']['timestamp']}")
            return model
        except Exception as e:
            logger.error(f"Load failed: {str(e)}", exc_info=True)
            raise