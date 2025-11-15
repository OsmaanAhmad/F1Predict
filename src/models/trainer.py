"""
Model Training Module
Trains machine learning models for F1 race prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains and evaluates F1 prediction models"""
    
    def __init__(self, model_config: Dict[str, Any], training_config: Dict[str, Any]):
        """
        Initialize model trainer
        
        Args:
            model_config: Model configuration dictionary
            training_config: Training configuration dictionary
        """
        self.model_config = model_config
        self.training_config = training_config
        self.model = None
        self.scaler = None
        self.feature_names = None
        
    def prepare_data(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for training
        
        Args:
            features_df: Feature matrix DataFrame
            
        Returns:
            Tuple of (X, y) - features and target
        """
        # Remove rows with missing target
        df = features_df.dropna(subset=['final_position']).copy()
        
        logger.info(f"Preparing data: {len(df)} samples")
        
        # Identify feature columns (exclude identifiers and target)
        exclude_cols = ['session_key', 'driver_number', 'final_position']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df['final_position'].copy()
        
        # Handle missing values in features
        # For now, use median imputation
        X = X.fillna(X.median())
        
        # Store feature names
        self.feature_names = feature_cols
        
        logger.info(f"Features: {len(feature_cols)}, Samples: {len(X)}")
        
        return X, y
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale features using configured scaler
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            Tuple of scaled (X_train, X_test)
        """
        if not self.training_config.get('scale_features', False):
            return X_train, X_test
        
        scaler_type = self.training_config.get('scaler_type', 'standard')
        
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            logger.warning(f"Unknown scaler type: {scaler_type}, using StandardScaler")
            self.scaler = StandardScaler()
        
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        logger.info(f"Features scaled using {scaler_type} scaler")
        
        return X_train_scaled, X_test_scaled
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
        """
        Train Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained Random Forest model
        """
        config = self.model_config.get('random_forest', {})
        
        model = RandomForestRegressor(
            n_estimators=config.get('n_estimators', 200),
            max_depth=config.get('max_depth', 15),
            min_samples_split=config.get('min_samples_split', 5),
            min_samples_leaf=config.get('min_samples_leaf', 2),
            random_state=config.get('random_state', 42),
            n_jobs=config.get('n_jobs', -1)
        )
        
        logger.info("Training Random Forest model")
        model.fit(X_train, y_train)
        
        return model
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBRegressor:
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained XGBoost model
        """
        config = self.model_config.get('xgboost', {})
        
        model = xgb.XGBRegressor(
            n_estimators=config.get('n_estimators', 300),
            max_depth=config.get('max_depth', 8),
            learning_rate=config.get('learning_rate', 0.05),
            subsample=config.get('subsample', 0.8),
            colsample_bytree=config.get('colsample_bytree', 0.8),
            objective=config.get('objective', 'reg:squarederror'),
            random_state=config.get('random_state', 42),
            n_jobs=config.get('n_jobs', -1)
        )
        
        logger.info("Training XGBoost model")
        model.fit(X_train, y_train)
        
        return model
    
    def train(self, features_df: pd.DataFrame, model_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Train model on features
        
        Args:
            features_df: Feature matrix DataFrame
            model_type: Model type ('random_forest' or 'xgboost'), defaults to config
            
        Returns:
            Dictionary with training results
        """
        # Prepare data
        X, y = self.prepare_data(features_df)
        
        # Split data
        test_size = self.training_config.get('test_size', 0.2)
        shuffle = self.training_config.get('shuffle', False)
        random_state = self.training_config.get('random_state', 42)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=shuffle, random_state=random_state
        )
        
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        # Scale features
        X_train, X_test = self.scale_features(X_train, X_test)
        
        # Train model
        if model_type is None:
            model_type = self.model_config.get('default_model', 'xgboost')
        
        if model_type == 'random_forest':
            self.model = self.train_random_forest(X_train, y_train)
        elif model_type == 'xgboost':
            self.model = self.train_xgboost(X_train, y_train)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Evaluate
        train_metrics = self.evaluate(X_train, y_train, dataset='train')
        test_metrics = self.evaluate(X_test, y_test, dataset='test')
        
        # Feature importance
        feature_importance = self.get_feature_importance()
        
        results = {
            'model_type': model_type,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_importance': feature_importance
        }
        
        return results
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, dataset: str = 'test') -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X: Features
            y: True target values
            dataset: Dataset name (for logging)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        y_pred = self.model.predict(X)
        
        # Basic regression metrics
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        # F1-specific metrics
        # Top 3 accuracy (predicted top 3 matches actual top 3)
        top3_accuracy = self.calculate_top_k_accuracy(y, y_pred, k=3)
        
        # Winner accuracy
        winner_accuracy = self.calculate_winner_accuracy(y, y_pred)
        
        # Podium precision
        podium_precision = self.calculate_podium_precision(y, y_pred)
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2,
            'top3_accuracy': top3_accuracy,
            'winner_accuracy': winner_accuracy,
            'podium_precision': podium_precision
        }
        
        logger.info(f"{dataset.upper()} Metrics - MAE: {mae:.3f}, RMSE: {rmse:.3f}, "
                   f"R²: {r2:.3f}, Top-3 Acc: {top3_accuracy:.3f}, Winner Acc: {winner_accuracy:.3f}")
        
        return metrics
    
    def calculate_top_k_accuracy(self, y_true: pd.Series, y_pred: np.ndarray, k: int = 3) -> float:
        """Calculate accuracy of top-k predictions"""
        # Round predictions to nearest integer position
        y_pred_int = np.round(y_pred).astype(int)
        
        # Get top k from predictions and actuals
        top_k_pred = set(np.argsort(y_pred)[:k])
        top_k_true = set(np.argsort(y_true.values)[:k])
        
        # Calculate overlap
        overlap = len(top_k_pred & top_k_true)
        accuracy = overlap / k
        
        return accuracy
    
    def calculate_winner_accuracy(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate accuracy of winner prediction"""
        pred_winner = np.argmin(y_pred)
        true_winner = y_true.values.argmin()
        
        return 1.0 if pred_winner == true_winner else 0.0
    
    def calculate_podium_precision(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate precision of podium predictions"""
        # Predicted podium (top 3)
        pred_podium = set(np.argsort(y_pred)[:3])
        true_podium = set(np.argsort(y_true.values)[:3])
        
        # Precision: how many predicted podium finishers actually finished in podium
        if len(pred_podium) == 0:
            return 0.0
        
        correct = len(pred_podium & true_podium)
        precision = correct / len(pred_podium)
        
        return precision
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from trained model
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        else:
            logger.warning("Model does not have feature_importances_ attribute")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    
    def save_model(self, path: str):
        """
        Save trained model to disk
        
        Args:
            path: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_config': self.model_config,
            'training_config': self.training_config
        }
        
        joblib.dump(model_data, save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, path: str):
        """
        Load trained model from disk
        
        Args:
            path: Path to load model from
        """
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.scaler = model_data.get('scaler')
        self.feature_names = model_data['feature_names']
        self.model_config = model_data.get('model_config', self.model_config)
        self.training_config = model_data.get('training_config', self.training_config)
        
        logger.info(f"Model loaded from {path}")
