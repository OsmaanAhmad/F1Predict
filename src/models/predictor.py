"""
Prediction Module
Makes predictions using trained models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import os

logger = logging.getLogger(__name__)


class RacePredictor:
    """Makes predictions for F1 race outcomes"""
    
    def __init__(self, model_trainer, data_dir: str = 'data/processed'):
        """
        Initialize predictor
        
        Args:
            model_trainer: Trained ModelTrainer instance
            data_dir: Directory containing processed data files
        """
        self.trainer = model_trainer
        self.data_dir = data_dir
        self.drivers_df = self._load_drivers()
        self.sessions_df = self._load_sessions()
    
    def _load_drivers(self) -> pd.DataFrame:
        """Load driver information"""
        drivers_path = os.path.join(self.data_dir, 'drivers_processed.csv')
        if os.path.exists(drivers_path):
            df = pd.read_csv(drivers_path)
            # Keep only unique driver_number and their names
            return df[['driver_number', 'full_name', 'name_acronym', 'team_name']].drop_duplicates('driver_number')
        else:
            logger.warning(f"Drivers file not found at {drivers_path}")
            return pd.DataFrame()
    
    def _load_sessions(self) -> pd.DataFrame:
        """Load session information"""
        sessions_path = os.path.join(self.data_dir, 'sessions_processed.csv')
        if os.path.exists(sessions_path):
            df = pd.read_csv(sessions_path)
            return df[['session_key', 'session_name', 'location', 'country_name', 'year', 'date_start']]
        else:
            logger.warning(f"Sessions file not found at {sessions_path}")
            return pd.DataFrame()
    
    def _add_driver_names(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Add driver names to predictions"""
        if not self.drivers_df.empty:
            predictions = predictions.merge(
                self.drivers_df,
                on='driver_number',
                how='left'
            )
        return predictions
    
    def get_session_info(self, session_key: int) -> Dict[str, Any]:
        """Get race information for a session"""
        if self.sessions_df.empty:
            return {'session_key': session_key}
        
        session = self.sessions_df[self.sessions_df['session_key'] == session_key]
        if session.empty:
            return {'session_key': session_key}
        
        session_row = session.iloc[0]
        return {
            'session_key': int(session_key),
            'race_name': f"{session_row['country_name']} GP" if pd.notna(session_row['country_name']) else session_row['location'],
            'location': session_row['location'],
            'country': session_row['country_name'],
            'year': int(session_row['year']) if pd.notna(session_row['year']) else None,
            'session_type': session_row['session_name'] if 'session_name' in session_row else 'Race',
            'date': session_row['date_start'] if 'date_start' in session_row else None
        }
        
    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for race outcomes
        
        Args:
            features_df: Feature matrix for prediction
            
        Returns:
            DataFrame with predictions
        """
        if self.trainer.model is None:
            raise ValueError("Model not trained yet")
        
        # Prepare features
        X = features_df[self.trainer.feature_names].copy()
        X = X.fillna(X.median())  # Handle missing values
        
        # Scale if needed
        if self.trainer.scaler is not None:
            X_scaled = pd.DataFrame(
                self.trainer.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X
        
        # Predict
        predictions = self.trainer.model.predict(X_scaled)
        
        # Create results DataFrame
        results = features_df[['session_key', 'driver_number']].copy()
        results['predicted_position'] = predictions
        
        # Round to nearest integer
        results['predicted_position_int'] = np.round(predictions).astype(int)
        
        # Clip to valid range (1-20 for F1)
        results['predicted_position_int'] = results['predicted_position_int'].clip(1, 20)
        
        # Add actual position if available
        if 'final_position' in features_df.columns:
            results['actual_position'] = features_df['final_position']
            results['error'] = results['actual_position'] - results['predicted_position']
        
        # Add driver names
        results = self._add_driver_names(results)
        
        return results
    
    def predict_race(self, features_df: pd.DataFrame, session_key: int) -> pd.DataFrame:
        """
        Predict outcomes for a specific race
        
        Args:
            features_df: Feature matrix
            session_key: Session key for the race
            
        Returns:
            DataFrame with race predictions sorted by predicted position
        """
        race_features = features_df[features_df['session_key'] == session_key].copy()
        
        if race_features.empty:
            logger.warning(f"No features found for session {session_key}")
            return pd.DataFrame()
        
        predictions = self.predict(race_features)
        predictions = predictions.sort_values('predicted_position')
        
        # Add predicted rank
        predictions['predicted_rank'] = range(1, len(predictions) + 1)
        
        logger.info(f"Predicted {len(predictions)} finishing positions for session {session_key}")
        
        return predictions
    
    def predict_podium(self, features_df: pd.DataFrame, session_key: int, 
                      return_probabilities: bool = False) -> Dict[str, Any]:
        """
        Predict podium finishers (top 3)
        
        Args:
            features_df: Feature matrix
            session_key: Session key for the race
            return_probabilities: Whether to return confidence scores
            
        Returns:
            Dictionary with podium predictions including driver names
        """
        predictions = self.predict_race(features_df, session_key)
        
        if predictions.empty:
            return {'podium': [], 'winner': None}
        
        # Top 3 predictions
        podium = predictions.head(3)
        
        # Helper function to get driver info
        def get_driver_info(row):
            if pd.isna(row['driver_number']):
                return None
            info = {'driver_number': int(row['driver_number'])}
            if 'full_name' in row and pd.notna(row['full_name']):
                info['name'] = row['full_name']
            if 'name_acronym' in row and pd.notna(row['name_acronym']):
                info['acronym'] = row['name_acronym']
            if 'team_name' in row and pd.notna(row['team_name']):
                info['team'] = row['team_name']
            return info
        
        result = {
            'winner': get_driver_info(podium.iloc[0]) if len(podium) > 0 else None,
            'second': get_driver_info(podium.iloc[1]) if len(podium) > 1 else None,
            'third': get_driver_info(podium.iloc[2]) if len(podium) > 2 else None,
            'podium': [get_driver_info(row) for _, row in podium.iterrows()]
        }
        
        if return_probabilities:
            # Simple confidence based on prediction gap
            if len(podium) > 0:
                gaps = podium['predicted_position'].diff().fillna(0).abs()
                result['confidence_scores'] = (1.0 / (1.0 + gaps)).tolist()
        
        return result
    
    def predict_top_k(self, features_df: pd.DataFrame, session_key: int, k: int = 10) -> List[int]:
        """
        Predict top K finishers
        
        Args:
            features_df: Feature matrix
            session_key: Session key for the race
            k: Number of top finishers to predict
            
        Returns:
            List of driver numbers in predicted order
        """
        predictions = self.predict_race(features_df, session_key)
        
        if predictions.empty:
            return []
        
        top_k = predictions.head(k)
        return top_k['driver_number'].tolist()
    
    def compare_predictions(self, features_df: pd.DataFrame, session_key: int) -> pd.DataFrame:
        """
        Compare predictions with actual results
        
        Args:
            features_df: Feature matrix with actual results
            session_key: Session key for the race
            
        Returns:
            DataFrame comparing predictions and actuals
        """
        predictions = self.predict_race(features_df, session_key)
        
        if 'actual_position' not in predictions.columns:
            logger.warning("No actual positions available for comparison")
            return predictions
        
        comparison = predictions[['driver_number', 'predicted_position_int', 'actual_position', 'error']].copy()
        comparison = comparison.sort_values('actual_position')
        
        # Calculate metrics
        mae = comparison['error'].abs().mean()
        correct_top3 = len(comparison[
            (comparison['actual_position'] <= 3) & 
            (comparison['predicted_position_int'] <= 3)
        ])
        
        logger.info(f"Race {session_key} - MAE: {mae:.2f}, Correct Top-3: {correct_top3}/3")
        
        return comparison
    
    def get_prediction_summary(self, predictions: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for predictions
        
        Args:
            predictions: Predictions DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_predictions': len(predictions),
            'predicted_positions': predictions['predicted_position'].describe().to_dict()
        }
        
        if 'actual_position' in predictions.columns:
            summary['mae'] = predictions['error'].abs().mean()
            summary['rmse'] = np.sqrt((predictions['error'] ** 2).mean())
            summary['max_error'] = predictions['error'].abs().max()
            summary['correct_predictions'] = len(predictions[predictions['error'].abs() < 0.5])
        
        return summary
