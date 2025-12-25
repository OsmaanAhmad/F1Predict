#!/usr/bin/env python3
"""
Automated Race Predictor using XGBoost
Handles the full pipeline: fetch data → build features → predict

Usage:
    python predict_race_xgboost.py --race "Abu Dhabi" --year 2025
    python predict_race_xgboost.py --meeting-key 1242
    python predict_race_xgboost.py --session-key 9900
"""

import argparse
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config
from src.data import OpenF1Client, DataPreprocessor
from src.features import FeatureEngineer
from src.models import ModelTrainer, RacePredictor


class AutomatedRacePredictor:
    """Automates the entire prediction pipeline for a new race"""
    
    def __init__(self, config=None):
        """Initialize with configuration"""
        self.config = config or load_config()
        self.api_config = self.config.get('api', {})
        self.data_config = self.config.get('data', {})
        
        # Initialize API client
        self.client = OpenF1Client(
            base_url=self.api_config.get('base_url', 'https://api.openf1.org/v1'),
            timeout=self.api_config.get('timeout', 30),
            max_retries=self.api_config.get('max_retries', 3),
            rate_limit_delay=self.api_config.get('rate_limit_delay', 0.5)
        )
        
        # Initialize components
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer(rolling_windows=[3, 5, 10])
        
        # Load historical data
        self.historical_features = self._load_historical_features()
        self.historical_data = self._load_historical_data()
        
    def _load_historical_features(self) -> pd.DataFrame:
        """Load pre-computed historical features"""
        features_path = Path('data/processed/features.csv')
        if features_path.exists():
            df = pd.read_csv(features_path)
            logger.info(f"Loaded {len(df)} historical feature records")
            return df
        logger.warning("No historical features found - predictions may be limited")
        return pd.DataFrame()
    
    def _load_historical_data(self) -> dict:
        """Load preprocessed historical data"""
        data = {}
        processed_path = Path('data/processed')
        
        for data_type in ['laps', 'positions', 'weather', 'pit_stops', 'sessions', 'drivers']:
            file_path = processed_path / f"{data_type}_processed.csv"
            if file_path.exists():
                data[data_type] = pd.read_csv(file_path)
                logger.info(f"Loaded {len(data[data_type])} historical {data_type} records")
        
        return data
    
    def find_race(self, race_name: str = None, year: int = 2025, meeting_key: int = None) -> dict:
        """
        Find a race by name or meeting key
        
        Returns:
            dict with meeting_key, session_keys, and race info
        """
        if meeting_key:
            sessions = self.client.get_sessions(year=year)
            race_sessions = [s for s in sessions if s.get('meeting_key') == meeting_key]
            if race_sessions:
                return {
                    'meeting_key': meeting_key,
                    'sessions': race_sessions,
                    'location': race_sessions[0].get('location', 'Unknown'),
                    'country': race_sessions[0].get('country_name', 'Unknown')
                }
        
        if race_name:
            sessions = self.client.get_sessions(year=year)
            
            # Common name mappings (user-friendly name -> API location/country)
            name_mappings = {
                'abu dhabi': ['yas island', 'united arab emirates'],
                'bahrain': ['sakhir', 'bahrain'],
                'saudi arabia': ['jeddah', 'saudi arabia'],
                'australia': ['melbourne', 'australia'],
                'japan': ['suzuka', 'japan'],
                'china': ['shanghai', 'china'],
                'usa': ['austin', 'miami', 'las vegas', 'united states'],
                'us': ['austin', 'miami', 'las vegas', 'united states'],
                'italy': ['monza', 'imola', 'italy'],
                'uk': ['silverstone', 'united kingdom'],
                'britain': ['silverstone', 'united kingdom'],
                'brazil': ['são paulo', 'sao paulo', 'brazil'],
                'qatar': ['lusail', 'qatar'],
                'singapore': ['marina bay', 'singapore'],
                'belgium': ['spa-francorchamps', 'spa', 'belgium'],
                'netherlands': ['zandvoort', 'netherlands'],
                'hungary': ['budapest', 'hungary'],
                'austria': ['spielberg', 'austria'],
                'canada': ['montréal', 'montreal', 'canada'],
                'spain': ['barcelona', 'spain'],
                'azerbaijan': ['baku', 'azerbaijan'],
                'mexico': ['mexico city', 'mexico'],
            }
            
            search_terms = [race_name.lower()]
            
            # Add mapped search terms
            for key, values in name_mappings.items():
                if race_name.lower() in key or key in race_name.lower():
                    search_terms.extend(values)
            
            # Search by location, country, or session name
            race_sessions = []
            for s in sessions:
                location = s.get('location', '').lower()
                country = s.get('country_name', '').lower()
                session_name = s.get('session_name', '').lower()
                
                for term in search_terms:
                    if (term in location or term in country or term in session_name or
                        location in term or country in term):
                        race_sessions.append(s)
                        break
            
            if race_sessions:
                meeting_key = race_sessions[0].get('meeting_key')
                return {
                    'meeting_key': meeting_key,
                    'sessions': [s for s in sessions if s.get('meeting_key') == meeting_key],
                    'location': race_sessions[0].get('location', 'Unknown'),
                    'country': race_sessions[0].get('country_name', 'Unknown')
                }
        
        return None
    
    def fetch_race_data(self, race_info: dict) -> dict:
        """
        Fetch all data for a race weekend
        
        Returns:
            dict with laps, positions, weather, drivers, pit_stops data
        """
        logger.info(f"\n📥 Fetching data for {race_info['location']}...")
        
        data = {
            'sessions': pd.DataFrame(race_info['sessions']),
            'laps': [],
            'positions': [],
            'weather': [],
            'drivers': [],
            'pit_stops': []
        }
        
        for session in race_info['sessions']:
            session_key = session['session_key']
            session_type = session.get('session_type', 'Unknown')
            logger.info(f"  Fetching {session_type} (session_key: {session_key})...")
            
            try:
                # Fetch each data type
                laps = self.client.get_laps(session_key=session_key)
                if laps:
                    for lap in laps:
                        lap['session_key'] = session_key
                    data['laps'].extend(laps)
                
                positions = self.client.get_positions(session_key=session_key)
                if positions:
                    data['positions'].extend(positions)
                
                weather = self.client.get_weather(session_key=session_key)
                if weather:
                    data['weather'].extend(weather)
                
                drivers = self.client.get_drivers(session_key=session_key)
                if drivers:
                    data['drivers'].extend(drivers)
                
                pit_stops = self.client.get_pit_stops(session_key=session_key)
                if pit_stops:
                    data['pit_stops'].extend(pit_stops)
                    
            except Exception as e:
                logger.warning(f"  Error fetching {session_type}: {e}")
        
        # Convert to DataFrames
        for key in ['laps', 'positions', 'weather', 'drivers', 'pit_stops']:
            data[key] = pd.DataFrame(data[key]) if data[key] else pd.DataFrame()
            logger.info(f"  ✓ {key}: {len(data[key])} records")
        
        return data
    
    def preprocess_race_data(self, raw_data: dict) -> dict:
        """Preprocess fetched race data"""
        logger.info("\n🔧 Preprocessing race data...")
        
        processed = {}
        
        # Preprocess laps
        if not raw_data['laps'].empty:
            processed['laps'] = self.preprocessor.preprocess_laps(raw_data['laps'])
        else:
            processed['laps'] = pd.DataFrame()
        
        # Preprocess positions
        if not raw_data['positions'].empty:
            processed['positions'] = self.preprocessor.preprocess_positions(raw_data['positions'])
        else:
            processed['positions'] = pd.DataFrame()
        
        # Preprocess weather
        if not raw_data['weather'].empty:
            processed['weather'] = self.preprocessor.preprocess_weather(raw_data['weather'])
        else:
            processed['weather'] = pd.DataFrame()
        
        # Preprocess pit stops
        if not raw_data['pit_stops'].empty:
            processed['pit_stops'] = self.preprocessor.preprocess_pit_stops(raw_data['pit_stops'])
        else:
            processed['pit_stops'] = pd.DataFrame()
        
        # Drivers don't need heavy preprocessing - just pass through
        processed['drivers'] = raw_data['drivers']
        
        # Sessions
        processed['sessions'] = raw_data['sessions']
        
        return processed
    
    def build_features_for_race(self, processed_data: dict, race_info: dict) -> pd.DataFrame:
        """
        Build features for a new race using historical data + current weekend data
        
        This combines:
        1. Historical performance features (from stored features.csv)
        2. Current weekend features (practice/qualifying from fetched data)
        """
        logger.info("\n⚙️ Building features for prediction...")
        
        # Find race session
        race_session = None
        quali_session = None
        practice_sessions = []
        
        for session in race_info['sessions']:
            if session.get('session_type') == 'Race':
                race_session = session
            elif session.get('session_type') == 'Qualifying':
                quali_session = session
            elif 'Practice' in session.get('session_type', ''):
                practice_sessions.append(session)
        
        # Get drivers participating in this race
        if not processed_data['drivers'].empty:
            drivers = processed_data['drivers']['driver_number'].unique()
        elif not processed_data['positions'].empty:
            drivers = processed_data['positions']['driver_number'].unique()
        elif not processed_data['laps'].empty:
            drivers = processed_data['laps']['driver_number'].unique()
        else:
            logger.error("No driver data found!")
            return pd.DataFrame()
        
        logger.info(f"Building features for {len(drivers)} drivers")
        
        features_list = []
        drivers_with_history = []
        drivers_without_history = []
        
        for driver_number in drivers:
            # Start with historical features (driver's past performance)
            driver_hist = self._get_historical_features(driver_number)
            
            # Track which drivers have historical data
            has_history = bool(driver_hist) and driver_hist.get('races_completed', 0) > 0
            
            # Add current weekend features
            weekend_features = self._build_weekend_features(
                driver_number, processed_data, quali_session, practice_sessions
            )
            
            # Get driver info
            driver_info = self._get_driver_info(driver_number, processed_data['drivers'])
            
            if has_history:
                drivers_with_history.append(driver_number)
            else:
                drivers_without_history.append(driver_number)
                # For drivers without history, use conservative defaults (back of grid)
                driver_hist = {
                    'avg_finish_position': 15.0,
                    'best_finish_position': 10.0,
                    'worst_finish_position': 20.0,
                    'finish_position_std': 3.0,
                    'races_completed': 0,
                    'podium_rate': 0.0,
                    'win_rate': 0.0,
                    'avg_finish_last_3': 15.0,
                    'avg_finish_last_5': 15.0,
                    'avg_finish_last_10': 15.0,
                    'trend_last_3': 0.0,
                    'trend_last_5': 0.0,
                    'trend_last_10': 0.0,
                    'team_avg_position': 15.0,
                    'team_consistency': 3.0,
                    'teammate_avg_position': 15.0
                }
            
            # Combine all features
            row = {
                'driver_number': driver_number,
                'session_key': race_session['session_key'] if race_session else race_info['meeting_key'],
                'has_historical_data': has_history,
                **driver_hist,
                **weekend_features,
                **driver_info
            }
            
            features_list.append(row)
        
        logger.info(f"  Drivers with historical data: {len(drivers_with_history)}")
        logger.info(f"  Drivers without history (rookies/reserves): {len(drivers_without_history)}")
        
        features_df = pd.DataFrame(features_list)
        logger.info(f"Built feature matrix with shape {features_df.shape}")
        
        return features_df
    
    def _get_historical_features(self, driver_number: int) -> dict:
        """Get historical features for a driver from stored data"""
        if self.historical_features.empty:
            return {}
        
        # Get most recent features for this driver
        driver_features = self.historical_features[
            self.historical_features['driver_number'] == driver_number
        ]
        
        if driver_features.empty:
            return {}
        
        # Get the most recent record
        latest = driver_features.iloc[-1]
        
        # Extract relevant historical features
        hist_cols = [
            'avg_finish_position', 'best_finish_position', 'worst_finish_position',
            'finish_position_std', 'races_completed', 'podium_rate', 'win_rate',
            'avg_finish_last_3', 'avg_finish_last_5', 'avg_finish_last_10',
            'trend_last_3', 'trend_last_5', 'trend_last_10',
            'team_avg_position', 'team_consistency', 'teammate_avg_position'
        ]
        
        return {col: latest.get(col, np.nan) for col in hist_cols if col in latest}
    
    def _build_weekend_features(self, driver_number: int, processed_data: dict,
                                quali_session: dict, practice_sessions: list) -> dict:
        """Build features from current weekend data (practice, qualifying)"""
        features = {}
        
        laps_df = processed_data.get('laps', pd.DataFrame())
        
        # Collect all weekend laps for this driver (practice + qualifying)
        driver_laps = laps_df[laps_df['driver_number'] == driver_number] if not laps_df.empty else pd.DataFrame()
        
        # Build lap-based features matching the model's expected format
        # NOTE: The training data has a bug where lap times were converted using
        # pd.to_timedelta() without unit='s', resulting in nanosecond interpretation.
        # We need to match this scaling for predictions to work correctly.
        # A proper fix would be to retrain the model with correct lap times.
        TRAINING_DATA_SCALE_FACTOR = 1e-9  # Convert seconds to nanoseconds then to "days"
        
        if not driver_laps.empty and 'lap_time_seconds' in driver_laps.columns:
            valid_laps = driver_laps['lap_time_seconds'].dropna()
            if len(valid_laps) > 0:
                # Apply the same broken scaling as training data
                scaled_laps = valid_laps * TRAINING_DATA_SCALE_FACTOR
                
                features['avg_lap_time'] = scaled_laps.mean()
                features['lap_time_std'] = scaled_laps.std() if len(valid_laps) > 1 else 0
                features['best_lap_time'] = scaled_laps.min()
                features['total_laps'] = len(valid_laps)
                
                # Consistency score (coefficient of variation)
                if features['avg_lap_time'] > 0:
                    features['consistency_score'] = features['lap_time_std'] / features['avg_lap_time']
                else:
                    features['consistency_score'] = 0
        
        # Apply same broken scaling for qualifying lap times
        TRAINING_DATA_SCALE_FACTOR = 1e-9
        
        # Qualifying features
        if quali_session and not laps_df.empty:
            quali_laps = laps_df[
                (laps_df['session_key'] == quali_session['session_key']) &
                (laps_df['driver_number'] == driver_number)
            ]
            
            if not quali_laps.empty and 'lap_time_seconds' in quali_laps.columns:
                features['qualifying_best_lap'] = quali_laps['lap_time_seconds'].min() * TRAINING_DATA_SCALE_FACTOR
                
                # Calculate qualifying position
                all_quali_laps = laps_df[laps_df['session_key'] == quali_session['session_key']]
                if not all_quali_laps.empty:
                    best_laps = all_quali_laps.groupby('driver_number')['lap_time_seconds'].min().sort_values()
                    if driver_number in best_laps.index:
                        features['qualifying_position'] = list(best_laps.index).index(driver_number) + 1
        
        # Weather features from weekend
        weather_df = processed_data.get('weather', pd.DataFrame())
        if not weather_df.empty:
            if 'air_temperature' in weather_df.columns:
                features['avg_air_temp'] = weather_df['air_temperature'].mean()
            if 'track_temperature' in weather_df.columns:
                features['avg_track_temp'] = weather_df['track_temperature'].mean()
            if 'humidity' in weather_df.columns:
                features['avg_humidity'] = weather_df['humidity'].mean()
            if 'rainfall' in weather_df.columns:
                features['rainfall'] = weather_df['rainfall'].sum()
            features['weather_variability'] = weather_df['air_temperature'].std() if 'air_temperature' in weather_df.columns else 0
        
        # Pit stop features (set to 0 for predictions since race hasn't happened)
        features['num_pit_stops'] = 0
        features['avg_pit_duration'] = 0
        features['total_pit_time'] = 0
        
        return features
    
    def _get_driver_info(self, driver_number: int, drivers_df: pd.DataFrame) -> dict:
        """Get driver name and team info"""
        if drivers_df.empty:
            return {'full_name': f'Driver #{driver_number}', 'team_name': 'Unknown'}
        
        driver = drivers_df[drivers_df['driver_number'] == driver_number]
        if driver.empty:
            # Try from historical data
            hist_drivers_path = Path('data/processed/drivers_processed.csv')
            if hist_drivers_path.exists():
                hist_drivers = pd.read_csv(hist_drivers_path)
                driver = hist_drivers[hist_drivers['driver_number'] == driver_number]
        
        if not driver.empty:
            return {
                'full_name': driver.iloc[0].get('full_name', f'Driver #{driver_number}'),
                'team_name': driver.iloc[0].get('team_name', 'Unknown'),
                'name_acronym': driver.iloc[0].get('name_acronym', '')
            }
        
        return {'full_name': f'Driver #{driver_number}', 'team_name': 'Unknown'}
    
    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions using trained XGBoost model"""
        logger.info("\n🤖 Loading XGBoost model and making predictions...")
        
        # Load model
        model_path = Path('data/models/xgboost_model.pkl')
        if not model_path.exists():
            logger.error(f"Model not found at {model_path}")
            logger.error("Please train the model first: python main.py --mode train")
            return pd.DataFrame()
        
        trainer = ModelTrainer({}, {})
        trainer.load_model(str(model_path))
        
        logger.info(f"Model expects {len(trainer.feature_names)} features")
        
        # Get feature columns that match training
        available_features = [f for f in trainer.feature_names if f in features_df.columns]
        missing_features = [f for f in trainer.feature_names if f not in features_df.columns]
        
        logger.info(f"Available features: {len(available_features)}, Missing: {len(missing_features)}")
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features[:10]}...")
            # Fill missing features with median from historical data or 0
            for f in missing_features:
                if not self.historical_features.empty and f in self.historical_features.columns:
                    features_df[f] = self.historical_features[f].median()
                else:
                    features_df[f] = 0
        
        # Prepare features
        X = features_df[trainer.feature_names].copy()
        
        # Debug: Show key features for a few drivers
        logger.info("\n📊 Debug - Key features by driver:")
        key_cols = ['podium_rate', 'avg_finish_position', 'qualifying_position']
        for dn in [1, 4, 44, 31, 18]:  # VER, NOR, HAM, OCO, STR
            driver_row = features_df[features_df['driver_number'] == dn]
            if not driver_row.empty:
                name = driver_row['full_name'].values[0][:15] if 'full_name' in driver_row else f'#{dn}'
                vals = [f"{col}={driver_row[col].values[0]:.2f}" if col in driver_row else f"{col}=N/A" 
                        for col in key_cols]
                logger.info(f"  {name}: {', '.join(vals)}")
        
        X = X.fillna(X.median())
        X = X.fillna(0)
        
        # Replace inf values
        X = X.replace([np.inf, -np.inf], 0)
        
        # Scale if needed
        if trainer.scaler is not None:
            X_scaled = pd.DataFrame(
                trainer.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X
        
        # Predict - model predicts finish position score
        # IMPORTANT: Lower predicted value = worse finish position (higher P#)
        # So we sort DESCENDING - higher score = better predicted finish
        predictions = trainer.model.predict(X_scaled)
        
        # Create results
        results = features_df[['driver_number', 'full_name', 'team_name']].copy()
        results['xgboost_score'] = predictions
        
        # Add qualifying position for context
        if 'qualifying_position' in features_df.columns:
            results['qualifying_position'] = features_df['qualifying_position'].values
        else:
            results['qualifying_position'] = np.nan
        
        # Add other key features for hybrid scoring
        for col in ['podium_rate', 'avg_finish_position', 'has_historical_data']:
            if col in features_df.columns:
                results[col] = features_df[col].values
        
        # HYBRID SCORING SYSTEM
        # The XGBoost model doesn't use qualifying position (it was NaN in training data)
        # but qualifying is HIGHLY predictive of race results (~70% correlation)
        # We create a hybrid score that combines:
        # 1. XGBoost prediction (historical performance) - already on 1-20 scale
        # 2. Qualifying position (current weekend form) - on 1-20 scale
        # 3. Penalty for drivers without historical data
        
        logger.info("\n📊 Building hybrid prediction (XGBoost + Qualifying + Historical)...")
        
        # XGBoost already predicts on a ~1-20 scale (final position)
        # Lower XGB score = better predicted position
        results['xgb_normalized'] = results['xgboost_score']
        
        # Fill missing qualifying positions with back of grid
        results['quali_adj'] = results['qualifying_position'].fillna(20)
        
        # Historical data penalty: drivers without history get worse score
        results['history_penalty'] = results['has_historical_data'].apply(
            lambda x: 0 if x else 5  # +5 positions penalty for no history
        )
        
        # Hybrid score formula:
        # 40% XGBoost (historical performance)
        # 50% Qualifying position (current weekend form - very predictive)
        # 10% History penalty (penalize unknown drivers)
        WEIGHT_XGBOOST = 0.40
        WEIGHT_QUALI = 0.50
        WEIGHT_HISTORY = 0.10
        
        results['predicted_score'] = (
            WEIGHT_XGBOOST * results['xgb_normalized'] +
            WEIGHT_QUALI * results['quali_adj'] +
            WEIGHT_HISTORY * results['history_penalty']
        )
        
        # Sort by predicted score ASCENDING (lower = better)
        results = results.sort_values('predicted_score', ascending=True)
        
        # Assign positions 1 to N based on sorted order
        results['predicted_position'] = range(1, len(results) + 1)
        
        # Debug: show top 5 drivers' scoring breakdown
        logger.info("\nTop 5 predicted drivers scoring breakdown:")
        logger.info(f"{'Driver':<20} {'XGB':<8} {'Quali':<8} {'Hybrid':<8}")
        for _, row in results.head(5).iterrows():
            name = row['full_name'][:18]
            xgb = row['xgb_normalized']
            quali = row['quali_adj']
            hybrid = row['predicted_score']
            logger.info(f"{name:<20} {xgb:<8.2f} {quali:<8.0f} {hybrid:<8.2f}")
        
        return results
    
    def display_predictions(self, predictions: pd.DataFrame, race_info: dict):
        """Display formatted predictions"""
        logger.info("\n" + "="*80)
        logger.info(f"🏆 PREDICTED RACE RESULTS - {race_info['location']}, {race_info['country']}")
        logger.info("="*80)
        
        for idx, row in predictions.head(20).iterrows():
            pos = int(row['predicted_position'])
            name = row.get('full_name', f"Driver #{int(row['driver_number'])}")
            team = row.get('team_name', '')[:20]
            driver_num = int(row['driver_number'])
            
            if pos == 1:
                emoji = "🥇"
            elif pos == 2:
                emoji = "🥈"
            elif pos == 3:
                emoji = "🥉"
            else:
                emoji = f"P{pos:2d}"
            
            logger.info(f"  {emoji}: {name:<25} #{driver_num:<3} {team}")
        
        # Podium
        logger.info("\n" + "="*80)
        logger.info("🏆 PREDICTED PODIUM")
        logger.info("="*80)
        
        podium = predictions.head(3)
        medals = ["🥇 Winner", "🥈 Second", "🥉 Third"]
        for (_, row), medal in zip(podium.iterrows(), medals):
            name = row.get('full_name', f"Driver #{int(row['driver_number'])}")
            logger.info(f"  {medal}: {name} (#{int(row['driver_number'])})")
        
        logger.info("="*80)
    
    def run(self, race_name: str = None, year: int = 2025, meeting_key: int = None,
            session_key: int = None) -> pd.DataFrame:
        """
        Run the full automated prediction pipeline
        
        Args:
            race_name: Name of the race (e.g., "Abu Dhabi", "Austin")
            year: Season year
            meeting_key: Direct meeting key if known
            session_key: Direct session key if known
            
        Returns:
            DataFrame with predictions
        """
        logger.info("\n" + "="*80)
        logger.info("🏎️  AUTOMATED XGBOOST RACE PREDICTOR")
        logger.info("="*80)
        
        # Step 1: Find the race
        race_info = self.find_race(race_name, year, meeting_key)
        if not race_info:
            logger.error(f"Could not find race: {race_name or meeting_key}")
            return pd.DataFrame()
        
        logger.info(f"\n📍 Found race: {race_info['location']}, {race_info['country']}")
        logger.info(f"   Sessions: {len(race_info['sessions'])}")
        
        # Step 2: Fetch race data
        raw_data = self.fetch_race_data(race_info)
        
        # Step 3: Preprocess data
        processed_data = self.preprocess_race_data(raw_data)
        
        # Step 4: Build features
        features_df = self.build_features_for_race(processed_data, race_info)
        
        if features_df.empty:
            logger.error("Could not build features - insufficient data")
            return pd.DataFrame()
        
        # Step 5: Make predictions
        predictions = self.predict(features_df)
        
        if predictions.empty:
            logger.error("Could not make predictions")
            return pd.DataFrame()
        
        # Step 6: Display results
        self.display_predictions(predictions, race_info)
        
        # Cleanup
        self.client.close()
        
        return predictions


def main():
    parser = argparse.ArgumentParser(
        description='Automated XGBoost Race Predictor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_race_xgboost.py --race "Abu Dhabi" --year 2025
  python predict_race_xgboost.py --race "Lusail" --year 2025
  python predict_race_xgboost.py --meeting-key 1242
        """
    )
    
    parser.add_argument('--race', type=str, help='Race name (e.g., "Abu Dhabi", "Austin")')
    parser.add_argument('--year', type=int, default=2025, help='Season year (default: 2025)')
    parser.add_argument('--meeting-key', type=int, help='Direct meeting key')
    parser.add_argument('--session-key', type=int, help='Direct session key')
    
    args = parser.parse_args()
    
    if not args.race and not args.meeting_key and not args.session_key:
        parser.print_help()
        print("\n❌ Please provide --race, --meeting-key, or --session-key")
        sys.exit(1)
    
    # Run predictor
    predictor = AutomatedRacePredictor()
    predictions = predictor.run(
        race_name=args.race,
        year=args.year,
        meeting_key=args.meeting_key,
        session_key=args.session_key
    )
    
    if not predictions.empty:
        print(f"\n✅ Prediction complete! {len(predictions)} drivers predicted.")


if __name__ == '__main__':
    main()
