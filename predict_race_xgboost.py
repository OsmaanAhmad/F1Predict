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
        
        for driver_number in drivers:
            # Start with historical features (driver's past performance)
            driver_hist = self._get_historical_features(driver_number)
            
            # Add current weekend features
            weekend_features = self._build_weekend_features(
                driver_number, processed_data, quali_session, practice_sessions
            )
            
            # Get driver info
            driver_info = self._get_driver_info(driver_number, processed_data['drivers'])
            
            # Combine all features
            row = {
                'driver_number': driver_number,
                'session_key': race_session['session_key'] if race_session else race_info['meeting_key'],
                **driver_hist,
                **weekend_features,
                **driver_info
            }
            
            features_list.append(row)
        
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
        
        # Qualifying features
        if quali_session and not laps_df.empty:
            quali_laps = laps_df[
                (laps_df['session_key'] == quali_session['session_key']) &
                (laps_df['driver_number'] == driver_number)
            ]
            
            if not quali_laps.empty and 'lap_time_seconds' in quali_laps.columns:
                features['qualifying_best_lap'] = quali_laps['lap_time_seconds'].min()
                
                # Calculate qualifying position
                all_quali_laps = laps_df[laps_df['session_key'] == quali_session['session_key']]
                if not all_quali_laps.empty:
                    best_laps = all_quali_laps.groupby('driver_number')['lap_time_seconds'].min().sort_values()
                    if driver_number in best_laps.index:
                        features['qualifying_position'] = list(best_laps.index).index(driver_number) + 1
        
        # Practice features (best lap from all practice sessions)
        practice_laps = []
        for practice in practice_sessions:
            if not laps_df.empty:
                p_laps = laps_df[
                    (laps_df['session_key'] == practice['session_key']) &
                    (laps_df['driver_number'] == driver_number)
                ]
                if not p_laps.empty and 'lap_time_seconds' in p_laps.columns:
                    practice_laps.extend(p_laps['lap_time_seconds'].dropna().tolist())
        
        if practice_laps:
            features['practice_best_lap'] = min(practice_laps)
            features['practice_avg_lap'] = np.mean(practice_laps)
            features['practice_lap_count'] = len(practice_laps)
        
        # Weather features from weekend
        weather_df = processed_data.get('weather', pd.DataFrame())
        if not weather_df.empty:
            if 'air_temperature' in weather_df.columns:
                features['avg_air_temp'] = weather_df['air_temperature'].mean()
            if 'track_temperature' in weather_df.columns:
                features['avg_track_temp'] = weather_df['track_temperature'].mean()
            if 'humidity' in weather_df.columns:
                features['avg_humidity'] = weather_df['humidity'].mean()
        
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
        
        # Get feature columns that match training
        available_features = [f for f in trainer.feature_names if f in features_df.columns]
        missing_features = [f for f in trainer.feature_names if f not in features_df.columns]
        
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features - filling with median/0")
            for f in missing_features:
                features_df[f] = 0
        
        # Prepare features
        X = features_df[trainer.feature_names].copy()
        X = X.fillna(X.median())
        X = X.fillna(0)
        
        # Scale if needed
        if trainer.scaler is not None:
            X_scaled = pd.DataFrame(
                trainer.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X
        
        # Predict
        predictions = trainer.model.predict(X_scaled)
        
        # Create results
        results = features_df[['driver_number', 'full_name', 'team_name']].copy()
        results['predicted_score'] = predictions
        results['predicted_position'] = results['predicted_score'].rank().astype(int)
        
        # Sort by predicted position
        results = results.sort_values('predicted_position')
        
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
