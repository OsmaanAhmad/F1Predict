"""
Future Race Predictor - Uses all available pre-race data
Automatically adapts to whatever practice/qualifying data exists
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from src.models import ModelTrainer, RacePredictor
from src.utils import load_config
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_race_weekend_data(meeting_key: int, data_dir: str = 'data/raw'):
    """Load all available data for a race weekend"""
    
    # Find the year for this meeting
    for year in [2023, 2024, 2025, 2026]:
        sessions_file = Path(data_dir) / str(year) / 'sessions.csv'
        if not sessions_file.exists():
            continue
            
        sessions_df = pd.read_csv(sessions_file)
        meeting_sessions = sessions_df[sessions_df['meeting_key'] == meeting_key]
        
        if meeting_sessions.empty:
            continue
        
        # Found the meeting - now load all available data
        data = {
            'sessions': meeting_sessions,
            'year': year,
            'meeting_key': meeting_key
        }
        
        # Load laps
        laps_file = Path(data_dir) / str(year) / 'laps.csv'
        if laps_file.exists():
            laps_df = pd.read_csv(laps_file)
            meeting_laps = laps_df[laps_df['meeting_key'] == meeting_key]
            if not meeting_laps.empty:
                data['laps'] = meeting_laps
                logger.info(f"✓ Loaded {len(meeting_laps)} laps from {meeting_laps['session_key'].nunique()} sessions")
        
        # Load positions
        positions_file = Path(data_dir) / str(year) / 'positions.csv'
        if positions_file.exists():
            positions_df = pd.read_csv(positions_file)
            meeting_positions = positions_df[positions_df['meeting_key'] == meeting_key]
            if not meeting_positions.empty:
                data['positions'] = meeting_positions
                logger.info(f"✓ Loaded {len(meeting_positions)} position records")
        
        # Load weather
        weather_file = Path(data_dir) / str(year) / 'weather.csv'
        if weather_file.exists():
            weather_df = pd.read_csv(weather_file)
            meeting_weather = weather_df[weather_df['meeting_key'] == meeting_key]
            if not meeting_weather.empty:
                data['weather'] = meeting_weather
                logger.info(f"✓ Loaded {len(meeting_weather)} weather records")
        
        # Load pit stops
        pit_stops_file = Path(data_dir) / str(year) / 'pit_stops.csv'
        if pit_stops_file.exists():
            pit_stops_df = pd.read_csv(pit_stops_file)
            meeting_pit_stops = pit_stops_df[pit_stops_df['meeting_key'] == meeting_key]
            if not meeting_pit_stops.empty:
                data['pit_stops'] = meeting_pit_stops
                logger.info(f"✓ Loaded {len(meeting_pit_stops)} pit stop records")
        
        return data
    
    return None


def categorize_sessions(sessions_df):
    """Categorize sessions by type"""
    session_types = {
        'practice': [],
        'qualifying': [],
        'sprint': [],
        'race': []
    }
    
    for _, session in sessions_df.iterrows():
        session_type = session['session_type'].lower()
        session_name = session['session_name'].lower()
        
        if 'practice' in session_type or 'practice' in session_name:
            session_types['practice'].append(session['session_key'])
        elif 'qualifying' in session_type or 'qualifying' in session_name:
            session_types['qualifying'].append(session['session_key'])
        elif 'sprint' in session_type or 'sprint' in session_name:
            session_types['sprint'].append(session['session_key'])
        elif 'race' in session_type:
            session_types['race'].append(session['session_key'])
    
    return session_types


def extract_weekend_features(data: dict) -> pd.DataFrame:
    """Extract features from all available weekend data"""
    
    sessions = data['sessions']
    session_types = categorize_sessions(sessions)
    
    # Log what data we have
    logger.info("\n📊 Available Session Data:")
    if session_types['practice']:
        logger.info(f"  • Practice: {len(session_types['practice'])} session(s)")
    if session_types['qualifying']:
        logger.info(f"  • Qualifying: {len(session_types['qualifying'])} session(s)")
    if session_types['sprint']:
        logger.info(f"  • Sprint: {len(session_types['sprint'])} session(s)")
    if session_types['race']:
        logger.info(f"  • Race: {len(session_types['race'])} session(s)")
    
    features_list = []
    
    # Get unique drivers from whatever data we have
    drivers = set()
    if 'laps' in data and not data['laps'].empty:
        drivers.update(data['laps']['driver_number'].unique())
    if 'positions' in data and not data['positions'].empty:
        drivers.update(data['positions']['driver_number'].unique())
    
    if not drivers:
        logger.warning("⚠️  No lap/position data available for this weekend yet")
        logger.info("📊 Will predict based on historical performance only")
        
        # Get drivers from historical data
        features_df = pd.read_csv('data/processed/features.csv')
        drivers = features_df['driver_number'].unique()
        logger.info(f"  • Using {len(drivers)} drivers from historical data\n")
    
    logger.info(f"  • Drivers: {len(drivers)} drivers found\n")
    
    # Extract features for each driver
    for driver in drivers:
        driver_features = {
            'driver_number': driver,
            'meeting_key': data['meeting_key']
        }
        
        # 1. PRACTICE SESSION FEATURES
        if 'laps' in data:
            laps_df = data['laps'].copy()
            
            # Convert lap duration to seconds
            if 'lap_duration' in laps_df.columns:
                laps_df['lap_time_seconds'] = pd.to_timedelta(
                    laps_df['lap_duration'], 
                    errors='coerce'
                ).dt.total_seconds()
            
            practice_laps = laps_df[
                (laps_df['driver_number'] == driver) & 
                (laps_df['session_key'].isin(session_types['practice']))
            ]
            
            if not practice_laps.empty and 'lap_time_seconds' in practice_laps.columns:
                valid_laps = practice_laps[practice_laps['lap_time_seconds'].notna()]['lap_time_seconds']
                if len(valid_laps) > 0:
                    driver_features['practice_best_lap'] = valid_laps.min()
                    driver_features['practice_avg_lap'] = valid_laps.mean()
                    driver_features['practice_median_lap'] = valid_laps.median()
                    driver_features['practice_lap_std'] = valid_laps.std()
                    driver_features['practice_laps_completed'] = len(valid_laps)
                    driver_features['practice_consistency'] = valid_laps.std() / valid_laps.mean() if valid_laps.mean() > 0 else np.nan
                    
                    # Sector speeds
                    if 'i1_speed' in practice_laps.columns:
                        driver_features['practice_avg_i1_speed'] = practice_laps['i1_speed'].mean()
                    if 'i2_speed' in practice_laps.columns:
                        driver_features['practice_avg_i2_speed'] = practice_laps['i2_speed'].mean()
                    if 'st_speed' in practice_laps.columns:
                        driver_features['practice_avg_st_speed'] = practice_laps['st_speed'].mean()
        
        # 2. QUALIFYING FEATURES
        if 'laps' in data:
            quali_laps = laps_df[
                (laps_df['driver_number'] == driver) & 
                (laps_df['session_key'].isin(session_types['qualifying']))
            ]
            
            if not quali_laps.empty and 'lap_time_seconds' in quali_laps.columns:
                valid_laps = quali_laps[quali_laps['lap_time_seconds'].notna()]['lap_time_seconds']
                if len(valid_laps) > 0:
                    driver_features['quali_best_lap'] = valid_laps.min()
                    driver_features['quali_avg_lap'] = valid_laps.mean()
                    driver_features['quali_laps_completed'] = len(valid_laps)
        
        # 3. GRID POSITION (from positions data)
        if 'positions' in data:
            positions_df = data['positions']
            quali_positions = positions_df[
                (positions_df['driver_number'] == driver) &
                (positions_df['session_key'].isin(session_types['qualifying']))
            ]
            if not quali_positions.empty and 'position' in quali_positions.columns:
                # Get best (lowest) position from qualifying
                best_quali_pos = quali_positions['position'].min()
                driver_features['grid_position'] = best_quali_pos
        
        # 4. SPRINT FEATURES (if available)
        if 'laps' in data and session_types['sprint']:
            sprint_laps = laps_df[
                (laps_df['driver_number'] == driver) & 
                (laps_df['session_key'].isin(session_types['sprint']))
            ]
            
            if not sprint_laps.empty and 'lap_time_seconds' in sprint_laps.columns:
                valid_laps = sprint_laps[sprint_laps['lap_time_seconds'].notna()]['lap_time_seconds']
                if len(valid_laps) > 0:
                    driver_features['sprint_best_lap'] = valid_laps.min()
                    driver_features['sprint_avg_lap'] = valid_laps.mean()
        
        # 5. WEATHER CONDITIONS (average across weekend)
        if 'weather' in data:
            weather_df = data['weather']
            if not weather_df.empty:
                if 'air_temperature' in weather_df.columns:
                    driver_features['avg_air_temp'] = weather_df['air_temperature'].mean()
                if 'track_temperature' in weather_df.columns:
                    driver_features['avg_track_temp'] = weather_df['track_temperature'].mean()
                if 'humidity' in weather_df.columns:
                    driver_features['avg_humidity'] = weather_df['humidity'].mean()
                if 'rainfall' in weather_df.columns:
                    driver_features['rainfall'] = weather_df['rainfall'].mean()
        
        features_list.append(driver_features)
    
    return pd.DataFrame(features_list)


def load_historical_features(driver_number: int, before_date: str = None) -> dict:
    """Load historical performance features for a driver"""
    
    features_df = pd.read_csv('data/processed/features.csv')
    
    # Get driver's historical races
    driver_history = features_df[features_df['driver_number'] == driver_number]
    
    if driver_history.empty:
        return {}
    
    # Get last 5-10 races for recent form
    recent_races = driver_history.tail(10)
    last_5 = driver_history.tail(5)
    
    historical = {}
    
    # Recent finishing positions
    if 'final_position' in recent_races.columns:
        positions = recent_races['final_position'].dropna()
        if len(positions) > 0:
            historical['avg_finish_last_10'] = positions.mean()
            historical['best_finish_last_10'] = positions.min()
            historical['worst_finish_last_10'] = positions.max()
            historical['finish_std_last_10'] = positions.std()
        
        last_5_positions = last_5['final_position'].dropna()
        if len(last_5_positions) > 0:
            historical['avg_finish_last_5'] = last_5_positions.mean()
            historical['best_finish_last_5'] = last_5_positions.min()
            
            # Recent form trend
            if len(last_5_positions) >= 3:
                historical['position_lag1'] = last_5_positions.iloc[-1] if len(last_5_positions) > 0 else np.nan
                historical['position_lag2'] = last_5_positions.iloc[-2] if len(last_5_positions) > 1 else np.nan
                historical['position_lag3'] = last_5_positions.iloc[-3] if len(last_5_positions) > 2 else np.nan
    
    # Podium rate
    if 'final_position' in recent_races.columns:
        podiums = (recent_races['final_position'] <= 3).sum()
        historical['podium_rate'] = podiums / len(recent_races)
        
        wins = (recent_races['final_position'] == 1).sum()
        historical['win_rate'] = wins / len(recent_races)
    
    # Consistency
    if 'consistency_score' in recent_races.columns:
        historical['avg_consistency'] = recent_races['consistency_score'].mean()
    
    # Qualifying performance
    if 'qualifying_position' in recent_races.columns:
        quali = recent_races['qualifying_position'].dropna()
        if len(quali) > 0:
            historical['avg_qualifying_position'] = quali.mean()
    
    return historical


def combine_features(weekend_features: pd.DataFrame, drivers_df: pd.DataFrame) -> pd.DataFrame:
    """Combine weekend features with historical data and driver info"""
    
    logger.info("🔄 Merging weekend data with historical performance...")
    
    all_features = []
    
    for _, driver_row in weekend_features.iterrows():
        driver_num = driver_row['driver_number']
        features = driver_row.to_dict()
        
        # Add historical features
        historical = load_historical_features(driver_num)
        features.update(historical)
        
        # Add driver name/team
        driver_info = drivers_df[drivers_df['driver_number'] == driver_num]
        if not driver_info.empty:
            driver_data = driver_info.iloc[0]
            features['full_name'] = driver_data.get('full_name', f"Driver #{driver_num}")
            features['team_name'] = driver_data.get('team_name', 'Unknown')
            features['name_acronym'] = driver_data.get('name_acronym', '')
        
        all_features.append(features)
    
    combined_df = pd.DataFrame(all_features)
    
    # Log feature availability
    logger.info(f"\n✓ Created feature matrix: {len(combined_df)} drivers")
    logger.info(f"  • Total features: {len(combined_df.columns)}")
    
    # Show which key features are available
    key_features = ['practice_best_lap', 'quali_best_lap', 'grid_position', 'avg_finish_last_5', 'podium_rate']
    available = [f for f in key_features if f in combined_df.columns and combined_df[f].notna().any()]
    logger.info(f"  • Key features available: {', '.join(available)}")
    
    return combined_df


def predict_race(meeting_key: int, location: str = None):
    """Predict race outcome using all available pre-race data"""
    
    logger.info("="*80)
    logger.info("🏁 FUTURE RACE PREDICTION ENGINE")
    logger.info("="*80)
    
    # Load weekend data
    logger.info(f"\n📡 Loading data for meeting {meeting_key}...")
    weekend_data = load_race_weekend_data(meeting_key)
    
    if not weekend_data:
        logger.error(f"❌ No data found for meeting {meeting_key}")
        return
    
    # Get race info
    race_info = weekend_data['sessions'].iloc[0]
    race_name = f"{race_info['country_name']} GP" if pd.notna(race_info['country_name']) else race_info['location']
    
    logger.info(f"\n🏁 PREDICTING: {weekend_data['year']} {race_name}")
    logger.info(f"📍 Location: {race_info['location']}")
    logger.info(f"📅 Date: {race_info['date_start'][:10]}")
    logger.info("="*80)
    
    # Extract features from weekend
    logger.info("\n🔍 Extracting Features from Weekend Sessions...")
    weekend_features = extract_weekend_features(weekend_data)
    
    if weekend_features.empty:
        logger.error("❌ Could not extract features from weekend data")
        return
    
    # Load driver info
    drivers_df = pd.read_csv('data/processed/drivers_processed.csv')
    drivers_df = drivers_df[['driver_number', 'full_name', 'name_acronym', 'team_name']].drop_duplicates('driver_number')
    
    # Combine with historical data
    prediction_features = combine_features(weekend_features, drivers_df)
    
    # Load trained model
    logger.info("\n🤖 Loading ML Model...")
    config = load_config('config.yaml')
    trainer = ModelTrainer(config['models'], config['training'])
    
    try:
        trainer.load_model('data/models/xgboost_model.pkl')
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        logger.info("\n💡 Train a model first: python main.py --mode train")
        return
    
    # Prepare features for prediction
    logger.info("\n🎯 Generating Predictions...")
    
    # Get the feature names the model was trained on
    model_features = trainer.feature_names
    
    # Create prediction DataFrame with model features
    X_pred = pd.DataFrame()
    for feature in model_features:
        if feature in prediction_features.columns:
            X_pred[feature] = prediction_features[feature]
        else:
            # Feature not available - use median from training data or 0
            X_pred[feature] = 0
    
    # Fill NaN values
    X_pred = X_pred.fillna(X_pred.median()).fillna(0)
    
    # Make predictions
    try:
        predictions = trainer.model.predict(X_pred)
        prediction_features['predicted_position'] = predictions
        prediction_features['predicted_position_int'] = np.round(predictions).astype(int).clip(1, 20)
        
        # Sort by predicted position
        prediction_features = prediction_features.sort_values('predicted_position')
        prediction_features['predicted_rank'] = range(1, len(prediction_features) + 1)
        
        # Display results
        logger.info("\n" + "="*80)
        logger.info("🏆 PREDICTED RACE RESULTS")
        logger.info("="*80)
        
        for idx, row in prediction_features.head(15).iterrows():
            pos = int(row['predicted_position_int'])
            name = row.get('full_name', f"Driver #{int(row['driver_number'])}")
            team = row.get('team_name', '')
            driver_num = int(row['driver_number'])
            
            emoji = "🥇" if pos == 1 else "🥈" if pos == 2 else "🥉" if pos == 3 else f"P{pos:2d}"
            
            # Add context from weekend
            context = []
            if 'quali_best_lap' in row and pd.notna(row['quali_best_lap']):
                context.append(f"Q: {row['quali_best_lap']:.3f}s")
            elif 'practice_best_lap' in row and pd.notna(row['practice_best_lap']):
                context.append(f"FP: {row['practice_best_lap']:.3f}s")
            
            if 'grid_position' in row and pd.notna(row['grid_position']):
                context.append(f"Grid: P{int(row['grid_position'])}")
            
            context_str = f" | {', '.join(context)}" if context else ""
            
            logger.info(f"  {emoji}: {name:<25} #{driver_num:<3} {team:<20}{context_str}")
        
        # Podium prediction
        logger.info("\n" + "="*80)
        logger.info("🏆 PREDICTED PODIUM")
        logger.info("="*80)
        
        podium = prediction_features.head(3)
        for i, (_, driver) in enumerate(podium.iterrows()):
            medals = ["🥇 Winner: ", "🥈 Second: ", "🥉 Third:  "]
            name = driver.get('full_name', f"Driver #{int(driver['driver_number'])}")
            logger.info(f"  {medals[i]}{name} (#{int(driver['driver_number'])})")
        
        logger.info("="*80)
        
        # Show prediction basis
        logger.info("\n💡 Prediction based on:")
        data_sources = []
        if 'practice_best_lap' in prediction_features.columns and prediction_features['practice_best_lap'].notna().any():
            data_sources.append("✓ Practice session lap times")
        if 'quali_best_lap' in prediction_features.columns and prediction_features['quali_best_lap'].notna().any():
            data_sources.append("✓ Qualifying performance")
        if 'grid_position' in prediction_features.columns and prediction_features['grid_position'].notna().any():
            data_sources.append("✓ Grid positions")
        if 'avg_finish_last_5' in prediction_features.columns and prediction_features['avg_finish_last_5'].notna().any():
            data_sources.append("✓ Historical race results (2023-2024)")
        if 'avg_air_temp' in prediction_features.columns and prediction_features['avg_air_temp'].notna().any():
            data_sources.append("✓ Weather conditions")
        
        for source in data_sources:
            logger.info(f"  {source}")
        
        if not data_sources:
            logger.warning("  ⚠️  Limited data available - predictions based primarily on historical form")
        
        logger.info("\n⚠️  Note: This is a probabilistic forecast. Actual results depend on")
        logger.info("    race strategy, incidents, weather changes, and other unpredictable factors.")
        logger.info("\n💡 Tip: Re-run after more sessions (FP2, FP3, Qualifying) for improved accuracy!\n")
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='Predict future F1 race using all available pre-race data'
    )
    parser.add_argument('--meeting', type=int, required=True,
                       help='Meeting key for the race weekend')
    parser.add_argument('--location', type=str, default=None,
                       help='Race location (optional, for display)')
    
    args = parser.parse_args()
    
    predict_race(args.meeting, args.location)


if __name__ == '__main__':
    main()
