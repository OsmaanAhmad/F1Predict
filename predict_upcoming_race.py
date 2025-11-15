"""
Predict upcoming race using practice/qualifying data and historical performance
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from src.models import ModelTrainer, RacePredictor
from src.utils import load_config
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_session_data(meeting_key: int, data_dir: str = 'data/raw'):
    """Load all session data for a race weekend"""
    sessions = {}
    
    # Find year from sessions file
    for year in [2023, 2024, 2025]:
        sessions_file = Path(data_dir) / str(year) / 'sessions.csv'
        if sessions_file.exists():
            df = pd.read_csv(sessions_file)
            meeting = df[df['meeting_key'] == meeting_key]
            if not meeting.empty:
                sessions['sessions'] = meeting
                
                # Load laps, positions for this meeting
                laps_file = Path(data_dir) / str(year) / 'laps.csv'
                if laps_file.exists():
                    laps = pd.read_csv(laps_file)
                    meeting_laps = laps[laps['meeting_key'] == meeting_key]
                    logger.info(f"Found {len(meeting_laps)} laps for meeting {meeting_key}")
                    sessions['laps'] = meeting_laps
                else:
                    logger.warning(f"Laps file not found: {laps_file}")
                
                positions_file = Path(data_dir) / str(year) / 'positions.csv'
                if positions_file.exists():
                    positions = pd.read_csv(positions_file)
                    sessions['positions'] = positions[positions['meeting_key'] == meeting_key]
                
                return sessions, year
    
    return None, None


def extract_practice_quali_features(sessions_data: dict) -> pd.DataFrame:
    """Extract features from practice and qualifying sessions"""
    
    features_list = []
    
    if 'laps' not in sessions_data or sessions_data['laps'].empty:
        logger.warning("No lap data available")
        return pd.DataFrame()
    
    laps_df = sessions_data['laps'].copy()
    sessions_df = sessions_data['sessions']
    
    # Convert lap_duration to seconds
    if 'lap_duration' in laps_df.columns:
        laps_df['lap_time_seconds'] = pd.to_timedelta(laps_df['lap_duration'], errors='coerce').dt.total_seconds()
    
    # Get practice and qualifying sessions
    practice_sessions = sessions_df[sessions_df['session_type'] == 'Practice']['session_key'].tolist()
    quali_sessions = sessions_df[sessions_df['session_type'].isin(['Qualifying', 'Sprint Qualifying'])]['session_key'].tolist()
    
    practice_laps = laps_df[laps_df['session_key'].isin(practice_sessions)]
    quali_laps = laps_df[laps_df['session_key'].isin(quali_sessions)]
    
    # Get unique drivers
    drivers = laps_df['driver_number'].unique()
    
    for driver in drivers:
        features = {'driver_number': driver}
        
        # Practice features
        driver_practice = practice_laps[practice_laps['driver_number'] == driver]
        if not driver_practice.empty and driver_practice['lap_time_seconds'].notna().any():
            valid_laps = driver_practice[driver_practice['lap_time_seconds'].notna()]['lap_time_seconds']
            features['practice_best_lap'] = valid_laps.min()
            features['practice_avg_lap'] = valid_laps.mean()
            features['practice_lap_std'] = valid_laps.std()
            features['practice_laps_completed'] = len(valid_laps)
        else:
            features['practice_best_lap'] = np.nan
            features['practice_avg_lap'] = np.nan
            features['practice_lap_std'] = np.nan
            features['practice_laps_completed'] = 0
        
        # Qualifying features
        driver_quali = quali_laps[quali_laps['driver_number'] == driver]
        if not driver_quali.empty and driver_quali['lap_time_seconds'].notna().any():
            valid_laps = driver_quali[driver_quali['lap_time_seconds'].notna()]['lap_time_seconds']
            features['quali_best_lap'] = valid_laps.min()
            features['quali_avg_lap'] = valid_laps.mean()
            features['quali_laps_completed'] = len(valid_laps)
        else:
            features['quali_best_lap'] = np.nan
            features['quali_avg_lap'] = np.nan
            features['quali_laps_completed'] = 0
        
        features_list.append(features)
    
    return pd.DataFrame(features_list)


def predict_upcoming_race(meeting_key: int, location: str = None):
    """Predict race outcome using practice/qualifying data"""
    
    logger.info("="*70)
    logger.info("🏁 PREDICTING UPCOMING RACE")
    logger.info("="*70)
    
    # Load session data
    sessions_data, year = load_session_data(meeting_key)
    
    if sessions_data is None:
        logger.error(f"No data found for meeting {meeting_key}")
        return
    
    # Get race info
    sessions_df = sessions_data['sessions']
    race_info = sessions_df.iloc[0]
    race_name = f"{race_info['country_name']} GP" if pd.notna(race_info['country_name']) else race_info['location']
    
    logger.info(f"\n🏁 RACE: {year} {race_name}")
    logger.info(f"📍 Location: {race_info['location']}")
    logger.info(f"📅 Date: {race_info['date_start'][:10]}")
    logger.info("="*70)
    
    # Extract features from practice and qualifying
    logger.info("\n📊 Analyzing Practice & Qualifying Data...")
    weekend_features = extract_practice_quali_features(sessions_data)
    
    if weekend_features.empty:
        logger.error("Could not extract features from session data")
        return
    
    logger.info(f"✓ Extracted features for {len(weekend_features)} drivers")
    
    # Load historical features
    logger.info("\n📈 Loading Historical Performance Data...")
    features_df = pd.read_csv('data/processed/features.csv')
    
    # Get latest historical performance for each driver
    latest_performance = features_df.groupby('driver_number').tail(5).groupby('driver_number').mean()
    
    # Merge weekend features with historical performance
    prediction_features = weekend_features.merge(
        latest_performance,
        on='driver_number',
        how='left',
        suffixes=('', '_hist')
    )
    
    # Fill missing historical data with medians
    for col in prediction_features.columns:
        if prediction_features[col].isna().all():
            prediction_features[col] = 0
        elif prediction_features[col].isna().any():
            prediction_features[col].fillna(prediction_features[col].median(), inplace=True)
    
    # Load trained model
    logger.info("\n🤖 Loading Trained Model...")
    trainer = ModelTrainer({}, {})
    trainer.load_model('data/models/xgboost_model.pkl')
    
    # Initialize predictor
    predictor = RacePredictor(trainer)
    
    # Make predictions
    logger.info("\n🎯 Generating Race Predictions...")
    
    # Create a mock session_key for prediction
    prediction_features['session_key'] = meeting_key
    
    # Predict
    try:
        predictions = predictor.predict(prediction_features)
        predictions = predictions.sort_values('predicted_position')
        
        # Get driver names
        drivers_df = pd.read_csv('data/processed/drivers_processed.csv')
        driver_names = drivers_df[['driver_number', 'full_name', 'name_acronym', 'team_name']].drop_duplicates('driver_number')
        predictions = predictions.merge(
            driver_names,
            on='driver_number',
            how='left'
        )
        
        # Display predictions
        logger.info("\n" + "="*70)
        logger.info("🏆 PREDICTED RACE RESULTS")
        logger.info("="*70)
        
        for idx, row in predictions.head(15).iterrows():
            pos = int(row['predicted_position_int'])
            driver_name = row.get('full_name', f"Driver #{int(row['driver_number'])}")
            if pd.isna(driver_name):
                driver_name = f"Driver #{int(row['driver_number'])}"
            driver_num = int(row['driver_number'])
            
            emoji = "🥇" if pos == 1 else "🥈" if pos == 2 else "🥉" if pos == 3 else f"P{pos:2d}"
            
            # Add practice/quali context
            quali_time = row.get('quali_best_lap', np.nan)
            quali_str = f" | Quali: {quali_time:.3f}s" if pd.notna(quali_time) else ""
            
            logger.info(f"  {emoji}: {driver_name} (#{driver_num}){quali_str}")
        
        # Podium prediction
        podium = predictions.head(3)
        logger.info("\n" + "="*70)
        logger.info("🏆 PREDICTED PODIUM")
        logger.info("="*70)
        
        if len(podium) > 0:
            p1 = podium.iloc[0]
            p1_name = p1.get('full_name', f"Driver #{int(p1['driver_number'])}")
            if pd.isna(p1_name):
                p1_name = f"Driver #{int(p1['driver_number'])}"
            logger.info(f"  🥇 Winner:  {p1_name} (#{int(p1['driver_number'])})")
        if len(podium) > 1:
            p2 = podium.iloc[1]
            p2_name = p2.get('full_name', f"Driver #{int(p2['driver_number'])}")
            if pd.isna(p2_name):
                p2_name = f"Driver #{int(p2['driver_number'])}"
            logger.info(f"  🥈 Second:  {p2_name} (#{int(p2['driver_number'])})")
        if len(podium) > 2:
            p3 = podium.iloc[2]
            p3_name = p3.get('full_name', f"Driver #{int(p3['driver_number'])}")
            if pd.isna(p3_name):
                p3_name = f"Driver #{int(p3['driver_number'])}"
            logger.info(f"  🥉 Third:   {p3_name} (#{int(p3['driver_number'])})")
        
        logger.info("="*70)
        
        logger.info("\n💡 Prediction based on:")
        logger.info("  ✓ Practice session performance")
        logger.info("  ✓ Qualifying/Sprint Qualifying results")
        logger.info("  ✓ Historical driver & team performance")
        logger.info("  ✓ Recent form (last 5 races)")
        
        logger.info("\n⚠️  Note: Predictions are probabilistic. Actual race results may vary")
        logger.info("    due to strategy, weather, incidents, and other race-day factors.\n")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Predict upcoming F1 race')
    parser.add_argument('--meeting', type=int, required=True,
                       help='Meeting key for the race weekend')
    parser.add_argument('--location', type=str, default=None,
                       help='Race location (optional, for display)')
    
    args = parser.parse_args()
    
    predict_upcoming_race(args.meeting, args.location)


if __name__ == '__main__':
    main()
