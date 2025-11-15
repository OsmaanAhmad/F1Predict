"""
Predict 2025 Austin GP using ALL available session data
Automatically loads FP1, FP2, FP3, Sprint Quali, Sprint Race, Main Quali
"""
import pandas as pd
import numpy as np
from src.models import ModelTrainer
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_austin_sessions():
    """Find all Austin 2025 session keys"""
    sessions = pd.read_csv('data/raw/2025/sessions.csv')
    austin = sessions[
        (sessions['meeting_key'] == 1271) | 
        (sessions['location'].str.contains('Austin', case=False, na=False))
    ]
    return austin


def load_all_austin_data():
    """Load ALL available Austin session data"""
    austin_sessions = find_austin_sessions()
    
    if austin_sessions.empty:
        logger.warning("No Austin sessions found!")
        return pd.DataFrame(), pd.DataFrame(), []
    
    logger.info(f"\n� Found {len(austin_sessions)} Austin 2025 Sessions:")
    for _, session in austin_sessions.iterrows():
        logger.info(f"  • {session['session_name']} (Session {session['session_key']})")
    
    all_laps = []
    all_positions = []
    loaded_sessions = []
    
    # Track which sessions we've processed
    session_keys_processed = set()
    
    # Try loading from individual session files first
    for _, session in austin_sessions.iterrows():
        session_key = session['session_key']
        session_name = session['session_name']
        session_keys_processed.add(session_key)
        
        # Check in data/raw/2025/
        lap_file = Path(f'data/raw/2025/laps_session_{session_key}.csv')
        pos_file = Path(f'data/raw/2025/positions_session_{session_key}.csv')
        
        if lap_file.exists():
            laps = pd.read_csv(lap_file)
            if not laps.empty:
                logger.info(f"  ✓ Loaded {len(laps)} laps from {session_name}")
                all_laps.append(laps)
                loaded_sessions.append(session_name)
        
        if pos_file.exists():
            positions = pd.read_csv(pos_file)
            if not positions.empty:
                all_positions.append(positions)
    
    # Also scan for any other lap files (API may use different session keys)
    logger.info("\n🔍 Scanning for additional session data files...")
    data_dir = Path('data/raw/2025')
    for lap_file in data_dir.glob('laps_session_*.csv'):
        # Extract session key from filename
        session_key = int(lap_file.stem.split('_')[-1])
        if session_key not in session_keys_processed:
            logger.info(f"  • Found additional session {session_key}")
            laps = pd.read_csv(lap_file)
            if not laps.empty:
                logger.info(f"  ✓ Loaded {len(laps)} laps from session {session_key}")
                all_laps.append(laps)
                loaded_sessions.append(f"Session {session_key}")
                session_keys_processed.add(session_key)
            
            # Also load positions if exists
            pos_file = data_dir / f'positions_session_{session_key}.csv'
            if pos_file.exists():
                positions = pd.read_csv(pos_file)
                if not positions.empty:
                    all_positions.append(positions)
    
    # Combine all data
    combined_laps = pd.concat(all_laps, ignore_index=True) if all_laps else pd.DataFrame()
    combined_positions = pd.concat(all_positions, ignore_index=True) if all_positions else pd.DataFrame()
    
    if not combined_laps.empty:
        logger.info(f"\n✓ Total: {len(combined_laps)} laps from {len(loaded_sessions)} sessions")
        logger.info(f"✓ Drivers: {combined_laps['driver_number'].nunique()}")
    
    return combined_laps, combined_positions, loaded_sessions


logger.info("="*80)
logger.info("🏁 2025 UNITED STATES GP - AUSTIN, TEXAS")
logger.info("="*80)
logger.info("📅 Race Date: October 20, 2025")
logger.info("📍 Circuit: Circuit of The Americas (COTA)")
logger.info("="*80)

# Load ALL available Austin data
logger.info("\n📡 Loading Austin 2025 Data...")

try:
    all_laps, all_positions, loaded_sessions = load_all_austin_data()
    
    if all_laps.empty:
        logger.error("\n❌ No Austin lap data found!")
        logger.info("\n💡 To fetch Austin data:")
        logger.info("  python fetch_live_data.py --session 9899 --save data/raw/2025  # FP1")
        logger.info("  python fetch_live_data.py --session 9900 --save data/raw/2025  # Sprint Quali")
        logger.info("  # Add more sessions as they become available")
        exit(1)
    
    # Lap duration is already in seconds - just rename for clarity
    all_laps['lap_time_seconds'] = pd.to_numeric(all_laps['lap_duration'], errors='coerce')
    
    # Load weather data if available
    all_weather = []
    data_dir = Path('data/raw/2025')
    for weather_file in data_dir.glob('weather_session_*.csv'):
        weather = pd.read_csv(weather_file)
        if not weather.empty:
            all_weather.append(weather)
    
    weather_df = pd.concat(all_weather, ignore_index=True) if all_weather else pd.DataFrame()
    
    # Load pit stop data if available
    pit_stops_file = Path('data/raw/2025/pit_stops.csv')
    pit_stops_df = pd.read_csv(pit_stops_file) if pit_stops_file.exists() else pd.DataFrame()
    
    logger.info("\n📊 Extracting Weekend Features...")
    
    # Separate qualifying session from other sessions
    # Session 9884 is the main race qualifying
    qualifying_positions = None
    if not all_positions.empty and 9884 in all_positions['session_key'].values:
        quali_pos = all_positions[all_positions['session_key'] == 9884].copy()
        quali_pos['date'] = pd.to_datetime(quali_pos['date'])
        # Get final qualifying positions (last timestamp for each driver)
        qualifying_positions = quali_pos.loc[quali_pos.groupby('driver_number')['date'].idxmax()]
        logger.info(f"✓ Using qualifying positions from session 9884 (Main Qualifying)")
    
    # Get qualifying lap times (best laps from qualifying session)
    qualifying_laps = None
    if not all_laps.empty and 9884 in all_laps['session_key'].values:
        qualifying_laps = all_laps[all_laps['session_key'] == 9884].copy()
        logger.info(f"✓ Using qualifying lap times from session 9884")
    
    # Get unique drivers
    drivers = all_laps['driver_number'].unique()
    
    features_list = []
    for driver in drivers:
        features = {'driver_number': driver}
        
        # All laps for this driver (all sessions)
        driver_laps = all_laps[all_laps['driver_number'] == driver].copy()
        
        # Get valid lap times from ALL sessions
        valid_times = driver_laps['lap_time_seconds'].dropna()
        valid_times = valid_times[valid_times > 0]
        
        if len(valid_times) > 0:
            features['best_lap_time'] = valid_times.min()
            features['avg_lap_time'] = valid_times.mean()
            features['lap_time_std'] = valid_times.std()
            features['total_laps'] = len(valid_times)
            features['consistency_score'] = valid_times.std() / valid_times.mean() if valid_times.mean() > 0 else 0
        
        # QUALIFYING POSITION - Use actual qualifying session (9884) if available
        if qualifying_positions is not None:
            driver_quali = qualifying_positions[qualifying_positions['driver_number'] == driver]
            if not driver_quali.empty:
                features['qualifying_position'] = driver_quali['position'].iloc[0]
        else:
            # Fallback: use best position across all sessions
            if not all_positions.empty:
                driver_positions = all_positions[all_positions['driver_number'] == driver]
                if not driver_positions.empty and 'position' in driver_positions.columns:
                    features['qualifying_position'] = driver_positions['position'].min()
        
        # QUALIFYING BEST LAP - Use best lap from qualifying session (9884) if available
        if qualifying_laps is not None:
            driver_quali_laps = qualifying_laps[qualifying_laps['driver_number'] == driver]
            quali_times = driver_quali_laps['lap_time_seconds'].dropna()
            quali_times = quali_times[quali_times > 0]
            if len(quali_times) > 0:
                features['qualifying_best_lap'] = quali_times.min()
        else:
            # Fallback: use best lap from all sessions
            if len(valid_times) > 0:
                features['qualifying_best_lap'] = valid_times.min()
        
        # Weather features (average across all Austin sessions)
        if not weather_df.empty:
            features['avg_air_temp'] = weather_df['air_temperature'].mean() if 'air_temperature' in weather_df.columns else 0
            features['avg_track_temp'] = weather_df['track_temperature'].mean() if 'track_temperature' in weather_df.columns else 0
            features['avg_humidity'] = weather_df['humidity'].mean() if 'humidity' in weather_df.columns else 0
            features['rainfall'] = weather_df['rainfall'].max() if 'rainfall' in weather_df.columns else 0
            if 'air_temperature' in weather_df.columns:
                features['weather_variability'] = weather_df['air_temperature'].std()
        
        # Pit stop features
        if not pit_stops_df.empty:
            driver_pitstops = pit_stops_df[pit_stops_df['driver_number'] == driver]
            if not driver_pitstops.empty and 'pit_duration' in driver_pitstops.columns:
                features['num_pit_stops'] = len(driver_pitstops)
                features['avg_pit_duration'] = driver_pitstops['pit_duration'].mean()
                features['total_pit_time'] = driver_pitstops['pit_duration'].sum()
        
        features_list.append(features)
    
    weekend_df = pd.DataFrame(features_list)
    
    logger.info(f"✓ Extracted features for {len(weekend_df)} drivers")
    
    # Load historical performance
    # DON'T use historical data - only use 2025 Austin weekend data
    logger.info("⚠️  Using ONLY 2025 Austin weekend data (no historical performance)")
    
    # Just use the weekend features as-is
    combined_features = weekend_df.to_dict('records')
    
    prediction_df = pd.DataFrame(combined_features)
    
    # Load driver names
    drivers_df = pd.read_csv('data/processed/drivers_processed.csv')
    drivers_info = drivers_df[['driver_number', 'full_name', 'team_name']].drop_duplicates('driver_number')
    prediction_df = prediction_df.merge(drivers_info, on='driver_number', how='left')
    
    # Load model
    logger.info("\n🤖 Loading ML Model...")
    trainer = ModelTrainer({}, {})
    trainer.load_model('data/models/xgboost_model.pkl')
    
    # Prepare features
    model_features = trainer.feature_names
    
    X_pred = pd.DataFrame()
    missing_features = []
    for feature in model_features:
        if feature in prediction_df.columns:
            X_pred[feature] = prediction_df[feature]
        else:
            missing_features.append(feature)
            # For missing features, use 0 (no historical data)
            X_pred[feature] = 0
    
    # Fill NaN values with column median or 0
    for col in X_pred.columns:
        if X_pred[col].isna().any():
            col_median = X_pred[col].median()
            if pd.notna(col_median):
                X_pred[col] = X_pred[col].fillna(col_median)
            else:
                X_pred[col] = X_pred[col].fillna(0)
    
    logger.info(f"\n📊 Using {len(prediction_df.columns)} features from 2025 Austin data only")
    logger.info(f"   Missing features filled with 0: {len(missing_features)}")
    
    # Predict
    logger.info("\n🎯 Generating Predictions...\n")
    predictions = trainer.model.predict(X_pred)
    prediction_df['predicted_position'] = predictions
    prediction_df['predicted_position_int'] = np.round(predictions).astype(int).clip(1, 20)
    
    # Sort by predicted position
    prediction_df = prediction_df.sort_values('predicted_position')
    
    # Display results
    logger.info("="*80)
    logger.info("🏆 PREDICTED RACE RESULTS")
    logger.info("="*80)
    
    for position_rank, (idx, row) in enumerate(prediction_df.head(15).iterrows(), 1):
        predicted_pos = row['predicted_position']
        name = row.get('full_name', f"Driver #{int(row['driver_number'])}")
        team = row.get('team_name', '')[:20]
        driver_num = int(row['driver_number'])
        
        # Use actual rank (1, 2, 3...) instead of rounded predicted position
        emoji = "🥇" if position_rank == 1 else "🥈" if position_rank == 2 else "🥉" if position_rank == 3 else f"P{position_rank:2d}"
        
        # Show weekend performance
        context = []
        if 'quali_best_lap' in row and pd.notna(row['quali_best_lap']):
            context.append(f"SQ: {row['quali_best_lap']:.3f}s")
        if 'practice_best_lap' in row and pd.notna(row['practice_best_lap']):
            context.append(f"FP1: {row['practice_best_lap']:.3f}s")
        if 'grid_position' in row and pd.notna(row['grid_position']):
            context.append(f"Grid: P{int(row['grid_position'])}")
        
        context_str = f" | {', '.join(context)}" if context else ""
        
        logger.info(f"  {emoji}: {name:<25} #{driver_num:<3} {team:<20}{context_str}")
    
    # Podium
    logger.info("\n" + "="*80)
    logger.info("🏆 PREDICTED PODIUM")
    logger.info("="*80)
    
    podium = prediction_df.head(3)
    medals = ["🥇 Winner: ", "🥈 Second: ", "🥉 Third:  "]
    for i, (_, driver) in enumerate(podium.iterrows()):
        name = driver.get('full_name', f"Driver #{int(driver['driver_number'])}")
        logger.info(f"  {medals[i]}{name} (#{int(driver['driver_number'])})")
    
    logger.info("="*80)
    
    logger.info("\n💡 Prediction based on:")
    for session_name in loaded_sessions:
        logger.info(f"  ✓ {session_name}")
    logger.info("  ✓ 2025 Austin weekend data ONLY (no historical performance)")
    logger.info("\n⚠️  Note: Main race hasn't happened yet. Predictions are probabilistic.")
    logger.info("    Purely based on this weekend's performance!\n")
    
except FileNotFoundError as e:
    logger.error(f"\n❌ Data files not found: {e}")
    logger.info("\n💡 First fetch the Austin 2025 data:")
    logger.info("    python fetch_live_data.py --session 9899 --save data/raw/2025")
    logger.info("    python fetch_live_data.py --session 9900 --save data/raw/2025")
except Exception as e:
    logger.error(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
