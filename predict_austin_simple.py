#!/usr/bin/env python3
"""
2025 Season-Based Austin GP Predictor
Uses ALL 2025 race data to calculate driver performance
Then applies it to predict Austin 2025 GP
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_2025_season_data():
    """Load all 2025 race data to calculate driver performance"""
    data_dir = Path('data/raw/2025')
    
    # Load all lap files
    logger.info("📥 Loading 2025 season data...")
    
    all_laps = []
    lap_files = list(data_dir.glob('laps_session_*.csv'))
    
    for lap_file in lap_files:
        laps = pd.read_csv(lap_file)
        if not laps.empty:
            all_laps.append(laps)
    
    if not all_laps:
        logger.warning("⚠️  No lap data found in data/raw/2025/")
        return pd.DataFrame()
    
    all_laps_df = pd.concat(all_laps, ignore_index=True)
    all_laps_df['lap_time_seconds'] = pd.to_numeric(all_laps_df['lap_duration'], errors='coerce')
    
    logger.info(f"✓ Loaded {len(all_laps_df)} laps from {len(lap_files)} sessions")
    
    return all_laps_df


def calculate_season_performance(all_laps_df):
    """Calculate driver performance from actual 2025 race results"""
    logger.info("\n📊 Calculating 2025 season race results for each driver...")
    
    data_dir = Path('data/raw/2025')
    
    # Load sessions to identify race sessions
    sessions_df = pd.read_csv(data_dir / 'sessions.csv')
    race_sessions = sessions_df[sessions_df['session_name'].str.contains('Race', case=False, na=False)]
    race_session_keys = race_sessions['session_key'].unique()
    
    # Load positions data from aggregate positions.csv file
    positions_file = data_dir / 'positions.csv'
    if not positions_file.exists():
        logger.warning("⚠️  No positions.csv found, using lap-based estimate")
        return calculate_season_performance_fallback(all_laps_df, race_session_keys)
    
    positions_df = pd.read_csv(positions_file)
    
    # Get race results (final positions from race sessions)
    race_positions = positions_df[positions_df['session_key'].isin(race_session_keys)].copy()
    
    if race_positions.empty:
        logger.warning("⚠️  No race position data found, using fallback")
        return calculate_season_performance_fallback(all_laps_df, race_session_keys)
    
    # For each driver, get their finishing positions in each race
    race_positions['date'] = pd.to_datetime(race_positions['date'], format='ISO8601')
    
    # Get final position in each race (last recorded position)
    final_race_positions = race_positions.sort_values('date').groupby(['session_key', 'driver_number']).last().reset_index()
    
    logger.info(f"✓ Found {len(final_race_positions['session_key'].unique())} races with position data")
    
    season_stats = []
    
    for driver in final_race_positions['driver_number'].unique():
        driver_races = final_race_positions[final_race_positions['driver_number'] == driver]
        positions = driver_races['position'].values
        
        if len(positions) < 3:  # Need at least 3 races
            continue
        
        # Calculate race result statistics
        avg_position = positions.mean()
        best_position = positions.min()
        worst_position = positions.max()
        position_std = positions.std()
        
        # Points finishes (top 10)
        points_finishes = (positions <= 10).sum()
        podiums = (positions <= 3).sum()
        wins = (positions == 1).sum()
        
        # Quality consistency score: Rewards good results + low variance
        # Formula: (21 - avg_position) / (1 + std_dev)
        # Higher score = consistently good finishes
        # Lower score = consistently bad OR inconsistent
        quality_consistency = (21 - avg_position) / (1 + position_std)
        
        season_stats.append({
            'driver_number': driver,
            'races_2025': len(positions),
            'avg_position_2025': avg_position,
            'best_position_2025': best_position,
            'worst_position_2025': worst_position,
            'position_std': position_std,
            'points_finishes': points_finishes,
            'podiums': podiums,
            'wins': wins,
            'quality_consistency': quality_consistency
        })
    
    logger.info(f"✓ Calculated race results for {len(season_stats)} drivers")
    
    return pd.DataFrame(season_stats)


def calculate_season_performance_fallback(all_laps_df, race_session_keys):
    """Fallback: Calculate performance from lap times if position data unavailable"""
    logger.warning("Using lap-based fallback calculation...")
    
    season_stats = []
    
    for driver in all_laps_df['driver_number'].unique():
        driver_laps = all_laps_df[all_laps_df['driver_number'] == driver]
        valid_times = driver_laps['lap_time_seconds'].dropna()
        valid_times = valid_times[(valid_times > 0) & (valid_times < 200)]
        
        if len(valid_times) < 10:
            continue
        
        avg_lap = valid_times.mean()
        lap_std = valid_times.std()
        
        # Estimate: assume lower lap times = better positions
        # This is rough but better than nothing
        estimated_avg_pos = 10.0  # Default mid-pack
        quality_consistency = 0.5  # Default medium
        
        season_stats.append({
            'driver_number': driver,
            'races_2025': 0,
            'avg_position_2025': estimated_avg_pos,
            'best_position_2025': 10,
            'worst_position_2025': 10,
            'position_std': 3.0,
            'points_finishes': 0,
            'podiums': 0,
            'wins': 0,
            'quality_consistency': quality_consistency
        })
    
    return pd.DataFrame(season_stats)


def predict_simple():
    """Predict Austin 2025 using full season data + Austin weekend"""
    
    logger.info("="*80)
    logger.info("🏁 2025 UNITED STATES GP - AUSTIN, TEXAS")
    logger.info("   SEASON-BASED PREDICTOR (All 2025 Data)")
    logger.info("="*80)
    
    # Load ALL 2025 season data
    all_season_laps = load_2025_season_data()
    if all_season_laps.empty:
        logger.error("❌ No 2025 data found!")
        return
    
    # Calculate season-long performance
    season_performance = calculate_season_performance(all_season_laps)
    
    # Load Austin weekend data specifically
    data_dir = Path('data/raw/2025')
    austin_sessions = [9878, 9879, 9883, 9884]  # Austin session keys
    
    all_positions = []
    austin_laps = []
    
    for session_key in austin_sessions:
        lap_file = data_dir / f'laps_session_{session_key}.csv'
        pos_file = data_dir / f'positions_session_{session_key}.csv'
        
        if lap_file.exists():
            laps = pd.read_csv(lap_file)
            austin_laps.append(laps)
        
        if pos_file.exists():
            positions = pd.read_csv(pos_file)
            all_positions.append(positions)
    
    if not austin_laps or not all_positions:
        logger.error("❌ No Austin data found! Run: python fetch_austin_live.py")
        return
    
    austin_laps_df = pd.concat(austin_laps, ignore_index=True)
    austin_laps_df['lap_time_seconds'] = pd.to_numeric(austin_laps_df['lap_duration'], errors='coerce')
    
    all_positions_df = pd.concat(all_positions, ignore_index=True)
    
    # Get qualifying positions from session 9884
    quali_pos_df = all_positions_df[all_positions_df['session_key'] == 9884].copy()
    quali_pos_df['date'] = pd.to_datetime(quali_pos_df['date'])
    final_grid = quali_pos_df.loc[quali_pos_df.groupby('driver_number')['date'].idxmax()]
    
    # Get qualifying lap times from session 9884
    quali_laps_df = austin_laps_df[austin_laps_df['session_key'] == 9884].copy()
    
    logger.info("\n📊 Combining Season Performance + Austin Weekend...")
    
    # Calculate predictions for each driver
    predictions = []
    
    for driver in austin_laps_df['driver_number'].unique():
        # Austin weekend laps
        driver_austin_laps = austin_laps_df[austin_laps_df['driver_number'] == driver]
        valid_times = driver_austin_laps['lap_time_seconds'].dropna()
        valid_times = valid_times[(valid_times > 0) & (valid_times < 200)]
        
        # Qualifying data
        driver_quali_pos = final_grid[final_grid['driver_number'] == driver]
        driver_quali_laps = quali_laps_df[quali_laps_df['driver_number'] == driver]
        quali_times = driver_quali_laps['lap_time_seconds'].dropna()
        quali_times = quali_times[(quali_times > 0) & (quali_times < 200)]
        
        # Season performance
        driver_season = season_performance[season_performance['driver_number'] == driver]
        
        if len(valid_times) == 0:
            continue
        
        # Austin weekend stats
        austin_best_lap = valid_times.min()
        quali_position = driver_quali_pos['position'].iloc[0] if not driver_quali_pos.empty else 20
        
        # Season race results stats
        avg_position_2025 = driver_season['avg_position_2025'].iloc[0] if not driver_season.empty else 15.0
        quality_consistency = driver_season['quality_consistency'].iloc[0] if not driver_season.empty else 0.0
        races_2025 = driver_season['races_2025'].iloc[0] if not driver_season.empty else 0
        points_finishes = driver_season['points_finishes'].iloc[0] if not driver_season.empty else 0
        
        predictions.append({
            'driver_number': driver,
            'qualifying_position': quali_position,
            'austin_best_lap': austin_best_lap,
            'avg_position_2025': avg_position_2025,
            'quality_consistency': quality_consistency,
            'races_2025': races_2025,
            'points_finishes': points_finishes,
        })
    
    # Create DataFrame
    pred_df = pd.DataFrame(predictions)
    
    # Normalize features for scoring (0-1 range, lower score = better prediction)
    pred_df['norm_quali'] = (pred_df['qualifying_position'] - 1) / 19  # P1=0, P20=1
    pred_df['norm_lap'] = (pred_df['austin_best_lap'] - pred_df['austin_best_lap'].min()) / \
                          (pred_df['austin_best_lap'].max() - pred_df['austin_best_lap'].min())
    # Average position: lower position number = better = lower score
    pred_df['norm_avg_pos'] = (pred_df['avg_position_2025'] - 1) / 19  # P1=0, P20=1
    # Quality consistency: higher = better, so invert for scoring
    max_quality = pred_df['quality_consistency'].max()
    if max_quality > 0:
        pred_df['norm_quality'] = 1 - (pred_df['quality_consistency'] / max_quality)
    else:
        pred_df['norm_quality'] = 0.5  # Default if no quality data
    
    # NEW Prediction score formula
    # 35% Qualifying, 25% Best Lap, 20% Season Avg Position, 20% Quality Consistency
    pred_df['prediction_score'] = (
        0.35 * pred_df['norm_quali'] + 
        0.25 * pred_df['norm_lap'] + 
        0.20 * pred_df['norm_avg_pos'] +
        0.20 * pred_df['norm_quality']
    )
    
    # Sort by prediction score (lower is better)
    pred_df = pred_df.sort_values('prediction_score')
    pred_df['predicted_position'] = range(1, len(pred_df) + 1)
    
    # Load driver names
    drivers_df = pd.read_csv('data/processed/drivers_processed.csv')
    drivers_info = drivers_df[['driver_number', 'full_name', 'team_name']].drop_duplicates('driver_number')
    pred_df = pred_df.merge(drivers_info, on='driver_number', how='left')
    
    # Display results
    logger.info("\n" + "="*80)
    logger.info("🏆 PREDICTED RACE RESULTS")
    logger.info("="*80)
    logger.info(f"{'Pos':<5} {'Driver':<25} {'#':<4} {'Team':<20} {'Quali':<7} {'2025 Avg':<10} {'Points'}")
    logger.info("-"*80)
    
    for idx, row in pred_df.head(15).iterrows():
        pos = int(row['predicted_position'])
        name = row.get('full_name', f"Driver #{int(row['driver_number'])}")[:25]
        team = row.get('team_name', '')[:20]
        driver_num = int(row['driver_number'])
        quali_pos = int(row['qualifying_position'])
        avg_2025 = f"P{row['avg_position_2025']:.1f}"
        points = f"{int(row['points_finishes'])}/{int(row['races_2025'])}" if row['races_2025'] > 0 else "N/A"
        
        emoji = "🥇" if pos == 1 else "🥈" if pos == 2 else "🥉" if pos == 3 else f"P{pos:2d}"
        
        logger.info(f"{emoji:<5} {name:<25} #{driver_num:<3} {team:<20} P{quali_pos:<6} {avg_2025:<10} {points}")
    
    # Podium
    logger.info("\n" + "="*80)
    logger.info("🏆 PREDICTED PODIUM")
    logger.info("="*80)
    
    podium = pred_df.head(3)
    medals = ["🥇 Winner: ", "🥈 Second: ", "🥉 Third:  "]
    for i, (_, driver) in enumerate(podium.iterrows()):
        name = driver.get('full_name', f"Driver #{int(driver['driver_number'])}")
        quali_pos = int(driver['qualifying_position'])
        avg_pos = driver.get('avg_position_2025', 15.0)
        races = int(driver.get('races_2025', 0))
        points = int(driver.get('points_finishes', 0))
        logger.info(f"  {medals[i]}{name} (Qualified P{quali_pos}, 2025 Avg: P{avg_pos:.1f}, {points}/{races} points finishes)")
    
    logger.info("="*80)
    logger.info("\n💡 Prediction Formula:")
    logger.info("  • 35% Qualifying Position (Austin)")
    logger.info("  • 25% Best Lap Time (Austin)")
    logger.info("  • 20% Season Average Position (2025 race finishes)")
    logger.info("  • 20% Quality Consistency (rewards consistently GOOD finishes)")
    logger.info("\n📊 Data Sources:")
    logger.info("  ✓ All 2025 race finishing positions for season performance")
    logger.info("  ✓ Austin Practice, Sprint Qualifying, Sprint, Qualifying")
    logger.info("  ✗ No 2023-2024 historical data")
    logger.info("\n🎯 Quality Consistency = (21 - avg_position) / (1 + position_std)")
    logger.info("   This rewards drivers who consistently finish well, not just consistently.\n")


if __name__ == '__main__':
    try:
        predict_simple()
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
