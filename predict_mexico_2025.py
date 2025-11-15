"""
Mexico GP 2025 Predictor
Uses 2025 season data + Mexico weekend data (when available)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_mexico_historical_performance():
    """Calculate driver performance at Mexico GP from 2023-2024 historical data"""
    logger.info("\n📊 Loading Mexico GP historical data (2023-2024)...")
    
    mexico_stats = []
    
    for year in [2023, 2024]:
        year_dir = Path(f'data/raw/{year}')
        if not year_dir.exists():
            continue
        
        # Load sessions and find Mexico GP
        sessions_df = pd.read_csv(year_dir / 'sessions.csv')
        mexico_sessions = sessions_df[
            (sessions_df['location'].str.contains('Mexico', case=False, na=False)) |
            (sessions_df['country_name'].str.contains('Mexico', case=False, na=False))
        ]
        
        if mexico_sessions.empty:
            continue
        
        # Load positions for Mexico race
        positions_file = year_dir / 'positions.csv'
        if not positions_file.exists():
            continue
        
        positions_df = pd.read_csv(positions_file)
        mexico_race_sessions = mexico_sessions[mexico_sessions['session_name'] == 'Race']
        
        if mexico_race_sessions.empty:
            continue
        
        race_key = mexico_race_sessions['session_key'].iloc[0]
        race_positions = positions_df[positions_df['session_key'] == race_key].copy()
        
        if not race_positions.empty:
            # Get final positions
            race_positions['date'] = pd.to_datetime(race_positions['date'], format='ISO8601')
            final_positions = race_positions.sort_values('date').groupby('driver_number').last()
            
            for driver_num, pos_data in final_positions.iterrows():
                mexico_stats.append({
                    'driver_number': driver_num,
                    'year': year,
                    'mexico_position': pos_data['position']
                })
            
            logger.info(f"✓ Loaded {year} Mexico GP results ({len(final_positions)} drivers)")
    
    if not mexico_stats:
        logger.warning("⚠️  No historical Mexico GP data found")
        return pd.DataFrame()
    
    # Calculate average Mexico performance per driver
    mexico_df = pd.DataFrame(mexico_stats)
    mexico_avg = mexico_df.groupby('driver_number').agg({
        'mexico_position': ['mean', 'min', 'count']
    }).reset_index()
    mexico_avg.columns = ['driver_number', 'mexico_avg_position', 'mexico_best_position', 'mexico_races']
    
    logger.info(f"✓ Calculated Mexico track performance for {len(mexico_avg)} drivers")
    
    return mexico_avg


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
    """Fallback: Use lap time statistics when position data is unavailable"""
    logger.warning("Using lap-based fallback calculation...")
    
    # Filter for race sessions only
    race_laps = all_laps_df[all_laps_df['session_key'].isin(race_session_keys)].copy()
    
    season_stats = []
    for driver in race_laps['driver_number'].unique():
        driver_laps = race_laps[race_laps['driver_number'] == driver]
        valid_times = driver_laps['lap_time_seconds'].dropna()
        valid_times = valid_times[(valid_times > 0) & (valid_times < 200)]
        
        if len(valid_times) < 10:
            continue
        
        season_stats.append({
            'driver_number': driver,
            'races_2025': 0,
            'avg_position_2025': 10.0,  # Default midfield
            'best_position_2025': 10,
            'worst_position_2025': 10,
            'position_std': 3.0,
            'points_finishes': 0,
            'podiums': 0,
            'wins': 0,
            'quality_consistency': 3.0  # Default mid-level consistency
        })
    
    return pd.DataFrame(season_stats)


def find_mexico_sessions():
    """Find Mexico GP session keys"""
    data_dir = Path('data/raw/2025')
    sessions_df = pd.read_csv(data_dir / 'sessions.csv')
    
    # Look for Mexico sessions
    mexico_sessions = sessions_df[
        (sessions_df['location'].str.contains('Mexico', case=False, na=False)) |
        (sessions_df['country_name'].str.contains('Mexico', case=False, na=False))
    ]
    
    if mexico_sessions.empty:
        logger.warning("⚠️  No Mexico GP sessions found yet")
        return None
    
    logger.info(f"✓ Found {len(mexico_sessions)} Mexico GP sessions")
    
    # Get session keys
    quali_session = mexico_sessions[mexico_sessions['session_name'] == 'Qualifying']
    quali_key = quali_session['session_key'].iloc[0] if not quali_session.empty else None
    
    sprint_quali_session = mexico_sessions[mexico_sessions['session_name'] == 'Sprint Qualifying']
    sprint_quali_key = sprint_quali_session['session_key'].iloc[0] if not sprint_quali_session.empty else None
    
    return {
        'all_sessions': mexico_sessions['session_key'].tolist(),
        'qualifying_key': quali_key or sprint_quali_key,  # Use sprint quali if no main quali
        'meeting_key': mexico_sessions['meeting_key'].iloc[0]
    }


def load_mexico_data():
    """Load all available Mexico GP data"""
    data_dir = Path('data/raw/2025')
    
    mexico_info = find_mexico_sessions()
    if not mexico_info:
        logger.info("💡 Mexico GP data not available yet - will use season averages only")
        return None, None, None
    
    # Load laps
    all_laps = []
    for session_key in mexico_info['all_sessions']:
        lap_file = data_dir / f'laps_session_{session_key}.csv'
        if lap_file.exists():
            laps = pd.read_csv(lap_file)
            all_laps.append(laps)
    
    if not all_laps:
        # Try aggregate laps file
        laps_df = pd.read_csv(data_dir / 'laps.csv')
        laps_df = laps_df[laps_df['session_key'].isin(mexico_info['all_sessions'])]
    else:
        laps_df = pd.concat(all_laps, ignore_index=True)
    
    # Load qualifying data
    quali_key = mexico_info['qualifying_key']
    if quali_key:
        quali_laps = laps_df[laps_df['session_key'] == quali_key].copy()
        
        # Get final grid positions
        final_grid = quali_laps.sort_values('lap_time_seconds').groupby('driver_number').first().reset_index()
        final_grid = final_grid.sort_values('lap_time_seconds')
        final_grid['position'] = range(1, len(final_grid) + 1)
    else:
        final_grid = None
    
    logger.info(f"✓ Loaded {len(laps_df)} laps from Mexico GP")
    
    return laps_df, final_grid, quali_laps if quali_key else None


def predict_mexico():
    """Predict Mexico GP results using 2025 season data + Mexico weekend"""
    
    logger.info("="*80)
    logger.info("🏁 2025 MEXICO CITY GP - AUTÓDROMO HERMANOS RODRÍGUEZ")
    logger.info("   SEASON-BASED PREDICTOR (All 2025 Data)")
    logger.info("="*80)
    
    # Load all 2025 season laps for performance calculation
    logger.info("📥 Loading 2025 season data...")
    data_dir = Path('data/raw/2025')
    
    all_season_laps = []
    for lap_file in data_dir.glob('laps_session_*.csv'):
        laps = pd.read_csv(lap_file)
        all_season_laps.append(laps)
    
    # Also load aggregate laps
    if (data_dir / 'laps.csv').exists():
        aggregate_laps = pd.read_csv(data_dir / 'laps.csv')
        all_season_laps.append(aggregate_laps)
    
    all_season_laps = pd.concat(all_season_laps, ignore_index=True).drop_duplicates()
    logger.info(f"✓ Loaded {len(all_season_laps)} total laps from season")
    
    # Calculate season performance from race results
    season_performance = calculate_season_performance(all_season_laps)
    
    # Load historical Mexico GP performance (2023-2024)
    mexico_historical = calculate_mexico_historical_performance()
    
    # Try to load Mexico weekend data
    logger.info("\n📊 Loading Mexico GP weekend data...")
    mexico_laps_df, final_grid, quali_laps_df = load_mexico_data()
    
    # Calculate predictions for each driver
    predictions = []
    
    # If we have Mexico data, use it; otherwise use season averages
    if mexico_laps_df is not None and not mexico_laps_df.empty:
        logger.info("✓ Mexico GP weekend data available - combining with season stats")
        
        for driver in season_performance['driver_number'].unique():
            # Mexico weekend laps
            driver_mexico_laps = mexico_laps_df[mexico_laps_df['driver_number'] == driver]
            valid_times = driver_mexico_laps['lap_time_seconds'].dropna()
            valid_times = valid_times[(valid_times > 0) & (valid_times < 200)]
            
            if len(valid_times) == 0:
                continue
            
            # Mexico weekend stats
            mexico_best_lap = valid_times.min()
            
            # Qualifying position
            if final_grid is not None:
                driver_quali_pos = final_grid[final_grid['driver_number'] == driver]
                quali_position = driver_quali_pos['position'].iloc[0] if not driver_quali_pos.empty else 20
            else:
                quali_position = 10  # Default midfield if no quali data
            
            # Season race results stats
            driver_season = season_performance[season_performance['driver_number'] == driver]
            avg_position_2025 = driver_season['avg_position_2025'].iloc[0] if not driver_season.empty else 15.0
            quality_consistency = driver_season['quality_consistency'].iloc[0] if not driver_season.empty else 0.0
            races_2025 = driver_season['races_2025'].iloc[0] if not driver_season.empty else 0
            points_finishes = driver_season['points_finishes'].iloc[0] if not driver_season.empty else 0
            
            predictions.append({
                'driver_number': driver,
                'qualifying_position': quali_position,
                'mexico_best_lap': mexico_best_lap,
                'avg_position_2025': avg_position_2025,
                'quality_consistency': quality_consistency,
                'races_2025': races_2025,
                'points_finishes': points_finishes,
            })
    else:
        logger.info("⚠️  Mexico GP 2025 data not available - using season + historical Mexico data")
        
        for _, driver_season in season_performance.iterrows():
            driver = driver_season['driver_number']
            avg_position_2025 = driver_season['avg_position_2025']
            quality_consistency = driver_season['quality_consistency']
            races_2025 = driver_season['races_2025']
            points_finishes = driver_season['points_finishes']
            
            # Check if driver has historical Mexico performance
            driver_mexico_hist = mexico_historical[mexico_historical['driver_number'] == driver]
            
            if not driver_mexico_hist.empty and not mexico_historical.empty:
                # Use historical Mexico performance to adjust estimates
                mexico_avg_pos = driver_mexico_hist['mexico_avg_position'].iloc[0]
                mexico_races = driver_mexico_hist['mexico_races'].iloc[0]
                
                # Blend season average with Mexico track performance
                # More Mexico races = more weight to Mexico history
                mexico_weight = min(0.4, mexico_races * 0.2)  # Max 40% weight
                estimated_quali = int((1 - mexico_weight) * avg_position_2025 + mexico_weight * mexico_avg_pos)
                
                logger.info(f"  Driver #{driver}: Mexico avg P{mexico_avg_pos:.1f} ({mexico_races} races) + Season avg P{avg_position_2025:.1f}")
            else:
                # No Mexico history, use season average only
                estimated_quali = int(avg_position_2025)
            
            estimated_lap_score = avg_position_2025 / 20.0  # Normalized 0-1
            
            predictions.append({
                'driver_number': driver,
                'qualifying_position': estimated_quali,
                'mexico_best_lap': None,  # Will be normalized differently
                'avg_position_2025': avg_position_2025,
                'quality_consistency': quality_consistency,
                'races_2025': races_2025,
                'points_finishes': points_finishes,
                'mexico_historical_avg': driver_mexico_hist['mexico_avg_position'].iloc[0] if not driver_mexico_hist.empty else None,
                'mexico_historical_races': driver_mexico_hist['mexico_races'].iloc[0] if not driver_mexico_hist.empty else 0,
            })
    
    # Create DataFrame
    pred_df = pd.DataFrame(predictions)
    
    if pred_df.empty:
        logger.error("❌ No predictions could be generated")
        return
    
    # Normalize features for scoring (0-1 range, lower score = better prediction)
    pred_df['norm_quali'] = (pred_df['qualifying_position'] - 1) / 19  # P1=0, P20=1
    
    # Normalize lap time (if available)
    if mexico_laps_df is not None and pred_df['mexico_best_lap'].notna().any():
        pred_df['norm_lap'] = (pred_df['mexico_best_lap'] - pred_df['mexico_best_lap'].min()) / \
                              (pred_df['mexico_best_lap'].max() - pred_df['mexico_best_lap'].min())
        lap_weight = 0.25
    else:
        # Use season average as proxy for lap performance
        pred_df['norm_lap'] = (pred_df['avg_position_2025'] - 1) / 19
        lap_weight = 0.15  # Lower weight when using estimated data
    
    # Average position: lower position number = better = lower score
    pred_df['norm_avg_pos'] = (pred_df['avg_position_2025'] - 1) / 19  # P1=0, P20=1
    
    # Quality consistency: higher = better, so invert for scoring
    max_quality = pred_df['quality_consistency'].max()
    if max_quality > 0:
        pred_df['norm_quality'] = 1 - (pred_df['quality_consistency'] / max_quality)
    else:
        pred_df['norm_quality'] = 0.5  # Default if no quality data
    
    # Prediction score formula
    # With Mexico data: 35% Quali, 25% Lap, 20% Season Avg, 20% Quality
    # Without Mexico data: 30% Quali, 15% Lap (est), 30% Season Avg, 25% Quality
    if mexico_laps_df is not None:
        pred_df['prediction_score'] = (
            0.35 * pred_df['norm_quali'] + 
            0.25 * pred_df['norm_lap'] + 
            0.20 * pred_df['norm_avg_pos'] +
            0.20 * pred_df['norm_quality']
        )
        formula_text = [
            "  • 35% Qualifying Position (Mexico)",
            "  • 25% Best Lap Time (Mexico)",
            "  • 20% Season Average Position (2025 race finishes)",
            "  • 20% Quality Consistency"
        ]
    else:
        pred_df['prediction_score'] = (
            0.30 * pred_df['norm_quali'] + 
            0.15 * pred_df['norm_lap'] + 
            0.30 * pred_df['norm_avg_pos'] +
            0.25 * pred_df['norm_quality']
        )
        if not mexico_historical.empty:
            formula_text = [
                "  • 30% Estimated Qualifying Position (season avg + Mexico 2023-2024 history)",
                "  • 15% Estimated Lap Performance (from season avg)",
                "  • 30% Season Average Position (2025 race finishes)",
                "  • 25% Quality Consistency (emphasized)"
            ]
        else:
            formula_text = [
                "  • 30% Estimated Qualifying Position (from season avg)",
                "  • 15% Estimated Lap Performance (from season avg)",
                "  • 30% Season Average Position (2025 race finishes)",
                "  • 25% Quality Consistency (emphasized)"
            ]
    
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
    
    # Show different headers based on whether we have Mexico historical data
    has_mexico_hist = 'mexico_historical_avg' in pred_df.columns and pred_df['mexico_historical_avg'].notna().any()
    
    if has_mexico_hist and mexico_laps_df is None:
        logger.info(f"{'Pos':<5} {'Driver':<25} {'#':<4} {'Team':<20} {'Quali':<7} {'2025':<8} {'MEX Hist'}")
        logger.info("-"*80)
        
        for idx, row in pred_df.head(15).iterrows():
            pos = int(row['predicted_position'])
            name = row.get('full_name', f"Driver #{int(row['driver_number'])}")[:25]
            team = row.get('team_name', '')[:20]
            driver_num = int(row['driver_number'])
            quali_pos = int(row['qualifying_position'])
            avg_2025 = f"P{row['avg_position_2025']:.1f}"
            
            if pd.notna(row.get('mexico_historical_avg')):
                mexico_hist = f"P{row['mexico_historical_avg']:.1f}"
            else:
                mexico_hist = "N/A"
            
            emoji = "🥇" if pos == 1 else "🥈" if pos == 2 else "🥉" if pos == 3 else f"P{pos:2d}"
            
            logger.info(f"{emoji:<5} {name:<25} #{driver_num:<3} {team:<20} P{quali_pos:<6} {avg_2025:<8} {mexico_hist}")
    else:
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
    for line in formula_text:
        logger.info(line)
    
    logger.info("\n📊 Data Sources:")
    if mexico_laps_df is not None:
        logger.info("  ✓ Mexico GP 2025 weekend data (Practice, Qualifying)")
    else:
        logger.info("  ⚠️  Mexico GP 2025 weekend data NOT YET AVAILABLE")
        logger.info("  📝 Using 2025 season performance + historical Mexico data")
    
    if not mexico_historical.empty:
        logger.info(f"  ✓ Mexico GP historical data (2023-2024) - {len(mexico_historical)} drivers")
    logger.info("  ✓ All 2025 race finishing positions for season performance")
    
    logger.info("\n🎯 Quality Consistency = (21 - avg_position) / (1 + position_std)")
    logger.info("   This rewards drivers who consistently finish well, not just consistently.")
    
    if not mexico_historical.empty and mexico_laps_df is None:
        logger.info("\n� Mexico Track History:")
        logger.info("   Qualifying estimates blend 2025 season avg with 2023-2024 Mexico performance")
        logger.info("   Drivers with better Mexico history get improved predictions")
    
    if mexico_laps_df is None:
        logger.info("\n�💡 TIP: Run fetch_mexico_live.py to get 2025 Mexico GP weekend data when available")
        logger.info("   Then re-run this script for even more accurate predictions!\n")


if __name__ == '__main__':
    try:
        predict_mexico()
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
