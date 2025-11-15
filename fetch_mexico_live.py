"""
Fetch live data for Mexico GP 2025
Auto-detects and downloads all Mexico sessions from OpenF1 API
"""

import pandas as pd
from pathlib import Path
import logging
from fetch_live_data import LiveDataFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_mexico_sessions():
    """Fetch all Mexico GP 2025 sessions"""
    
    logger.info("="*80)
    logger.info("🏁 FETCHING MEXICO CITY GP 2025 DATA")
    logger.info("="*80)
    
    # Initialize fetcher
    fetcher = LiveDataFetcher()
    data_dir = Path('data/raw/2025')
    
    # Check if sessions file exists
    sessions_file = data_dir / 'sessions.csv'
    if not sessions_file.exists():
        logger.error("\n❌ sessions.csv not found!")
        logger.info("💡 Run 'python main.py --mode collect' first to get session data.\n")
        return
    
    # Load sessions and find Mexico
    sessions_df = pd.read_csv(sessions_file)
    
    mexico_sessions = sessions_df[
        (sessions_df['location'].str.contains('Mexico', case=False, na=False)) |
        (sessions_df['country_name'].str.contains('Mexico', case=False, na=False))
    ]
    
    if mexico_sessions.empty:
        logger.warning("\n⚠️  No Mexico GP sessions found!")
        logger.info("💡 The Mexico GP may not have started yet or data is not available.")
        logger.info("📅 Typically scheduled for late October 2025")
        logger.info("🔄 Try running this script again when the race weekend begins.\n")
        return
    
    logger.info(f"\n✓ Found {len(mexico_sessions)} Mexico GP sessions:")
    logger.info("-"*80)
    
    for _, session in mexico_sessions.iterrows():
        session_type = session['session_type']
        session_name = session['session_name']
        session_key = session['session_key']
        date_start = session['date_start']
        logger.info(f"  • {session_name:20s} (Key: {session_key}) - {date_start}")
    
    logger.info("-"*80)
    
    # Fetch data for each session
    logger.info("\n📥 Fetching session data...")
    
    for _, session in mexico_sessions.iterrows():
        session_key = session['session_key']
        session_name = session['session_name']
        
        logger.info(f"\n🔄 Fetching {session_name} (session {session_key})...")
        
        try:
            # Fetch laps
            logger.info("  📊 Fetching laps...")
            laps = fetcher.fetch_live_laps(session_key=session_key)
            if laps is not None and not laps.empty:
                output_file = data_dir / f'laps_session_{session_key}.csv'
                laps.to_csv(output_file, index=False)
                logger.info(f"  ✓ Saved {len(laps)} laps to {output_file.name}")
            else:
                logger.warning(f"  ⚠️  No lap data available for session {session_key}")
            
            # Fetch positions
            logger.info("  📍 Fetching positions...")
            positions = fetcher.fetch_live_positions(session_key=session_key)
            if positions is not None and not positions.empty:
                output_file = data_dir / f'positions_session_{session_key}.csv'
                positions.to_csv(output_file, index=False)
                logger.info(f"  ✓ Saved {len(positions)} position records to {output_file.name}")
            else:
                logger.warning(f"  ⚠️  No position data available for session {session_key}")
            
            # Fetch weather
            logger.info("  🌤️  Fetching weather...")
            weather = fetcher.fetch_live_weather(session_key=session_key)
            if weather is not None and not weather.empty:
                output_file = data_dir / f'weather_session_{session_key}.csv'
                weather.to_csv(output_file, index=False)
                logger.info(f"  ✓ Saved {len(weather)} weather records to {output_file.name}")
            else:
                logger.warning(f"  ⚠️  No weather data available for session {session_key}")
                
        except Exception as e:
            logger.error(f"  ❌ Error fetching session {session_key}: {e}")
            continue
    
    logger.info("\n" + "="*80)
    logger.info("✅ MEXICO GP DATA FETCH COMPLETE!")
    logger.info("="*80)
    logger.info("\n💡 Next steps:")
    logger.info("   1. Run: python predict_mexico_2025.py")
    logger.info("   2. See updated predictions with Mexico weekend data!\n")


if __name__ == '__main__':
    try:
        fetch_mexico_sessions()
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
