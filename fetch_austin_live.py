#!/usr/bin/env python3
"""
Fetch ALL available live data for Austin 2025 GP
Automatically discovers and downloads all completed sessions
"""

import pandas as pd
import logging
from pathlib import Path
from fetch_live_data import LiveDataFetcher

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fetch_all_austin_sessions():
    """Fetch all available Austin 2025 session data"""
    
    logger.info("="*80)
    logger.info("🏁 AUSTIN 2025 GP - LIVE DATA FETCHER")
    logger.info("="*80)
    
    # Find Austin sessions
    sessions_file = Path('data/raw/2025/sessions.csv')
    if not sessions_file.exists():
        logger.error("❌ sessions.csv not found. Run 'python main.py --mode collect' first!")
        return
    
    sessions = pd.read_csv(sessions_file)
    austin_sessions = sessions[
        (sessions['meeting_key'] == 1271) | 
        (sessions['location'].str.contains('Austin', case=False, na=False))
    ]
    
    if austin_sessions.empty:
        logger.error("❌ No Austin sessions found in sessions.csv")
        return
    
    logger.info(f"\n📋 Found {len(austin_sessions)} Austin 2025 Sessions:")
    for _, session in austin_sessions.iterrows():
        logger.info(f"  • {session['session_name']:20s} (Session {session['session_key']})")
    
    # Initialize fetcher
    fetcher = LiveDataFetcher()
    save_dir = Path('data/raw/2025')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Track success/failures
    successful = []
    failed = []
    already_exists = []
    
    logger.info("\n🔄 Fetching session data...\n")
    
    for _, session in austin_sessions.iterrows():
        session_key = session['session_key']
        session_name = session['session_name']
        
        # Check if already downloaded
        lap_file = save_dir / f'laps_session_{session_key}.csv'
        pos_file = save_dir / f'positions_session_{session_key}.csv'
        
        if lap_file.exists() and pos_file.exists():
            logger.info(f"✓ {session_name:20s} - Already downloaded (session {session_key})")
            already_exists.append(session_name)
            continue
        
        logger.info(f"⏳ Fetching {session_name} (session {session_key})...")
        
        try:
            # Fetch complete session data
            result = fetcher.fetch_session_complete_data(session_key, str(save_dir))
            
            # Count records
            laps_count = len(result.get('laps', pd.DataFrame()))
            positions_count = len(result.get('positions', pd.DataFrame()))
            weather_count = len(result.get('weather', pd.DataFrame()))
            
            if laps_count > 0 or positions_count > 0:
                logger.info(f"✅ {session_name:20s} - Downloaded!")
                logger.info(f"   Laps: {laps_count}, Positions: {positions_count}, Weather: {weather_count}")
                successful.append(session_name)
            else:
                logger.warning(f"⚠️  {session_name:20s} - No data available (session hasn't started yet)")
                failed.append(session_name)
        
        except Exception as e:
            logger.error(f"❌ {session_name:20s} - Failed: {e}")
            failed.append(session_name)
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("📊 SUMMARY")
    logger.info("="*80)
    logger.info(f"✅ Successfully downloaded: {len(successful)}")
    for name in successful:
        logger.info(f"   • {name}")
    
    if already_exists:
        logger.info(f"\n📁 Already had data for: {len(already_exists)}")
        for name in already_exists:
            logger.info(f"   • {name}")
    
    if failed:
        logger.info(f"\n⚠️  Not available yet: {len(failed)}")
        for name in failed:
            logger.info(f"   • {name}")
    
    total_available = len(successful) + len(already_exists)
    logger.info(f"\n🏆 Total sessions with data: {total_available}/{len(austin_sessions)}")
    
    if total_available > 0:
        logger.info("\n✨ Ready to predict! Run:")
        logger.info("   python predict_austin_2025.py")
    else:
        logger.info("\n⏳ No session data available yet. Check back after sessions complete!")
    
    logger.info("="*80)


if __name__ == '__main__':
    fetch_all_austin_sessions()
