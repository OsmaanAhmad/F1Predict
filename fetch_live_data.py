"""
Fetch live data from OpenF1 API during active sessions
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveDataFetcher:
    """Fetch live F1 data from OpenF1 API"""
    
    def __init__(self, base_url: str = "https://api.openf1.org/v1"):
        self.base_url = base_url
    
    def get_latest_session(self):
        """Get the most recent or currently active session"""
        try:
            # Get sessions from today and recent past
            url = f"{self.base_url}/sessions"
            params = {
                'date_start>=': (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            sessions = response.json()
            
            if not sessions:
                logger.warning("No recent sessions found")
                return None
            
            # Sort by date, get most recent
            sessions_df = pd.DataFrame(sessions)
            sessions_df['date_start'] = pd.to_datetime(sessions_df['date_start'])
            latest = sessions_df.sort_values('date_start', ascending=False).iloc[0]
            
            return latest.to_dict()
            
        except Exception as e:
            logger.error(f"Error fetching latest session: {e}")
            return None
    
    def fetch_live_laps(self, session_key: int):
        """Fetch live lap data for a session"""
        try:
            url = f"{self.base_url}/laps"
            params = {'session_key': session_key}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            laps = response.json()
            
            if laps:
                logger.info(f"✓ Fetched {len(laps)} laps for session {session_key}")
                return pd.DataFrame(laps)
            else:
                logger.warning(f"No lap data available for session {session_key}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching laps: {e}")
            return pd.DataFrame()
    
    def fetch_live_positions(self, session_key: int):
        """Fetch live position data"""
        try:
            url = f"{self.base_url}/position"
            params = {'session_key': session_key}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            positions = response.json()
            
            if positions:
                logger.info(f"✓ Fetched {len(positions)} position updates")
                return pd.DataFrame(positions)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return pd.DataFrame()
    
    def fetch_live_weather(self, session_key: int):
        """Fetch live weather data"""
        try:
            url = f"{self.base_url}/weather"
            params = {'session_key': session_key}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            weather = response.json()
            
            if weather:
                logger.info(f"✓ Fetched weather data")
                return pd.DataFrame(weather)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching weather: {e}")
            return pd.DataFrame()
    
    def fetch_session_complete_data(self, session_key: int, save_dir: str = None):
        """Fetch all available data for a session"""
        logger.info(f"\n=== Fetching Live Data for Session {session_key} ===")
        
        data = {}
        
        # Fetch all data types
        data['laps'] = self.fetch_live_laps(session_key)
        data['positions'] = self.fetch_live_positions(session_key)
        data['weather'] = self.fetch_live_weather(session_key)
        
        # Save if directory provided
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            for data_type, df in data.items():
                if not df.empty:
                    filepath = save_path / f"{data_type}_session_{session_key}.csv"
                    df.to_csv(filepath, index=False)
                    logger.info(f"Saved {data_type} to {filepath}")
        
        # Summary
        logger.info("\n=== Data Fetch Summary ===")
        for data_type, df in data.items():
            logger.info(f"{data_type}: {len(df)} records")
        
        return data
    
    def monitor_live_session(self, session_key: int, interval: int = 30):
        """
        Monitor a live session and fetch data periodically
        
        Args:
            session_key: Session to monitor
            interval: Seconds between updates
        """
        logger.info(f"\n🔴 LIVE: Monitoring session {session_key}")
        logger.info(f"Updates every {interval} seconds. Press Ctrl+C to stop.\n")
        
        try:
            while True:
                data = self.fetch_session_complete_data(session_key)
                
                # Show current standings if position data available
                if not data['positions'].empty:
                    latest_positions = data['positions'].sort_values('date').groupby('driver_number').tail(1)
                    latest_positions = latest_positions.sort_values('position')
                    
                    print("\n📊 Current Positions:")
                    for _, row in latest_positions.head(10).iterrows():
                        print(f"  {int(row['position']):2d}. Driver #{int(row['driver_number'])}")
                
                # Show lap count
                if not data['laps'].empty:
                    total_laps = len(data['laps'])
                    drivers = data['laps']['driver_number'].nunique()
                    print(f"\n⏱️  Total laps completed: {total_laps} ({drivers} drivers)")
                
                print(f"\n⏳ Next update in {interval} seconds...\n")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("\n✋ Stopped monitoring")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch live F1 data')
    parser.add_argument('--session', type=int, help='Session key to fetch')
    parser.add_argument('--monitor', action='store_true', help='Monitor live session')
    parser.add_argument('--interval', type=int, default=30, help='Update interval for monitoring (seconds)')
    parser.add_argument('--save', type=str, help='Directory to save data')
    
    args = parser.parse_args()
    
    fetcher = LiveDataFetcher()
    
    if args.session:
        if args.monitor:
            fetcher.monitor_live_session(args.session, args.interval)
        else:
            data = fetcher.fetch_session_complete_data(args.session, args.save)
            
            # Show summary
            print("\n" + "="*70)
            print("📊 DATA FETCH SUMMARY")
            print("="*70)
            for data_type, df in data.items():
                if not df.empty:
                    print(f"✓ {data_type}: {len(df)} records")
                else:
                    print(f"✗ {data_type}: No data available")
            print("="*70)
    else:
        # Find latest session
        session = fetcher.get_latest_session()
        if session:
            print(f"\n🏁 Latest Session: {session['meeting_name']} - {session['session_name']}")
            print(f"📅 Date: {session['date_start']}")
            print(f"🔑 Session Key: {session['session_key']}")
            
            fetch = input("\nFetch data for this session? (y/n): ")
            if fetch.lower() == 'y':
                data = fetcher.fetch_session_complete_data(session['session_key'], args.save)


if __name__ == '__main__':
    main()
