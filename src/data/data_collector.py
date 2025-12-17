"""
Data Collection Module
Orchestrates data collection from OpenF1 API and saves to local storage
"""

import os
import json
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from .api_client import OpenF1Client

logger = logging.getLogger(__name__)


class DataCollector:
    """Collects and stores F1 data from OpenF1 API"""
    
    def __init__(self, api_client: OpenF1Client, raw_data_path: str):
        """
        Initialize data collector
        
        Args:
            api_client: OpenF1 API client instance
            raw_data_path: Path to store raw data
        """
        self.api_client = api_client
        self.raw_data_path = Path(raw_data_path)
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        
    def collect_season_data(self, year: int, session_types: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Collect all data for a specific season
        
        Args:
            year: Season year
            session_types: List of session types to collect (default: ['Race'])
            
        Returns:
            Dictionary of DataFrames with collected data
        """
        if session_types is None:
            session_types = ['Race']
            
        logger.info(f"Collecting data for {year} season")
        
        # Get all sessions for the year
        sessions = self.api_client.get_sessions(year=year)
        sessions_df = pd.DataFrame(sessions)
        
        # Filter by session type
        race_sessions = sessions_df[sessions_df['session_type'].isin(session_types)]
        logger.info(f"Found {len(race_sessions)} sessions")
        
        # Initialize data containers
        all_laps = []
        all_positions = []
        all_weather = []
        all_pit_stops = []
        all_intervals = []
        all_drivers = []
        
        # Collect data for each session using parallel processing
        # Reduced to 2 workers to avoid API rate limiting
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit all session collection tasks
            future_to_session = {
                executor.submit(self._collect_session_data, row['session_key'], row.get('session_name', 'Unknown')): row
                for _, row in race_sessions.iterrows()
            }
            
            # Process completed tasks
            for future in as_completed(future_to_session):
                try:
                    session_data = future.result()
                    
                    # Aggregate data from each session
                    if session_data['laps']:
                        all_laps.extend(session_data['laps'])
                    if session_data['positions']:
                        all_positions.extend(session_data['positions'])
                    if session_data['weather']:
                        all_weather.extend(session_data['weather'])
                    if session_data['pit_stops']:
                        all_pit_stops.extend(session_data['pit_stops'])
                    if session_data['intervals']:
                        all_intervals.extend(session_data['intervals'])
                    if session_data['drivers']:
                        all_drivers.extend(session_data['drivers'])
                        
                except Exception as e:
                    session = future_to_session[future]
                    logger.error(f"Error collecting data for session {session.get('session_key')}: {e}")
        
        # Convert to DataFrames
        data = {
            'sessions': sessions_df,
            'laps': pd.DataFrame(all_laps) if all_laps else pd.DataFrame(),
            'positions': pd.DataFrame(all_positions) if all_positions else pd.DataFrame(),
            'weather': pd.DataFrame(all_weather) if all_weather else pd.DataFrame(),
            'pit_stops': pd.DataFrame(all_pit_stops) if all_pit_stops else pd.DataFrame(),
            'intervals': pd.DataFrame(all_intervals) if all_intervals else pd.DataFrame(),
            'drivers': pd.DataFrame(all_drivers) if all_drivers else pd.DataFrame(),
        }
        
        # Save raw data
        self.save_raw_data(data, year)
        
        return data
    
    def _collect_session_data(self, session_key: int, session_name: str) -> Dict[str, List]:
        """
        Collect all data types for a single session using parallel requests
        
        Args:
            session_key: Unique session identifier
            session_name: Name of the session
            
        Returns:
            Dictionary with lists of data for each type
        """
        logger.info(f"Collecting data for {session_name} (session_key: {session_key})")
        
        session_data = {
            'laps': [],
            'positions': [],
            'weather': [],
            'pit_stops': [],
            'intervals': [],
            'drivers': []
        }
        
        # Use ThreadPoolExecutor to fetch all data types in parallel
        with ThreadPoolExecutor(max_workers=6) as executor:
            # Submit all API calls for this session
            futures = {
                'laps': executor.submit(self.api_client.get_laps, session_key=session_key),
                'positions': executor.submit(self.api_client.get_positions, session_key=session_key),
                'weather': executor.submit(self.api_client.get_weather, session_key=session_key),
                'pit_stops': executor.submit(self.api_client.get_pit_stops, session_key=session_key),
                'intervals': executor.submit(self.api_client.get_intervals, session_key=session_key),
                'drivers': executor.submit(self.api_client.get_drivers, session_key=session_key)
            }
            
            # Collect results as they complete
            for data_type, future in futures.items():
                try:
                    result = future.result(timeout=60)  # 60 second timeout per request
                    if result:
                        session_data[data_type] = result
                        logger.debug(f"  ✓ {data_type}: {len(result)} records")
                except Exception as e:
                    logger.warning(f"  ✗ {data_type} failed for session {session_key}: {e}")
        
        return session_data
    
    def save_raw_data(self, data: Dict[str, pd.DataFrame], year: int):
        """
        Save raw data to disk
        
        Args:
            data: Dictionary of DataFrames to save
            year: Season year
        """
        year_path = self.raw_data_path / str(year)
        year_path.mkdir(parents=True, exist_ok=True)
        
        for data_type, df in data.items():
            if not df.empty:
                file_path = year_path / f"{data_type}.csv"
                df.to_csv(file_path, index=False)
                logger.info(f"Saved {len(df)} records to {file_path}")
            else:
                logger.warning(f"No data to save for {data_type}")
    
    def load_raw_data(self, year: int) -> Dict[str, pd.DataFrame]:
        """
        Load raw data from disk
        
        Args:
            year: Season year
            
        Returns:
            Dictionary of DataFrames
        """
        year_path = self.raw_data_path / str(year)
        
        if not year_path.exists():
            logger.error(f"No data found for year {year}")
            return {}
        
        data = {}
        data_types = ['sessions', 'laps', 'positions', 'weather', 'pit_stops', 'intervals', 'drivers']
        
        for data_type in data_types:
            file_path = year_path / f"{data_type}.csv"
            if file_path.exists():
                data[data_type] = pd.read_csv(file_path)
                logger.info(f"Loaded {len(data[data_type])} records from {file_path}")
            else:
                logger.warning(f"File not found: {file_path}")
                data[data_type] = pd.DataFrame()
        
        return data
    
    def collect_multiple_seasons(self, years: List[int], session_types: Optional[List[str]] = None) -> Dict[int, Dict[str, pd.DataFrame]]:
        """
        Collect data for multiple seasons
        
        Args:
            years: List of season years
            session_types: List of session types to collect
            
        Returns:
            Dictionary mapping year to data dictionaries
        """
        all_data = {}
        
        for year in years:
            logger.info(f"Collecting data for {year}")
            all_data[year] = self.collect_season_data(year, session_types)
        
        return all_data
