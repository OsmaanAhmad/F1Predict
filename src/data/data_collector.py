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
        
        # Collect data for each session
        for _, session in race_sessions.iterrows():
            session_key = session['session_key']
            session_name = session.get('session_name', 'Unknown')
            
            logger.info(f"Collecting data for {session_name} (session_key: {session_key})")
            
            try:
                # Get laps
                laps = self.api_client.get_laps(session_key=session_key)
                if laps:
                    all_laps.extend(laps)
                    
                # Get positions
                positions = self.api_client.get_positions(session_key=session_key)
                if positions:
                    all_positions.extend(positions)
                    
                # Get weather
                weather = self.api_client.get_weather(session_key=session_key)
                if weather:
                    all_weather.extend(weather)
                    
                # Get pit stops
                pit_stops = self.api_client.get_pit_stops(session_key=session_key)
                if pit_stops:
                    all_pit_stops.extend(pit_stops)
                    
                # Get intervals
                intervals = self.api_client.get_intervals(session_key=session_key)
                if intervals:
                    all_intervals.extend(intervals)
                    
                # Get drivers
                drivers = self.api_client.get_drivers(session_key=session_key)
                if drivers:
                    all_drivers.extend(drivers)
                    
            except Exception as e:
                logger.error(f"Error collecting data for session {session_key}: {e}")
                continue
        
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
