"""
Data Preprocessing Module
Cleans and prepares raw F1 data for feature engineering
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocesses raw F1 data"""
    
    def __init__(self):
        self.processed_data = {}
        
    def preprocess_laps(self, laps_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess lap data
        
        Args:
            laps_df: Raw laps DataFrame
            
        Returns:
            Preprocessed laps DataFrame
        """
        if laps_df.empty:
            logger.warning("Empty laps DataFrame")
            return laps_df
        
        df = laps_df.copy()
        
        # Convert duration columns to seconds
        if 'lap_duration' in df.columns:
            df['lap_time_seconds'] = pd.to_timedelta(df['lap_duration'], errors='coerce').dt.total_seconds()
        
        if 'duration_sector_1' in df.columns:
            df['sector_1_seconds'] = pd.to_timedelta(df['duration_sector_1'], errors='coerce').dt.total_seconds()
        
        if 'duration_sector_2' in df.columns:
            df['sector_2_seconds'] = pd.to_timedelta(df['duration_sector_2'], errors='coerce').dt.total_seconds()
        
        if 'duration_sector_3' in df.columns:
            df['sector_3_seconds'] = pd.to_timedelta(df['duration_sector_3'], errors='coerce').dt.total_seconds()
        
        # Remove invalid laps (outliers, pit laps)
        if 'is_pit_out_lap' in df.columns:
            df = df[df['is_pit_out_lap'] == False]
        
        # Remove laps with missing times
        if 'lap_time_seconds' in df.columns:
            df = df[df['lap_time_seconds'].notna()]
            
            # Remove extreme outliers (e.g., > 2 minutes)
            df = df[df['lap_time_seconds'] < 120]
        
        # Convert date columns
        if 'date_start' in df.columns:
            df['date_start'] = pd.to_datetime(df['date_start'], format='ISO8601', errors='coerce')
        
        logger.info(f"Preprocessed {len(df)} laps")
        return df
    
    def preprocess_positions(self, positions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess position data
        
        Args:
            positions_df: Raw positions DataFrame
            
        Returns:
            Preprocessed positions DataFrame
        """
        if positions_df.empty:
            logger.warning("Empty positions DataFrame")
            return positions_df
        
        df = positions_df.copy()
        
        # Convert date columns
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='ISO8601', errors='coerce')
        
        # Ensure position is numeric
        if 'position' in df.columns:
            df['position'] = pd.to_numeric(df['position'], errors='coerce')
        
        logger.info(f"Preprocessed {len(df)} position records")
        return df
    
    def preprocess_weather(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess weather data
        
        Args:
            weather_df: Raw weather DataFrame
            
        Returns:
            Preprocessed weather DataFrame
        """
        if weather_df.empty:
            logger.warning("Empty weather DataFrame")
            return weather_df
        
        df = weather_df.copy()
        
        # Convert date columns
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='ISO8601', errors='coerce')
        
        # Ensure numeric columns
        numeric_cols = ['air_temperature', 'track_temperature', 'humidity', 'pressure', 'wind_speed']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill missing values with forward fill (weather changes gradually)
        df = df.fillna(method='ffill')
        
        logger.info(f"Preprocessed {len(df)} weather records")
        return df
    
    def preprocess_pit_stops(self, pit_stops_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess pit stop data
        
        Args:
            pit_stops_df: Raw pit stops DataFrame
            
        Returns:
            Preprocessed pit stops DataFrame
        """
        if pit_stops_df.empty:
            logger.warning("Empty pit stops DataFrame")
            return pit_stops_df
        
        df = pit_stops_df.copy()
        
        # Convert date columns
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='ISO8601', errors='coerce')
        
        # Convert pit duration to seconds
        if 'pit_duration' in df.columns:
            df['pit_duration_seconds'] = pd.to_timedelta(df['pit_duration'], errors='coerce').dt.total_seconds()
        
        logger.info(f"Preprocessed {len(df)} pit stop records")
        return df
    
    def get_race_results(self, positions_df: pd.DataFrame, session_key: int) -> pd.DataFrame:
        """
        Extract final race results from position data
        
        Args:
            positions_df: Preprocessed positions DataFrame
            session_key: Session key for the race
            
        Returns:
            DataFrame with final positions for each driver
        """
        if positions_df.empty:
            return pd.DataFrame()
        
        # Filter by session
        session_positions = positions_df[positions_df['session_key'] == session_key].copy()
        
        if session_positions.empty:
            return pd.DataFrame()
        
        # Get the last position for each driver
        session_positions = session_positions.sort_values('date')
        final_positions = session_positions.groupby('driver_number').last().reset_index()
        
        # Select relevant columns
        result_cols = ['driver_number', 'position']
        if 'date' in final_positions.columns:
            result_cols.append('date')
            
        results = final_positions[result_cols].copy()
        results = results.sort_values('position')
        
        return results
    
    def preprocess_all(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Preprocess all data types
        
        Args:
            raw_data: Dictionary of raw DataFrames
            
        Returns:
            Dictionary of preprocessed DataFrames
        """
        processed = {}
        
        if 'laps' in raw_data:
            processed['laps'] = self.preprocess_laps(raw_data['laps'])
        
        if 'positions' in raw_data:
            processed['positions'] = self.preprocess_positions(raw_data['positions'])
        
        if 'weather' in raw_data:
            processed['weather'] = self.preprocess_weather(raw_data['weather'])
        
        if 'pit_stops' in raw_data:
            processed['pit_stops'] = self.preprocess_pit_stops(raw_data['pit_stops'])
        
        # Pass through sessions and drivers without modification
        if 'sessions' in raw_data:
            processed['sessions'] = raw_data['sessions'].copy()
        
        if 'drivers' in raw_data:
            processed['drivers'] = raw_data['drivers'].copy()
        
        if 'intervals' in raw_data:
            processed['intervals'] = raw_data['intervals'].copy()
        
        self.processed_data = processed
        return processed
