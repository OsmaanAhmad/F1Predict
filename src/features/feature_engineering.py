"""
Feature Engineering Module
Creates predictive features from preprocessed F1 data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates features for F1 race prediction"""
    
    def __init__(self, rolling_windows: List[int] = [3, 5, 10]):
        """
        Initialize feature engineer
        
        Args:
            rolling_windows: Window sizes for rolling statistics
        """
        self.rolling_windows = rolling_windows
        
    def create_lap_features(self, laps_df: pd.DataFrame, session_key: int, driver_number: int) -> Dict[str, float]:
        """
        Create lap-based features for a driver in a session
        
        Args:
            laps_df: Preprocessed laps DataFrame
            session_key: Session identifier
            driver_number: Driver number
            
        Returns:
            Dictionary of lap features
        """
        driver_laps = laps_df[
            (laps_df['session_key'] == session_key) & 
            (laps_df['driver_number'] == driver_number)
        ].copy()
        
        if driver_laps.empty or 'lap_time_seconds' not in driver_laps.columns:
            return {
                'avg_lap_time': np.nan,
                'lap_time_std': np.nan,
                'best_lap_time': np.nan,
                'consistency_score': np.nan,
                'total_laps': 0
            }
        
        lap_times = driver_laps['lap_time_seconds'].dropna()
        
        features = {
            'avg_lap_time': lap_times.mean(),
            'lap_time_std': lap_times.std(),
            'best_lap_time': lap_times.min(),
            'consistency_score': lap_times.std() / lap_times.mean() if lap_times.mean() > 0 else np.nan,
            'total_laps': len(lap_times)
        }
        
        return features
    
    def create_qualifying_features(self, sessions_df: pd.DataFrame, laps_df: pd.DataFrame, 
                                   race_session_key: int, driver_number: int) -> Dict[str, float]:
        """
        Create qualifying-based features
        
        Args:
            sessions_df: Sessions DataFrame
            laps_df: Laps DataFrame
            race_session_key: Race session key
            driver_number: Driver number
            
        Returns:
            Dictionary of qualifying features
        """
        # Find the qualifying session for this race
        race_session = sessions_df[sessions_df['session_key'] == race_session_key]
        if race_session.empty:
            return {'qualifying_position': np.nan, 'qualifying_best_lap': np.nan}
        
        # Get qualifying session (same event, different session type)
        race_info = race_session.iloc[0]
        qualifying_sessions = sessions_df[
            (sessions_df['session_name'] == race_info.get('session_name')) &
            (sessions_df['session_type'] == 'Qualifying')
        ]
        
        if qualifying_sessions.empty:
            return {'qualifying_position': np.nan, 'qualifying_best_lap': np.nan}
        
        qual_session_key = qualifying_sessions.iloc[0]['session_key']
        
        # Get driver's best qualifying lap
        qual_laps = laps_df[
            (laps_df['session_key'] == qual_session_key) & 
            (laps_df['driver_number'] == driver_number)
        ]
        
        if qual_laps.empty or 'lap_time_seconds' not in qual_laps.columns:
            return {'qualifying_position': np.nan, 'qualifying_best_lap': np.nan}
        
        best_lap = qual_laps['lap_time_seconds'].min()
        
        # Calculate qualifying position based on best lap times
        all_qual_laps = laps_df[laps_df['session_key'] == qual_session_key]
        if not all_qual_laps.empty and 'lap_time_seconds' in all_qual_laps.columns:
            driver_best_laps = all_qual_laps.groupby('driver_number')['lap_time_seconds'].min().sort_values()
            qual_position = list(driver_best_laps.index).index(driver_number) + 1 if driver_number in driver_best_laps.index else np.nan
        else:
            qual_position = np.nan
        
        return {
            'qualifying_position': qual_position,
            'qualifying_best_lap': best_lap
        }
    
    def create_historical_features(self, positions_df: pd.DataFrame, sessions_df: pd.DataFrame,
                                   current_session_key: int, driver_number: int) -> Dict[str, float]:
        """
        Create features based on historical performance
        
        Args:
            positions_df: Positions DataFrame
            sessions_df: Sessions DataFrame
            current_session_key: Current race session key
            driver_number: Driver number
            
        Returns:
            Dictionary of historical features
        """
        # Get all races before current one
        current_session = sessions_df[sessions_df['session_key'] == current_session_key]
        if current_session.empty:
            return self._empty_historical_features()
        
        current_date = pd.to_datetime(current_session.iloc[0].get('date_start'))
        
        past_sessions = sessions_df[
            (sessions_df['session_type'] == 'Race') &
            (pd.to_datetime(sessions_df['date_start']) < current_date)
        ]
        
        if past_sessions.empty:
            return self._empty_historical_features()
        
        # Get driver's past race results
        past_results = []
        for _, session in past_sessions.iterrows():
            session_positions = positions_df[
                (positions_df['session_key'] == session['session_key']) &
                (positions_df['driver_number'] == driver_number)
            ]
            
            if not session_positions.empty:
                final_pos = session_positions.iloc[-1]['position']
                if pd.notna(final_pos):
                    past_results.append(final_pos)
        
        if not past_results:
            return self._empty_historical_features()
        
        # Create rolling statistics
        features = {
            'avg_finish_position': np.mean(past_results),
            'best_finish_position': np.min(past_results),
            'worst_finish_position': np.max(past_results),
            'finish_position_std': np.std(past_results),
            'races_completed': len(past_results),
        }
        
        # Add rolling window features
        for window in self.rolling_windows:
            if len(past_results) >= window:
                recent_results = past_results[-window:]
                features[f'avg_finish_last_{window}'] = np.mean(recent_results)
                features[f'trend_last_{window}'] = recent_results[-1] - recent_results[0]  # Negative = improving
            else:
                features[f'avg_finish_last_{window}'] = np.nan
                features[f'trend_last_{window}'] = np.nan
        
        # Podium rate
        features['podium_rate'] = sum(1 for pos in past_results if pos <= 3) / len(past_results)
        features['win_rate'] = sum(1 for pos in past_results if pos == 1) / len(past_results)
        
        return features
    
    def _empty_historical_features(self) -> Dict[str, float]:
        """Return empty historical features"""
        features = {
            'avg_finish_position': np.nan,
            'best_finish_position': np.nan,
            'worst_finish_position': np.nan,
            'finish_position_std': np.nan,
            'races_completed': 0,
            'podium_rate': 0.0,
            'win_rate': 0.0,
        }
        
        for window in self.rolling_windows:
            features[f'avg_finish_last_{window}'] = np.nan
            features[f'trend_last_{window}'] = np.nan
        
        return features
    
    def create_team_features(self, positions_df: pd.DataFrame, drivers_df: pd.DataFrame,
                            sessions_df: pd.DataFrame, current_session_key: int, 
                            driver_number: int) -> Dict[str, float]:
        """
        Create team-based features
        
        Args:
            positions_df: Positions DataFrame
            drivers_df: Drivers DataFrame
            sessions_df: Sessions DataFrame
            current_session_key: Current race session key
            driver_number: Driver number
            
        Returns:
            Dictionary of team features
        """
        # Get driver's team
        driver_info = drivers_df[
            (drivers_df['session_key'] == current_session_key) &
            (drivers_df['driver_number'] == driver_number)
        ]
        
        if driver_info.empty or 'team_name' not in driver_info.columns:
            return {
                'team_avg_position': np.nan,
                'team_consistency': np.nan,
                'teammate_avg_position': np.nan
            }
        
        team_name = driver_info.iloc[0]['team_name']
        
        # Get all drivers from the same team in past races
        current_session = sessions_df[sessions_df['session_key'] == current_session_key]
        if current_session.empty:
            return {'team_avg_position': np.nan, 'team_consistency': np.nan, 'teammate_avg_position': np.nan}
        
        current_date = pd.to_datetime(current_session.iloc[0].get('date_start'))
        past_sessions = sessions_df[
            (sessions_df['session_type'] == 'Race') &
            (pd.to_datetime(sessions_df['date_start']) < current_date)
        ]
        
        team_positions = []
        teammate_positions = []
        
        for _, session in past_sessions.iterrows():
            session_drivers = drivers_df[
                (drivers_df['session_key'] == session['session_key']) &
                (drivers_df['team_name'] == team_name)
            ]
            
            for _, teammate in session_drivers.iterrows():
                teammate_num = teammate['driver_number']
                positions = positions_df[
                    (positions_df['session_key'] == session['session_key']) &
                    (positions_df['driver_number'] == teammate_num)
                ]
                
                if not positions.empty:
                    final_pos = positions.iloc[-1]['position']
                    if pd.notna(final_pos):
                        team_positions.append(final_pos)
                        if teammate_num != driver_number:
                            teammate_positions.append(final_pos)
        
        features = {
            'team_avg_position': np.mean(team_positions) if team_positions else np.nan,
            'team_consistency': np.std(team_positions) if team_positions else np.nan,
            'teammate_avg_position': np.mean(teammate_positions) if teammate_positions else np.nan
        }
        
        return features
    
    def create_weather_features(self, weather_df: pd.DataFrame, session_key: int) -> Dict[str, float]:
        """
        Create weather-based features
        
        Args:
            weather_df: Weather DataFrame
            session_key: Session identifier
            
        Returns:
            Dictionary of weather features
        """
        session_weather = weather_df[weather_df['session_key'] == session_key]
        
        if session_weather.empty:
            return {
                'avg_air_temp': np.nan,
                'avg_track_temp': np.nan,
                'avg_humidity': np.nan,
                'rainfall': 0,
                'weather_variability': np.nan
            }
        
        features = {}
        
        if 'air_temperature' in session_weather.columns:
            features['avg_air_temp'] = session_weather['air_temperature'].mean()
        else:
            features['avg_air_temp'] = np.nan
            
        if 'track_temperature' in session_weather.columns:
            features['avg_track_temp'] = session_weather['track_temperature'].mean()
        else:
            features['avg_track_temp'] = np.nan
            
        if 'humidity' in session_weather.columns:
            features['avg_humidity'] = session_weather['humidity'].mean()
        else:
            features['avg_humidity'] = np.nan
        
        if 'rainfall' in session_weather.columns:
            features['rainfall'] = session_weather['rainfall'].sum()
        else:
            features['rainfall'] = 0
        
        # Weather variability (temperature change)
        if 'track_temperature' in session_weather.columns:
            features['weather_variability'] = session_weather['track_temperature'].std()
        else:
            features['weather_variability'] = np.nan
        
        return features
    
    def create_pit_stop_features(self, pit_stops_df: pd.DataFrame, session_key: int, 
                                 driver_number: int) -> Dict[str, float]:
        """
        Create pit stop features
        
        Args:
            pit_stops_df: Pit stops DataFrame
            session_key: Session identifier
            driver_number: Driver number
            
        Returns:
            Dictionary of pit stop features
        """
        driver_pits = pit_stops_df[
            (pit_stops_df['session_key'] == session_key) &
            (pit_stops_df['driver_number'] == driver_number)
        ]
        
        if driver_pits.empty:
            return {
                'num_pit_stops': 0,
                'avg_pit_duration': np.nan,
                'total_pit_time': 0
            }
        
        features = {
            'num_pit_stops': len(driver_pits)
        }
        
        if 'pit_duration_seconds' in driver_pits.columns:
            features['avg_pit_duration'] = driver_pits['pit_duration_seconds'].mean()
            features['total_pit_time'] = driver_pits['pit_duration_seconds'].sum()
        else:
            features['avg_pit_duration'] = np.nan
            features['total_pit_time'] = 0
        
        return features
    
    def create_feature_matrix(self, processed_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create complete feature matrix for all drivers and races
        
        Args:
            processed_data: Dictionary of preprocessed DataFrames
            
        Returns:
            Feature matrix DataFrame
        """
        sessions_df = processed_data.get('sessions', pd.DataFrame())
        laps_df = processed_data.get('laps', pd.DataFrame())
        positions_df = processed_data.get('positions', pd.DataFrame())
        weather_df = processed_data.get('weather', pd.DataFrame())
        pit_stops_df = processed_data.get('pit_stops', pd.DataFrame())
        drivers_df = processed_data.get('drivers', pd.DataFrame())
        
        # Get all race sessions
        race_sessions = sessions_df[sessions_df['session_type'] == 'Race']
        
        features_list = []
        
        logger.info(f"Creating features for {len(race_sessions)} races")
        
        for _, session in race_sessions.iterrows():
            session_key = session['session_key']
            session_name = session.get('session_name', 'Unknown')
            
            # Get all drivers in this session
            session_drivers = drivers_df[drivers_df['session_key'] == session_key]
            
            if session_drivers.empty:
                # Fallback: get drivers from positions
                session_positions = positions_df[positions_df['session_key'] == session_key]
                driver_numbers = session_positions['driver_number'].unique()
            else:
                driver_numbers = session_drivers['driver_number'].unique()
            
            logger.info(f"Processing {session_name} with {len(driver_numbers)} drivers")
            
            # Weather features (same for all drivers in a session)
            weather_features = self.create_weather_features(weather_df, session_key)
            
            for driver_number in driver_numbers:
                # Create all feature groups
                lap_features = self.create_lap_features(laps_df, session_key, driver_number)
                qual_features = self.create_qualifying_features(sessions_df, laps_df, session_key, driver_number)
                hist_features = self.create_historical_features(positions_df, sessions_df, session_key, driver_number)
                team_features = self.create_team_features(positions_df, drivers_df, sessions_df, session_key, driver_number)
                pit_features = self.create_pit_stop_features(pit_stops_df, session_key, driver_number)
                
                # Get target (final position)
                final_positions = positions_df[
                    (positions_df['session_key'] == session_key) &
                    (positions_df['driver_number'] == driver_number)
                ]
                
                if not final_positions.empty:
                    final_position = final_positions.iloc[-1]['position']
                else:
                    final_position = np.nan
                
                # Combine all features
                row = {
                    'session_key': session_key,
                    'driver_number': driver_number,
                    'final_position': final_position,
                    **lap_features,
                    **qual_features,
                    **hist_features,
                    **team_features,
                    **weather_features,
                    **pit_features
                }
                
                features_list.append(row)
        
        features_df = pd.DataFrame(features_list)
        logger.info(f"Created feature matrix with shape {features_df.shape}")
        
        return features_df
