"""
OpenF1 API Client
Handles all interactions with the OpenF1 API
"""

import requests
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class OpenF1Client:
    """Client for interacting with the OpenF1 API"""
    
    def __init__(self, base_url: str = "https://api.openf1.org/v1", 
                 timeout: int = 30, max_retries: int = 3, 
                 rate_limit_delay: float = 1.0):
        """
        Initialize the OpenF1 API client
        
        Args:
            base_url: Base URL for the OpenF1 API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            rate_limit_delay: Delay between requests in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> List[Dict]:
        """
        Make a request to the OpenF1 API with retry logic
        
        Args:
            endpoint: API endpoint (e.g., 'sessions', 'laps')
            params: Query parameters
            
        Returns:
            List of data dictionaries from the API
        """
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Request to {url} with params {params} (attempt {attempt + 1})")
                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
                
        return []
    
    def get_sessions(self, year: Optional[int] = None, 
                     session_type: Optional[str] = None,
                     country: Optional[str] = None) -> List[Dict]:
        """
        Get session data (practice, qualifying, race)
        
        Args:
            year: Season year
            session_type: Type of session (Practice, Qualifying, Race, Sprint)
            country: Country name
            
        Returns:
            List of session dictionaries
        """
        params = {}
        if year:
            params['year'] = year
        if session_type:
            params['session_type'] = session_type
        if country:
            params['country_name'] = country
            
        logger.info(f"Fetching sessions for year={year}, type={session_type}")
        return self._make_request('sessions', params)
    
    def get_laps(self, session_key: Optional[int] = None,
                 driver_number: Optional[int] = None) -> List[Dict]:
        """
        Get lap time data
        
        Args:
            session_key: Unique session identifier
            driver_number: Driver number
            
        Returns:
            List of lap dictionaries
        """
        params = {}
        if session_key:
            params['session_key'] = session_key
        if driver_number:
            params['driver_number'] = driver_number
            
        logger.info(f"Fetching laps for session={session_key}, driver={driver_number}")
        return self._make_request('laps', params)
    
    def get_positions(self, session_key: int, driver_number: Optional[int] = None) -> List[Dict]:
        """
        Get driver position data throughout the session
        
        Args:
            session_key: Unique session identifier
            driver_number: Driver number
            
        Returns:
            List of position dictionaries
        """
        params = {'session_key': session_key}
        if driver_number:
            params['driver_number'] = driver_number
            
        logger.info(f"Fetching positions for session={session_key}")
        return self._make_request('position', params)
    
    def get_weather(self, session_key: int) -> List[Dict]:
        """
        Get weather data for a session
        
        Args:
            session_key: Unique session identifier
            
        Returns:
            List of weather dictionaries
        """
        params = {'session_key': session_key}
        logger.info(f"Fetching weather for session={session_key}")
        return self._make_request('weather', params)
    
    def get_pit_stops(self, session_key: int, driver_number: Optional[int] = None) -> List[Dict]:
        """
        Get pit stop data
        
        Args:
            session_key: Unique session identifier
            driver_number: Driver number
            
        Returns:
            List of pit stop dictionaries
        """
        params = {'session_key': session_key}
        if driver_number:
            params['driver_number'] = driver_number
            
        logger.info(f"Fetching pit stops for session={session_key}")
        return self._make_request('pit', params)
    
    def get_intervals(self, session_key: int, driver_number: Optional[int] = None) -> List[Dict]:
        """
        Get time intervals between drivers
        
        Args:
            session_key: Unique session identifier
            driver_number: Driver number
            
        Returns:
            List of interval dictionaries
        """
        params = {'session_key': session_key}
        if driver_number:
            params['driver_number'] = driver_number
            
        logger.info(f"Fetching intervals for session={session_key}")
        return self._make_request('intervals', params)
    
    def get_drivers(self, session_key: Optional[int] = None, 
                    driver_number: Optional[int] = None) -> List[Dict]:
        """
        Get driver information
        
        Args:
            session_key: Unique session identifier
            driver_number: Driver number
            
        Returns:
            List of driver dictionaries
        """
        params = {}
        if session_key:
            params['session_key'] = session_key
        if driver_number:
            params['driver_number'] = driver_number
            
        logger.info(f"Fetching drivers for session={session_key}")
        return self._make_request('drivers', params)
    
    def close(self):
        """Close the session"""
        self.session.close()
