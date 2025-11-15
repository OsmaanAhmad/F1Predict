"""
Utility Helper Functions
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: Dict[str, Any]):
    """
    Setup logging configuration
    
    Args:
        config: Configuration dictionary
    """
    log_config = config.get('logging', {})
    
    level = getattr(logging, log_config.get('level', 'INFO'))
    format_str = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = log_config.get('file')
    
    # Create logs directory if needed
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=handlers
    )
    
    # Suppress verbose libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)


def ensure_directories(config: Dict[str, Any]):
    """
    Ensure all required directories exist
    
    Args:
        config: Configuration dictionary
    """
    data_config = config.get('data', {})
    
    dirs = [
        data_config.get('raw_data_path', 'data/raw'),
        data_config.get('processed_data_path', 'data/processed'),
        data_config.get('model_path', 'data/models'),
        'logs'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def format_driver_name(driver_number: int, drivers_df=None) -> str:
    """
    Format driver name from driver number
    
    Args:
        driver_number: Driver number
        drivers_df: Optional drivers DataFrame to lookup name
        
    Returns:
        Formatted driver identifier
    """
    if drivers_df is not None and not drivers_df.empty:
        driver_info = drivers_df[drivers_df['driver_number'] == driver_number]
        if not driver_info.empty:
            row = driver_info.iloc[0]
            if 'name_acronym' in row:
                return f"{row['name_acronym']} (#{driver_number})"
            elif 'full_name' in row:
                return f"{row['full_name']} (#{driver_number})"
    
    return f"Driver #{driver_number}"


def format_race_name(session_info: Dict[str, Any]) -> str:
    """
    Format race name from session information
    
    Args:
        session_info: Session information dictionary
        
    Returns:
        Formatted race name
    """
    name = session_info.get('session_name', 'Unknown Race')
    year = session_info.get('year', '')
    
    if year:
        return f"{year} {name}"
    return name


def calculate_points(position: int) -> int:
    """
    Calculate F1 championship points for a position
    
    Args:
        position: Finishing position
        
    Returns:
        Points awarded
    """
    points_system = {
        1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
        6: 8, 7: 6, 8: 4, 9: 2, 10: 1
    }
    
    return points_system.get(position, 0)
