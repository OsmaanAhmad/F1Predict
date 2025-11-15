"""
F1 Race Outcome Predictor - Main Entry Point
"""

import argparse
import logging
from pathlib import Path
import pandas as pd

from src.utils import load_config, setup_logging, ensure_directories
from src.data import OpenF1Client, DataCollector, DataPreprocessor
from src.features import FeatureEngineer
from src.models import ModelTrainer, RacePredictor

logger = logging.getLogger(__name__)


def collect_data(config):
    """Collect data from OpenF1 API"""
    logger.info("=== Starting Data Collection ===")
    
    # Initialize API client
    api_config = config.get('api', {})
    client = OpenF1Client(
        base_url=api_config.get('base_url'),
        timeout=api_config.get('timeout'),
        max_retries=api_config.get('max_retries'),
        rate_limit_delay=api_config.get('rate_limit_delay')
    )
    
    # Initialize data collector
    data_config = config.get('data', {})
    collector = DataCollector(
        api_client=client,
        raw_data_path=data_config.get('raw_data_path')
    )
    
    # Collect data for specified seasons
    seasons = data_config.get('seasons', [2023, 2024])
    logger.info(f"Collecting data for seasons: {seasons}")
    
    for season in seasons:
        logger.info(f"\n--- Collecting {season} Season ---")
        collector.collect_season_data(season, session_types=['Race'])
    
    client.close()
    logger.info("=== Data Collection Complete ===\n")


def preprocess_data(config):
    """Preprocess collected data"""
    logger.info("=== Starting Data Preprocessing ===")
    
    data_config = config.get('data', {})
    seasons = data_config.get('seasons', [2023, 2024])
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load and process data for each season
    collector = DataCollector(
        api_client=None,
        raw_data_path=data_config.get('raw_data_path')
    )
    
    all_processed_data = {}
    
    for season in seasons:
        logger.info(f"\n--- Processing {season} Season ---")
        raw_data = collector.load_raw_data(season)
        
        if not raw_data:
            logger.warning(f"No raw data found for {season}")
            continue
        
        processed = preprocessor.preprocess_all(raw_data)
        all_processed_data[season] = processed
    
    # Combine all seasons and save
    processed_path = Path(data_config.get('processed_data_path'))
    processed_path.mkdir(parents=True, exist_ok=True)
    
    for data_type in ['laps', 'positions', 'weather', 'pit_stops', 'sessions', 'drivers']:
        combined = pd.concat([
            data.get(data_type, pd.DataFrame()) 
            for data in all_processed_data.values()
        ], ignore_index=True)
        
        if not combined.empty:
            output_file = processed_path / f"{data_type}_processed.csv"
            combined.to_csv(output_file, index=False)
            logger.info(f"Saved {len(combined)} {data_type} records to {output_file}")
    
    logger.info("=== Data Preprocessing Complete ===\n")


def engineer_features(config):
    """Create features from preprocessed data"""
    logger.info("=== Starting Feature Engineering ===")
    
    data_config = config.get('data', {})
    feature_config = config.get('features', {})
    
    processed_path = Path(data_config.get('processed_data_path'))
    
    # Load preprocessed data
    processed_data = {}
    for data_type in ['laps', 'positions', 'weather', 'pit_stops', 'sessions', 'drivers']:
        file_path = processed_path / f"{data_type}_processed.csv"
        if file_path.exists():
            processed_data[data_type] = pd.read_csv(file_path)
            logger.info(f"Loaded {len(processed_data[data_type])} {data_type} records")
    
    # Initialize feature engineer
    rolling_windows = feature_config.get('rolling_windows', [3, 5, 10])
    engineer = FeatureEngineer(rolling_windows=rolling_windows)
    
    # Create feature matrix
    features_df = engineer.create_feature_matrix(processed_data)
    
    # Save features
    features_file = processed_path / 'features.csv'
    features_df.to_csv(features_file, index=False)
    logger.info(f"Saved {len(features_df)} feature records to {features_file}")
    
    # Print feature summary
    logger.info(f"\nFeature Matrix Shape: {features_df.shape}")
    logger.info(f"Number of features: {len([c for c in features_df.columns if c not in ['session_key', 'driver_number', 'final_position']])}")
    logger.info(f"Samples with target: {features_df['final_position'].notna().sum()}")
    
    logger.info("=== Feature Engineering Complete ===\n")


def train_model(config, model_type=None):
    """Train prediction model"""
    logger.info("=== Starting Model Training ===")
    
    data_config = config.get('data', {})
    model_config = config.get('models', {})
    training_config = config.get('training', {})
    
    # Load features
    features_file = Path(data_config.get('processed_data_path')) / 'features.csv'
    features_df = pd.read_csv(features_file)
    logger.info(f"Loaded {len(features_df)} feature records")
    
    # Initialize trainer
    trainer = ModelTrainer(model_config, training_config)
    
    # Train model
    if model_type is None:
        model_type = model_config.get('default_model', 'xgboost')
    
    results = trainer.train(features_df, model_type=model_type)
    
    # Print results
    logger.info(f"\n--- Training Results ({model_type}) ---")
    logger.info(f"Train samples: {results['train_samples']}")
    logger.info(f"Test samples: {results['test_samples']}")
    logger.info("\nTrain Metrics:")
    for metric, value in results['train_metrics'].items():
        logger.info(f"  {metric}: {value:.4f}")
    logger.info("\nTest Metrics:")
    for metric, value in results['test_metrics'].items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Print top features
    logger.info("\nTop 10 Most Important Features:")
    top_features = results['feature_importance'].head(10)
    for idx, row in top_features.iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save model
    model_path = Path(data_config.get('model_path')) / f'{model_type}_model.pkl'
    trainer.save_model(str(model_path))
    logger.info(f"\nModel saved to {model_path}")
    
    logger.info("=== Model Training Complete ===\n")


def make_predictions(config, session_key=None, model_type=None):
    """Make predictions for a race"""
    logger.info("=== Making Predictions ===")
    
    data_config = config.get('data', {})
    model_config = config.get('models', {})
    training_config = config.get('training', {})
    
    # Load features
    features_file = Path(data_config.get('processed_data_path')) / 'features.csv'
    features_df = pd.read_csv(features_file)
    
    # Load model
    if model_type is None:
        model_type = model_config.get('default_model', 'xgboost')
    
    model_path = Path(data_config.get('model_path')) / f'{model_type}_model.pkl'
    
    trainer = ModelTrainer(model_config, training_config)
    trainer.load_model(str(model_path))
    
    # Initialize predictor
    predictor = RacePredictor(trainer)
    
    # Get session key if not provided
    if session_key is None:
        # Use the most recent session
        session_key = features_df['session_key'].max()
    
    logger.info(f"Predicting for session {session_key}")
    
    # Get race information
    race_info = predictor.get_session_info(session_key)
    
    # Display race details
    logger.info("\n" + "=" * 60)
    if 'year' in race_info and race_info['year']:
        race_title = f"{race_info['year']} {race_info['race_name']}"
    else:
        race_title = race_info.get('race_name', f"Session {session_key}")
    
    logger.info(f"🏁 PREDICTIONS FOR: {race_title}")
    if 'location' in race_info:
        logger.info(f"📍 Location: {race_info['location']}")
    if 'date' in race_info and race_info['date']:
        logger.info(f"📅 Date: {race_info['date'][:10]}")
    logger.info("=" * 60)
    
    # Make predictions
    predictions = predictor.predict_race(features_df, session_key)
    
    if predictions.empty:
        logger.warning("No predictions generated")
        return
    
    # Print predictions
    logger.info("\n--- Race Predictions ---")
    for idx, row in predictions.head(10).iterrows():
        driver = row['driver_number']
        pred_pos = row['predicted_position_int']
        
        # Get driver name if available
        driver_str = f"#{driver}"
        if 'full_name' in row and pd.notna(row['full_name']):
            driver_str = f"{row['full_name']} (#{driver})"
        elif 'name_acronym' in row and pd.notna(row['name_acronym']):
            driver_str = f"{row['name_acronym']} (#{driver})"
        
        actual_str = ""
        if 'actual_position' in row and pd.notna(row['actual_position']):
            actual_pos = int(row['actual_position'])
            error = row['error']
            actual_str = f" | Actual: P{actual_pos} (Error: {error:+.1f})"
        
        logger.info(f"  P{pred_pos}: {driver_str}{actual_str}")
    
    # Podium prediction
    podium = predictor.predict_podium(features_df, session_key)
    logger.info("\n--- Podium Prediction ---")
    
    def format_podium_driver(driver_info):
        if driver_info is None:
            return "N/A"
        if 'name' in driver_info:
            return f"{driver_info['name']} (#{driver_info['driver_number']})"
        return f"Driver #{driver_info['driver_number']}"
    
    logger.info(f"  🥇 Winner: {format_podium_driver(podium['winner'])}")
    logger.info(f"  🥈 2nd: {format_podium_driver(podium['second'])}")
    logger.info(f"  🥉 3rd: {format_podium_driver(podium['third'])}")
    
    # Comparison if actual results available
    if 'actual_position' in predictions.columns:
        comparison = predictor.compare_predictions(features_df, session_key)
        logger.info("\nPrediction accuracy metrics logged above")
    
    logger.info("=== Predictions Complete ===\n")


def run_full_pipeline(config):
    """Run the complete pipeline"""
    logger.info("=" * 60)
    logger.info("RUNNING FULL F1 PREDICTION PIPELINE")
    logger.info("=" * 60 + "\n")
    
    collect_data(config)
    preprocess_data(config)
    engineer_features(config)
    train_model(config)
    make_predictions(config)
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='F1 Race Outcome Predictor')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['collect', 'preprocess', 'features', 'train', 'predict', 'full'],
                       help='Pipeline mode to run')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, default=None,
                       choices=['random_forest', 'xgboost'],
                       help='Model type to use')
    parser.add_argument('--session', type=int, default=None,
                       help='Session key for prediction')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(config)
    
    # Ensure directories exist
    ensure_directories(config)
    
    logger.info(f"Starting F1 Predictor in '{args.mode}' mode")
    
    # Run appropriate mode
    if args.mode == 'collect':
        collect_data(config)
    elif args.mode == 'preprocess':
        preprocess_data(config)
    elif args.mode == 'features':
        engineer_features(config)
    elif args.mode == 'train':
        train_model(config, model_type=args.model)
    elif args.mode == 'predict':
        make_predictions(config, session_key=args.session, model_type=args.model)
    elif args.mode == 'full':
        run_full_pipeline(config)
    
    logger.info("Done!")


if __name__ == '__main__':
    main()
