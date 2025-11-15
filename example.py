"""
Example Script: Simple F1 Race Prediction
Demonstrates basic usage of the F1Predict package
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.data import OpenF1Client, DataCollector, DataPreprocessor
from src.features import FeatureEngineer
from src.models import ModelTrainer, RacePredictor
from src.utils import load_config, setup_logging

def main():
    """
    Simple example: Collect data, train model, make predictions
    """
    
    # Load configuration
    print("Loading configuration...")
    config = load_config('config.yaml')
    setup_logging(config)
    
    # Step 1: Collect sample data (2024 season)
    print("\n=== Step 1: Collecting Data ===")
    client = OpenF1Client()
    collector = DataCollector(client, 'data/raw')
    
    # Collect one season
    raw_data = collector.collect_season_data(2024, session_types=['Race'])
    client.close()
    
    print(f"Collected data for {len(raw_data.get('sessions', []))} sessions")
    
    # Step 2: Preprocess
    print("\n=== Step 2: Preprocessing ===")
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess_all(raw_data)
    
    print("Data preprocessing complete")
    
    # Step 3: Feature Engineering
    print("\n=== Step 3: Feature Engineering ===")
    engineer = FeatureEngineer(rolling_windows=[3, 5])
    features_df = engineer.create_feature_matrix(processed_data)
    
    print(f"Created {features_df.shape[1]} features for {features_df.shape[0]} samples")
    
    # Step 4: Train Model
    print("\n=== Step 4: Training Model ===")
    trainer = ModelTrainer(
        model_config=config['models'],
        training_config=config['training']
    )
    
    results = trainer.train(features_df, model_type='xgboost')
    
    print(f"\nModel Performance:")
    print(f"  Training MAE: {results['train_metrics']['mae']:.3f}")
    print(f"  Test MAE: {results['test_metrics']['mae']:.3f}")
    print(f"  Test Top-3 Accuracy: {results['test_metrics']['top3_accuracy']:.1%}")
    
    # Step 5: Make Predictions
    print("\n=== Step 5: Making Predictions ===")
    predictor = RacePredictor(trainer)
    
    # Get most recent session
    latest_session = features_df['session_key'].max()
    
    # Predict podium
    podium = predictor.predict_podium(features_df, latest_session)
    
    print(f"\nPredicted Podium for Session {latest_session}:")
    print(f"  🥇 Winner: Driver #{podium['winner']}")
    print(f"  🥈 Second: Driver #{podium['second']}")
    print(f"  🥉 Third: Driver #{podium['third']}")
    
    # Full race prediction
    predictions = predictor.predict_race(features_df, latest_session)
    
    print(f"\nTop 10 Predicted Finishers:")
    for idx, row in predictions.head(10).iterrows():
        print(f"  P{row['predicted_position_int']:2d}: Driver #{int(row['driver_number'])}")
    
    print("\n✅ Example complete!")
    print("\nNext steps:")
    print("  - Collect more seasons for better predictions")
    print("  - Tune hyperparameters in config.yaml")
    print("  - Explore data in Jupyter notebooks")
    print("  - Add custom features for your domain knowledge")

if __name__ == '__main__':
    main()
