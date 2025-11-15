# F1 Race Outcome Predictor - Project Structure

## 📁 Complete Directory Structure

```
F1Predict/
│
├── README.md                       # Project overview and documentation
├── QUICKSTART.md                   # Quick start guide
├── requirements.txt                # Python dependencies
├── config.yaml                     # Configuration settings
├── .gitignore                      # Git ignore rules
├── main.py                         # Main entry point script
├── example.py                      # Example usage script
│
├── data/                           # Data directory
│   ├── raw/                        # Raw data from OpenF1 API
│   │   └── .gitkeep
│   ├── processed/                  # Preprocessed and cleaned data
│   │   └── .gitkeep
│   └── models/                     # Trained model artifacts
│       └── .gitkeep
│
├── src/                            # Source code
│   ├── __init__.py
│   │
│   ├── data/                       # Data collection and preprocessing
│   │   ├── __init__.py
│   │   ├── api_client.py          # OpenF1 API client
│   │   ├── data_collector.py      # Data collection orchestration
│   │   └── preprocessor.py        # Data cleaning and preprocessing
│   │
│   ├── features/                   # Feature engineering
│   │   ├── __init__.py
│   │   └── feature_engineering.py # Feature creation logic
│   │
│   ├── models/                     # Machine learning models
│   │   ├── __init__.py
│   │   ├── trainer.py             # Model training
│   │   └── predictor.py           # Prediction logic
│   │
│   └── utils/                      # Utility functions
│       ├── __init__.py
│       └── helpers.py             # Helper functions
│
├── notebooks/                      # Jupyter notebooks
│   ├── 01_data_exploration.ipynb  # Data exploration and analysis
│   ├── 02_feature_engineering.ipynb (to be created)
│   └── 03_model_training.ipynb    (to be created)
│
├── tests/                          # Unit tests
│   ├── __init__.py
│   └── test_api_client.py         # API client tests
│
└── logs/                           # Log files (created at runtime)
    └── f1predict.log
```

## 🔑 Key Components

### 1. Data Collection (`src/data/`)
- **api_client.py**: Robust OpenF1 API client with retry logic
- **data_collector.py**: Orchestrates data collection for multiple seasons
- **preprocessor.py**: Cleans and prepares raw data

### 2. Feature Engineering (`src/features/`)
- **feature_engineering.py**: Creates predictive features:
  - Lap-based: average pace, consistency, best lap
  - Historical: past performance, rolling averages
  - Team: team performance, teammate comparison
  - Weather: temperature, humidity, rainfall
  - Pit stops: strategy and efficiency metrics

### 3. Machine Learning (`src/models/`)
- **trainer.py**: Model training with:
  - Random Forest Regressor
  - XGBoost Regressor
  - Feature scaling options
  - Cross-validation support
  - Comprehensive evaluation metrics
- **predictor.py**: Makes predictions for:
  - Race finishing positions
  - Podium finishers
  - Winner predictions

### 4. Configuration (`config.yaml`)
Centralized configuration for:
- API settings (timeouts, rate limits)
- Data collection (seasons, session types)
- Feature engineering (rolling windows, feature types)
- Model hyperparameters
- Training parameters

### 5. Entry Points

#### Command Line (`main.py`)
```bash
python main.py --mode [collect|preprocess|features|train|predict|full]
```

#### Python API (`example.py`)
```python
from src.data import OpenF1Client
from src.models import ModelTrainer
# ... use programmatically
```

### 6. Analysis Notebooks
- Interactive exploration in Jupyter
- Visualization of data and results
- Iterative model development

## 🚀 Usage Workflow

```
1. Data Collection
   └─> main.py --mode collect
       └─> Fetches from OpenF1 API
           └─> Saves to data/raw/

2. Preprocessing
   └─> main.py --mode preprocess
       └─> Cleans and validates
           └─> Saves to data/processed/

3. Feature Engineering
   └─> main.py --mode features
       └─> Creates predictive features
           └─> Saves features.csv

4. Model Training
   └─> main.py --mode train
       └─> Trains ML models
           └─> Saves to data/models/

5. Predictions
   └─> main.py --mode predict
       └─> Generates race predictions
           └─> Displays results
```

## 📊 Data Flow

```
OpenF1 API
    ↓
[Raw Data] → data/raw/{year}/
    ↓
[Preprocessing] → Cleaning, Validation
    ↓
[Processed Data] → data/processed/
    ↓
[Feature Engineering] → Create Features
    ↓
[Feature Matrix] → features.csv
    ↓
[Model Training] → Train & Evaluate
    ↓
[Trained Model] → data/models/
    ↓
[Predictions] → Race Outcomes
```

## 🎯 Features Created

### Driver Features
- Average lap time
- Lap time consistency (std dev)
- Best lap time
- Qualifying position
- Historical finishing positions
- Rolling averages (3, 5, 10 races)
- Podium rate
- Win rate

### Team Features
- Team average position
- Team consistency
- Teammate performance

### Race Features
- Weather conditions (temp, humidity)
- Pit stop count
- Pit stop duration
- Weather variability

## 📈 Model Evaluation Metrics

- **MAE** (Mean Absolute Error): Average position error
- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **R² Score**: Explained variance
- **Top-3 Accuracy**: Correctly predict podium finishers
- **Winner Accuracy**: Correctly predict race winner
- **Podium Precision**: Precision of podium predictions

## 🛠️ Technologies Used

- **Python 3.9+**
- **pandas**: Data manipulation
- **NumPy**: Numerical computing
- **scikit-learn**: ML models and preprocessing
- **XGBoost**: Gradient boosting
- **requests**: API calls
- **matplotlib/seaborn**: Visualization
- **PyYAML**: Configuration
- **Jupyter**: Interactive analysis

## 📝 Configuration Options

Edit `config.yaml` to customize:

```yaml
api:
  base_url: "https://api.openf1.org/v1"
  rate_limit_delay: 1.0

data:
  seasons: [2023, 2024]

features:
  rolling_windows: [3, 5, 10]

models:
  default_model: 'xgboost'
  xgboost:
    n_estimators: 300
    max_depth: 8
    learning_rate: 0.05
```

## 🔬 Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest --cov=src tests/
```

## 📚 Documentation

- **README.md**: Comprehensive project overview
- **QUICKSTART.md**: Installation and basic usage
- **Code comments**: Detailed docstrings
- **Type hints**: Function signatures
- **Notebooks**: Interactive tutorials

## 🎓 Learning Resources

- [OpenF1 API Documentation](https://openf1.org/)
- [scikit-learn User Guide](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- Formula 1 domain knowledge for feature ideas

---

**Ready to predict F1 races!** 🏎️🏁
