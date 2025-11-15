# 🏎️ F1 Race Prediction System

Machine learning system for predicting Formula 1 race outcomes using XGBoost and custom statistical models.

## 🚀 Quick Start

### macOS/Linux

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/F1Predict.git
cd F1Predict

# 2. Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Run the interactive system
./venv/bin/python3 f1_predict_system.py
```

### Windows (PowerShell)

```powershell
# 1. Clone the repository
git clone https://github.com/yourusername/F1Predict.git
cd F1Predict

# 2. Set PowerShell execution policy (if needed)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 3. Create virtual environment and install dependencies
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 4. Run the interactive system
python f1_predict_system.py
```

That's it! The interactive menu will guide you through everything.

## 📋 What You Can Do

The system provides an easy-to-use menu:

1. **📥 Fetch Data** - Download F1 race data from OpenF1 API
2. **🤖 Train Models** - Train the XGBoost prediction model
3. **🔮 Predict Race** - Predict any 2025 F1 race outcome
4. **🚪 Exit**

## 💡 Example Usage

### First Time Setup
```bash
./venv/bin/python3 f1_predict_system.py

# Then in the menu:
1. Select "Fetch Data" → "Collect sessions"
2. Select "Train Models" → Confirm training
3. Select "Predict Race" → Choose your race
```

### Features

- **2 Prediction Models:**
  - **XGBoost**: ML model trained on historical data
  - **Statistical**: Custom weighted formula for live races

- **Complete 2025 Calendar:** All 24 races from Bahrain to Abu Dhabi

- **Real-time Data:** Fetches live telemetry from OpenF1 API

## 📊 What It Analyzes

- 22,585+ laps across multiple seasons
- Driver performance metrics
- Weather conditions
- Pit stop strategies
- Track-specific performance
- 31+ engineered features per driver

## 📁 Project Structure

```
F1Predict/
├── f1_predict_system.py    # 👈 Main interactive system
├── main.py                  # Core pipeline
├── requirements.txt         # Dependencies
├── config.yaml             # Configuration
└── src/                    # Source code
    ├── data/               # Data collection
    ├── features/           # Feature engineering
    ├── models/             # ML models
    └── utils/              # Utilities
```

## 🐛 Troubleshooting

**"No module named 'xyz'"**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

**"No features found"**
- XGBoost needs processed data
- Use Statistical Model for future races instead
- Or run: `python main.py --mode process`

**"No sessions found"**
```bash
python main.py --mode collect
```

## 📖 Documentation
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Project organization

## 🎯 Use Cases

- Race strategy planning
- Fantasy F1 team selection
- Performance analysis
- Research & analytics


## 🙏 Credits

- **OpenF1 API** - Free F1 data (no API key needed)
- **XGBoost** - Machine learning framework
- **Python Libraries** - pandas, scikit-learn, numpy

---

**⚠️ Disclaimer**: For educational purposes only. Predictions are probabilistic estimates.

