"""
F1 Predict - Complete Interactive System

Main menu system for:
1. Fetch data (collect sessions, fetch live data for specific race)
2. Train models (XGBoost)
3. Predict race (XGBoost or Custom Statistical model)
"""

import sys
import subprocess
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get the correct Python interpreter path
PYTHON_CMD = sys.executable  # This will use the same Python that's running this script


def clear_screen():
    """Clear terminal screen"""
    print("\n" * 2)


def print_header():
    """Print system header"""
    print("=" * 80)
    print("🏎️  F1 PREDICT - COMPLETE PREDICTION SYSTEM")
    print("=" * 80)


def gather_sessions():
    """Load all sessions from data/raw/*/sessions.csv"""
    all_sessions = []
    
    for year in [2023, 2024, 2025]:
        sessions_file = Path(f'data/raw/{year}/sessions.csv')
        if sessions_file.exists():
            try:
                df = pd.read_csv(sessions_file)
                df['year'] = year
                all_sessions.append(df)
            except Exception as e:
                logger.warning(f"Error reading {sessions_file}: {e}")
    
    if not all_sessions:
        return pd.DataFrame(), pd.DataFrame()
    
    all_sessions_df = pd.concat(all_sessions, ignore_index=True)
    
    # Group by meeting to get unique races
    meetings = (all_sessions_df.groupby(['year', 'meeting_key', 'location'], as_index=False)
                .agg(
                    date_start=('date_start', 'min'),
                    session_count=('session_key', 'count'),
                    country=('country_name', 'first')
                ))
    meetings = meetings.sort_values(['year', 'date_start'])
    
    return all_sessions_df, meetings


def get_2025_full_calendar():
    """Get complete 2025 F1 calendar including races not yet in sessions.csv"""
    # Load existing sessions
    all_sessions, meetings = gather_sessions()
    
    # Full 2025 calendar with expected dates
    full_calendar = [
        {"location": "Sakhir", "country": "Bahrain", "date": "2025-02-26"},
        {"location": "Melbourne", "country": "Australia", "date": "2025-03-14"},
        {"location": "Shanghai", "country": "China", "date": "2025-03-21"},
        {"location": "Suzuka", "country": "Japan", "date": "2025-04-04"},
        {"location": "Jeddah", "country": "Saudi Arabia", "date": "2025-04-18"},
        {"location": "Miami", "country": "United States", "date": "2025-05-02"},
        {"location": "Imola", "country": "Italy", "date": "2025-05-16"},
        {"location": "Monaco", "country": "Monaco", "date": "2025-05-23"},
        {"location": "Barcelona", "country": "Spain", "date": "2025-05-30"},
        {"location": "Montréal", "country": "Canada", "date": "2025-06-13"},
        {"location": "Spielberg", "country": "Austria", "date": "2025-06-27"},
        {"location": "Silverstone", "country": "United Kingdom", "date": "2025-07-04"},
        {"location": "Spa-Francorchamps", "country": "Belgium", "date": "2025-07-25"},
        {"location": "Budapest", "country": "Hungary", "date": "2025-08-01"},
        {"location": "Zandvoort", "country": "Netherlands", "date": "2025-08-29"},
        {"location": "Monza", "country": "Italy", "date": "2025-09-05"},
        {"location": "Baku", "country": "Azerbaijan", "date": "2025-09-19"},
        {"location": "Marina Bay", "country": "Singapore", "date": "2025-10-03"},
        {"location": "Austin", "country": "United States", "date": "2025-10-17"},
        {"location": "Mexico City", "country": "Mexico", "date": "2025-10-24"},
        {"location": "São Paulo", "country": "Brazil", "date": "2025-11-07"},
        {"location": "Las Vegas", "country": "United States", "date": "2025-11-22"},
        {"location": "Lusail", "country": "Qatar", "date": "2025-11-30"},
        {"location": "Yas Island", "country": "United Arab Emirates", "date": "2025-12-07"},
    ]
    
    # Create dataframe from full calendar
    calendar_df = pd.DataFrame(full_calendar)
    
    # Get 2025 meetings from sessions
    meetings_2025 = meetings[meetings['year'] == 2025].copy()
    
    # Merge with full calendar to add meeting_keys where available
    # Use fuzzy matching on location name
    result_rows = []
    for idx, cal_race in calendar_df.iterrows():
        # Try to find matching race in sessions
        matched = meetings_2025[meetings_2025['location'].str.contains(cal_race['location'].split()[0], case=False, na=False)]
        
        if not matched.empty:
            # Use data from sessions.csv
            race = matched.iloc[0]
            result_rows.append({
                'location': cal_race['location'],
                'country': cal_race['country'],
                'date': cal_race['date'],
                'meeting_key': race['meeting_key'],
                'has_data': True
            })
        else:
            # Race not yet in sessions.csv
            result_rows.append({
                'location': cal_race['location'],
                'country': cal_race['country'],
                'date': cal_race['date'],
                'meeting_key': None,
                'has_data': False
            })
    
    return pd.DataFrame(result_rows), all_sessions


def run_command(cmd, description):
    """Run a shell command with error handling"""
    logger.info(f"\n{'='*80}")
    logger.info(f"Running: {description}")
    logger.info(f"Command: {cmd}")
    logger.info(f"{'='*80}\n")
    
    result = subprocess.run(cmd, shell=True)
    return result.returncode


def menu_fetch_data():
    """Data fetching menu"""
    clear_screen()
    print_header()
    print("\n📥 DATA FETCHING")
    print("-" * 80)
    print("1. Collect all sessions metadata (updates sessions.csv)")
    print("2. Fetch live data for specific race")
    print("3. Back to main menu")
    
    choice = input("\nSelect option [1-3]: ").strip()
    
    if choice == '1':
        logger.info("\n🔄 Collecting sessions metadata...")
        rc = run_command(f"{PYTHON_CMD} main.py --mode collect", "Collect Sessions")
        if rc == 0:
            logger.info("\n✅ Sessions metadata collected successfully!")
        else:
            logger.error("\n❌ Failed to collect sessions")
        input("\nPress Enter to continue...")
        return
    
    elif choice == '2':
        # Show available meetings with full calendar
        calendar_2025, all_sessions = get_2025_full_calendar()
        
        if calendar_2025.empty:
            logger.error("No calendar data available")
            input("\nPress Enter to continue...")
            return
        
        print("\n📋 2025 F1 Race Calendar:")
        print("-" * 80)
        for idx, row in calendar_2025.iterrows():
            location = row['location']
            country = row['country']
            date = row['date']
            has_data = row['has_data']
            status = "✓" if has_data else "○"
            
            print(f"[{idx:2d}] {status} {location:25s} ({country:20s}) - {date}")
        
        print("\n✓ = Data available | ○ = Not yet available")
        
        selection = input("\nEnter meeting number to fetch data: ").strip()
        
        try:
            idx = int(selection)
            selected = calendar_2025.iloc[idx]
            location = selected['location']
            country = selected['country']
            has_data = selected['has_data']
            meeting_key = selected['meeting_key']
            
            if not has_data or pd.isna(meeting_key):
                logger.error(f"\n❌ {location} data not yet available in API!")
                logger.info("Run option 1 first to update sessions metadata")
                input("\nPress Enter to continue...")
                return
            
            meeting_key = int(meeting_key)
            
            print(f"\n📡 Fetching data for 2025 {location} (meeting_key: {meeting_key})...")
            
            # Get all session keys for this meeting
            meeting_sessions = all_sessions[all_sessions['meeting_key'] == meeting_key]
            
            if meeting_sessions.empty:
                logger.error("No sessions found for this meeting")
                input("\nPress Enter to continue...")
                return
            
            # Fetch data for each session
            for _, session in meeting_sessions.iterrows():
                session_key = int(session['session_key'])
                session_name = session['session_name']
                
                logger.info(f"\n📊 Fetching {session_name} (session {session_key})...")
                cmd = f"{PYTHON_CMD} fetch_live_data.py --session {session_key} --save data/raw/2025"
                subprocess.run(cmd, shell=True)
            
            logger.info(f"\n✅ Data fetched for {location}!")
            
        except (ValueError, IndexError):
            logger.error("Invalid selection")
        
        input("\nPress Enter to continue...")
        return
    
    elif choice == '3':
        return
    
    else:
        logger.warning("Invalid option")
        input("\nPress Enter to continue...")


def menu_train_models():
    """Model training menu"""
    clear_screen()
    print_header()
    print("\n🤖 MODEL TRAINING")
    print("-" * 80)
    print("1. Train XGBoost model (uses 2023-2024 historical data)")
    print("2. Check if model exists")
    print("3. Back to main menu")
    
    choice = input("\nSelect option [1-3]: ").strip()
    
    if choice == '1':
        logger.info("\n🔄 Training XGBoost model...")
        logger.info("This will:")
        logger.info("  1. Load processed features")
        logger.info("  2. Train XGBoost with cross-validation")
        logger.info("  3. Save model to data/models/xgboost_model.pkl")
        
        confirm = input("\nProceed with training? [y/N]: ").strip().lower()
        if confirm == 'y':
            rc = run_command(f"{PYTHON_CMD} main.py --mode train", "Train XGBoost Model")
            if rc == 0:
                logger.info("\n✅ Model trained successfully!")
            else:
                logger.error("\n❌ Training failed")
        else:
            logger.info("Training cancelled")
        
        input("\nPress Enter to continue...")
        return
    
    elif choice == '2':
        model_path = Path('data/models/xgboost_model.pkl')
        if model_path.exists():
            size = model_path.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"\n✅ XGBoost model found: {model_path}")
            logger.info(f"   Size: {size:.2f} MB")
        else:
            logger.warning(f"\n⚠️  XGBoost model NOT found at {model_path}")
            logger.info("   Run option 1 to train the model")
        
        input("\nPress Enter to continue...")
        return
    
    elif choice == '3':
        return
    
    else:
        logger.warning("Invalid option")
        input("\nPress Enter to continue...")


def find_race_session_key(all_sessions, meeting_key):
    """Find the race session key for a meeting"""
    meeting_sessions = all_sessions[all_sessions['meeting_key'] == meeting_key]
    
    # Try to find Race session
    race = meeting_sessions[meeting_sessions['session_name'].str.contains('Race', case=False, na=False)]
    if not race.empty:
        # Prefer main race over sprint
        main_race = race[~race['session_name'].str.contains('Sprint', case=False, na=False)]
        if not main_race.empty:
            return int(main_race.iloc[0]['session_key'])
        return int(race.iloc[0]['session_key'])
    
    # Fallback to last session
    if not meeting_sessions.empty:
        return int(meeting_sessions.iloc[-1]['session_key'])
    
    return None


def menu_predict():
    """Prediction menu"""
    clear_screen()
    print_header()
    print("\n🔮 RACE PREDICTION")
    print("-" * 80)
    print("1. Predict with XGBoost (ML model using historical data)")
    print("2. Predict with Custom Statistical Model (2025 season data)")
    print("3. Back to main menu")
    
    model_choice = input("\nSelect prediction model [1-3]: ").strip()
    
    if model_choice == '3':
        return
    
    if model_choice not in ['1', '2']:
        logger.warning("Invalid option")
        input("\nPress Enter to continue...")
        return
    
    # Get full 2025 calendar
    calendar_2025, all_sessions = get_2025_full_calendar()
    
    if calendar_2025.empty:
        logger.error("No calendar data available")
        input("\nPress Enter to continue...")
        return
    
    # Show available meetings
    print("\n📋 2025 F1 Race Calendar:")
    print("-" * 80)
    for idx, row in calendar_2025.iterrows():
        location = row['location']
        country = row['country']
        date = row['date']
        has_data = row['has_data']
        status = "✓" if has_data else "○"
        
        print(f"[{idx:2d}] {status} {location:25s} ({country:20s}) - {date}")
    
    print("\n✓ = Data available | ○ = Not yet available")
    
    selection = input("\nEnter meeting number to predict: ").strip()
    
    try:
        idx = int(selection)
        selected = calendar_2025.iloc[idx]
        location = selected['location']
        country = selected['country']
        has_data = selected['has_data']
        meeting_key = selected['meeting_key']
        
        if not has_data or pd.isna(meeting_key):
            logger.error(f"\n❌ No data available for {location} yet!")
            logger.info("Please fetch the data first (Main Menu -> Option 1 -> Collect sessions)")
            input("\nPress Enter to continue...")
            return
        
        meeting_key = int(meeting_key)
        print(f"\n🏁 Predicting: 2025 {location}, {country} (meeting_key: {meeting_key})")
        
        if model_choice == '1':
            # XGBoost prediction - check if features exist first
            model_path = Path('data/models/xgboost_model.pkl')
            if not model_path.exists():
                logger.error("\n❌ XGBoost model not found!")
                logger.info("Please train the model first (Main Menu -> Option 2)")
                input("\nPress Enter to continue...")
                return
            
            # Find race session
            race_session_key = find_race_session_key(all_sessions, meeting_key)
            if race_session_key is None:
                logger.error("No race session found for this meeting")
                input("\nPress Enter to continue...")
                return
            
            # Check if features exist for this session
            features_file = Path('data/processed/features.csv')
            if features_file.exists():
                features_df = pd.read_csv(features_file)
                session_features = features_df[features_df['session_key'] == race_session_key]
                
                if session_features.empty:
                    logger.error(f"\n❌ No processed features found for this race!")
                    logger.info("\n💡 XGBoost requires processed features from completed races.")
                    logger.info("   For future/ongoing races, use option 2 (Custom Statistical Model) instead.")
                    logger.info("\n   Or if race is complete:")
                    logger.info("   1. Fetch race data (Main Menu -> Option 1)")
                    logger.info("   2. Process data: python main.py --mode process")
                    logger.info("   3. Try prediction again")
                    input("\nPress Enter to continue...")
                    return
            
            logger.info(f"Using race session key: {race_session_key}")
            
            description = f"XGBoost Prediction - {location}"
            cmd = f"{PYTHON_CMD} main.py --mode predict --session {race_session_key}"
            returncode = run_command(cmd, description)
        
        else:  # model_choice == '2'
            # Custom Statistical Model prediction
            # Custom Statistical Model
            # Check if we have specific predictor for this location
            location_lower = location.lower().replace(' ', '_')
            
            # Map locations to predictor scripts
            if 'austin' in location_lower:
                script = 'predict_austin_simple.py'
            elif 'mexico' in location_lower:
                script = 'predict_mexico_2025.py'
            else:
                # Use generic predict_future_race.py
                script = 'predict_future_race.py'
            
            script_path = Path(script)
            if script_path.exists():
                if script == 'predict_future_race.py':
                    cmd = f"{PYTHON_CMD} {script} --meeting {meeting_key} --location \"{location}\""
                else:
                    cmd = f"{PYTHON_CMD} {script}"
                
                run_command(cmd, f"Custom Statistical Prediction - {location}")
            else:
                logger.error(f"\n❌ Prediction script not found: {script}")
                logger.info("Available scripts: predict_austin_simple.py, predict_mexico_2025.py")
        
    except (ValueError, IndexError) as e:
        logger.error(f"Invalid selection: {e}")
    
    input("\nPress Enter to continue...")


def main_menu():
    """Main menu loop"""
    while True:
        clear_screen()
        print_header()
        print("\n📋 MAIN MENU")
        print("-" * 80)
        print("1. 📥 Fetch Data (collect sessions, download race data)")
        print("2. 🤖 Train Models (train XGBoost)")
        print("3. 🔮 Predict Race (choose model and race)")
        print("4. 🚪 Exit")
        print("-" * 80)
        
        choice = input("\nSelect option [1-4]: ").strip()
        
        if choice == '1':
            menu_fetch_data()
        elif choice == '2':
            menu_train_models()
        elif choice == '3':
            menu_predict()
        elif choice == '4':
            print("\n👋 Goodbye! Good luck with your predictions! 🏎️💨\n")
            sys.exit(0)
        else:
            logger.warning("Invalid option. Please select 1-4.")
            input("\nPress Enter to continue...")


if __name__ == '__main__':
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!\n")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
