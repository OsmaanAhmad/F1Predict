#!/usr/bin/env python
"""
Quick test script to verify F1Predict installation
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from src.data import OpenF1Client, DataCollector, DataPreprocessor
        print("  ✅ Data modules")
    except Exception as e:
        print(f"  ❌ Data modules: {e}")
        return False
    
    try:
        from src.features import FeatureEngineer
        print("  ✅ Features module")
    except Exception as e:
        print(f"  ❌ Features module: {e}")
        return False
    
    try:
        from src.models import ModelTrainer, RacePredictor
        print("  ✅ Models modules")
    except Exception as e:
        print(f"  ❌ Models modules: {e}")
        return False
    
    try:
        from src.utils import load_config
        print("  ✅ Utils module")
    except Exception as e:
        print(f"  ❌ Utils module: {e}")
        return False
    
    return True

def test_dependencies():
    """Test that all dependencies are installed"""
    print("\nTesting dependencies...")
    
    dependencies = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'xgboost': 'xgboost',
        'requests': 'requests',
        'yaml': 'pyyaml',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
    }
    
    all_ok = True
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError as e:
            print(f"  ❌ {name}: {e}")
            all_ok = False
    
    return all_ok

def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        from src.utils import load_config
        config = load_config('config.yaml')
        print(f"  ✅ Configuration loaded")
        print(f"     - Seasons: {config.get('data', {}).get('seasons')}")
        print(f"     - Default model: {config.get('models', {}).get('default_model')}")
        return True
    except Exception as e:
        print(f"  ❌ Configuration: {e}")
        return False

def test_api_client():
    """Test API client initialization"""
    print("\nTesting API client...")
    
    try:
        from src.data import OpenF1Client
        client = OpenF1Client()
        print(f"  ✅ API client initialized")
        print(f"     - Base URL: {client.base_url}")
        client.close()
        return True
    except Exception as e:
        print(f"  ❌ API client: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("F1PREDICT INSTALLATION TEST")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Dependencies", test_dependencies),
        ("Configuration", test_config),
        ("API Client", test_api_client),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            results.append(test_func())
        except Exception as e:
            print(f"\n❌ {name} test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    if all(results):
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYour F1Predict installation is ready to use!")
        print("\nNext steps:")
        print("  1. Run: python main.py --mode collect")
        print("  2. Run: python main.py --mode full")
        print("  3. Explore: jupyter notebook notebooks/01_data_exploration.ipynb")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 60)
        print("\nPlease check the errors above and:")
        print("  1. Make sure all dependencies are installed")
        print("  2. Check that you're in the virtual environment")
        print("  3. See QUICKSTART.md for installation instructions")
        return 1

if __name__ == '__main__':
    sys.exit(main())
