#!/bin/bash

# F1Predict Environment Setup Helper
# Makes it easy to activate the virtual environment

echo "🏎️  F1Predict - Activating Virtual Environment"
echo "================================================"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if virtual environment exists
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python3 -m venv venv"
    exit 1
fi

# Activate virtual environment
source "$SCRIPT_DIR/venv/bin/activate"

# Check if activation was successful
if [ "$VIRTUAL_ENV" != "" ]; then
    echo "✅ Virtual environment activated!"
    echo ""
    echo "Python: $(which python)"
    echo "Version: $(python --version)"
    echo ""
    echo "Quick commands:"
    echo "  python test_installation.py    - Test installation"
    echo "  python main.py --mode collect  - Collect F1 data"
    echo "  python main.py --mode full     - Run full pipeline"
    echo "  jupyter notebook              - Launch notebooks"
    echo "  deactivate                    - Exit virtual environment"
    echo ""
else
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

# Start a new shell with the environment activated
exec $SHELL
