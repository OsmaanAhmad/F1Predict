#!/bin/bash
# Quick runner script that always uses the venv Python

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: ./install.sh"
    exit 1
fi

# Run with venv Python
exec "$VENV_PYTHON" "$@"
