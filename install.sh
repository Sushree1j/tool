#!/bin/bash
# Installation script for Stock Price Prediction Tool

set -e  # Exit on error

echo "=========================================="
echo "Stock Price Prediction Tool - Installation"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Extract major and minor version
major=$(echo $python_version | cut -d. -f1)
minor=$(echo $python_version | cut -d. -f2)

if [ "$major" -lt 3 ] || ([ "$major" -eq 3 ] && [ "$minor" -lt 8 ]); then
    echo "❌ Error: Python 3.8 or higher is required"
    echo "   Current version: $python_version"
    exit 1
fi

echo "✓ Python version is compatible"
echo ""

# Ask for installation type
echo "Choose installation type:"
echo "  1) Full installation (includes deep learning)"
echo "  2) Minimal installation (faster, no LSTM)"
echo "  3) Development installation (includes dev tools)"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "Installing full dependencies..."
        pip install -r requirements.txt
        ;;
    2)
        echo ""
        echo "Installing minimal dependencies..."
        pip install -r requirements-minimal.txt
        ;;
    3)
        echo ""
        echo "Installing full dependencies + dev tools..."
        pip install -r requirements.txt
        pip install pytest black flake8
        ;;
    *)
        echo "Invalid choice. Installing minimal dependencies..."
        pip install -r requirements-minimal.txt
        ;;
esac

echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
echo ""

# Run check script
python check_requirements.py

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Try a quick prediction:"
echo "     python main.py --ticker AAPL"
echo ""
echo "  2. Run an example:"
echo "     python examples/basic_prediction.py"
echo ""
echo "  3. Read the documentation:"
echo "     cat README.md"
echo ""
echo "For help: python main.py --help"
echo ""
