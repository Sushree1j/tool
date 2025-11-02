#!/usr/bin/env python3
"""
Check Requirements Script
Verifies that all required packages are installed and working
"""

import sys
from importlib import import_module

# Define required packages
REQUIRED_PACKAGES = {
    'core': [
        ('pandas', 'Data processing'),
        ('numpy', 'Numerical operations'),
        ('yfinance', 'Stock data collection'),
    ],
    'ml': [
        ('sklearn', 'Machine learning (scikit-learn)'),
        ('xgboost', 'XGBoost model'),
    ],
    'visualization': [
        ('matplotlib', 'Plotting'),
        ('seaborn', 'Statistical visualization'),
    ],
    'technical': [
        ('ta', 'Technical analysis indicators'),
    ],
    'utilities': [
        ('tqdm', 'Progress bars'),
        ('yaml', 'Configuration (pyyaml)'),
        ('joblib', 'Model serialization'),
    ],
}

OPTIONAL_PACKAGES = {
    'deep_learning': [
        ('tensorflow', 'TensorFlow (for LSTM)'),
        ('keras', 'Keras (for LSTM)'),
    ],
    'advanced_viz': [
        ('plotly', 'Interactive plots'),
    ],
    'jupyter': [
        ('jupyter', 'Jupyter notebook'),
        ('notebook', 'Notebook server'),
    ],
}


def check_package(package_name: str, display_name: str) -> bool:
    """
    Check if a package can be imported
    
    Args:
        package_name (str): Package import name
        display_name (str): Human-readable package name
        
    Returns:
        bool: True if package is available
    """
    try:
        module = import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"  ✓ {display_name:<40} (v{version})")
        return True
    except ImportError:
        print(f"  ✗ {display_name:<40} (NOT INSTALLED)")
        return False


def main():
    """Main function"""
    print("=" * 80)
    print("STOCK PREDICTION TOOL - REQUIREMENTS CHECK")
    print("=" * 80)
    
    all_ok = True
    missing_core = []
    missing_optional = []
    
    # Check required packages
    print("\n📦 Required Packages:")
    print("-" * 80)
    
    for category, packages in REQUIRED_PACKAGES.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        for package_name, display_name in packages:
            if not check_package(package_name, display_name):
                all_ok = False
                missing_core.append(package_name)
    
    # Check optional packages
    print("\n\n🔧 Optional Packages:")
    print("-" * 80)
    
    for category, packages in OPTIONAL_PACKAGES.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        for package_name, display_name in packages:
            if not check_package(package_name, display_name):
                missing_optional.append(package_name)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if all_ok:
        print("\n✅ All required packages are installed!")
        print("   You can use all core features of the tool.")
    else:
        print("\n❌ Some required packages are missing!")
        print("\nMissing required packages:")
        for pkg in missing_core:
            print(f"  • {pkg}")
        
        print("\nTo install missing packages:")
        print("  pip install -r requirements.txt")
        print("\nOr install individually:")
        for pkg in missing_core:
            print(f"  pip install {pkg}")
    
    if missing_optional:
        print("\n⚠️  Some optional packages are missing:")
        for pkg in missing_optional:
            print(f"  • {pkg}")
        
        print("\nThese packages are optional. Install them for additional features:")
        print("  pip install -r requirements.txt")
    
    # Check Python version
    print("\n" + "=" * 80)
    print("PYTHON VERSION")
    print("=" * 80)
    
    py_version = sys.version_info
    print(f"\nPython version: {py_version.major}.{py_version.minor}.{py_version.micro}")
    
    if py_version.major == 3 and py_version.minor >= 8:
        print("✅ Python version is compatible (3.8+)")
    else:
        print("⚠️  Python 3.8 or higher is recommended")
        print(f"   Your version: {py_version.major}.{py_version.minor}")
    
    # Final message
    print("\n" + "=" * 80)
    
    if all_ok:
        print("\n🎉 You're all set! Run the tool with:")
        print("   python main.py --ticker AAPL --visualize")
        print("\nFor more examples, see:")
        print("   python examples/basic_prediction.py")
        sys.exit(0)
    else:
        print("\n❗ Please install missing packages before running the tool.")
        sys.exit(1)


if __name__ == "__main__":
    main()
