# Examples Directory

This directory contains example scripts demonstrating how to use the Stock Price Prediction Tool.

## Available Examples

### 1. Basic Prediction (`basic_prediction.py`)

The simplest way to use the tool. Demonstrates:
- Creating a predictor
- Fetching data
- Engineering features
- Training a model
- Viewing results

**Run it:**
```bash
python examples/basic_prediction.py
```

**What it does:**
- Predicts Apple (AAPL) stock
- Uses 2 years of historical data
- Trains a Random Forest model
- Displays results in console

**Customization:**
Edit the script to change:
- Ticker symbol (line 19)
- Data period (line 25)
- Model type (line 33)

---

### 2. Compare Stocks (`compare_stocks.py`)

Compare predictions across multiple stocks. Demonstrates:
- Processing multiple tickers
- Collecting results
- Creating comparison tables
- Identifying best predictions

**Run it:**
```bash
python examples/compare_stocks.py
```

**What it does:**
- Compares AAPL, GOOGL, MSFT, TSLA
- Uses XGBoost (fast and accurate)
- Shows comparison table
- Identifies best performer

**Customization:**
Edit the `tickers` list (line 62):
```python
tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']  # Add your stocks here
```

---

### 3. Custom Training (`custom_training.py`)

Advanced model customization. Demonstrates:
- Using individual components
- Custom model parameters
- Feature importance analysis
- Manual visualization
- Saving models

**Run it:**
```bash
python examples/custom_training.py
```

**What it does:**
- Trains Random Forest with 200 trees
- Trains XGBoost with slower learning rate
- Shows feature importance
- Compares model performance
- Optionally saves models

**Customization:**
Adjust model parameters:
```python
# Random Forest (lines 43-45)
rf_model = RandomForestModel(
    n_estimators=200,  # More trees
    max_depth=20,      # Deeper trees
)

# XGBoost (lines 65-69)
xgb_model = XGBoostModel(
    n_estimators=150,
    learning_rate=0.05,  # Slower learning
    max_depth=8,
)
```

---

## Creating Your Own Examples

### Template Script

```python
#!/usr/bin/env python3
"""
My Custom Stock Prediction Script
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predictor import StockPredictor

def main():
    """Main function"""
    
    # Your ticker
    ticker = 'YOUR_TICKER_HERE'
    
    # Create predictor
    predictor = StockPredictor(ticker)
    
    # Fetch data
    predictor.fetch_data(period='2y')
    
    # Engineer features
    predictor.engineer_features()
    
    # Train model (choose one)
    # predictor.train_linear_regression()
    # predictor.train_random_forest()
    predictor.train_xgboost()  # Recommended
    # predictor.train_lstm()
    # predictor.train_all_models()
    
    # Show results
    predictor.print_summary()
    
    # Optional: Visualize
    # predictor.visualize_results()

if __name__ == "__main__":
    main()
```

### Tips for Custom Scripts

1. **Import from src**: Always add the parent directory to path
2. **Error handling**: Wrap in try-except for robustness
3. **Progress feedback**: Print messages to show progress
4. **Save results**: Consider saving models or predictions
5. **Documentation**: Add docstrings and comments

### Common Customizations

**Different time periods:**
```python
predictor.fetch_data(period='1y')   # 1 year
predictor.fetch_data(period='5y')   # 5 years
predictor.fetch_data(period='max')  # All available
```

**Model selection by use case:**
```python
# Fast prototyping
predictor.train_linear_regression()

# Balanced performance
predictor.train_random_forest()

# Best accuracy
predictor.train_xgboost()

# Time series patterns (slow)
predictor.train_lstm()

# Compare all models
predictor.train_all_models(include_lstm=False)
```

**Save trained models:**
```python
for model_name, result in predictor.results.items():
    model = result['model']
    filename = f"{ticker}_{model_name.lower()}.pkl"
    model.save_model(filename)
    print(f"Saved: {filename}")
```

## Advanced Usage

### Using Individual Components

For more control, use components directly:

```python
from src.data_collector import StockDataCollector
from src.feature_engineering import FeatureEngineer
from src.models import XGBoostModel
from src.visualizer import StockVisualizer

# 1. Collect data
collector = StockDataCollector()
df = collector.fetch_stock_data('AAPL', period='3y')

# 2. Engineer features
engineer = FeatureEngineer(df)
engineer.add_moving_averages()
engineer.add_rsi()
engineer.add_macd()
# ... add other indicators as needed
engineer.add_target_variable()
features_df = engineer.get_feature_dataframe()

# 3. Train model
model = XGBoostModel(n_estimators=150)
X_train, X_test, y_train, y_test = model.prepare_data(
    features_df,
    engineer.get_feature_names()
)
model.train(X_train, y_train)
predictions = model.predict(X_test)

# 4. Evaluate
metrics = model.evaluate(y_test, predictions)
print(f"R² Score: {metrics['R2']:.4f}")
print(f"RMSE: {metrics['RMSE']:.2f}")

# 5. Visualize
visualizer = StockVisualizer()
visualizer.plot_predictions(y_test, predictions, 'AAPL', 'XGBoost')
```

## Need Help?

- **Documentation**: See [README.md](../README.md)
- **Quick Start**: See [QUICKSTART.md](../QUICKSTART.md)
- **Issues**: [GitHub Issues](https://github.com/Sushree1j/tool/issues)

## Contributing

Have a useful example? See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on submitting new examples.

---

**Happy Coding! 🚀**
