# Quick Start Guide

## Installation

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Verify Installation**:
```bash
python -c "import yfinance; import sklearn; import tensorflow; print('All dependencies installed!')"
```

## Usage

### 1. Quick Prediction (Command Line)

```bash
# Predict Apple stock with all models
python main.py --ticker AAPL --visualize

# Predict Tesla with XGBoost only
python main.py --ticker TSLA --model xgb

# Multiple stocks comparison
python main.py --ticker AAPL GOOGL MSFT --period 2y
```

### 2. Using Jupyter Notebook

```bash
jupyter notebook notebooks/stock_analysis.ipynb
```

### 3. Python Script

```python
from src.predictor import StockPredictor

# Create predictor
predictor = StockPredictor('AAPL')

# Fetch and prepare data
predictor.fetch_data(period='5y')
predictor.engineer_features()

# Train models
predictor.train_all_models()

# View results
predictor.print_summary()
predictor.visualize_results()
```

## Command Line Options

```
--ticker      Stock ticker(s) (e.g., AAPL, GOOGL)
--period      Data period (1mo, 3mo, 6mo, 1y, 2y, 5y, 10y)
--model       Model type (lr, rf, xgb, lstm, all)
--visualize   Show plots
--save-models Save trained models
```

## Examples

### Example 1: Quick Analysis
```bash
python main.py --ticker AAPL --model xgb --visualize
```

### Example 2: Multiple Stocks
```bash
python main.py --ticker AAPL GOOGL MSFT AMZN --period 2y
```

### Example 3: LSTM Model
```bash
python main.py --ticker TSLA --model lstm --period 5y --visualize
```

### Example 4: Save Models
```bash
python main.py --ticker AAPL --model all --save-models
```

## Project Structure

```
.
├── main.py                 # Main entry point
├── requirements.txt        # Dependencies
├── src/                    # Source code
│   ├── data_collector.py   # Data fetching
│   ├── feature_engineering.py  # Feature creation
│   ├── models.py           # ML models
│   ├── visualizer.py       # Plotting
│   └── predictor.py        # Main logic
├── notebooks/              # Jupyter notebooks
│   └── stock_analysis.ipynb
├── data/                   # Downloaded data (created automatically)
└── models/                 # Saved models (created automatically)
```

## Tips

1. **Start Small**: Begin with 1-2 years of data for faster training
2. **Try Different Models**: Each model has strengths for different patterns
3. **Visualize**: Always use `--visualize` to understand the predictions
4. **Technical Indicators**: The system automatically creates 30+ indicators
5. **Save Models**: Use `--save-models` to reuse trained models

## Troubleshooting

**Issue**: `Module not found`
```bash
pip install -r requirements.txt
```

**Issue**: Slow training
- Reduce data period: `--period 1y`
- Use simpler model: `--model lr` or `--model rf`

**Issue**: Poor predictions
- Try different models
- Increase data period
- Check stock volatility

## Next Steps

1. Explore the Jupyter notebook for detailed analysis
2. Experiment with different stocks and time periods
3. Compare model performances
4. Customize technical indicators in `feature_engineering.py`

## Disclaimer

⚠️ **Important**: This is for educational purposes only. Stock predictions are uncertain. Never make investment decisions based solely on these predictions!
