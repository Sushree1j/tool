# Quick Start Guide

Get started with the Stock Price Prediction Tool in minutes!

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Sushree1j/tool.git
cd tool
```

### Step 2: Install Dependencies

**Option A: Full Installation (includes deep learning)**
```bash
pip install -r requirements.txt
```

**Option B: Minimal Installation (faster, no LSTM)**
```bash
pip install -r requirements-minimal.txt
```

**Option C: Install as Package**
```bash
pip install -e .
```

### Step 3: Verify Installation

```bash
# Check if all packages are installed
python check_requirements.py

# Or quick check
python -c "import yfinance, sklearn, matplotlib; print('✅ Ready to go!')"
```

## First Prediction (30 seconds)

Run your first stock prediction with one command:

```bash
python main.py --ticker AAPL
```

That's it! The tool will:
1. Download Apple stock data (5 years by default)
2. Calculate 30+ technical indicators
3. Train multiple ML models
4. Display prediction results

## Basic Usage Examples

### Example 1: Quick Prediction with Visualization

```bash
python main.py --ticker AAPL --visualize
```

Shows interactive charts of predictions and technical indicators.

### Example 2: Predict a Different Stock

```bash
python main.py --ticker TSLA --period 2y --model xgb
```

Predicts Tesla stock using XGBoost with 2 years of data.

### Example 3: Compare Multiple Stocks

```bash
python main.py --ticker AAPL GOOGL MSFT --period 1y
```

Compares predictions for Apple, Google, and Microsoft.

### Example 4: Save Trained Models

```bash
python main.py --ticker AAPL --model all --save-models
```

Trains all models and saves them for later use.

## Using Example Scripts

The `examples/` directory contains ready-to-run scripts:

### Basic Prediction
```bash
python examples/basic_prediction.py
```

### Compare Stocks
```bash
python examples/compare_stocks.py
```

### Custom Training
```bash
python examples/custom_training.py
```

## Configuration

Customize the tool by editing `config.yaml`:

```yaml
data:
  default_period: '5y'  # Change default data period
  
models:
  random_forest:
    n_estimators: 100   # Adjust model parameters
    max_depth: 15
```

## Command Line Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--ticker` | Stock ticker(s) | AAPL | `--ticker AAPL GOOGL` |
| `--period` | Data period | 5y | `--period 2y` |
| `--model` | Model type | all | `--model xgb` |
| `--visualize` | Show plots | False | `--visualize` |
| `--save-models` | Save models | False | `--save-models` |
| `--config` | Config file | config.yaml | `--config my_config.yaml` |
| `--verbose` | Debug mode | False | `--verbose` |

### Model Types

- `lr` - Linear Regression (fast, simple)
- `rf` - Random Forest (balanced)
- `xgb` - XGBoost (most accurate)
- `lstm` - LSTM Deep Learning (requires TensorFlow)
- `all` - Train all models

### Data Periods

- `1mo`, `3mo`, `6mo` - Short term
- `1y`, `2y`, `5y` - Medium to long term
- `10y`, `max` - Maximum available data

## Python API

### Simple Usage

```python
from src import StockPredictor

# Create and run predictor
predictor = StockPredictor('AAPL')
predictor.fetch_data(period='2y')
predictor.engineer_features()
predictor.train_xgboost()
predictor.print_summary()
```

### Advanced Usage

```python
from src import StockDataCollector, FeatureEngineer, XGBoostModel

# 1. Collect data
collector = StockDataCollector()
df = collector.fetch_stock_data('AAPL', period='3y')

# 2. Engineer features
engineer = FeatureEngineer(df)
engineer.add_all_features()
features_df = engineer.get_feature_dataframe()

# 3. Train custom model
model = XGBoostModel(n_estimators=200, learning_rate=0.05)
X_train, X_test, y_train, y_test = model.prepare_data(
    features_df,
    engineer.get_feature_names()
)
model.train(X_train, y_train)
predictions = model.predict(X_test)
metrics = model.evaluate(y_test, predictions)

print(f"R² Score: {metrics['R2']:.4f}")
```

## Jupyter Notebook

Explore interactively with Jupyter:

```bash
jupyter notebook notebooks/stock_analysis.ipynb
```

The notebook includes:
- Data exploration
- Feature analysis
- Model comparisons
- Interactive visualizations

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError`
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt
```

### No Data for Ticker

**Problem**: "No data found for ticker"
```bash
# Solution: Verify ticker symbol
# Check on Yahoo Finance: https://finance.yahoo.com/
# Use correct format (e.g., AAPL not APPLE)
```

### Slow Performance

**Solutions**:
1. Use shorter period: `--period 1y`
2. Use faster model: `--model rf` or `--model xgb`
3. Skip LSTM: Avoid `--model all` or `--model lstm`
4. Use minimal installation: `pip install -r requirements-minimal.txt`

### Visualization Not Working

**Problem**: Plots not showing
```bash
# Solution 1: Install visualization packages
pip install matplotlib seaborn

# Solution 2: Run without visualization
python main.py --ticker AAPL  # No --visualize flag
```

## Tips for Best Results

1. **Start Small**: Use 1-2 years of data initially
   ```bash
   python main.py --ticker AAPL --period 1y
   ```

2. **Try XGBoost First**: Best accuracy/speed balance
   ```bash
   python main.py --ticker AAPL --model xgb
   ```

3. **Visualize Results**: Always check predictions visually
   ```bash
   python main.py --ticker AAPL --visualize
   ```

4. **Compare Periods**: Test different time ranges
   ```bash
   python main.py --ticker AAPL --period 2y
   python main.py --ticker AAPL --period 5y
   ```

5. **Save Good Models**: Keep trained models for reuse
   ```bash
   python main.py --ticker AAPL --model xgb --save-models
   ```

## Next Steps

1. ✅ **Run your first prediction** (see Example 1 above)
2. 📚 **Read the full README** for detailed documentation
3. 🔬 **Explore examples** in the `examples/` directory
4. 📊 **Try Jupyter notebook** for interactive analysis
5. ⚙️ **Customize settings** in `config.yaml`
6. 🤝 **Contribute** - see CONTRIBUTING.md

## Getting Help

- **Documentation**: See [README.md](README.md)
- **Issues**: [GitHub Issues](https://github.com/Sushree1j/tool/issues)
- **Examples**: Check `examples/` directory

## Disclaimer

⚠️ **Educational Use Only**: This tool is for learning purposes. Never make investment decisions based solely on these predictions. Always consult financial advisors.

---

**Happy Predicting! 📈**
