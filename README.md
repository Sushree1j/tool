# Stock Price Prediction Project

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A comprehensive, user-friendly data science project for predicting stock prices using machine learning techniques. This tool makes stock price prediction accessible to both beginners and experienced developers.

## ✨ Features

- **📊 Real-time Data Collection**: Fetch historical stock data using yfinance
- **🔧 Technical Indicators**: Automatically calculate 30+ indicators including moving averages, RSI, MACD, Bollinger Bands
- **🤖 Multiple ML Models**: 
  - Linear Regression (baseline)
  - Random Forest (ensemble)
  - XGBoost (gradient boosting)
  - LSTM (deep learning, optional)
- **📈 Rich Visualization**: Interactive charts and prediction plots with matplotlib, seaborn, and plotly
- **📉 Model Evaluation**: Comprehensive metrics including RMSE, MAE, R², and MAPE
- **⚙️ Configuration Support**: Customize settings via YAML configuration file
- **🛠️ Easy to Use**: Simple CLI interface with helpful examples

## 🚀 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sushree1j/tool.git
   cd tool
   ```

2. **Install dependencies**
   ```bash
   # Option 1: Full installation (includes deep learning)
   pip install -r requirements.txt
   
   # Option 2: Minimal installation (faster, no deep learning)
   pip install -r requirements-minimal.txt
   
   # Option 3: Install as package
   pip install -e .
   ```

3. **Verify installation**
   ```bash
   python -c "import yfinance; import sklearn; print('✅ Installation successful!')"
   ```

### Basic Usage

**Predict Apple stock price (simplest command):**
```bash
python main.py --ticker AAPL --model xgb
```

**Predict with visualization:**
```bash
python main.py --ticker AAPL --visualize
```

**Compare multiple stocks:**
```bash
python main.py --ticker AAPL GOOGL MSFT --period 2y
```

**Train all models:**
```bash
python main.py --ticker TSLA --model all --visualize --save-models
```

## 📖 Documentation

### Command Line Options

```
--ticker TICKER [TICKER ...]  Stock ticker symbol(s) (e.g., AAPL, GOOGL)
--period PERIOD               Data period: 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max
--model MODEL                 Model: lr, rf, xgb, lstm, all
--visualize                   Show prediction plots
--save-models                 Save trained models to disk
--days DAYS                   Number of days to predict ahead (default: 30)
```

### Examples

See the `examples/` directory for more detailed examples:

- **basic_prediction.py**: Simple prediction example
- **compare_stocks.py**: Compare multiple stocks
- **custom_training.py**: Train models with custom parameters

Run an example:
```bash
python examples/basic_prediction.py
```

## 🏗️ Project Structure

```
.
├── src/                      # Source code
│   ├── __init__.py          # Package initialization
│   ├── data_collector.py    # Fetch stock data
│   ├── feature_engineering.py  # Create technical indicators
│   ├── models.py            # ML model implementations
│   ├── predictor.py         # Main prediction orchestrator
│   ├── visualizer.py        # Plotting functions
│   └── config.py            # Configuration management
├── examples/                # Example scripts
│   ├── basic_prediction.py
│   ├── compare_stocks.py
│   └── custom_training.py
├── notebooks/               # Jupyter notebooks
│   └── stock_analysis.ipynb
├── data/                    # Downloaded stock data (auto-created)
├── models/                  # Saved models (auto-created)
├── config.yaml              # Configuration file
├── requirements.txt         # Full dependencies
├── requirements-minimal.txt # Minimal dependencies
├── setup.py                 # Package setup
├── main.py                  # CLI entry point
├── README.md                # This file
├── QUICKSTART.md           # Quick start guide
└── CONTRIBUTING.md         # Contribution guidelines
```

## 🔧 Configuration

Customize the tool by editing `config.yaml`:

```yaml
data:
  directory: 'data'
  default_period: '5y'

models:
  random_forest:
    n_estimators: 100
    max_depth: 15
  
  xgboost:
    learning_rate: 0.1
    max_depth: 6

visualization:
  save_plots: false
  dpi: 300
```

## 🐍 Python API Usage

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

### Using Individual Components

```python
from src.data_collector import StockDataCollector
from src.feature_engineering import FeatureEngineer
from src.models import XGBoostModel

# Collect data
collector = StockDataCollector()
df = collector.fetch_stock_data('AAPL', period='2y')

# Engineer features
engineer = FeatureEngineer(df)
engineer.add_all_features()
features_df = engineer.get_feature_dataframe()

# Train model
model = XGBoostModel(n_estimators=150)
X_train, X_test, y_train, y_test = model.prepare_data(
    features_df, 
    engineer.get_feature_names()
)
model.train(X_train, y_train)
predictions = model.predict(X_test)
```

## 📊 Technical Indicators Used

- **Moving Averages**: SMA (5, 10, 20, 50, 200), EMA (5, 10, 20, 50, 200)
- **Momentum**: RSI (Relative Strength Index)
- **Trend**: MACD (Moving Average Convergence Divergence)
- **Volatility**: Bollinger Bands, ATR (Average True Range)
- **Volume**: OBV (On-Balance Volume)
- **Price Changes**: 1-day, 5-day, 20-day percentage changes

## 🤖 Models Explained

| Model | Best For | Speed | Accuracy | Notes |
|-------|----------|-------|----------|-------|
| **Linear Regression** | Baseline, trending markets | ⚡⚡⚡ | ⭐⭐ | Simple, interpretable |
| **Random Forest** | General purpose, non-linear patterns | ⚡⚡ | ⭐⭐⭐⭐ | Good balance, feature importance |
| **XGBoost** | High accuracy, complex patterns | ⚡⚡ | ⭐⭐⭐⭐⭐ | Best overall performance |
| **LSTM** | Time series, sequential patterns | ⚡ | ⭐⭐⭐⭐ | Requires more data, TensorFlow |

## 🎯 Performance Tips

1. **Start Small**: Begin with 1-2 years of data for faster training
2. **Try Different Models**: Each model has strengths for different patterns
3. **Visualize**: Always use `--visualize` to understand predictions
4. **Compare Periods**: Test with different time periods (1y, 2y, 5y)
5. **Use XGBoost**: Generally provides best accuracy/speed balance
6. **Save Models**: Use `--save-models` to reuse trained models

## 🐛 Troubleshooting

### Installation Issues

**Problem**: `ModuleNotFoundError` for a package
```bash
# Solution: Install missing package
pip install package-name

# Or reinstall all dependencies
pip install -r requirements.txt --force-reinstall
```

**Problem**: TensorFlow/Keras not installing
```bash
# Solution: Use minimal requirements (no deep learning)
pip install -r requirements-minimal.txt
```

### Runtime Issues

**Problem**: No data found for ticker
```bash
# Solution: Verify ticker symbol is correct
# Try on Yahoo Finance: https://finance.yahoo.com/
# Use correct symbol (e.g., AAPL not APPLE)
```

**Problem**: Slow training
```bash
# Solutions:
# 1. Reduce data period
python main.py --ticker AAPL --period 1y

# 2. Use simpler model
python main.py --ticker AAPL --model lr

# 3. Skip LSTM
python main.py --ticker AAPL --model xgb
```

**Problem**: Poor predictions
```
Solutions:
1. Try different models
2. Increase data period (more historical data)
3. Check stock volatility (highly volatile = harder to predict)
4. Use ensemble (--model all)
```

## 🔒 Security & Privacy

- ✅ No API keys required
- ✅ No personal data collected
- ✅ All data processing is local
- ✅ Uses public Yahoo Finance data
- ✅ No external data sharing

## ⚠️ Disclaimer

**IMPORTANT**: This project is for **educational purposes only**. 

- Stock market predictions are inherently uncertain
- Past performance does not guarantee future results
- **Never make investment decisions based solely on these predictions**
- Always consult with qualified financial advisors
- The authors are not responsible for any financial losses

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Contribution Guide
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add: amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Data provided by [yfinance](https://github.com/ranaroussi/yfinance)
- Technical indicators from [ta](https://github.com/bukosabino/ta)
- ML frameworks: scikit-learn, XGBoost, TensorFlow
- Visualization: matplotlib, seaborn, plotly

## 📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/Sushree1j/tool/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Sushree1j/tool/discussions)

## 🗺️ Roadmap

### Planned Features
- [ ] Real-time prediction dashboard
- [ ] Sentiment analysis from news/social media
- [ ] Portfolio optimization
- [ ] Backtesting strategies
- [ ] REST API for predictions
- [ ] Docker containerization
- [ ] Model hyperparameter tuning
- [ ] More technical indicators
- [ ] Support for cryptocurrency

## ⭐ Star History

If you find this project helpful, please consider giving it a star! ⭐

---

**Made with ❤️ for the data science and finance community**
