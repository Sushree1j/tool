# Stock Price Prediction Project

A comprehensive data science project for predicting stock prices using machine learning techniques.

## 🆕 NEW: Google Stock Prediction Notebook with BUY/SELL Signals!

**Ready for paper trading!** Check out our new comprehensive notebook:

📓 **`notebooks/google_stock_prediction.ipynb`**

### What's New:
- ✅ **8 ML Models** (Linear, Ridge, Lasso, Random Forest, XGBoost, Gradient Boosting, Extra Trees, AdaBoost)
- ✅ **Clear BUY/SELL/HOLD recommendations** with confidence scores
- ✅ **10 years of Google (GOOGL) data** for maximum accuracy
- ✅ **40+ technical indicators** (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
- ✅ **Paper trading instructions** with entry/exit prices
- ✅ **Ensemble predictions** combining all models
- ✅ **Professional visualizations** and performance metrics

### Quick Start:
```bash
cd notebooks
jupyter notebook google_stock_prediction.ipynb
# Run all cells and get your BUY/SELL signal!
```

📚 **Full documentation**: See [GOOGLE_STOCK_QUICKSTART.md](GOOGLE_STOCK_QUICKSTART.md) and [notebooks/README.md](notebooks/README.md)

---

## Features

- **Real-time Data Collection**: Fetch historical stock data using yfinance
- **Technical Indicators**: Calculate moving averages, RSI, MACD, Bollinger Bands
- **Multiple ML Models** (8 models available!): 
  - Linear Regression (baseline)
  - Ridge Regression (L2 regularization)
  - Lasso Regression (L1 regularization)
  - Random Forest (ensemble)
  - XGBoost (gradient boosting)
  - Gradient Boosting (optimized)
  - Extra Trees (extremely randomized)
  - AdaBoost (adaptive boosting)
  - LSTM (Deep Learning - optional)
- **Visualization**: Interactive charts and prediction plots
- **Model Evaluation**: RMSE, MAE, R² scores

## Project Structure

```
.
├── data/                    # Store downloaded stock data
├── models/                  # Saved trained models
├── notebooks/              # Jupyter notebooks for exploration
│   ├── google_stock_prediction.ipynb  # 🆕 NEW! Google stock with BUY/SELL signals
│   └── README.md           # Notebook documentation
├── src/                    # Source code
│   ├── data_collector.py   # Fetch stock data
│   ├── feature_engineering.py  # Create technical indicators
│   ├── models.py           # ML model implementations (8 models!)
│   ├── visualizer.py       # Plotting functions
│   └── predictor.py        # Main prediction script
├── requirements.txt        # Project dependencies
├── main.py                # Run the complete pipeline
├── GOOGLE_STOCK_QUICKSTART.md  # 🆕 Quick start guide for new notebook
└── README.md              # This file

```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Use the New Google Stock Prediction Notebook (Recommended!)

```bash
# Navigate to notebooks directory
cd notebooks

# Open the comprehensive Google stock notebook
jupyter notebook google_stock_prediction.ipynb

# Run all cells and get your BUY/SELL signal!
```

See [GOOGLE_STOCK_QUICKSTART.md](GOOGLE_STOCK_QUICKSTART.md) for detailed instructions.

### Option 2: Command Line Interface

```bash
# Predict stock prices for a specific ticker
python main.py --ticker AAPL --days 30

# Train and evaluate multiple models
python src/predictor.py --ticker TSLA --model all
```

## Usage Examples

### Predict Apple Stock Price
```python
python main.py --ticker AAPL --days 30 --model lstm
```

### Compare Multiple Stocks
```python
python main.py --ticker AAPL GOOGL MSFT --days 60
```

## Models Explained

1. **Linear Regression**: Baseline model for trend prediction
2. **Random Forest**: Ensemble method handling non-linear relationships
3. **LSTM**: Deep learning model capturing temporal patterns
4. **XGBoost**: Gradient boosting for high accuracy

## Technical Indicators Used

- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Bollinger Bands
- Volume indicators

## Disclaimer

⚠️ **Important**: This project is for educational purposes only. Stock market predictions are inherently uncertain. Never make investment decisions based solely on these predictions. Always consult with financial advisors.

## Future Enhancements

- [ ] Sentiment analysis from news and social media
- [ ] Real-time prediction dashboard
- [ ] Portfolio optimization
- [ ] Backtesting strategies
- [ ] API for predictions

## License

MIT License
