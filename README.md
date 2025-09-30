# Stock Price Prediction Project

A comprehensive data science project for predicting stock prices using machine learning techniques.

## Features

- **Real-time Data Collection**: Fetch historical stock data using yfinance
- **Technical Indicators**: Calculate moving averages, RSI, MACD, Bollinger Bands
- **Multiple ML Models**: 
  - Linear Regression
  - Random Forest
  - LSTM (Deep Learning)
  - XGBoost
- **Visualization**: Interactive charts and prediction plots
- **Model Evaluation**: RMSE, MAE, R² scores

## Project Structure

```
.
├── data/                    # Store downloaded stock data
├── models/                  # Saved trained models
├── notebooks/              # Jupyter notebooks for exploration
│   └── stock_analysis.ipynb
├── src/                    # Source code
│   ├── data_collector.py   # Fetch stock data
│   ├── feature_engineering.py  # Create technical indicators
│   ├── models.py           # ML model implementations
│   ├── visualizer.py       # Plotting functions
│   └── predictor.py        # Main prediction script
├── requirements.txt        # Project dependencies
└── main.py                # Run the complete pipeline

```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Predict stock prices for a specific ticker
python main.py --ticker AAPL --days 30

# Train and evaluate multiple models
python src/predictor.py --ticker TSLA --model all

# Explore data in Jupyter notebook
jupyter notebook notebooks/stock_analysis.ipynb
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
