# 📈 Google Stock Trading System - Advanced ML Prediction

## 🎯 Overview

This Jupyter notebook is an **advanced machine learning system** designed to analyze Google (GOOGL) stock data and provide clear **BUY/SELL/HOLD recommendations** for paper trading practice. It trains 8 different ML models on 10 years of historical data with 31 technical indicators to predict next-day stock prices.

---

## 🤖 What This System Does

1. **Downloads 10 years** of Google stock data from Yahoo Finance
2. **Creates 31 technical indicators** (RSI, MACD, Bollinger Bands, etc.)
3. **Trains 8 ML models** simultaneously for accurate predictions
4. **Generates ensemble predictions** weighted by model performance
5. **Provides clear trading recommendations** with confidence levels
6. **Shows entry/exit prices** and stop-loss suggestions for paper trading

---

## 📊 Machine Learning Models Used

| # | Model | Type | Purpose |
|---|-------|------|---------|
| 1 | **Linear Regression** | Baseline | Simple linear trend prediction |
| 2 | **Random Forest** | Ensemble | 200 decision trees for robust predictions |
| 3 | **XGBoost** | Gradient Boosting | Fast, optimized boosting algorithm |
| 4 | **Gradient Boosting** | Ensemble | Sequential error correction |
| 5 | **LSTM Neural Network** | Deep Learning | Captures time-series patterns |
| 6 | **Support Vector Regression** | Kernel-based | Non-linear pattern recognition |
| 7 | **AdaBoost** | Boosting | Adaptive weight adjustment |
| 8 | **Extra Trees** | Ensemble | Randomized decision trees |

---

## 🔧 Technical Indicators (31 Features)

### Momentum Indicators
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- MACD Signal Line
- Stochastic Oscillator (K & D)

### Volatility Indicators
- Bollinger Bands (Upper, Middle, Lower)
- ATR (Average True Range)
- High-Low Spread

### Trend Indicators
- SMA (Simple Moving Average) - 20 & 50 periods
- EMA (Exponential Moving Average) - 12 & 26 periods
- ADX (Average Directional Index)

### Volume Indicators
- OBV (On-Balance Volume)
- Volume Change

### Price Features
- Daily Returns
- 5-day & 10-day Price Changes
- Lag features (1, 2, 3, 5, 10 days)

---

## 🚀 How to Use

### Prerequisites
```bash
# Required Python packages
pip install yfinance pandas numpy scikit-learn tensorflow xgboost pandas-ta matplotlib seaborn
```

### Running the Notebook

1. **Open the notebook**: `google_trading_system.ipynb`

2. **Run cells sequentially** (Shift + Enter):
   - Cell 1: Import libraries
   - Cell 2: Download stock data
   - Cell 3-4: Exploratory data analysis
   - Cell 5-6: Feature engineering
   - Cell 7: Prepare training data
   - Cells 8-15: Train all 8 models
   - Cell 16-17: Compare model performance
   - Cell 18: Make predictions
   - Cell 19: **Get BUY/SELL/HOLD recommendation** ⭐
   - Cell 20: Visualize predictions

3. **Read the final recommendation** in Step 9 output

### Expected Runtime
- **Total time**: ~2-5 minutes (depending on hardware)
- **Data download**: 10-30 seconds
- **Model training**: 1-3 minutes
- **LSTM training**: Longest (30 epochs)

---

## 📢 Understanding Recommendations

### 🟢 STRONG BUY
- **Conditions**: Predicted gain >2% AND ≥62.5% models agree (5+ bullish)
- **Confidence**: HIGH
- **Action**: Enter position at current price

### 🟢 BUY
- **Conditions**: Predicted gain >0.5% AND ≥50% models agree (4+ bullish)
- **Confidence**: MEDIUM
- **Action**: Consider entering with smaller position

### 🟡 HOLD
- **Conditions**: Prediction <0.5% or low consensus
- **Confidence**: LOW
- **Action**: Wait for clearer signal

### 🔴 SELL
- **Conditions**: Predicted loss >0.5% AND ≥50% models agree (4+ bearish)
- **Confidence**: MEDIUM
- **Action**: Consider exiting or avoid entry

### 🔴 STRONG SELL
- **Conditions**: Predicted loss >2% AND ≥62.5% models agree (5+ bearish)
- **Confidence**: HIGH
- **Action**: Exit positions or short (if experienced)

---

## 🛠️ Troubleshooting Common Errors

### ❌ Error: `NameError: name 'sys' is not defined`
**Fix**: Cell 1 now includes `import sys`

### ❌ Error: `IndexError: index 0 is out of bounds`
**Cause**: Data download failed (empty DataFrame)

**Solutions**:
1. **Check internet connection**
2. **Verify ticker symbol** (GOOGL is correct for Alphabet Inc.)
3. **Try alternative download** (notebook has fallback with explicit dates)
4. **Check Yahoo Finance status** (sometimes API has temporary issues)

**Manual fix**:
```python
# If download fails, try this in Cell 2:
from datetime import datetime, timedelta
end_date = datetime.now()
start_date = end_date - timedelta(days=365*10)
df = yf.download('GOOGL', start=start_date, end=end_date, interval='1d')
```

### ❌ Error: `AttributeError` or `KeyError` in technical indicators
**Fix**: Make sure you have latest pandas-ta version:
```bash
pip install --upgrade pandas-ta
```

### ❌ Error: TensorFlow/Keras warnings
**Note**: These are usually harmless warnings. To suppress:
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

### ⚠️ Empty DataFrame Warning
If you see "❌ Error: No data was downloaded!", the notebook will:
1. Show diagnostic messages
2. Attempt alternative download method with explicit dates
3. Raise clear error if still failing

---

## 📈 How the Prediction System Works

### 1. Data Collection
- **Source**: Yahoo Finance API via `yfinance`
- **Period**: 10 years (maximum history for accuracy)
- **Interval**: Daily (1d)
- **Columns**: Open, High, Low, Close, Volume

### 2. Feature Engineering
- Original data: 5 columns
- After feature engineering: **31 features**
- Target variable: Next day's closing price
- Data cleaning: Removes NaN values from rolling calculations

### 3. Train-Test Split
- **Training set**: 80% (first 8 years)
- **Test set**: 20% (last 2 years)
- **Validation**: Time-series split (no random shuffle)

### 4. Model Training
- Each model trained independently
- Hyperparameters pre-tuned for stock prediction
- Evaluation metrics: R², RMSE, MAE, MAPE

### 5. Ensemble Prediction
- **Weighted average** of all 8 models
- Weights based on R² scores (better models have more influence)
- More robust than single model prediction

### 6. Recommendation Logic
```
IF ensemble_change > 2% AND consensus ≥ 62.5%:
    → STRONG BUY/SELL
ELIF ensemble_change > 0.5% AND consensus ≥ 50%:
    → BUY/SELL
ELSE:
    → HOLD
```

---

## 📊 Model Performance Metrics

### R² Score (Coefficient of Determination)
- **Range**: 0 to 1 (higher = better)
- **Meaning**: How well model explains variance
- **Good**: >0.90 for stock prediction

### RMSE (Root Mean Squared Error)
- **Unit**: Dollars ($)
- **Meaning**: Average prediction error magnitude
- **Good**: <$5 for GOOGL price range

### MAE (Mean Absolute Error)
- **Unit**: Dollars ($)
- **Meaning**: Average absolute prediction error
- **Good**: <$3 for GOOGL

### MAPE (Mean Absolute Percentage Error)
- **Unit**: Percentage (%)
- **Meaning**: Average percentage error
- **Good**: <2% for stock prediction

---

## 🎓 Paper Trading Instructions

### Using TradingView Paper Trading

1. **Sign up** at [TradingView.com](https://www.tradingview.com)
2. **Enable Paper Trading** in account settings
3. **Search for GOOGL** on the platform
4. **Follow notebook recommendations**:
   - Entry price: Current price from notebook
   - Target: Ensemble predicted price
   - Stop-loss: 2% below entry (risk management)

### Recommended Position Sizing
- **High confidence**: 5-10% of paper portfolio
- **Medium confidence**: 2-5% of paper portfolio
- **Low confidence**: Skip or 1% for testing

### Track Your Results
Create a trading journal:
```
Date | Action | Entry Price | Target | Stop Loss | Actual Result | P&L
-----|--------|-------------|--------|-----------|---------------|----
```

---

## 🔄 Customization Options

### Change Stock Ticker
```python
# In Cell 2, modify:
TICKER = 'AAPL'  # Apple
TICKER = 'MSFT'  # Microsoft
TICKER = 'TSLA'  # Tesla
```

### Adjust Data Period
```python
# In Cell 2, modify:
PERIOD = '5y'   # 5 years (faster)
PERIOD = '10y'  # 10 years (more data, slower)
PERIOD = 'max'  # Maximum available
```

### Modify Recommendation Thresholds
```python
# In Cell 19, adjust:
if ensemble_change_pct > 2.0:  # Change 2.0 to 1.5 for more signals
if consensus_pct >= 62.5:      # Change 62.5 to 50 for looser consensus
```

### Add More Models
You can add your own models in Step 6:
```python
# Model 9: Your Custom Model
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
models['KNN'] = knn
predictions['KNN'] = knn_pred
metrics['KNN'] = evaluate_model(y_test, knn_pred, 'KNN')
```

---

## ⚠️ Important Disclaimers

### Educational Purpose Only
- This system is for **LEARNING** and **PAPER TRADING** only
- **NOT financial advice** - do not use real money initially
- Always practice for 3-6 months before considering real trading

### Limitations
- **Past performance ≠ future results**
- Models trained on historical data may not predict black swan events
- Market conditions change (regime shifts)
- News/events can invalidate predictions instantly

### Risk Management
- Never risk more than 1-2% of portfolio per trade
- Always use stop-loss orders
- Diversify across multiple stocks
- Never trade based on emotion
- Review and adjust strategy regularly

---

## 📚 Additional Resources

### Learn More About
- **Technical Analysis**: [Investopedia Technical Analysis](https://www.investopedia.com/technical-analysis-4689657)
- **Machine Learning for Trading**: [Quantopian Lectures](https://www.quantopian.com/lectures)
- **Paper Trading**: [TradingView Paper Trading Guide](https://www.tradingview.com/paper-trading/)

### Python Libraries Documentation
- [yfinance](https://pypi.org/project/yfinance/) - Stock data download
- [pandas-ta](https://github.com/twopirllc/pandas-ta) - Technical indicators
- [scikit-learn](https://scikit-learn.org/) - ML models
- [TensorFlow/Keras](https://www.tensorflow.org/) - Deep learning
- [XGBoost](https://xgboost.readthedocs.io/) - Gradient boosting

---

## 🐛 Reporting Issues

If you encounter errors not covered in troubleshooting:

1. **Check cell outputs** for specific error messages
2. **Verify all packages** are installed correctly
3. **Ensure Python version** is 3.8+ (3.12+ recommended)
4. **Check internet connection** for data download
5. **Restart kernel** and run all cells from top

### Common Package Conflicts
If you get import errors:
```bash
# Create fresh environment
python -m venv trading_env
source trading_env/bin/activate  # On Windows: trading_env\Scripts\activate
pip install -r requirements.txt
```

---

## 📝 Version History

- **v1.0** (2025-09-30): Initial release
  - 8 ML models
  - 31 technical indicators
  - 10 years historical data
  - BUY/SELL/HOLD recommendations
  - Error handling for data download
  - Comprehensive documentation

---

## 🎯 Next Steps After Running

1. **Compare prediction with actual next-day price** tomorrow
2. **Log accuracy** in a spreadsheet
3. **Calculate model win rate** over 30 days
4. **Adjust parameters** based on results
5. **Try different stocks** to test robustness
6. **Build a portfolio strategy** using multiple stock predictions

---

## 💡 Pro Tips

- **Run daily before market open** (9:00 AM EST)
- **Track predictions vs actual** for 30+ days to assess accuracy
- **Don't chase trades** - wait for HIGH confidence signals
- **Combine with fundamental analysis** (earnings, news, etc.)
- **Use multiple timeframes** - add weekly/monthly analysis
- **Backtest thoroughly** before risking real money
- **Keep learning** - markets evolve, so should your strategy

---

## 🤝 Contributing

Improvements welcome! Ideas:
- Add sentiment analysis from news
- Include options pricing data
- Multi-stock portfolio optimization
- Real-time data streaming
- Backtesting framework
- Walk-forward optimization

---

**Remember: The best trader is an educated trader. Learn, practice, and stay disciplined! 🚀📈**

---

*Last Updated: September 30, 2025*
