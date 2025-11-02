"""
Feature Engineering Module
Creates technical indicators and features for stock prediction
"""

import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator


class FeatureEngineer:
    """Create technical indicators and features from stock data"""
    
    def __init__(self, df):
        """
        Initialize feature engineer
        
        Args:
            df (pd.DataFrame): Stock data with OHLCV columns
        """
        self.df = df.copy()
    
    def add_moving_averages(self, windows=[5, 10, 20, 50, 200]):
        """
        Add Simple Moving Averages
        
        Args:
            windows (list): List of window sizes
        """
        for window in windows:
            self.df[f'SMA_{window}'] = SMAIndicator(
                close=self.df['Close'], 
                window=window
            ).sma_indicator()
            
            self.df[f'EMA_{window}'] = EMAIndicator(
                close=self.df['Close'], 
                window=window
            ).ema_indicator()
        
        return self
    
    def add_rsi(self, window=14):
        """
        Add Relative Strength Index
        
        Args:
            window (int): RSI window
        """
        rsi = RSIIndicator(close=self.df['Close'], window=window)
        self.df['RSI'] = rsi.rsi()
        
        return self
    
    def add_macd(self, fast=12, slow=26, signal=9):
        """
        Add MACD indicators
        
        Args:
            fast (int): Fast period
            slow (int): Slow period
            signal (int): Signal period
        """
        macd = MACD(
            close=self.df['Close'],
            window_fast=fast,
            window_slow=slow,
            window_sign=signal
        )
        
        self.df['MACD'] = macd.macd()
        self.df['MACD_signal'] = macd.macd_signal()
        self.df['MACD_diff'] = macd.macd_diff()
        
        return self
    
    def add_bollinger_bands(self, window=20, std=2):
        """
        Add Bollinger Bands
        
        Args:
            window (int): Moving average window
            std (int): Number of standard deviations
        """
        bb = BollingerBands(
            close=self.df['Close'],
            window=window,
            window_dev=std
        )
        
        self.df['BB_upper'] = bb.bollinger_hband()
        self.df['BB_middle'] = bb.bollinger_mavg()
        self.df['BB_lower'] = bb.bollinger_lband()
        self.df['BB_width'] = bb.bollinger_wband()
        
        return self
    
    def add_stochastic(self, window=14, smooth=3):
        """
        Add Stochastic Oscillator
        
        Args:
            window (int): Window size
            smooth (int): Smoothing window
        """
        stoch = StochasticOscillator(
            high=self.df['High'],
            low=self.df['Low'],
            close=self.df['Close'],
            window=window,
            smooth_window=smooth
        )
        
        self.df['Stoch'] = stoch.stoch()
        self.df['Stoch_signal'] = stoch.stoch_signal()
        
        return self
    
    def add_atr(self, window=14):
        """
        Add Average True Range (volatility indicator)
        
        Args:
            window (int): Window size
        """
        atr = AverageTrueRange(
            high=self.df['High'],
            low=self.df['Low'],
            close=self.df['Close'],
            window=window
        )
        
        self.df['ATR'] = atr.average_true_range()
        
        return self
    
    def add_obv(self):
        """Add On-Balance Volume"""
        obv = OnBalanceVolumeIndicator(
            close=self.df['Close'],
            volume=self.df['Volume']
        )
        
        self.df['OBV'] = obv.on_balance_volume()
        
        return self
    
    def add_price_changes(self):
        """Add price change features"""
        self.df['Price_Change'] = self.df['Close'].pct_change()
        self.df['Price_Change_1d'] = self.df['Close'].pct_change(periods=1)
        self.df['Price_Change_5d'] = self.df['Close'].pct_change(periods=5)
        self.df['Price_Change_20d'] = self.df['Close'].pct_change(periods=20)
        
        return self
    
    def add_volume_features(self):
        """Add volume-based features"""
        self.df['Volume_Change'] = self.df['Volume'].pct_change()
        self.df['Volume_MA_5'] = self.df['Volume'].rolling(window=5).mean()
        self.df['Volume_MA_20'] = self.df['Volume'].rolling(window=20).mean()
        
        return self
    
    def add_target_variable(self, days_ahead=1):
        """
        Add target variable for prediction
        
        Args:
            days_ahead (int): Number of days to predict ahead
        """
        self.df['Target'] = self.df['Close'].shift(-days_ahead)
        self.df['Target_Change'] = ((self.df['Target'] - self.df['Close']) / self.df['Close']) * 100
        
        return self
    
    def add_all_features(self):
        """Add all technical indicators"""
        print("Adding technical indicators...")
        
        self.add_moving_averages()
        self.add_rsi()
        self.add_macd()
        self.add_bollinger_bands()
        self.add_stochastic()
        self.add_atr()
        self.add_obv()
        self.add_price_changes()
        self.add_volume_features()
        self.add_target_variable()
        
        # Remove NaN values
        initial_rows = len(self.df)
        self.df = self.df.dropna()
        print(f"Removed {initial_rows - len(self.df)} rows with NaN values")
        
        return self
    
    def get_feature_dataframe(self):
        """
        Get the dataframe with all features
        
        Returns:
            pd.DataFrame: Dataframe with features
        """
        return self.df
    
    def get_feature_names(self, exclude=['Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'Target_Change']):
        """
        Get list of feature names
        
        Args:
            exclude (list): Columns to exclude
        
        Returns:
            list: Feature names
        """
        return [col for col in self.df.columns if col not in exclude]


if __name__ == "__main__":
    # Example usage
    from .data_collector import StockDataCollector
    
    # Fetch data
    collector = StockDataCollector()
    df = collector.load_stock_data('AAPL', period='2y')
    
    # Engineer features
    engineer = FeatureEngineer(df)
    engineer.add_all_features()
    
    # Get result
    result_df = engineer.get_feature_dataframe()
    print(f"\nDataFrame shape: {result_df.shape}")
    print(f"\nFeatures: {engineer.get_feature_names()}")
    print(f"\nLast 5 rows:")
    print(result_df.tail())
