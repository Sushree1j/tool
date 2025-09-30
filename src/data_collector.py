"""
Stock Data Collection Module
Fetches historical stock data from Yahoo Finance
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os


class StockDataCollector:
    """Collect and manage stock market data"""
    
    def __init__(self, data_dir='data'):
        """
        Initialize the data collector
        
        Args:
            data_dir (str): Directory to save data files
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def fetch_stock_data(self, ticker, period='5y', interval='1d'):
        """
        Fetch historical stock data
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            period (str): Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
            pd.DataFrame: Stock data with OHLCV columns
        """
        try:
            print(f"Fetching data for {ticker}...")
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
            
            if df.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            
            # Save to CSV
            filename = f"{self.data_dir}/{ticker}_{period}_{interval}.csv"
            df.to_csv(filename)
            print(f"Data saved to {filename}")
            
            return df
        
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
    def fetch_multiple_stocks(self, tickers, period='5y', interval='1d'):
        """
        Fetch data for multiple stocks
        
        Args:
            tickers (list): List of ticker symbols
            period (str): Data period
            interval (str): Data interval
        
        Returns:
            dict: Dictionary with ticker as key and DataFrame as value
        """
        stock_data = {}
        for ticker in tickers:
            df = self.fetch_stock_data(ticker, period, interval)
            if df is not None:
                stock_data[ticker] = df
        
        return stock_data
    
    def load_stock_data(self, ticker, period='5y', interval='1d'):
        """
        Load stock data from file or fetch if not available
        
        Args:
            ticker (str): Stock ticker symbol
            period (str): Data period
            interval (str): Data interval
        
        Returns:
            pd.DataFrame: Stock data
        """
        filename = f"{self.data_dir}/{ticker}_{period}_{interval}.csv"
        
        if os.path.exists(filename):
            print(f"Loading data from {filename}")
            df = pd.read_csv(filename, index_col=0, parse_dates=True)
            return df
        else:
            return self.fetch_stock_data(ticker, period, interval)
    
    def get_stock_info(self, ticker):
        """
        Get detailed stock information
        
        Args:
            ticker (str): Stock ticker symbol
        
        Returns:
            dict: Stock information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            relevant_info = {
                'symbol': info.get('symbol', 'N/A'),
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'current_price': info.get('currentPrice', 'N/A'),
                '52_week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
                '52_week_low': info.get('fiftyTwoWeekLow', 'N/A'),
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'dividend_yield': info.get('dividendYield', 'N/A'),
            }
            
            return relevant_info
        
        except Exception as e:
            print(f"Error getting info for {ticker}: {str(e)}")
            return None


if __name__ == "__main__":
    # Example usage
    collector = StockDataCollector()
    
    # Fetch Apple stock data
    apple_data = collector.fetch_stock_data('AAPL', period='2y')
    print(f"\nApple Stock Data Shape: {apple_data.shape}")
    print(apple_data.head())
    
    # Get stock info
    apple_info = collector.get_stock_info('AAPL')
    print(f"\nApple Stock Info:")
    for key, value in apple_info.items():
        print(f"{key}: {value}")
