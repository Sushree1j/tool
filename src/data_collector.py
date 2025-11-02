"""
Stock Data Collection Module
Fetches historical stock data from Yahoo Finance
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataCollector:
    """Collect and manage stock market data"""
    
    VALID_PERIODS = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    VALID_INTERVALS = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
    
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize the data collector
        
        Args:
            data_dir (str): Directory to save data files
        """
        self.data_dir = data_dir
        try:
            os.makedirs(data_dir, exist_ok=True)
            logger.info(f"Data directory initialized: {data_dir}")
        except Exception as e:
            logger.error(f"Failed to create data directory: {e}")
            raise
    
    def _validate_ticker(self, ticker: str) -> str:
        """
        Validate and clean ticker symbol
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            str: Cleaned ticker symbol
            
        Raises:
            ValueError: If ticker is invalid
        """
        if not ticker or not isinstance(ticker, str):
            raise ValueError("Ticker must be a non-empty string")
        
        ticker = ticker.strip().upper()
        
        if not ticker.replace('-', '').replace('.', '').isalnum():
            raise ValueError(f"Invalid ticker symbol: {ticker}")
        
        return ticker
    
    def _validate_period(self, period: str) -> None:
        """Validate period parameter"""
        if period not in self.VALID_PERIODS:
            raise ValueError(
                f"Invalid period '{period}'. Must be one of: {', '.join(self.VALID_PERIODS)}"
            )
    
    def _validate_interval(self, interval: str) -> None:
        """Validate interval parameter"""
        if interval not in self.VALID_INTERVALS:
            raise ValueError(
                f"Invalid interval '{interval}'. Must be one of: {', '.join(self.VALID_INTERVALS)}"
            )
    
    def fetch_stock_data(
        self, 
        ticker: str, 
        period: str = '5y', 
        interval: str = '1d'
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical stock data
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            period (str): Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
            pd.DataFrame: Stock data with OHLCV columns, or None if failed
        """
        try:
            # Validate inputs
            ticker = self._validate_ticker(ticker)
            self._validate_period(period)
            self._validate_interval(interval)
            
            logger.info(f"Fetching data for {ticker} (period={period}, interval={interval})...")
            
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"No data found for ticker {ticker}")
                return None
            
            logger.info(f"Successfully fetched {len(df)} rows for {ticker}")
            
            # Save to CSV
            filename = os.path.join(self.data_dir, f"{ticker}_{period}_{interval}.csv")
            df.to_csv(filename)
            logger.info(f"Data saved to {filename}")
            
            return df
        
        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None
    
    def fetch_multiple_stocks(
        self, 
        tickers: List[str], 
        period: str = '5y', 
        interval: str = '1d'
    ) -> Dict[str, pd.DataFrame]:
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
        failed_tickers = []
        
        for ticker in tickers:
            try:
                df = self.fetch_stock_data(ticker, period, interval)
                if df is not None:
                    stock_data[ticker] = df
                else:
                    failed_tickers.append(ticker)
            except Exception as e:
                logger.error(f"Failed to fetch {ticker}: {e}")
                failed_tickers.append(ticker)
        
        if failed_tickers:
            logger.warning(f"Failed to fetch data for: {', '.join(failed_tickers)}")
        
        return stock_data
    
    def load_stock_data(
        self, 
        ticker: str, 
        period: str = '5y', 
        interval: str = '1d',
        force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Load stock data from file or fetch if not available
        
        Args:
            ticker (str): Stock ticker symbol
            period (str): Data period
            interval (str): Data interval
            force_refresh (bool): Force fetch new data even if cached
        
        Returns:
            pd.DataFrame: Stock data, or None if failed
        """
        ticker = self._validate_ticker(ticker)
        filename = os.path.join(self.data_dir, f"{ticker}_{period}_{interval}.csv")
        
        if not force_refresh and os.path.exists(filename):
            try:
                logger.info(f"Loading cached data from {filename}")
                df = pd.read_csv(filename, index_col=0, parse_dates=True)
                return df
            except Exception as e:
                logger.warning(f"Failed to load cached data: {e}. Fetching fresh data...")
        
        return self.fetch_stock_data(ticker, period, interval)
    
    def get_stock_info(self, ticker: str) -> Optional[Dict]:
        """
        Get detailed stock information
        
        Args:
            ticker (str): Stock ticker symbol
        
        Returns:
            dict: Stock information, or None if failed
        """
        try:
            ticker = self._validate_ticker(ticker)
            logger.info(f"Fetching info for {ticker}")
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info:
                logger.warning(f"No info available for {ticker}")
                return None
            
            relevant_info = {
                'symbol': info.get('symbol', 'N/A'),
                'name': info.get('longName', info.get('shortName', 'N/A')),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'current_price': info.get('currentPrice', info.get('regularMarketPrice', 'N/A')),
                '52_week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
                '52_week_low': info.get('fiftyTwoWeekLow', 'N/A'),
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'dividend_yield': info.get('dividendYield', 'N/A'),
            }
            
            return relevant_info
        
        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error getting info for {ticker}: {e}")
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
