"""
Main Prediction Script
Orchestrates the entire prediction pipeline with comprehensive error handling
"""

import argparse
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from .data_collector import StockDataCollector
from .feature_engineering import FeatureEngineer
from .models import LinearRegressionModel, RandomForestModel, XGBoostModel, LSTMModel
from .visualizer import StockVisualizer
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class StockPredictor:
    """Main class for stock price prediction with comprehensive error handling"""
    
    def __init__(self, ticker: str):
        """
        Initialize predictor
        
        Args:
            ticker (str): Stock ticker symbol
            
        Raises:
            ValueError: If ticker is invalid
        """
        if not ticker or not isinstance(ticker, str):
            raise ValueError("Ticker must be a non-empty string")
        
        self.ticker = ticker.strip().upper()
        self.collector = StockDataCollector()
        self.visualizer = StockVisualizer()
        self.df: Optional[pd.DataFrame] = None
        self.features_df: Optional[pd.DataFrame] = None
        self.feature_names: Optional[list] = None
        self.results: Dict[str, Any] = {}
        
        logger.info(f"StockPredictor initialized for {self.ticker}")
    
    def fetch_data(self, period: str = '5y') -> pd.DataFrame:
        """
        Fetch stock data
        
        Args:
            period (str): Historical data period
            
        Returns:
            pd.DataFrame: Stock data
            
        Raises:
            ValueError: If data fetching fails
        """
        print(f"\n{'='*60}")
        print(f"Fetching data for {self.ticker}...")
        print(f"{'='*60}")
        
        try:
            self.df = self.collector.load_stock_data(self.ticker, period=period)
            
            if self.df is None or self.df.empty:
                raise ValueError(f"Failed to fetch data for {self.ticker}")
            
            print(f"✓ Data fetched: {len(self.df)} rows")
            
            # Get and display stock info
            try:
                info = self.collector.get_stock_info(self.ticker)
                if info:
                    print(f"\n📊 Stock Information:")
                    print(f"  Name: {info['name']}")
                    print(f"  Sector: {info['sector']}")
                    if info['current_price'] != 'N/A':
                        print(f"  Current Price: ${info['current_price']}")
            except Exception as e:
                logger.debug(f"Could not fetch stock info: {e}")
            
            return self.df
            
        except Exception as e:
            logger.error(f"Error fetching data for {self.ticker}: {e}")
            raise ValueError(f"Failed to fetch data for {self.ticker}: {e}")
    
    def engineer_features(self) -> pd.DataFrame:
        """
        Create technical indicators
        
        Returns:
            pd.DataFrame: Features dataframe
            
        Raises:
            RuntimeError: If feature engineering fails
            ValueError: If data hasn't been fetched
        """
        if self.df is None or self.df.empty:
            raise ValueError("No data available. Call fetch_data() first.")
        
        print(f"\n{'='*60}")
        print("Engineering features...")
        print(f"{'='*60}")
        
        try:
            engineer = FeatureEngineer(self.df)
            engineer.add_all_features(show_progress=True)
            self.features_df = engineer.get_feature_dataframe()
            self.feature_names = engineer.get_feature_names()
            
            print(f"\n✓ Features created: {len(self.feature_names)}")
            print(f"✓ Final dataset: {self.features_df.shape[0]} rows × {self.features_df.shape[1]} columns")
            
            return self.features_df
            
        except Exception as e:
            logger.error(f"Error engineering features: {e}")
            raise RuntimeError(f"Feature engineering failed: {e}")
    
    def train_linear_regression(self):
        """Train Linear Regression model"""
        print(f"\n{'='*60}")
        print("LINEAR REGRESSION MODEL")
        print(f"{'='*60}")
        
        model = LinearRegressionModel()
        X_train, X_test, y_train, y_test = model.prepare_data(
            self.features_df, self.feature_names
        )
        
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = model.evaluate(y_test, y_pred)
        self.results['Linear Regression'] = {
            'model': model,
            'y_test': y_test,
            'y_pred': y_pred,
            'metrics': metrics
        }
        
        print(f"\nMetrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        return model, metrics
    
    def train_random_forest(self):
        """Train Random Forest model"""
        print(f"\n{'='*60}")
        print("RANDOM FOREST MODEL")
        print(f"{'='*60}")
        
        model = RandomForestModel(n_estimators=100, max_depth=15)
        X_train, X_test, y_train, y_test = model.prepare_data(
            self.features_df, self.feature_names
        )
        
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = model.evaluate(y_test, y_pred)
        self.results['Random Forest'] = {
            'model': model,
            'y_test': y_test,
            'y_pred': y_pred,
            'metrics': metrics
        }
        
        print(f"\nMetrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Feature importance
        importance = model.get_feature_importance()
        print(f"\nTop 10 Important Features:")
        print(importance.head(10))
        
        return model, metrics
    
    def train_xgboost(self):
        """Train XGBoost model"""
        print(f"\n{'='*60}")
        print("XGBOOST MODEL")
        print(f"{'='*60}")
        
        model = XGBoostModel(n_estimators=100, learning_rate=0.1, max_depth=6)
        X_train, X_test, y_train, y_test = model.prepare_data(
            self.features_df, self.feature_names
        )
        
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = model.evaluate(y_test, y_pred)
        self.results['XGBoost'] = {
            'model': model,
            'y_test': y_test,
            'y_pred': y_pred,
            'metrics': metrics
        }
        
        print(f"\nMetrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        return model, metrics
    
    def train_lstm(self, epochs=50):
        """Train LSTM model"""
        print(f"\n{'='*60}")
        print("LSTM MODEL")
        print(f"{'='*60}")
        
        model = LSTMModel(sequence_length=60)
        X_train, X_test, y_train, y_test = model.prepare_data(
            self.features_df, self.feature_names
        )
        
        model.train(X_train, y_train, epochs=epochs)
        y_pred = model.predict(X_test)
        
        metrics = model.evaluate(y_test, y_pred)
        self.results['LSTM'] = {
            'model': model,
            'y_test': model.scaler.inverse_transform(y_test.reshape(-1, 1)),
            'y_pred': y_pred,
            'metrics': metrics
        }
        
        print(f"\nMetrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        return model, metrics
    
    def train_all_models(self, include_lstm=False):
        """Train all models"""
        self.train_linear_regression()
        self.train_random_forest()
        self.train_xgboost()
        
        if include_lstm:
            self.train_lstm(epochs=50)
    
    def visualize_results(self):
        """Visualize all results"""
        print(f"\n{'='*60}")
        print("Generating visualizations...")
        print(f"{'='*60}")
        
        # Stock price history
        self.visualizer.plot_stock_price(self.df, self.ticker)
        
        # Technical indicators
        self.visualizer.plot_technical_indicators(self.features_df, self.ticker)
        
        # Predictions for each model
        for model_name, result in self.results.items():
            self.visualizer.plot_predictions(
                result['y_test'],
                result['y_pred'],
                self.ticker,
                model_name
            )
        
        # Model comparison
        metrics_dict = {name: result['metrics'] for name, result in self.results.items()}
        self.visualizer.plot_model_comparison(metrics_dict, self.ticker)
    
    def print_summary(self):
        """Print summary of results"""
        print(f"\n{'='*60}")
        print(f"PREDICTION SUMMARY FOR {self.ticker}")
        print(f"{'='*60}\n")
        
        # Create comparison table
        comparison_df = pd.DataFrame({
            model_name: result['metrics']
            for model_name, result in self.results.items()
        }).T
        
        print(comparison_df.to_string())
        
        # Best model
        best_model = comparison_df['R2'].idxmax()
        print(f"\n🏆 Best Model (by R²): {best_model}")
        print(f"   R² Score: {comparison_df.loc[best_model, 'R2']:.4f}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Stock Price Prediction')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--period', type=str, default='5y', help='Data period')
    parser.add_argument('--model', type=str, default='all', 
                       choices=['lr', 'rf', 'xgb', 'lstm', 'all'],
                       help='Model to train')
    parser.add_argument('--visualize', action='store_true', help='Show visualizations')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = StockPredictor(args.ticker)
    
    # Fetch and prepare data
    predictor.fetch_data(period=args.period)
    predictor.engineer_features()
    
    # Train models
    if args.model == 'all':
        predictor.train_all_models(include_lstm=True)
    elif args.model == 'lr':
        predictor.train_linear_regression()
    elif args.model == 'rf':
        predictor.train_random_forest()
    elif args.model == 'xgb':
        predictor.train_xgboost()
    elif args.model == 'lstm':
        predictor.train_lstm()
    
    # Print summary
    predictor.print_summary()
    
    # Visualize
    if args.visualize:
        predictor.visualize_results()


if __name__ == "__main__":
    main()
