#!/usr/bin/env python3
"""
Compare Multiple Stocks Example
Demonstrates how to compare predictions for multiple stocks
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predictor import StockPredictor
import pandas as pd

def compare_stocks(tickers, period='2y'):
    """
    Compare predictions for multiple stocks
    
    Args:
        tickers (list): List of stock ticker symbols
        period (str): Historical data period
    """
    results = {}
    
    for ticker in tickers:
        print(f"\n{'#' * 80}")
        print(f"# Processing {ticker}")
        print(f"{'#' * 80}")
        
        try:
            # Create predictor
            predictor = StockPredictor(ticker)
            
            # Fetch and prepare data
            predictor.fetch_data(period=period)
            predictor.engineer_features()
            
            # Train XGBoost model (fast and accurate)
            predictor.train_xgboost()
            
            # Store results
            if 'XGBoost' in predictor.results:
                results[ticker] = predictor.results['XGBoost']['metrics']
            
        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}")
            continue
    
    # Print comparison
    if results:
        print(f"\n{'=' * 80}")
        print("COMPARISON RESULTS")
        print(f"{'=' * 80}\n")
        
        comparison_df = pd.DataFrame(results).T
        print(comparison_df.to_string())
        
        # Find best stock
        best_stock = comparison_df['R2'].idxmax()
        print(f"\n🏆 Best Prediction (by R²): {best_stock}")
        print(f"   R² Score: {comparison_df.loc[best_stock, 'R2']:.4f}")

def main():
    """Main function"""
    
    print("=" * 80)
    print("MULTIPLE STOCKS COMPARISON EXAMPLE")
    print("=" * 80)
    
    # Define stocks to compare
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    
    print(f"\nComparing stocks: {', '.join(tickers)}")
    print(f"This may take a few minutes...\n")
    
    compare_stocks(tickers, period='2y')
    
    print(f"\n{'=' * 80}")
    print("✅ COMPARISON COMPLETE!")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main()
