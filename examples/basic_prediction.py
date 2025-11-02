#!/usr/bin/env python3
"""
Basic Stock Prediction Example
Demonstrates the simplest way to use the stock prediction tool
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predictor import StockPredictor

def main():
    """Run basic stock prediction"""
    
    print("=" * 80)
    print("BASIC STOCK PREDICTION EXAMPLE")
    print("=" * 80)
    
    # Initialize predictor for Apple stock
    ticker = 'AAPL'
    print(f"\n1. Creating predictor for {ticker}...")
    predictor = StockPredictor(ticker)
    
    # Fetch data
    print("\n2. Fetching historical data...")
    predictor.fetch_data(period='2y')
    
    # Engineer features
    print("\n3. Engineering features...")
    predictor.engineer_features()
    
    # Train a simple model (Random Forest)
    print("\n4. Training Random Forest model...")
    predictor.train_random_forest()
    
    # Print results
    print("\n5. Results:")
    predictor.print_summary()
    
    print("\n" + "=" * 80)
    print("✅ PREDICTION COMPLETE!")
    print("=" * 80)
    
    # Optional: Visualize results
    # Uncomment the line below to show charts
    # predictor.visualize_results()

if __name__ == "__main__":
    main()
