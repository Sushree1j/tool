#!/usr/bin/env python3
"""
Custom Model Training Example
Demonstrates how to customize model parameters
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_collector import StockDataCollector
from src.feature_engineering import FeatureEngineer
from src.models import RandomForestModel, XGBoostModel
from src.visualizer import StockVisualizer

def custom_training_example():
    """Train models with custom parameters"""
    
    print("=" * 80)
    print("CUSTOM MODEL TRAINING EXAMPLE")
    print("=" * 80)
    
    ticker = 'AAPL'
    
    # 1. Collect data
    print(f"\n1. Collecting data for {ticker}...")
    collector = StockDataCollector()
    df = collector.fetch_stock_data(ticker, period='3y')
    
    if df is None:
        print("Failed to fetch data. Exiting.")
        return
    
    # 2. Engineer features
    print("\n2. Engineering features...")
    engineer = FeatureEngineer(df)
    engineer.add_all_features()
    features_df = engineer.get_feature_dataframe()
    feature_names = engineer.get_feature_names()
    
    # 3. Train Random Forest with custom parameters
    print("\n3. Training Random Forest with custom parameters...")
    rf_model = RandomForestModel(
        n_estimators=200,  # More trees
        max_depth=20,      # Deeper trees
        model_dir='models'
    )
    
    X_train, X_test, y_train, y_test = rf_model.prepare_data(
        features_df, feature_names, test_size=0.2
    )
    
    rf_model.train(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_metrics = rf_model.evaluate(y_test, rf_pred)
    
    print("\nRandom Forest Metrics:")
    for key, value in rf_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # 4. Train XGBoost with custom parameters
    print("\n4. Training XGBoost with custom parameters...")
    xgb_model = XGBoostModel(
        n_estimators=150,
        learning_rate=0.05,  # Slower learning
        max_depth=8,
        model_dir='models'
    )
    
    X_train, X_test, y_train, y_test = xgb_model.prepare_data(
        features_df, feature_names, test_size=0.2
    )
    
    xgb_model.train(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_metrics = xgb_model.evaluate(y_test, xgb_pred)
    
    print("\nXGBoost Metrics:")
    for key, value in xgb_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # 5. Visualize predictions
    print("\n5. Visualizing predictions...")
    visualizer = StockVisualizer()
    
    # Compare models
    metrics_dict = {
        'Random Forest': rf_metrics,
        'XGBoost': xgb_metrics
    }
    visualizer.plot_model_comparison(metrics_dict, ticker)
    
    # Show feature importance
    print("\n6. Top 10 Important Features (XGBoost):")
    importance = xgb_model.get_feature_importance()
    print(importance.head(10))
    
    # Optional: Save models
    save_models = input("\nDo you want to save the trained models? (y/n): ")
    if save_models.lower() == 'y':
        rf_model.save_model(f'{ticker}_rf_custom.pkl')
        xgb_model.save_model(f'{ticker}_xgb_custom.pkl')
        print("Models saved successfully!")
    
    print("\n" + "=" * 80)
    print("✅ CUSTOM TRAINING COMPLETE!")
    print("=" * 80)

def main():
    """Main function"""
    custom_training_example()

if __name__ == "__main__":
    main()
