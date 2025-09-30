"""
Stock Price Prediction - Main Entry Point
Simple interface for running predictions
"""

import sys
import argparse
from src.predictor import StockPredictor


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Stock Price Prediction Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict Apple stock with all models
  python main.py --ticker AAPL --visualize
  
  # Predict Tesla with XGBoost only
  python main.py --ticker TSLA --model xgb
  
  # Compare multiple stocks
  python main.py --ticker AAPL GOOGL MSFT --period 2y
        """
    )
    
    parser.add_argument('--ticker', nargs='+', default=['AAPL'], 
                       help='Stock ticker symbol(s)')
    parser.add_argument('--period', type=str, default='5y',
                       choices=['1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max'],
                       help='Historical data period')
    parser.add_argument('--model', type=str, default='all',
                       choices=['lr', 'rf', 'xgb', 'lstm', 'all'],
                       help='Model to train (lr=Linear Regression, rf=Random Forest, xgb=XGBoost, lstm=LSTM)')
    parser.add_argument('--days', type=int, default=30,
                       help='Number of days to predict ahead')
    parser.add_argument('--visualize', action='store_true',
                       help='Show visualization plots')
    parser.add_argument('--save-models', action='store_true',
                       help='Save trained models to disk')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("STOCK PRICE PREDICTION TOOL")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Tickers: {', '.join(args.ticker)}")
    print(f"  Period: {args.period}")
    print(f"  Model: {args.model}")
    print(f"  Prediction Days: {args.days}")
    print(f"  Visualize: {args.visualize}")
    
    # Process each ticker
    for ticker in args.ticker:
        try:
            print(f"\n{'#' * 80}")
            print(f"# Processing {ticker}")
            print(f"{'#' * 80}")
            
            # Initialize predictor
            predictor = StockPredictor(ticker)
            
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
            
            # Save models
            if args.save_models:
                for model_name, result in predictor.results.items():
                    model = result['model']
                    filename = f"{ticker}_{model_name.replace(' ', '_').lower()}.pkl"
                    model.save_model(filename)
            
            # Visualize
            if args.visualize:
                predictor.visualize_results()
                
        except Exception as e:
            print(f"\n❌ Error processing {ticker}: {str(e)}")
            continue
    
    print(f"\n{'=' * 80}")
    print("✅ Prediction complete!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
