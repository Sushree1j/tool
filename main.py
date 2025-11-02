"""
Stock Price Prediction - Main Entry Point
Simple interface for running predictions with comprehensive error handling
"""

import sys
import argparse
import logging
from typing import List

try:
    from src.predictor import StockPredictor
    from src.config import get_config
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("\nPlease ensure all dependencies are installed:")
    print("  pip install -r requirements.txt")
    print("\nOr for minimal installation:")
    print("  pip install -r requirements-minimal.txt")
    sys.exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_argument_parser() -> argparse.ArgumentParser:
    """
    Set up command line argument parser
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description='Stock Price Prediction Tool - Predict stock prices using machine learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick prediction for Apple stock
  python main.py --ticker AAPL
  
  # Predict with visualization
  python main.py --ticker AAPL --visualize
  
  # Predict Tesla with XGBoost only
  python main.py --ticker TSLA --model xgb
  
  # Compare multiple stocks
  python main.py --ticker AAPL GOOGL MSFT --period 2y
  
  # Train all models and save them
  python main.py --ticker AAPL --model all --save-models --visualize

For more examples, see the examples/ directory.
For detailed documentation, see README.md
        """
    )
    
    parser.add_argument(
        '--ticker', 
        nargs='+', 
        default=['AAPL'], 
        help='Stock ticker symbol(s) (e.g., AAPL, GOOGL, MSFT)',
        metavar='TICKER'
    )
    parser.add_argument(
        '--period', 
        type=str, 
        default='5y',
        choices=['1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max'],
        help='Historical data period (default: 5y)',
        metavar='PERIOD'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='all',
        choices=['lr', 'rf', 'xgb', 'lstm', 'all'],
        help='Model to train: lr=Linear Regression, rf=Random Forest, xgb=XGBoost, lstm=LSTM, all=All models (default: all)',
        metavar='MODEL'
    )
    parser.add_argument(
        '--days', 
        type=int, 
        default=30,
        help='Number of days to predict ahead (default: 30)',
        metavar='DAYS'
    )
    parser.add_argument(
        '--visualize', 
        action='store_true',
        help='Show visualization plots (requires matplotlib)'
    )
    parser.add_argument(
        '--save-models', 
        action='store_true',
        help='Save trained models to disk for later use'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)',
        metavar='CONFIG'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging (debug mode)'
    )
    
    return parser


def validate_ticker(ticker: str) -> bool:
    """
    Basic ticker validation
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        bool: True if ticker appears valid
    """
    if not ticker or not isinstance(ticker, str):
        return False
    
    ticker = ticker.strip().upper()
    
    # Basic validation: should be alphanumeric with possible hyphens/dots
    return len(ticker) > 0 and ticker.replace('-', '').replace('.', '').isalnum()


def process_ticker(
    ticker: str, 
    args: argparse.Namespace,
    config: dict
) -> bool:
    """
    Process prediction for a single ticker
    
    Args:
        ticker (str): Stock ticker symbol
        args (Namespace): Command line arguments
        config (dict): Configuration dictionary
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"\n{'#' * 80}")
        print(f"# Processing {ticker}")
        print(f"{'#' * 80}")
        
        # Initialize predictor
        logger.info(f"Initializing predictor for {ticker}")
        predictor = StockPredictor(ticker)
        
        # Fetch and prepare data
        logger.info("Fetching historical data...")
        predictor.fetch_data(period=args.period)
        
        logger.info("Engineering features...")
        predictor.engineer_features()
        
        # Train models
        logger.info("Training model(s)...")
        if args.model == 'all':
            # Check if LSTM is enabled in config
            include_lstm = config.get('models.lstm.enabled', False)
            if include_lstm:
                logger.info("LSTM is enabled - training may take longer...")
            predictor.train_all_models(include_lstm=include_lstm)
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
        
        # Save models if requested
        if args.save_models:
            logger.info("Saving trained models...")
            for model_name, result in predictor.results.items():
                try:
                    model = result['model']
                    filename = f"{ticker}_{model_name.replace(' ', '_').lower()}.pkl"
                    model.save_model(filename)
                    logger.info(f"Saved model: {filename}")
                except Exception as e:
                    logger.warning(f"Failed to save {model_name}: {e}")
        
        # Visualize if requested
        if args.visualize:
            try:
                logger.info("Generating visualizations...")
                predictor.visualize_results()
            except Exception as e:
                logger.error(f"Visualization error: {e}")
                print(f"⚠️  Visualization failed: {e}")
                print("Continuing without visualization...")
        
        logger.info(f"Successfully completed prediction for {ticker}")
        return True
                
    except ValueError as e:
        logger.error(f"Validation error for {ticker}: {e}")
        print(f"\n❌ Validation Error: {e}")
        return False
    except Exception as e:
        logger.error(f"Error processing {ticker}: {e}", exc_info=True)
        print(f"\n❌ Error processing {ticker}: {str(e)}")
        print("This stock may not have sufficient data or the ticker may be invalid.")
        return False


def main():
    """Main entry point"""
    # Parse arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")
    
    # Load configuration
    try:
        from src.config import get_config
        config = get_config(args.config)
        logger.info(f"Configuration loaded from {args.config}")
    except Exception as e:
        logger.warning(f"Failed to load config: {e}. Using defaults.")
        config = {}
    
    # Print header
    print("=" * 80)
    print("STOCK PRICE PREDICTION TOOL")
    print("=" * 80)
    print(f"\n⚙️  Configuration:")
    print(f"  Tickers: {', '.join(args.ticker)}")
    print(f"  Period: {args.period}")
    print(f"  Model: {args.model}")
    print(f"  Prediction Days: {args.days}")
    print(f"  Visualize: {args.visualize}")
    print(f"  Save Models: {args.save_models}")
    
    # Validate tickers
    invalid_tickers = [t for t in args.ticker if not validate_ticker(t)]
    if invalid_tickers:
        logger.error(f"Invalid ticker symbols: {', '.join(invalid_tickers)}")
        print(f"\n❌ Invalid ticker symbols: {', '.join(invalid_tickers)}")
        print("Please use valid stock ticker symbols (e.g., AAPL, GOOGL, MSFT)")
        sys.exit(1)
    
    # Process each ticker
    successful = []
    failed = []
    
    for ticker in args.ticker:
        ticker = ticker.upper().strip()
        if process_ticker(ticker, args, config):
            successful.append(ticker)
        else:
            failed.append(ticker)
    
    # Print final summary
    print(f"\n{'=' * 80}")
    print("FINAL SUMMARY")
    print(f"{'=' * 80}")
    
    if successful:
        print(f"\n✅ Successfully processed {len(successful)} ticker(s):")
        for ticker in successful:
            print(f"   • {ticker}")
    
    if failed:
        print(f"\n❌ Failed to process {len(failed)} ticker(s):")
        for ticker in failed:
            print(f"   • {ticker}")
    
    print(f"\n{'=' * 80}")
    if failed:
        print("⚠️  Completed with some errors")
        print(f"{'=' * 80}")
        sys.exit(1)
    else:
        print("✅ All predictions completed successfully!")
        print(f"{'=' * 80}")
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n❌ Unexpected error: {e}")
        print("\nFor help, run: python main.py --help")
        print("For bugs, please report at: https://github.com/Sushree1j/tool/issues")
        sys.exit(1)
