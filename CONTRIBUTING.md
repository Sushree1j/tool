# Contributing to Stock Price Prediction Tool

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/Sushree1j/tool.git
   cd tool
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   # Or for minimal installation
   pip install -r requirements-minimal.txt
   ```

3. **Set up development environment**
   ```bash
   pip install -r requirements.txt
   # Install development dependencies
   pip install pytest black flake8
   ```

## Development Workflow

1. **Create a new branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, readable code
   - Add docstrings to functions and classes
   - Include type hints where appropriate
   - Follow PEP 8 style guidelines

3. **Test your changes**
   ```bash
   # Run existing tests (if available)
   pytest tests/
   
   # Test manually
   python main.py --ticker AAPL --model rf --visualize
   ```

4. **Format your code**
   ```bash
   # Format with black
   black src/ main.py
   
   # Check with flake8
   flake8 src/ main.py
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: Brief description of your changes"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Go to GitHub and create a pull request
   - Provide a clear description of your changes
   - Reference any related issues

## Code Style Guidelines

### Python Style
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use meaningful variable and function names

### Documentation
- Add docstrings to all public functions and classes
- Use Google-style docstrings
- Include type hints for function parameters and return values

Example:
```python
def fetch_stock_data(ticker: str, period: str = '5y') -> pd.DataFrame:
    """
    Fetch historical stock data
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Data period
        
    Returns:
        pd.DataFrame: Stock data with OHLCV columns
        
    Raises:
        ValueError: If ticker is invalid
    """
    pass
```

### Commit Messages
Use clear, descriptive commit messages:
- **Add**: When adding new features
- **Fix**: When fixing bugs
- **Update**: When updating existing features
- **Refactor**: When refactoring code
- **Docs**: When updating documentation

Example:
```
Add: Support for custom technical indicators
Fix: Handle missing data in LSTM model
Update: Improve error messages in data collector
```

## Areas for Contribution

### High Priority
- [ ] Add unit tests
- [ ] Add integration tests
- [ ] Improve error handling
- [ ] Add more technical indicators
- [ ] Optimize model performance
- [ ] Add model hyperparameter tuning

### Medium Priority
- [ ] Add more visualization options
- [ ] Create web dashboard
- [ ] Add real-time prediction API
- [ ] Implement sentiment analysis
- [ ] Add portfolio optimization
- [ ] Create Docker container

### Low Priority
- [ ] Add more documentation
- [ ] Create video tutorials
- [ ] Add more example scripts
- [ ] Improve CLI interface
- [ ] Add configuration validation

## Testing Guidelines

### Unit Tests
Create unit tests for individual functions:
```python
def test_validate_ticker():
    """Test ticker validation"""
    collector = StockDataCollector()
    assert collector._validate_ticker('AAPL') == 'AAPL'
    assert collector._validate_ticker(' aapl ') == 'AAPL'
    
    with pytest.raises(ValueError):
        collector._validate_ticker('')
```

### Integration Tests
Test complete workflows:
```python
def test_full_prediction_pipeline():
    """Test complete prediction pipeline"""
    predictor = StockPredictor('AAPL')
    predictor.fetch_data(period='1mo')
    predictor.engineer_features()
    predictor.train_random_forest()
    assert 'Random Forest' in predictor.results
```

## Adding New Features

### New Technical Indicators
1. Add indicator calculation in `src/feature_engineering.py`
2. Update `add_all_features()` method
3. Document the indicator in README.md
4. Add example usage

### New Models
1. Create model class in `src/models.py`
2. Inherit from `StockPredictionModel`
3. Implement `train()` and `predict()` methods
4. Add to `StockPredictor` class
5. Update documentation

### New Visualization
1. Add plot function in `src/visualizer.py`
2. Follow existing naming conventions
3. Include docstring and examples
4. Add to visualization options

## Questions or Issues?

- **Bug reports**: Open an issue with detailed description and steps to reproduce
- **Feature requests**: Open an issue describing the feature and use case
- **Questions**: Open a discussion or issue

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow
- Focus on the code, not the person

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Thank You!

Your contributions help make this project better for everyone! 🎉
