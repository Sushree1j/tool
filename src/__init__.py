"""
Stock Prediction Package
"""

from .data_collector import StockDataCollector
from .feature_engineering import FeatureEngineer
from .models import (
    LinearRegressionModel,
    RandomForestModel,
    XGBoostModel,
    LSTMModel
)
from .predictor import StockPredictor
from .visualizer import StockVisualizer
from .config import Config, get_config

__version__ = '1.0.0'
__all__ = [
    'StockDataCollector',
    'FeatureEngineer',
    'LinearRegressionModel',
    'RandomForestModel',
    'XGBoostModel',
    'LSTMModel',
    'StockPredictor',
    'StockVisualizer',
    'Config',
    'get_config',
]
