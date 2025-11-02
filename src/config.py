"""
Configuration Management Module
Loads and manages configuration from YAML file
"""

import yaml
import os
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for the stock prediction tool"""
    
    DEFAULT_CONFIG = {
        'data': {
            'directory': 'data',
            'cache_enabled': True,
            'default_period': '5y',
            'default_interval': '1d'
        },
        'models': {
            'linear_regression': {'enabled': True},
            'random_forest': {
                'enabled': True,
                'n_estimators': 100,
                'max_depth': 15,
                'random_state': 42
            },
            'xgboost': {
                'enabled': True,
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42
            },
            'lstm': {
                'enabled': False,
                'sequence_length': 60,
                'epochs': 50,
                'batch_size': 32,
                'validation_split': 0.1
            }
        },
        'training': {
            'test_size': 0.2,
            'random_state': 42
        },
        'visualization': {
            'figure_size': [14, 8],
            'dpi': 300,
            'save_plots': False
        },
        'logging': {
            'level': 'INFO',
            'console': True
        }
    }
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize configuration
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Returns:
            dict: Configuration dictionary
        """
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Configuration loaded from {self.config_path}")
                return self._merge_with_defaults(config)
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}. Using defaults.")
                return self.DEFAULT_CONFIG.copy()
        else:
            logger.info("Config file not found. Using default configuration.")
            return self.DEFAULT_CONFIG.copy()
    
    def _merge_with_defaults(self, config: Dict) -> Dict:
        """
        Merge loaded config with defaults
        
        Args:
            config (dict): Loaded configuration
            
        Returns:
            dict: Merged configuration
        """
        merged = self.DEFAULT_CONFIG.copy()
        
        def deep_merge(base, override):
            """Recursively merge dictionaries"""
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    base[key] = deep_merge(base[key], value)
                else:
                    base[key] = value
            return base
        
        return deep_merge(merged, config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key
        
        Args:
            key (str): Configuration key (e.g., 'data.directory')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value
        
        Args:
            key (str): Configuration key (e.g., 'data.directory')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: str = None) -> None:
        """
        Save configuration to YAML file
        
        Args:
            path (str): Path to save configuration (defaults to config_path)
        """
        path = path or self.config_path
        
        try:
            with open(path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            logger.info(f"Configuration saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access"""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-like setting"""
        self.set(key, value)


# Global configuration instance
_config = None


def get_config(config_path: str = 'config.yaml') -> Config:
    """
    Get global configuration instance
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        Config: Configuration instance
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config


if __name__ == "__main__":
    # Test configuration
    config = Config()
    print("Data directory:", config.get('data.directory'))
    print("Random Forest max_depth:", config.get('models.random_forest.max_depth'))
    print("Test size:", config.get('training.test_size'))
