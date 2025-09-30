"""
Machine Learning Models for Stock Price Prediction
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os


class StockPredictionModel:
    """Base class for stock prediction models"""
    
    def __init__(self, model_dir='models'):
        """
        Initialize model
        
        Args:
            model_dir (str): Directory to save models
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def prepare_data(self, df, feature_cols, target_col='Target', test_size=0.2):
        """
        Prepare data for training
        
        Args:
            df (pd.DataFrame): Input dataframe
            feature_cols (list): List of feature column names
            target_col (str): Target column name
            test_size (float): Test set size
        
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        # Remove rows with NaN in target
        df = df.dropna(subset=[target_col])
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Split data (time series - no shuffling)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.feature_names = feature_cols
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def evaluate(self, y_true, y_pred):
        """
        Evaluate model performance
        
        Args:
            y_true: True values
            y_pred: Predicted values
        
        Returns:
            dict: Evaluation metrics
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
    
    def save_model(self, filename):
        """Save model to file"""
        filepath = os.path.join(self.model_dir, filename)
        joblib.dump({'model': self.model, 'scaler': self.scaler, 'features': self.feature_names}, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filename):
        """Load model from file"""
        filepath = os.path.join(self.model_dir, filename)
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['features']
        print(f"Model loaded from {filepath}")


class LinearRegressionModel(StockPredictionModel):
    """Linear Regression model"""
    
    def __init__(self, model_dir='models'):
        super().__init__(model_dir)
        self.model = LinearRegression()
    
    def train(self, X_train, y_train):
        """Train the model"""
        print("Training Linear Regression model...")
        self.model.fit(X_train, y_train)
        print("Training completed!")
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)


class RandomForestModel(StockPredictionModel):
    """Random Forest model"""
    
    def __init__(self, n_estimators=100, max_depth=10, model_dir='models'):
        super().__init__(model_dir)
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
    
    def train(self, X_train, y_train):
        """Train the model"""
        print("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        print("Training completed!")
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """Get feature importance"""
        if self.feature_names:
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance
        return None


class XGBoostModel(StockPredictionModel):
    """XGBoost model"""
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=6, model_dir='models'):
        super().__init__(model_dir)
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
    
    def train(self, X_train, y_train):
        """Train the model"""
        print("Training XGBoost model...")
        self.model.fit(X_train, y_train, verbose=False)
        print("Training completed!")
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """Get feature importance"""
        if self.feature_names:
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance
        return None


class RidgeModel(StockPredictionModel):
    """Ridge Regression model"""
    
    def __init__(self, alpha=1.0, model_dir='models'):
        super().__init__(model_dir)
        self.model = Ridge(alpha=alpha, random_state=42)
    
    def train(self, X_train, y_train):
        """Train the model"""
        print("Training Ridge Regression model...")
        self.model.fit(X_train, y_train)
        print("Training completed!")
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)


class LassoModel(StockPredictionModel):
    """Lasso Regression model"""
    
    def __init__(self, alpha=0.1, model_dir='models'):
        super().__init__(model_dir)
        self.model = Lasso(alpha=alpha, random_state=42)
    
    def train(self, X_train, y_train):
        """Train the model"""
        print("Training Lasso Regression model...")
        self.model.fit(X_train, y_train)
        print("Training completed!")
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)


class GradientBoostingModel(StockPredictionModel):
    """Gradient Boosting model"""
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=5, model_dir='models'):
        super().__init__(model_dir)
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
    
    def train(self, X_train, y_train):
        """Train the model"""
        print("Training Gradient Boosting model...")
        self.model.fit(X_train, y_train)
        print("Training completed!")
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """Get feature importance"""
        if self.feature_names:
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance
        return None


class ExtraTreesModel(StockPredictionModel):
    """Extra Trees model"""
    
    def __init__(self, n_estimators=100, max_depth=20, model_dir='models'):
        super().__init__(model_dir)
        self.model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
    
    def train(self, X_train, y_train):
        """Train the model"""
        print("Training Extra Trees model...")
        self.model.fit(X_train, y_train)
        print("Training completed!")
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """Get feature importance"""
        if self.feature_names:
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance
        return None


class AdaBoostModel(StockPredictionModel):
    """AdaBoost model"""
    
    def __init__(self, n_estimators=100, learning_rate=0.1, model_dir='models'):
        super().__init__(model_dir)
        self.model = AdaBoostRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42
        )
    
    def train(self, X_train, y_train):
        """Train the model"""
        print("Training AdaBoost model...")
        self.model.fit(X_train, y_train)
        print("Training completed!")
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)


class SVRModel(StockPredictionModel):
    """Support Vector Regression model"""
    
    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1, model_dir='models'):
        super().__init__(model_dir)
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    
    def train(self, X_train, y_train):
        """Train the model"""
        print("Training SVR model...")
        self.model.fit(X_train, y_train)
        print("Training completed!")
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)


class LSTMModel:
    """LSTM Deep Learning model"""
    
    def __init__(self, sequence_length=60, model_dir='models'):
        """
        Initialize LSTM model
        
        Args:
            sequence_length (int): Number of time steps to look back
            model_dir (str): Directory to save models
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def prepare_sequences(self, data):
        """
        Prepare sequences for LSTM
        
        Args:
            data: Input data
        
        Returns:
            np.array: Sequences
        """
        sequences = []
        targets = []
        
        for i in range(self.sequence_length, len(data)):
            sequences.append(data[i-self.sequence_length:i])
            targets.append(data[i])
        
        return np.array(sequences), np.array(targets)
    
    def prepare_data(self, df, feature_cols, target_col='Target', test_size=0.2):
        """Prepare data for LSTM"""
        df = df.dropna(subset=[target_col])
        
        # Use only Close price for simple LSTM
        data = df['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = self.prepare_sequences(scaled_data)
        
        # Split data
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, input_shape):
        """Build LSTM model"""
        self.model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1)
        ])
        
        self.model.compile(optimizer='adam', loss='mean_squared_error')
    
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1):
        """Train LSTM model"""
        print("Training LSTM model...")
        
        if self.model is None:
            self.build_model((X_train.shape[1], X_train.shape[2]))
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        print("Training completed!")
        return history
    
    def predict(self, X):
        """Make predictions"""
        predictions = self.model.predict(X)
        return self.scaler.inverse_transform(predictions)
    
    def evaluate(self, y_true, y_pred):
        """Evaluate model"""
        y_true_unscaled = self.scaler.inverse_transform(y_true.reshape(-1, 1))
        
        rmse = np.sqrt(mean_squared_error(y_true_unscaled, y_pred))
        mae = mean_absolute_error(y_true_unscaled, y_pred)
        r2 = r2_score(y_true_unscaled, y_pred)
        mape = np.mean(np.abs((y_true_unscaled - y_pred) / y_true_unscaled)) * 100
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
    
    def save_model(self, filename):
        """Save model"""
        filepath = os.path.join(self.model_dir, filename)
        self.model.save(filepath)
        scaler_path = os.path.join(self.model_dir, f"{filename}_scaler.pkl")
        joblib.dump(self.scaler, scaler_path)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filename):
        """Load model"""
        filepath = os.path.join(self.model_dir, filename)
        self.model = keras.models.load_model(filepath)
        scaler_path = os.path.join(self.model_dir, f"{filename}_scaler.pkl")
        self.scaler = joblib.load(scaler_path)
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    print("Stock Prediction Models Module")
    print("Available models: LinearRegression, Ridge, Lasso, RandomForest, XGBoost,")
    print("                  GradientBoosting, ExtraTrees, AdaBoost, SVR, LSTM")
