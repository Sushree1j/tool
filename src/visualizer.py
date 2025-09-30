"""
Visualization Module for Stock Analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set style
sns.set_style('darkgrid')
plt.style.use('seaborn-v0_8-darkgrid')


class StockVisualizer:
    """Visualize stock data and predictions"""
    
    def __init__(self, figsize=(14, 8)):
        """
        Initialize visualizer
        
        Args:
            figsize (tuple): Default figure size
        """
        self.figsize = figsize
    
    def plot_stock_price(self, df, ticker, save_path=None):
        """
        Plot stock closing price
        
        Args:
            df (pd.DataFrame): Stock data
            ticker (str): Stock ticker
            save_path (str): Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(df.index, df['Close'], label='Close Price', linewidth=2)
        ax.set_title(f'{ticker} Stock Price History', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_ohlc_volume(self, df, ticker, days=180, save_path=None):
        """
        Plot OHLC with volume
        
        Args:
            df (pd.DataFrame): Stock data
            ticker (str): Stock ticker
            days (int): Number of recent days to plot
            save_path (str): Path to save figure
        """
        df_plot = df.tail(days)
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'{ticker} Price', 'Volume')
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df_plot.index,
                open=df_plot['Open'],
                high=df_plot['High'],
                low=df_plot['Low'],
                close=df_plot['Close'],
                name='OHLC'
            ),
            row=1, col=1
        )
        
        # Volume chart
        colors = ['red' if close < open else 'green' 
                  for close, open in zip(df_plot['Close'], df_plot['Open'])]
        
        fig.add_trace(
            go.Bar(x=df_plot.index, y=df_plot['Volume'], name='Volume', marker_color=colors),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f'{ticker} Stock Analysis',
            xaxis_rangeslider_visible=False,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    def plot_technical_indicators(self, df, ticker, save_path=None):
        """
        Plot technical indicators
        
        Args:
            df (pd.DataFrame): Stock data with indicators
            ticker (str): Stock ticker
            save_path (str): Path to save figure
        """
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        
        # Price and Moving Averages
        axes[0].plot(df.index, df['Close'], label='Close', linewidth=2)
        if 'SMA_20' in df.columns:
            axes[0].plot(df.index, df['SMA_20'], label='SMA 20', alpha=0.7)
        if 'SMA_50' in df.columns:
            axes[0].plot(df.index, df['SMA_50'], label='SMA 50', alpha=0.7)
        if 'EMA_20' in df.columns:
            axes[0].plot(df.index, df['EMA_20'], label='EMA 20', alpha=0.7, linestyle='--')
        axes[0].set_title(f'{ticker} - Price and Moving Averages', fontweight='bold')
        axes[0].set_ylabel('Price ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # RSI
        if 'RSI' in df.columns:
            axes[1].plot(df.index, df['RSI'], label='RSI', color='purple', linewidth=2)
            axes[1].axhline(70, color='r', linestyle='--', alpha=0.5, label='Overbought')
            axes[1].axhline(30, color='g', linestyle='--', alpha=0.5, label='Oversold')
            axes[1].set_title('Relative Strength Index (RSI)', fontweight='bold')
            axes[1].set_ylabel('RSI')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # MACD
        if 'MACD' in df.columns:
            axes[2].plot(df.index, df['MACD'], label='MACD', linewidth=2)
            axes[2].plot(df.index, df['MACD_signal'], label='Signal', linewidth=2)
            axes[2].bar(df.index, df['MACD_diff'], label='Histogram', alpha=0.3)
            axes[2].set_title('MACD', fontweight='bold')
            axes[2].set_ylabel('MACD')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        # Bollinger Bands
        if 'BB_upper' in df.columns:
            axes[3].plot(df.index, df['Close'], label='Close', linewidth=2)
            axes[3].plot(df.index, df['BB_upper'], label='Upper Band', alpha=0.5, linestyle='--')
            axes[3].plot(df.index, df['BB_middle'], label='Middle Band', alpha=0.5)
            axes[3].plot(df.index, df['BB_lower'], label='Lower Band', alpha=0.5, linestyle='--')
            axes[3].fill_between(df.index, df['BB_lower'], df['BB_upper'], alpha=0.1)
            axes[3].set_title('Bollinger Bands', fontweight='bold')
            axes[3].set_ylabel('Price ($)')
            axes[3].set_xlabel('Date')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_predictions(self, y_true, y_pred, ticker, model_name, save_path=None):
        """
        Plot actual vs predicted prices
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            ticker (str): Stock ticker
            model_name (str): Model name
            save_path (str): Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)
        
        # Actual vs Predicted
        ax1.plot(y_true, label='Actual', linewidth=2, alpha=0.8)
        ax1.plot(y_pred, label='Predicted', linewidth=2, alpha=0.8)
        ax1.set_title(f'{ticker} - {model_name} Predictions', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Prediction Error
        error = y_true - y_pred
        ax2.plot(error, label='Prediction Error', color='red', alpha=0.7)
        ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax2.fill_between(range(len(error)), error, alpha=0.3, color='red')
        ax2.set_title('Prediction Error', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Error ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_comparison(self, results_dict, ticker, save_path=None):
        """
        Compare multiple models
        
        Args:
            results_dict (dict): Dictionary with model names as keys and metrics as values
            ticker (str): Stock ticker
            save_path (str): Path to save figure
        """
        metrics = ['RMSE', 'MAE', 'R2', 'MAPE']
        models = list(results_dict.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            values = [results_dict[model][metric] for model in models]
            
            bars = axes[idx].bar(models, values, alpha=0.7, color='steelblue')
            axes[idx].set_title(f'{metric} Comparison', fontweight='bold')
            axes[idx].set_ylabel(metric)
            axes[idx].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                             f'{value:.2f}',
                             ha='center', va='bottom')
        
        plt.suptitle(f'{ticker} - Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance(self, importance_df, top_n=20, save_path=None):
        """
        Plot feature importance
        
        Args:
            importance_df (pd.DataFrame): Feature importance dataframe
            top_n (int): Number of top features to show
            save_path (str): Path to save figure
        """
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.barh(range(len(top_features)), top_features['importance'], alpha=0.7)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_correlation_matrix(self, df, save_path=None):
        """
        Plot correlation matrix
        
        Args:
            df (pd.DataFrame): Dataframe with features
            save_path (str): Path to save figure
        """
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Calculate correlation
        corr = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(corr, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


if __name__ == "__main__":
    print("Stock Visualization Module")
