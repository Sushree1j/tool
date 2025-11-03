"""
Setup script for Stock Price Prediction Tool
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_file(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

setup(
    name='stock-prediction-tool',
    version='1.0.0',
    description='A comprehensive stock price prediction tool using machine learning',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    author='Stock Prediction Team',
    author_email='',
    url='https://github.com/Sushree1j/tool',
    packages=find_packages(),
    install_requires=[
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'yfinance>=0.2.28',
        'scikit-learn>=1.3.0',
        'xgboost>=2.0.0',
        'ta>=0.11.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'tqdm>=4.66.0',
        'pyyaml>=6.0',
        'joblib>=1.3.0',
    ],
    extras_require={
        'deep-learning': [
            'tensorflow>=2.13.0',
            'keras>=2.13.0',
        ],
        'advanced-viz': [
            'plotly>=5.14.0',
        ],
        'jupyter': [
            'jupyter>=1.0.0',
            'notebook>=7.0.0',
            'ipywidgets>=8.0.0',
        ],
        'dev': [
            'pytest>=7.0.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'stock-predict=main:cli_entry',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Developers',
        'Topic :: Office/Business :: Financial :: Investment',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    python_requires='>=3.8',
    keywords='stock prediction machine-learning finance trading technical-analysis',
)
