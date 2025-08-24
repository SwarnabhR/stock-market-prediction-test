# setup.py
from setuptools import setup, find_packages
import os

setup(
    name="stock-predictor",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "tensorflow[and-cuda]>=2.20.0",  # Latest with GPU support
        "protobuf>=5.28.0",              # Compatible with TF 2.20+
        "pandas>=2.0.0",                 # Latest pandas
        "numpy>=1.24.0",                 # Latest numpy
        "scikit-learn>=1.3.0",           # Latest sklearn
        "matplotlib>=3.7.0",             # Latest matplotlib
        "keras-tuner>=1.4.0",            # For hyperparameter tuning
        "seaborn>=0.12.0",               # For better plotting
        "yfinance>=0.2.0",               # For data fetching
        "requests>=2.31.0",              # For API calls
    ],
    python_requires=">=3.9",
    author="Roy",
    description="AI-powered BSE/NSE stock price prediction platform",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
)
