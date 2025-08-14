import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import warnings 
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    A comprehensive data preprocessing class for Indian stock market analysis.
    
    This class handles the complete preprocessing pipeline for BSE/NSE stock data,
    including technical indicator calculation, feature engineering, and LSTM sequence
    preparation for machine learning models.
    
    Features:
    - Technical indicators (RSI, MACD, Bollinger Bands, Moving Averages)
    - Indian market-specific time features
    - LSTM sequence generation with proper scaling
    - Batch processing of multiple stock datasets
    - Production-ready error handling and logging
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize DataPreprocessor with configuration settings.

        Args:
            config_path (str): Path to YAML configuration file containing
                             storage paths and processing parameters.
                             Defaults to "config/config.yaml".
                             
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If configuration file is malformed
        """
        import yaml
        
        # Load configuration from YAML file
        with open(config_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)
        
        # Setup directory paths from configuration
        storage_config = self.config.get('storage', {})
        self.raw_data_path = Path(storage_config.get('raw_data_path', '../data/raw'))
        self.processed_data_path = Path(storage_config.get('processed_data_path', '../data/processed'))
        
        # Create processed data directory if it doesn't exist
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging for tracking preprocessing operations
        self.logger = logging.getLogger(__name__)
        self.logger.info("DataPreprocessor initialized successfully")

    def load_raw_data(self, file_path: Path) -> pd.DataFrame:
        """
        Load and prepare raw stock data from parquet file.

        Args:
            file_path (Path): Path to the parquet file containing raw stock data

        Returns:
            pd.DataFrame: Loaded and time-sorted dataframe with proper datetime index
            
        Raises:
            Exception: If file loading fails or data format is invalid
        """
        try:
            # Load data from parquet file
            df = pd.read_parquet(file_path)
            self.logger.info(f"Loaded {len(df)} rows from {file_path}")
            
            # Ensure Datetime column is properly formatted and sorted
            if 'Datetime' in df.columns:
                df['Datetime'] = pd.to_datetime(df['Datetime'])
                df = df.sort_values('Datetime').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data from {file_path}: {e}")
            raise
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators for Indian stock market analysis.
        
        This method adds the following technical indicators commonly used in 
        Indian stock market analysis:
        - Simple Moving Averages (SMA): 5, 10, 20, 50 periods
        - Exponential Moving Averages (EMA): 12, 26 periods  
        - MACD (Moving Average Convergence Divergence)
        - RSI (Relative Strength Index): 14-day standard
        - Bollinger Bands with position indicators
        - Price change and volatility metrics
        - Volume-based indicators

        Args:
            df (pd.DataFrame): Stock data with OHLCV columns (Open, High, Low, Close, Volume)

        Returns:
            pd.DataFrame: Original data with technical indicators added as new columns
            
        Raises:
            Exception: If technical indicator calculation fails
        """
        try:
            # Create copy to avoid modifying original data
            data = df.copy()
            
            # === MOVING AVERAGES ===
            # Simple Moving Averages - commonly used periods in Indian markets
            data['SMA_5'] = data['Close'].rolling(window=5).mean()    # Short-term trend
            data['SMA_10'] = data['Close'].rolling(window=10).mean()  # Short-term trend
            data['SMA_20'] = data['Close'].rolling(window=20).mean()  # Medium-term trend
            data['SMA_50'] = data['Close'].rolling(window=50).mean()  # Long-term trend
            
            # Exponential Moving Averages - gives more weight to recent prices
            data['EMA_12'] = data['Close'].ewm(span=12).mean()  # Fast EMA for MACD
            data['EMA_26'] = data['Close'].ewm(span=26).mean()  # Slow EMA for MACD
            
            # === MACD (Moving Average Convergence Divergence) ===
            # Primary MACD line: difference between fast and slow EMA
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            # Signal line: 9-period EMA of MACD line
            data['MACD_signal'] = data['MACD'].ewm(span=9).mean()
            # Histogram: difference between MACD and signal line
            data['MACD_Histogram'] = data['MACD'] - data['MACD_signal']
            
            # === RSI (Relative Strength Index) ===
            # 14-day RSI - measures overbought/oversold conditions
            data['RSI'] = self._calculate_rsi(data['Close'], period=14)
            
            # === BOLLINGER BANDS ===
            # Statistical bands around moving average for volatility analysis
            bb_data = self._calculate_bollinger_bands(data['Close'], period=20, std_dev=2)
            data = pd.concat([data, bb_data], axis=1)
            
            # === PRICE-BASED FEATURES ===
            # Daily price change percentage
            data['Price_Change'] = data['Close'].pct_change()
            # 5-day moving average of price changes (smoothed momentum)
            data['Price_Change_MA'] = data['Price_Change'].rolling(window=5).mean()
            # 20-day rolling volatility (standard deviation of returns)
            data['Volatility'] = data['Price_Change'].rolling(window=20).std()
            
            # === VOLUME INDICATORS ===
            # 10-day simple moving average of volume
            data['Volume_SMA'] = data['Volume'].rolling(window=10).mean()
            # Volume ratio: current volume vs average volume
            data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
            
            # === HIGH-LOW SPREAD ANALYSIS ===
            # Daily trading range as percentage of closing price
            data['HL_Spread'] = (data['High'] - data['Low']) / data['Close']
            # 5-day moving average of high-low spread
            data['HL_Spread_MA'] = data['HL_Spread'].rolling(window=5).mean()
            
            self.logger.info("Technical indicators calculated successfully")
            return data
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            raise
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI) for momentum analysis.
        
        RSI is a momentum oscillator that measures the speed and magnitude of 
        price changes. Values above 70 typically indicate overbought conditions,
        while values below 30 indicate oversold conditions.

        Args:
            prices (pd.Series): Series of closing prices
            period (int): Number of periods for RSI calculation. Defaults to 14.

        Returns:
            pd.Series: RSI values (0-100 scale)
        """
        # Calculate price differences (gains and losses)
        delta = prices.diff()
        
        # Separate gains and losses, replacing negative values with 0
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Calculate Relative Strength (RS) and RSI
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> pd.DataFrame:
        """
        Calculate Bollinger Bands for volatility and mean reversion analysis.
        
        Bollinger Bands consist of:
        - Middle Band: Simple Moving Average
        - Upper Band: Middle Band + (Standard Deviation × multiplier)
        - Lower Band: Middle Band - (Standard Deviation × multiplier)
        
        Args:
            prices (pd.Series): Series of closing prices
            period (int): Period for moving average calculation. Defaults to 20.
            std_dev (int): Standard deviation multiplier. Defaults to 2.

        Returns:
            pd.DataFrame: DataFrame with Bollinger Band indicators
        """
        # Calculate middle band (Simple Moving Average)
        sma = prices.rolling(window=period).mean()
        # Calculate rolling standard deviation
        std = prices.rolling(window=period).std()
        
        # Create Bollinger Bands DataFrame
        bb_data = pd.DataFrame({
            'BB_Upper': sma + (std * std_dev),      # Upper band
            'BB_Lower': sma - (std * std_dev),      # Lower band  
            'BB_Middle': sma,                        # Middle band (SMA)
            'BB_Width': (sma + (std * std_dev)) - (sma - (std * std_dev)),  # Band width
            'BB_Position': (prices - (sma - (std * std_dev))) / 
                          ((sma + (std * std_dev)) - (sma - (std * std_dev)))  # Price position within bands
        })
        
        return bb_data
    
    def create_sequences_for_lstm(self, df: pd.DataFrame, sequence_length: int = 60, 
                                  prediction_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create time series sequences for LSTM model training.
        
        This method prepares data for deep learning models by:
        1. Selecting relevant features for stock prediction
        2. Normalizing features using MinMaxScaler (0-1 range)
        3. Creating sliding window sequences for time series learning
        4. Preparing target values for supervised learning

        Args:
            df (pd.DataFrame): Preprocessed stock data with technical indicators
            sequence_length (int): Number of time steps to look back for each sequence.
                                 Defaults to 60 (approximately 3 months of daily data).
            prediction_horizon (int): Number of days ahead to predict. Defaults to 1.

        Returns:
            Tuple[np.ndarray, np.ndarray, MinMaxScaler]: 
                - X: Input sequences (samples, time_steps, features)
                - y: Target values (samples,)  
                - scaler: Fitted scaler for inverse transformation
                
        Raises:
            ValueError: If insufficient data for sequence creation
            Exception: If sequence generation fails
        """
        try:
            # Select features most relevant for LSTM stock prediction
            feature_columns = [
                # Basic OHLCV data
                'Open', 'High', 'Low', 'Close', 'Volume',
                # Moving averages for trend analysis
                'SMA_5', 'SMA_10', 'SMA_20', 
                # Technical indicators for momentum and volatility
                'RSI', 'MACD',
                # Price dynamics and volatility
                'Price_Change', 'Volatility', 
                # Volume analysis
                'Volume_Ratio', 
                # Trading range analysis
                'HL_Spread'
            ]
            
            # Filter to only include columns that exist in the dataframe
            available_features = [col for col in feature_columns if col in df.columns]
            
            # Remove rows with NaN values to ensure clean training data
            clean_data = df[available_features].dropna()
            
            # Validate sufficient data for sequence creation
            min_required_rows = sequence_length + prediction_horizon
            if len(clean_data) < min_required_rows:
                raise ValueError(f"Insufficient data for sequences. Need at least {min_required_rows} rows, got {len(clean_data)}")
            
            # Normalize features to 0-1 range (critical for LSTM performance)
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(clean_data)
            
            # Create sequences using sliding window approach
            X, y = [], []
            for i in range(sequence_length, len(scaled_data) - prediction_horizon + 1):
                # Input sequence: previous 'sequence_length' time steps
                X.append(scaled_data[i-sequence_length:i])
                # Target: Close price at prediction horizon (index 3 = Close column)
                y.append(scaled_data[i+prediction_horizon-1, 3])
            
            # Convert to numpy arrays for neural network compatibility
            X, y = np.array(X), np.array(y)
            
            self.logger.info(f"Created {len(X)} sequences for LSTM training")
            self.logger.info(f"Sequence shape: {X.shape}, Target shape: {y.shape}")
            
            return X, y, scaler
            
        except Exception as e:
            self.logger.error(f"Error creating LSTM sequences: {e}")
            raise
    
    def preprocess_for_ml(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Complete preprocessing pipeline for machine learning model training.
        
        This method orchestrates the full preprocessing workflow:
        1. Data sorting and cleaning
        2. Technical indicator calculation
        3. Missing value handling
        4. Indian market-specific feature engineering
        5. Final data validation and cleaning

        Args:
            df (pd.DataFrame): Raw stock data from data pipeline
            symbol (str): Stock symbol for logging and identification

        Returns:
            pd.DataFrame: ML-ready dataset with all features and proper formatting
            
        Raises:
            Exception: If any preprocessing step fails
        """
        try:
            self.logger.info(f"Starting preprocessing for {symbol}")
            
            # Ensure data is chronologically sorted for time series analysis
            df = df.sort_values('Datetime').reset_index(drop=True)
            
            # Calculate all technical indicators
            df_with_indicators = self.calculate_technical_indicators(df)
            
            # Handle missing values using forward fill then backward fill
            # Forward fill: use last known value (common in financial data)
            # Backward fill: handle any remaining NaN at beginning of series
            df_processed = df_with_indicators.fillna(method='ffill').fillna(method='bfill')
            
            # Add time-based features specific to Indian stock market
            if 'Datetime' in df_processed.columns:
                # Day of week (0=Monday, 6=Sunday) - captures weekly patterns
                df_processed['Day_of_Week'] = df_processed['Datetime'].dt.dayofweek
                # Month (1-12) - captures seasonal patterns
                df_processed['Month'] = df_processed['Datetime'].dt.month
                # Quarter (1-4) - captures quarterly reporting effects
                df_processed['Quarter'] = df_processed['Datetime'].dt.quarter
                
                # Indian market-specific calendar features
                # Month-end effect: increased trading activity at month end
                df_processed['Is_Month_End'] = df_processed['Datetime'].dt.is_month_end.astype(int)
                # Quarter-end effect: important for earnings and results
                df_processed['Is_Quarter_End'] = df_processed['Datetime'].dt.is_quarter_end.astype(int)
            
            # Final data cleaning: remove rows where essential indicators are missing
            # These are critical features that must be present for reliable predictions
            essential_cols = ['Close', 'SMA_20', 'RSI', 'MACD']
            available_essential = [col for col in essential_cols if col in df_processed.columns]
            df_processed = df_processed.dropna(subset=available_essential)
            
            self.logger.info(f"Preprocessing completed for {symbol}. Final dataset: {len(df_processed)} rows")
            
            return df_processed
            
        except Exception as e:
            self.logger.error(f"Error in ML preprocessing for {symbol}: {e}")
            raise
    
    def save_processed_data(self, df: pd.DataFrame, symbol: str, data_type: str = "processed") -> Path:
        """
        Save preprocessed data to parquet format with organized naming convention.
        
        Files are saved with timestamp to avoid conflicts and enable version tracking.
        Parquet format is used for efficient storage and fast loading of time series data.

        Args:
            df (pd.DataFrame): Processed data to save
            symbol (str): Stock symbol for filename identification
            data_type (str): Type identifier for filename. Defaults to "processed".

        Returns:
            Path: Full path where the data was saved
            
        Raises:
            Exception: If file saving fails
        """
        try:
            # Clean symbol for filename (remove exchange suffixes and special characters)
            clean_symbol = symbol.replace('.NS', '').replace('.BO', '').replace('^', 'INDEX_')
            
            # Create timestamp for unique filename
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{clean_symbol}_{data_type}_{timestamp}.parquet"
            file_path = self.processed_data_path / filename
            
            # Save to parquet format (efficient for time series data)
            df.to_parquet(file_path, index=False)
            
            self.logger.info(f"Processed data saved: {file_path}")
            self.logger.info(f"Data shape: {df.shape}")
            
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error saving processed data for {symbol}: {e}")
            raise
    
    def process_all_raw_data(self) -> List[Path]:
        """
        Batch process all raw stock data files and save processed versions.
        
        This method provides complete automation of the preprocessing pipeline:
        1. Discovers all raw parquet files in the yahoo_finance directory
        2. Processes each file through the complete ML preprocessing pipeline
        3. Saves processed data with consistent naming convention
        4. Provides comprehensive logging and error handling

        Returns:
            List[Path]: List of file paths where processed data was saved
            
        Raises:
            Exception: If batch processing fails
        """
        try:
            # Locate raw data directory
            raw_yahoo_path = self.raw_data_path / "yahoo_finance"
            processed_files = []
            
            # Validate that raw data directory exists
            if not raw_yahoo_path.exists():
                self.logger.warning(f"Raw data directory not found: {raw_yahoo_path}")
                return processed_files
            
            # Process each parquet file in the directory
            for raw_file in raw_yahoo_path.glob("*.parquet"):
                self.logger.info(f"Processing {raw_file.name}...")
                
                # Load raw stock data
                raw_df = self.load_raw_data(raw_file)
                
                # Extract symbol from data (prefer data column over filename)
                symbol = raw_df['Symbol'].iloc[0] if 'Symbol' in raw_df.columns else raw_file.stem
                
                # Run complete preprocessing pipeline
                processed_df = self.preprocess_for_ml(raw_df, symbol)
                
                # Save processed data with ML-ready tag
                saved_path = self.save_processed_data(processed_df, symbol, "ml_ready")
                processed_files.append(saved_path)
                
                self.logger.info(f"✅ Completed processing: {symbol}")
            
            # Log batch processing summary
            self.logger.info(f"Batch processing completed: {len(processed_files)} files processed successfully")
            
            return processed_files
            
        except Exception as e:
            self.logger.error(f"Error in batch processing: {e}")
            raise