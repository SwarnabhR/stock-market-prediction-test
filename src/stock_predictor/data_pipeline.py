import logging, yaml, time
import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import Dict, List
import pytz

class StockDataPipeline:
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the StockDataPipeline with configuration and logging setup

        Args:
            config_path (str, optional): Path to the YAML configuration. Defaults to "/config/config.yaml".
        """
        # Load configuration from YAML file
        self.config_path = Path(config_path)
        try:
            with open(self.config_path, 'r') as config_file:
                self.config = yaml.safe_load(config_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
        
        # Set up logging configuration
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO').upper())
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file = log_config.get('file', 'logs/data_pipeline.log')
        
        # Create logs directory if it doesn't exist
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(log_format)
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(log_level)
        self.logger.info("StockDataPipeline initialized successfully")
        
        # Initialize API clients based on configuration
        self._setup_api_clients()
        
        # Create data directories if they don't exist
        self._setup_data_directories()

    def _setup_api_clients(self):
        """Set up API clients for data sources."""
        try:
            # Yahoo Finance configuration
            yahoo_config = self.config.get('apis', {}).get('yahoo_finance', {})
            self.yahoo_timeout = yahoo_config.get('timeout', 30)
            self.yahoo_retry_attempts = yahoo_config.get('retry_attempts', 3)
            
            self.logger.info("API clients setup completed")
        except Exception as e:
            self.logger.error(f"Error setting up API clients: {e}")
            raise
    
    def _setup_data_directories(self):
        """Create necessary data directories."""
        try:
            storage_config = self.config.get('storage', {})
            self.raw_data_path = Path(storage_config.get('raw_data_path', 'data/raw'))
            self.processed_data_path = Path(storage_config.get('processed_data_path', 'data/processed'))
            
            # Create directories
            self.raw_data_path.mkdir(parents=True, exist_ok=True)
            self.processed_data_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Data directories created: {self.raw_data_path}, {self.processed_data_path}")
        except Exception as e:
            self.logger.error(f"Error setting up data directories: {e}")
            raise
        
    
    def fetch_yahoo_data(self, symbol: str, period: str) -> pd.DataFrame:
        """
        Fetch historical stock data from Yahoo Finance using yfinance for BSE/NSE stocks.

        Args:
            symbol (str): Indian stock symbol (e.g., 'RELIANCE.NS', 'TCS.BO')
            period (str): Time period for data.
            Valid periods: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'

        Returns:
            pd.DataFrame: DataFrame with Indian market data [Open, High, Low, Close, Volume]
            
        Raises:
            ValueError: If invalid symbol or period provided
            Exception: If data retrieval fails after retries
        """
        self.logger.info(f"Fetching BSE/NSE data for {symbol} with period {period}")
        
        # Validate Indian stock symbol format
        if not (symbol.endswith('.NS') or symbol.endswith('.BO') or symbol.startswith('^')):
            raise ValueError(f"Invalid Indian stock symbol: {symbol}. Use .NS for NSE, .BO for BSE, or ^ for indices")
        
        # Validate inputs
        valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
        if period not in valid_periods:
            raise ValueError(f"Invalid period: {period}. Valid periods: {valid_periods}")
        
        # Retry logic for handling rate limits and network issues
        retry_attempts = self.yahoo_retry_attempts
        last_exception = None
        
        for attempt in range(retry_attempts):
            try:
                self.logger.info(f"Attempt {attempt + 1}/{retry_attempts} for {symbol}")
                
                # Create yfinance Ticker object
                ticker = yf.Ticker(symbol)
                
                # Fetch historical data with timeout
                data = ticker.history(
                    period = period,
                    timeout = self.yahoo_timeout
                )
                
                # Validate data quality
                if data.empty:
                    raise ValueError(f"No data found for symbol {symbol}")
                
                if not self.validate_data(data):
                    raise ValueError(f"Data validation failed for {symbol}")
                
                # Clean and prepare data 
                cleaned_data = self._clean_yahoo_data(data, symbol)
                self.logger.info(f"Successfully fetched {len(cleaned_data)} records for {symbol}")
                return cleaned_data
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {str(e)}")
                
                if attempt < retry_attempts - 1: # Don't sleep on last attempt
                    # Exponential backoff for rate limiting issues 
                    sleep_time = min(2 ** attempt, 30) # Cap at 30 seconds
                    self.logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
        # All retry attempts failed
        self.logger.error(f"Failed to fetch data for {symbol} after {retry_attempts} attempts")
        raise Exception(f"Failed to fetch Yahoo Finance data for {symbol}: {last_exception}")
    
    def _individual_fallback(self, symbols: List[str], period: str) -> Dict[str, pd.DataFrame]:
        """
        Fallback to individual symbol downloads with delays.
        """
        results = {}
        for symbol in symbols:
            try:
                time.sleep(2)
                data = self.fetch_yahoo_data(symbol, period)
                results[symbol] = data
                self.save_data(data, symbol, "yahoo_finance")
                self.logger.info(f"Individual fallback completed: {symbol}")
            except Exception as e:
                self.logger.error(f"Individual fallback failed for {symbol}: {e}")
        return results
    
    def fetch_multiple_symbols(self, symbols: List[str], period: str = "1y", retry_depth: int = 0) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols with proper error handling.

        Args:
            symbols (List[str]): List of stock symbols.
            period (str, optional): Time period for data. Defaults to "1y".

        Returns:
            Dict[str, pd.DataFrame]: Dict mapping symbol to DataFrame
        """
        self.logger.info("Bulk download %d symbols", len(symbols))
        
        # Add rate limiting protection
        time.sleep(1)
        
        try:
            raw = yf.download(
                tickers=" ".join(symbols),
                period=period,
                group_by="ticker",
                threads=True,
                auto_adjust=False,
                progress=False
            )
        except Exception as e:
            self.logger.error(f"Bulk download failed: {e}")
            return self._individual_fallback(symbols, period)
        
        results, failures = {}, []
        
        for sym in symbols:
            try:
                df = raw[sym] if len(symbols) > 1 else raw
                df = df.dropna(how="all")
                if df.empty or not self.validate_data(df):
                    raise ValueError("empty or failed validation")
                cleaned = self._clean_yahoo_data(df, sym)
                results[sym] = cleaned
                self.save_data(cleaned, sym, "yahoo_finance")
                self.logger.info("Completed %s", sym)
            except Exception as e:
                failures.append((sym, str(e)))
                self.logger.error("Failed %s: %s", sym, e)
        self.logger.info("Finished: %d/%d succeeded", len(results),len(symbols))
        if failures and retry_depth < 2: # Max 2 retry levels
            self.logger.warning("Retrying %d failed symbols after 30s pause", len(failures))
            time.sleep(30)
            retry_syms = [s for s, _ in failures]
            retry_results = self.fetch_multiple_symbols(retry_syms, period, retry_depth + 1)
            results.update(retry_results)
        elif failures:
            self.logger.warning(f"Giving up on {len(failures)} symbols after max retries")
        return results
    
    def _clean_yahoo_data(self, data:pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Clean and standardize Yahoo Finance data.

        Args:
            data (pd.DataFrame): Raw data from yfinance
            symbol (str): Stock symbol for logging

        Returns:
            pd.DataFrame: Cleaned data with standardized format
        """
        try:
            # Make a copy to avoid modifying original data
            cleaned_data = data.copy()
            
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in cleaned_data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns for {symbol}: {missing_columns}")
            
            # Remove any rows with all NaN values
            cleaned_data = cleaned_data.dropna(how='all')
            
            # Forward fill missing values
            cleaned_data = cleaned_data.ffill()
            
            # Add symbol column for identification
            cleaned_data['Symbol'] = symbol
            cleaned_data['Source'] = 'yahoo_finance'
            
            # Round price columns to 2 decimal places (INR format)
            price_columns = ['Open', 'High', 'Low', 'Close']
            cleaned_data[price_columns] = cleaned_data[price_columns].round(2)
            
            # Ensure volume is integer
            cleaned_data['Volume'] = cleaned_data['Volume'].astype(int)
            
            # Reset index to make Date a column instead of index
            cleaned_data = cleaned_data.reset_index()
            
            # Rename Date column for consistency
            if 'Date' in cleaned_data.columns:
                cleaned_data = cleaned_data.rename(columns={'Date': 'Datetime'})
            elif cleaned_data.index.name == 'Date':
                cleaned_data.index.name = 'Datetime'
                
            # BSE/NSE specific enhancements
            cleaned_data = self._add_indian_market_metadata(cleaned_data, symbol)
            
            self.logger.info(f"Data cleaned successfully for {symbol}")
            return cleaned_data
        except Exception as e:
            self.logger.error(f"Error cleaning data for {symbol}: {e}")
            raise
    
    def _add_indian_market_metadata(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Add Indian market specific metadata to the dataset.

        Args:
            data (pd.DataFrame): Cleaned market data
            symbol (str): Stock symbol

        Returns:
            pd.DataFrame: Data with Indian market metadata
        """
        try:
            # Add exchange information
            if symbol.endswith('.NS'):
                data['Exchange'] = 'NSE'
            elif symbol.endswith('.BO'):
                data['Exchange'] = 'BSE'
            elif symbol.startswith('^'):
                data['Exchange'] = 'INDEX'
            else:
                data['Exchange'] = 'UNKNOWN'
            
            # Add currency
            data['Currency'] = 'INR'
            
            # Convert timezone to IST if datetime column exists
            if 'Datetime' in data.columns:
                try:
                    import pytz
                    ist_tz = pytz.timezone('Asia/Kolkata')
                    
                    # If timezone naive, assume UTC then convert to IST
                    if data['Datetime'].dt.tz is None or str(data['Datetime'].dt.tz) == 'UTC':
                        data['Datetime'] = data['Datetime'].dt.tz_localize('UTC').dt.tz_convert(ist_tz)
                    else:
                        data['Datetime'] = data['Datetime'].dt.tz_convert(ist_tz)
                        
                    self.logger.info(f"Converted timestamps to IST for {symbol}")
                except Exception as tz_error:
                    self.logger.warning(f"Timezone conversion failed for {symbol}: {tz_error}")
            return data
        except Exception as e:
            self.logger.error(f"Error adding Indian market metadata for {symbol}: {e}")
            return data # Return original data if metadata addition fails
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Perform basic data validation checks on financial DataFrame.

        Args:
            df (pd.DataFrame): Dataframe

        Returns:
            bool: True if data passes validation, False otherwise
        """
        try:
            # Check if DataFrame is empty
            if df.empty:
                self.logger.warning("DataFrame is empty")
                return False
            # Check for required financial columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.warning(f"Missing required columnds: {missing_columns}")
                return False
            # Check for all NaN rows 
            if df[required_columns].isnull().all(axis=1).any():
                self.logger.warning("Found rows with all NaN values")
                return False
            # Validate price data (should be non-negative)
            price_columns = ['Open', 'High', 'Low', 'Close']
            for col in price_columns:
                if (df[col] < 0).any():
                    self.logger.warning(f"Found negative values in {col} column")
                    return False
            # Validate volume (should be non-negative integer)
            if (df['Volume'] < 0).any():
                self.logger.warning("Found negative volume values")
                return False
            # Cehck High >= Low logic
            if (df['High'] < df['Low']).any():
                self.logger.warning("Found High prices lower than Low prices")
                return False
            
            self.logger.info("Data validation passed")
            return True
        except Exception as e:
            self.logger.error(f"Error during data validation: {e}")
            return False
    
    def save_data(self, df: pd.DataFrame, symbol: str, source: str):
        """
        Save DataFrame to parquet format in organized structure.

        Args:
            df (pd.DataFrame): Data to save
            symbol (str): Stock symbol (e.g., 'RELIANCE.NS')
            source (str): Data Source ('yahoo_finance')
        """
        try:
            # Clean symbol for filename (remove .NS/.BO and special characters)
            clean_symbol = symbol.replace('.NS', '').replace('.BO', '').replace('^', 'INDEX_')
            
            # Create timestamp for filename
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Determine file path based on source
            if source == 'yahoo_finance':
                base_path = self.raw_data_path
            else:
                base_path = self.processed_data_path
            
            # Create source-specific subdirectory
            source_dir = base_path / source
            source_dir.mkdir(parents=True, exist_ok=True)
            
            # Create filename with symbol and timestamp 
            filename = f"{clean_symbol}_{timestamp}.parquet"
            file_path = source_dir / filename
            
            # Save to parquet format
            df.to_parquet(file_path, index=False)
            
            self.logger.info(f"Data saved successfully: {file_path}")
            self.logger.info(f"Saved {len(df)} records for {symbol}")
            return file_path
        except Exception as e:
            self.logger.error(f"Error saving data for {symbol}: {e}")
            raise