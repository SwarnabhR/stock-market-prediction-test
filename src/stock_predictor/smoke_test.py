from stock_predictor.data_pipeline import StockDataPipeline

def main() -> None:
    pipe = StockDataPipeline("config/config.yaml")
    
    symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
    pipe.fetch_multiple_symbols(symbols, period="6mo")

if __name__ == "__main__":
    main()