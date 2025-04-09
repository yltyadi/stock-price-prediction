import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os


class StockDataCollector:
    def __init__(self, symbols, start_date=None, end_date=None):
        self.symbols = symbols
        self.start_date = start_date or (
            datetime.now() - timedelta(days=365 * 5)
        ).strftime("%Y-%m-%d")
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"
        )
        os.makedirs(self.data_dir, exist_ok=True)

    def download_data(self):
        """Download historical data for multiple stocks"""
        for symbol in self.symbols:
            try:
                # Download data
                stock = yf.Ticker(symbol)
                df = stock.history(start=self.start_date, end=self.end_date)

                # Basic preprocessing
                df = df.dropna()
                df = df.reset_index()

                # Save to CSV
                output_path = os.path.join(self.data_dir, f"{symbol}_data.csv")
                df.to_csv(output_path, index=False)
                print(f"Successfully downloaded data for {symbol}")

            except Exception as e:
                print(f"Error downloading data for {symbol}: {str(e)}")


def main():
    # List of companies from different sectors
    symbols = [
        "AAPL",  # Technology
        "MSFT",  # Technology
        "JPM",  # Finance
        "BAC",  # Finance
        "JNJ",  # Healthcare
        "PFE",  # Healthcare
        "XOM",  # Energy
        "CVX",  # Energy
        "PG",  # Consumer Goods
        "KO",  # Consumer Goods
        "WMT",  # Retail
        "AMZN",  # Retail
        "BA",  # Industrial
        "GE",  # Industrial
        "VZ",  # Telecommunications
        "T",  # Telecommunications
        "HD",  # Home Improvement
        "LOW",  # Home Improvement
        "MCD",  # Restaurants
        "SBUX",  # Restaurants
    ]

    collector = StockDataCollector(symbols)
    collector.download_data()


if __name__ == "__main__":
    main()
