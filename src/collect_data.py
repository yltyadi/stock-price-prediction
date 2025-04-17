import yfinance as yf
from datetime import datetime, timedelta
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class StockDataCollector:
    def __init__(self, symbols, start_date=None, end_date=None):
        """A class used for collecting historical data for stocks for a given period of time"""
        self.symbols = symbols
        self.start_date = start_date or (
            datetime.now() - timedelta(days=365 * 5)
        ).strftime("%Y-%m-%d")
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.data_dir = os.path.join(PROJECT_ROOT, "data")
        os.makedirs(self.data_dir, exist_ok=True)

    def download_data(self):
        """Downloading historical data for multiple stocks"""
        for symbol in self.symbols:
            try:
                # Fetching data from yfinance
                stock = yf.Ticker(symbol)
                df = stock.history(start=self.start_date, end=self.end_date)
                df = df.dropna()
                df = df.reset_index()

                # Saving each stock's data to csv files
                output_path = os.path.join(self.data_dir, f"{symbol}_data.csv")
                df.to_csv(output_path, index=False)
                print(f"Successfully downloaded data for {symbol}")

            except Exception as e:
                print(f"Error downloading data for {symbol}: {str(e)}")
