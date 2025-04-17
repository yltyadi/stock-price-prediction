import pandas as pd
import numpy as np

FUTURE_PERIOD = 5  # 5 days into the future for prediction


class FeatureEngineer:
    """A class used to transform raw data into features suitable for our ML model"""

    def __init__(self):
        pass

    def calculate_technical_indicators(self, df):
        """Calculating technical indicators for stock price prediction"""
        # Make a copy of the dataframe
        df = df.copy()

        # Simple Moving Averages
        df["SMA_5"] = df["Close"].rolling(window=5).mean()
        df["SMA_20"] = df["Close"].rolling(window=20).mean()
        df["SMA_50"] = df["Close"].rolling(window=50).mean()

        # Exponential Moving Averages
        df["EMA_5"] = df["Close"].ewm(span=5, adjust=False).mean()
        df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()

        # Relative Strength Index (RSI)
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # Moving Average Convergence Divergence (MACD)
        exp1 = df["Close"].ewm(span=12, adjust=False).mean()
        exp2 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        df["BB_middle"] = df["Close"].rolling(window=20).mean()
        df["BB_upper"] = df["BB_middle"] + 2 * df["Close"].rolling(window=20).std()
        df["BB_lower"] = df["BB_middle"] - 2 * df["Close"].rolling(window=20).std()

        # Price Rate of Change
        df["ROC"] = df["Close"].pct_change(periods=10) * 100

        # Average True Range (ATR)
        high_low = df["High"] - df["Low"]
        high_close = np.abs(df["High"] - df["Close"].shift())
        low_close = np.abs(df["Low"] - df["Close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df["ATR"] = true_range.rolling(14).mean()

        # Volume Features
        df["Volume_MA_5"] = df["Volume"].rolling(window=5).mean()
        df["Volume_MA_20"] = df["Volume"].rolling(window=20).mean()

        # Price Momentum
        df["Momentum"] = df["Close"] - df["Close"].shift(4)

        # Drop NaN values
        df = df.dropna()

        return df

    def create_target_variable(self, df, forecast_period=FUTURE_PERIOD):
        """Creating target variable for prediction, forecast_period is the prediction period in days"""
        df["Target"] = df["Close"].shift(-forecast_period)
        df = df.dropna()
        return df

    def prepare_features(self, df, forecast_period=FUTURE_PERIOD):
        """Prepare features for model training by using methods above"""
        df = self.calculate_technical_indicators(df)
        df = self.create_target_variable(df, forecast_period)

        # features for model training
        feature_columns = [
            "Close",
            "SMA_5",
            "SMA_20",
            "SMA_50",
            "EMA_5",
            "EMA_20",
            "RSI",
            "MACD",
            "Signal_Line",
            "BB_middle",
            "BB_upper",
            "BB_lower",
            "ROC",
            "ATR",
            "Volume_MA_5",
            "Volume_MA_20",
            "Momentum",
        ]

        X = df[feature_columns]
        y = df["Target"]

        return X, y
