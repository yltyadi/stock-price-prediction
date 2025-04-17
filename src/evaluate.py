import os
import pandas as pd
from datetime import datetime
import sys

from collect_data import StockDataCollector
from feature_engineering import FeatureEngineer, FUTURE_PERIOD
from train_models import StockPricePredictor
from visualize import ModelVisualizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)


def evaluate_stock(symbol):
    # Initialize components
    feature_engineer = FeatureEngineer()
    predictor = StockPricePredictor()
    visualizer = ModelVisualizer()

    # Load data
    data_path = os.path.join(PROJECT_ROOT, "data", f"{symbol}_data.csv")
    if not os.path.exists(data_path):
        print(f"Data not found for {symbol}. Downloading...")
        collector = StockDataCollector([symbol])
        collector.download_data()

    # Read and prepare data
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])

    # Feature engineering
    X, y = feature_engineer.prepare_features(df)
    dates = df["Date"].iloc[:-FUTURE_PERIOD]  # Adjust dates for the forecast period

    # Prepare data for training
    X_train, X_test, y_train, y_test = predictor.prepare_data(X, y)

    # Train and evaluate Linear Regression
    print(f"Training Linear Regression model for {symbol}...")
    linear_model = predictor.train_linear_regression(X_train, y_train)
    linear_metrics, y_pred_linear = predictor.evaluate_model(
        linear_model, X_test, y_test
    )

    # Train and evaluate Kernel Ridge Regression
    print(f"Training Kernel Ridge Regression model for {symbol}...")
    kernel_model, best_params = predictor.optimize_kernel_ridge(
        X_train, y_train, X_test, y_test
    )
    kernel_metrics, y_pred_kernel = predictor.evaluate_model(
        kernel_model, X_test, y_test
    )

    # Prepare metrics for visualization
    metrics_dict = {
        "MSE": {"Linear": linear_metrics["MSE"], "Kernel": kernel_metrics["MSE"]},
        "RMSE": {"Linear": linear_metrics["RMSE"], "Kernel": kernel_metrics["RMSE"]},
        "MAE": {"Linear": linear_metrics["MAE"], "Kernel": kernel_metrics["MAE"]},
        "R2": {"Linear": linear_metrics["R2"], "Kernel": kernel_metrics["R2"]},
    }

    # Create visualizations
    test_dates = dates[-len(y_test) :]
    visualizer.plot_predictions(
        y_test, y_pred_linear, y_pred_kernel, test_dates, symbol
    )
    visualizer.plot_error_distribution(y_test, y_pred_linear, y_pred_kernel, symbol)
    visualizer.plot_metrics_comparison(metrics_dict)

    # Generate performance report
    report = visualizer.create_performance_report(metrics_dict, symbol)
    print("\nPerformance Report:")
    print(report)

    return metrics_dict, best_params


def main():
    # folder setup
    os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "plots"), exist_ok=True)

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
    # collecting stock data
    collector = StockDataCollector(symbols)
    collector.download_data()

    results = {}
    for symbol in symbols:
        print(f"\nEvaluating {symbol}...")
        try:
            metrics, params = evaluate_stock(symbol)
            results[symbol] = {"metrics": metrics, "best_params": params}
        except Exception as e:
            print(f"Error evaluating {symbol}: {str(e)}")

    # Save overall results
    results_df = pd.DataFrame()
    for symbol, data in results.items():
        metrics = data["metrics"]
        row = {
            "Symbol": symbol,
            "Linear_RMSE": metrics["RMSE"]["Linear"],
            "Kernel_RMSE": metrics["RMSE"]["Kernel"],
            "Linear_R2": metrics["R2"]["Linear"],
            "Kernel_R2": metrics["R2"]["Kernel"],
        }
        results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)

    results_path = os.path.join(PROJECT_ROOT, "results", "overall_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nOverall results saved to {results_path}")


if __name__ == "__main__":
    main()
