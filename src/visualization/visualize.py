import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import os
from matplotlib.dates import YearLocator, DateFormatter


class ModelVisualizer:
    def __init__(self):
        self.plots_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "plots"
        )
        os.makedirs(self.plots_dir, exist_ok=True)

        # Set style for all plots
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

    def plot_predictions(
        self, y_true, y_pred_linear, y_pred_kernel, dates, symbol, save=True
    ):
        """Plot actual vs predicted prices for both models"""
        plt.figure(figsize=(15, 8))
        plt.plot(dates, y_true, label="Actual", linewidth=2)
        plt.plot(
            dates, y_pred_linear, label="Linear Regression", linewidth=2, linestyle="--"
        )
        plt.plot(dates, y_pred_kernel, label="Kernel Ridge", linewidth=2, linestyle=":")

        plt.title(f"Stock Price Predictions for {symbol}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)

        # Format x-axis with dates
        ax = plt.gca()
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)

        # Adjust layout to prevent date labels from being cut off
        plt.tight_layout()

        if save:
            plt.savefig(
                os.path.join(self.plots_dir, f"{symbol}_predictions.png"),
                bbox_inches="tight",
                dpi=300,
            )
        plt.close()

    def plot_error_distribution(
        self, y_true, y_pred_linear, y_pred_kernel, symbol, save=True
    ):
        """Plot error distribution for both models"""
        errors_linear = y_true - y_pred_linear
        errors_kernel = y_true - y_pred_kernel

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        sns.histplot(errors_linear, kde=True)
        plt.title("Linear Regression Error Distribution")
        plt.xlabel("Error")

        plt.subplot(1, 2, 2)
        sns.histplot(errors_kernel, kde=True)
        plt.title("Kernel Ridge Error Distribution")
        plt.xlabel("Error")

        plt.tight_layout()

        if save:
            plt.savefig(
                os.path.join(self.plots_dir, f"{symbol}_error_distribution.png"),
                bbox_inches="tight",
                dpi=300,
            )
        plt.close()

    def plot_metrics_comparison(self, metrics_dict, save=True):
        """Plot comparison of metrics between models"""
        metrics_df = pd.DataFrame(metrics_dict).T

        plt.figure(figsize=(12, 6))
        metrics_df.plot(kind="bar")
        plt.title("Model Performance Metrics Comparison")
        plt.xlabel("Metric")
        plt.ylabel("Value")
        plt.legend(title="Model")
        plt.xticks(rotation=45)
        plt.grid(True, axis="y")

        if save:
            plt.savefig(
                os.path.join(self.plots_dir, "metrics_comparison.png"),
                bbox_inches="tight",
                dpi=300,
            )
        plt.close()

    def create_performance_report(self, metrics_dict, symbol):
        """Create a performance report comparing both models"""
        report = f"Performance Report for {symbol}\n"
        report += "=" * 50 + "\n\n"

        for metric, values in metrics_dict.items():
            report += f"{metric}:\n"
            report += f"  Linear Regression: {values['Linear']:.4f}\n"
            report += f"  Kernel Ridge: {values['Kernel']:.4f}\n"
            improvement = (
                (values["Kernel"] - values["Linear"]) / values["Linear"]
            ) * 100
            report += f"  Improvement: {improvement:.2f}%\n\n"

        report_path = os.path.join(self.plots_dir, f"{symbol}_performance_report.txt")
        with open(report_path, "w") as f:
            f.write(report)

        return report
