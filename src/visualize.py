import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import os
from matplotlib.dates import YearLocator, DateFormatter
from sklearn.metrics import mean_squared_error, r2_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ModelVisualizer:
    def __init__(self):
        self.plots_dir = os.path.join(PROJECT_ROOT, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

        # Set style for all plots
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

    def plot_predictions(
        self, y_true, y_pred_linear, y_pred_kernel, dates, symbol, save=True
    ):
        """Plot actual vs predicted prices for both models"""
        plt.figure(figsize=(18, 9))

        # Calculate dynamic y-axis limits
        min_val = min(y_true.min(), y_pred_linear.min(), y_pred_kernel.min())
        max_val = max(y_true.max(), y_pred_linear.max(), y_pred_kernel.max())
        margin = (max_val - min_val) * 0.1

        # Calculate metrics
        rmse_lin = np.sqrt(mean_squared_error(y_true, y_pred_linear))
        rmse_krn = np.sqrt(mean_squared_error(y_true, y_pred_kernel))
        r2_lin = r2_score(y_true, y_pred_linear)
        r2_krn = r2_score(y_true, y_pred_kernel)

        plt.plot(dates, y_true, label="Actual", linewidth=2.5, color="#1f77b4")
        plt.plot(
            dates,
            y_pred_linear,
            label=f"Linear (RMSE: {rmse_lin:.2f}, R²: {r2_lin:.2f})",
            linewidth=2,
            linestyle="--",
            color="#ff7f0e",
        )
        plt.plot(
            dates,
            y_pred_kernel,
            label=f"Kernel (RMSE: {rmse_krn:.2f}, R²: {r2_krn:.2f})",
            linewidth=2,
            linestyle="dashdot",
            marker="o",
            markersize=4,
            color="#2ca02c",
        )

        plt.title(
            f"{symbol} Price Predictions\n{dates.iloc[0].date()} to {dates.iloc[-1].date()}"
        )
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Price (USD)", fontsize=12)
        plt.legend(fontsize=10, loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.ylim(min_val - margin, max_val + margin)

        # Add volatility indicator
        volatility = y_true.rolling(20).std().iloc[-1]
        plt.annotate(
            f"20D Vol: {volatility:.2f}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=10,
            color="#d62728",
        )

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

        for metric in ["MSE", "RMSE", "MAE"]:
            linear_val = metrics_dict[metric]["Linear"]
            kernel_val = metrics_dict[metric]["Kernel"]
            error_reduction = abs((kernel_val - linear_val) / linear_val * 100)

            report += f"{metric}:\n"
            report += f"  Linear Regression: {linear_val:.4f}\n"
            report += f"  Kernel Ridge: {kernel_val:.4f}\n"
            report += f"  Error Reduction: {error_reduction:.2f}%\n\n"

        # R² is handled differently as higher is better
        linear_r2 = metrics_dict["R2"]["Linear"]
        kernel_r2 = metrics_dict["R2"]["Kernel"]
        r2_improvement = (kernel_r2 - linear_r2) / linear_r2 * 100

        report += f"R2:\n"
        report += f"  Linear Regression: {linear_r2:.4f}\n"
        report += f"  Kernel Ridge: {kernel_r2:.4f}\n"
        report += f"  Accuracy Improvement: {r2_improvement:.2f}%\n\n"

        # Save the report to a file in the plots directory
        report_path = os.path.join(self.plots_dir, f"{symbol}_performance_report.txt")
        with open(report_path, "w") as f:
            f.write(report)

        return report
