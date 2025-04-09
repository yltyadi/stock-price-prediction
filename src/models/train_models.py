import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import os
import joblib


class StockPricePredictor:
    def __init__(self):
        self.linear_model = None
        self.kernel_model = None
        self.scaler = StandardScaler()
        self.models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models"
        )
        os.makedirs(self.models_dir, exist_ok=True)

    def prepare_data(self, X, y, test_size=0.2):
        """Prepare and split data for training"""
        X_scaled = self.scaler.fit_transform(X)
        return train_test_split(X_scaled, y, test_size=test_size, random_state=42)

    def train_linear_regression(self, X_train, y_train):
        """Train baseline Linear Regression model"""
        self.linear_model = LinearRegression()
        self.linear_model.fit(X_train, y_train)
        return self.linear_model

    def optimize_kernel_ridge(self, X_train, y_train, X_val, y_val):
        """Optimize Kernel Ridge Regression using Optuna"""

        def objective(trial):
            kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
            alpha = trial.suggest_loguniform("alpha", 1e-4, 1.0)
            gamma = (
                trial.suggest_loguniform("gamma", 1e-4, 1.0)
                if kernel in ["rbf", "poly"]
                else None
            )
            degree = trial.suggest_int("degree", 2, 5) if kernel == "poly" else None

            params = {"alpha": alpha}
            if gamma is not None:
                params["gamma"] = gamma
            if degree is not None:
                params["degree"] = degree

            model = KernelRidge(kernel=kernel, **params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            return mean_squared_error(y_val, y_pred)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50)

        best_params = study.best_params
        self.kernel_model = KernelRidge(
            kernel=best_params["kernel"], alpha=best_params["alpha"]
        )
        if "gamma" in best_params:
            self.kernel_model.gamma = best_params["gamma"]
        if "degree" in best_params:
            self.kernel_model.degree = best_params["degree"]

        self.kernel_model.fit(X_train, y_train)
        return self.kernel_model, best_params

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        metrics = {
            "MSE": mean_squared_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "MAE": mean_absolute_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred),
        }
        return metrics, y_pred

    def save_models(self, symbol):
        """Save trained models and scaler"""
        if self.linear_model is not None:
            joblib.dump(
                self.linear_model,
                os.path.join(self.models_dir, f"{symbol}_linear_model.joblib"),
            )
        if self.kernel_model is not None:
            joblib.dump(
                self.kernel_model,
                os.path.join(self.models_dir, f"{symbol}_kernel_model.joblib"),
            )
        joblib.dump(
            self.scaler, os.path.join(self.models_dir, f"{symbol}_scaler.joblib")
        )

    def load_models(self, symbol):
        """Load trained models and scaler"""
        self.linear_model = joblib.load(
            os.path.join(self.models_dir, f"{symbol}_linear_model.joblib")
        )
        self.kernel_model = joblib.load(
            os.path.join(self.models_dir, f"{symbol}_kernel_model.joblib")
        )
        self.scaler = joblib.load(
            os.path.join(self.models_dir, f"{symbol}_scaler.joblib")
        )
        return self.linear_model, self.kernel_model, self.scaler
