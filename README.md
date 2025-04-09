# Stock Price Prediction using Linear and Kernel Ridge Regression

This project implements a machine learning system for predicting stock prices using Linear Regression as a baseline model and Kernel Ridge Regression for enhanced performance. The system analyzes historical stock data from multiple companies across different sectors to provide accurate price predictions.

## Features

- Data collection using yfinance for multiple companies
- Feature engineering with technical indicators
- Baseline Linear Regression model
- Advanced Kernel Ridge Regression model
- Hyperparameter optimization using GridSearchCV and Optuna
- Comprehensive model evaluation and visualization

## Project Structure

```
.
├── data/               # Stock data and processed datasets
├── models/             # Trained model files
├── notebooks/          # Jupyter notebooks for analysis
├── src/                # Source code
│   ├── data/           # Data collection and processing
│   ├── features/       # Feature engineering
│   ├── models/         # Model implementation
│   └── visualization/  # Plotting and visualization
├── tests/              # Unit tests
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd stock-price-prediction
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run:
```bash
sh start.sh
```

## Model Evaluation

The project evaluates models using the following metrics:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.