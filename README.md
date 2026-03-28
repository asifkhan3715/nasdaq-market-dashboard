# NASDAQ Stock Market Analysis & Prediction Dashboard

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)](https://streamlit.io/)
[![Deployment](https://img.shields.io/badge/Deployed_on-Streamlit_Cloud-black)](https://nasdaq-market-dashboard-asif.streamlit.app/)

**Data Mining (DS-311) Semester Project** — *National University of Sciences & Technology (NUST)*

## Overview

This project is a comprehensive machine learning and quantitative analysis dashboard designed to analyze and predict the movement of NASDAQ (`^IXIC`) and other global assets. Built with Streamlit, the application leverages historical financial data and advanced feature engineering to perform two primary predictive tasks:

1. **Classification (Trend Prediction):** Forecasts whether the asset's closing price will be higher (`1`) or lower (`0`) on the subsequent trading day.
2. **Regression (Price Forecasting):** Predicts the exact continuous numerical closing price of the asset for the next day.

**[View Live Application →](https://nasdaq-market-dashboard-asif.streamlit.app/)**

---

## Key Features

- **Live Market Integration:** Dynamically fetches real-time OHLCV data from Yahoo Finance via the `yfinance` API for any valid ticker symbol (e.g., `AAPL`, `BTC-USD`, `^IXIC`).
- **Modular Architecture:** Seamlessly toggle between Classification and Regression prediction environments within a unified UI.
- **Automated Feature Engineering:** Calculates advanced technical indicators on the fly, including RSI, MACD, Bollinger Bands, and Simple Moving Averages.
- **Interactive Visualizations:** Utilizes Plotly for highly interactive time-series visualizations, allowing granular zooming, hovering, and trend analysis.
- **Custom Dataset Support:** Includes a data ingestion module for users to upload and analyze proprietary CSV datasets.

---

## Technical Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Frontend | Streamlit |
| Data Processing | Pandas, NumPy, Yfinance, TA-Lib (`ta`) |
| Machine Learning | Scikit-Learn, XGBoost |
| Deep Learning | TensorFlow / Keras |
| Visualization | Plotly, Matplotlib, Seaborn |

---

## Project Architecture
```text
nasdaq-market-dashboard/
│
├── app.py                           # Main Streamlit application source
├── requirements.txt                 # Production dependencies
├── README.md                        # Project documentation
│
├── dm_project_classification.ipynb  # Notebook: Classifier training & evaluation
├── dm_project_regression.ipynb      # Notebook: Regressor training & evaluation
│
├── saved_models_classification/     # Serialized classification models
│   ├── rf_classifier_model.pkl
│   ├── xgb_classifier_model.pkl
│   ├── knn_classifier_model.pkl
│   ├── svm_classifier_model.pkl
│   ├── lstm_classifier_model.keras
│   ├── knn_scaler.pkl
│   └── lstm_scaler.pkl
│
└── saved_models_regression/         # Serialized regression models
    ├── linear_reg.pkl
    ├── rf_reg.pkl
    ├── svr_reg.pkl
    ├── lstm_reg.h5
    └── scaler_reg.pkl
```

---

## Machine Learning Models

### 1. Classification Module — Trend Prediction

| Model | Description | Primary Use Case |
|---|---|---|
| **Random Forest** | Ensemble of decision trees | Robust baseline for non-linear trend classification |
| **XGBoost** | Gradient boosting framework | High-performance accuracy on structured tabular financial data |
| **KNN** | K-Nearest Neighbors | Distance-based classification using scaled feature spaces |
| **SVM** | Support Vector Machine | Defining complex decision boundaries in high-dimensional space |
| **LSTM** | Long Short-Term Memory RNN | Deep learning for capturing sequential time-series patterns |

### 2. Regression Module — Price Forecasting

| Model | Description | Primary Use Case |
|---|---|---|
| **Linear Regression** | Simple linear approach | Baseline metric for general price trend direction |
| **Random Forest Reg** | Non-linear ensemble | Capturing complex feature interactions without heavy scaling |
| **SVR** | Support Vector Regression | Robust continuous forecasting via hyperplanes |
| **LSTM Regressor** | Deep neural network | Predicting exact numerical values from temporal sequences |

---

## Feature Engineering Pipeline

The system automatically extracts and engineers the following quantitative features from raw OHLC data before inference:

- **RSI** *(Relative Strength Index)* — Quantifies momentum to identify overbought or oversold conditions.
- **MACD** *(Moving Average Convergence Divergence)* — Trend-following momentum indicator capturing the relationship between two moving averages.
- **Bollinger Bands Width** — Measures market volatility and standard deviation from the mean.
- **Distance from SMA 50** — Calculates the percentage deviation from the 50-day Simple Moving Average.
- **Smooth Price** — A 10-day rolling mean used to define the ground-truth trend, smoothing out daily noise.

---

## Local Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/asifkhan3715/nasdaq-market-dashboard.git
cd nasdaq-market-dashboard
```

### 2. Create a Virtual Environment

> Highly recommended to prevent dependency conflicts with system packages.

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```

The dashboard will automatically launch in your default web browser at `http://localhost:8501`.

---

## Deployment Notes

This application is actively deployed on **Streamlit Community Cloud**.

- **App Hibernation:** If the application receives no traffic for a few consecutive days, it will automatically enter hibernation mode. Visiting the link will wake it up, though the initial spin-up may take 1–2 minutes.
- **Resource Limits:** Streamlit Cloud provides roughly 1 GB of RAM. To prevent memory exhaustion when running deep learning models (LSTMs), the application loads model artifacts dynamically into memory only upon user execution, rather than globally at startup.

---

## Contributors

- **Asif Khan**
- **Muhammad Ahmad**
