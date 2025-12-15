# NASDAQ Stock Market Analysis & Prediction
### Data Mining (DS-311) Semester Project
**National University of Sciences & Technology (NUST)**

## 📖 Project Overview
This project is a comprehensive Data Mining application designed to analyze and predict the movement of NASDAQ (`^IXIC`) and other global assets. It leverages historical financial data to perform two primary tasks:

1. **Classification (Trend Prediction):** Predicts whether the stock price will close **HIGHER (1)** or **LOWER (0)** on the next trading day.
2. **Regression (Price Forecasting):** Predicts the **exact closing price** of the asset for the next day.

The solution is deployed as an interactive **Streamlit Dashboard**, allowing users to test models on live market data dynamically, visualizing trends, prediction accuracy, and feature distributions.

## 🚀 Key Features
- **Plug-and-Play Dashboard:** A professional, user-friendly interface to switch between Classification and Regression modules seamlessly.
- **Live Data Integration:** Fetches real-time data from Yahoo Finance (`yfinance`) for any ticker symbol (e.g., `AAPL`, `BTC-USD`, `^IXIC`).
- **Custom CSV Support:** Allows users to upload their own datasets for analysis.
- **Advanced Technical Indicators:** Automatically calculates RSI, MACD, Bollinger Bands, and SMA on the fly during feature engineering.
- **Interactive Visualizations:** Powered by **Plotly** for zooming, hovering, and deep-dive analysis of time-series data.
- **Model Comparison:** Supports multiple algorithms including Random Forest, XGBoost, SVM, and LSTM (Deep Learning).

## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **Frontend:** Streamlit
- **Data Handling:** Pandas, NumPy, Yfinance
- **Machine Learning:** Scikit-Learn (Random Forest, SVM, KNN, Linear Regression, XGBoost)
- **Deep Learning:** TensorFlow/Keras (LSTM)
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Technical Analysis:** TA-Lib (via `ta` library)

## 📂 Project Structure
Ensure your project directory is organized as follows for the application to function correctly:

Data_Mining_Project/

│

├── app.py                          # Main Streamlit Application Source Code

├── requirements.txt                # List of Python dependencies

├── README.md                       # Project Documentation

│

├── dm_project_classification.ipynb # Notebook used to train/save classifiers

├── dm_project_regression.ipynb     # Notebook used to train/save regressors

│

├── saved_models_classification/    # Folder for Classification Artifacts

│   ├── rf_classifier_model.pkl

│   ├── xgb_classifier_model.pkl

│   ├── knn_classifier_model.pkl

│   ├── svm_classifier_model.pkl

│   ├── lstm_classifier_model.keras

│   ├── knn_scaler.pkl

│   └── lstm_scaler.pkl

│

└── saved_models_regression/        # Folder for Regression Artifacts

├── linear_reg.pkl

├── rf_reg.pkl

├── svr_reg.pkl

├── lstm_reg.h5

└── scaler_reg.pkl

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/nasdaq-market-dashboard.git
cd nasdaq-market-dashboard


2. Create a Virtual Environment (Recommended)
Isolating dependencies is highly recommended to avoid conflicts.



Linux/Mac:

python3 -m venv venv
source venv/bin/activate


Windows:

python -m venv venv
venv\Scripts\activate


3. Install Dependencies


pip install -r requirements.txt


🏃‍♂️ How to Run the App
Once the dependencies are installed and the virtual environment is active, run the following command:



streamlit run app.py


A new tab will automatically open in your default browser at http://localhost:8501.



🧠 Models Implemented


Classification Module


Model

Description

Use Case

Random Forest

Ensemble of decision trees.

Robust baseline for trend classification.

XGBoost

Gradient boosting framework.

High accuracy on structured tabular data.

KNN

K-Nearest Neighbors.

Distance-based classification (Requires Scaled Data).

SVM

Support Vector Machine.

Effective for defining decision boundaries in high-dimensional space.

LSTM

Long Short-Term Memory RNN.

Deep Learning for capturing sequential time-series patterns.



Regression Module


Model

Description

Use Case

Linear Regression

Simple linear approach.

Baseline for general price trend direction.

Random Forest Reg

Non-linear ensemble.

Captures complex interactions without heavy scaling.

SVR

Support Vector Regression.

Robust forecasting by finding a hyperplane in N-dimensional space.

LSTM Regressor

Deep Neural Network.

Predicts numerical values based on historical sequences.



📊 Feature Engineering
The application automatically engineers the following features from raw OHLC (Open, High, Low, Close) data:



RSI (Relative Strength Index): Measures momentum (Overbought/Oversold conditions).
MACD (Moving Average Convergence Divergence): Trend-following momentum indicator.
Bollinger Bands Width: Measures market volatility.
Distance from SMA 50: Measures deviation from the 50-day Simple Moving Average.
Smooth Price: A 10-day rolling mean used to define the "Ground Truth" trend.


👥 Contributors
Asif Khan 
Muhammad Ahmad


