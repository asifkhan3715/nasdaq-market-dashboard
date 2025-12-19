import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import pickle
import os
from scipy import stats
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
from tensorflow.keras.models import load_model

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Market Intelligence Dashboard",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. THEME
# ==========================================
primary_color = "#00CC96"
danger_color = "#EF553B"
background_color = "#0E1117"
card_background = "#161B22"
text_color = "#FAFAFA"
secondary_text = "#8B949E"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {background_color};
        color: {text_color};
    }}

    div[data-testid="stMetric"] {{
        background-color: {card_background};
        border: 1px solid #30363D;
        padding: 20px;
        border-radius: 10px;
    }}

    h1, h2, h3 {{
        color: {text_color};
    }}

    p {{
        color: {secondary_text};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📈 Market Intelligence Dashboard")
st.markdown("**Advanced Predictive Analytics & Forecasting**")
st.markdown("---")

# ==========================================
# 3. MODEL LOADING
# ==========================================
@st.cache_resource
def load_class_models():
    models = {}

    if os.path.exists("knn_scaler.pkl"):
        with open("knn_scaler.pkl", "rb") as f:
            models["knn_scaler"] = pickle.load(f)

    if os.path.exists("lstm_scaler.pkl"):
        with open("lstm_scaler.pkl", "rb") as f:
            models["lstm_scaler"] = pickle.load(f)

    file_map = {
        "Random Forest": "rf_classifier_model.pkl",
        "XGBoost": "xgb_classifier_model.pkl",
        "KNN": "knn_classifier_model.pkl",
        "SVM": "svm_classifier_model.pkl",
    }

    for name, file in file_map.items():
        if os.path.exists(file):
            with open(file, "rb") as f:
                models[name] = pickle.load(f)

    if os.path.exists("lstm_classifier_model.keras"):
        models["LSTM"] = load_model("lstm_classifier_model.keras", compile=False)

    return models


@st.cache_resource
def load_reg_models():
    models = {}

    if os.path.exists("scaler_reg.pkl"):
        with open("scaler_reg.pkl", "rb") as f:
            models["scaler"] = pickle.load(f)

    file_map = {
        "Linear Regression": "linear_reg.pkl",
        "Random Forest": "rf_reg.pkl",
        "SVR": "svr_reg.pkl",
    }

    for name, file in file_map.items():
        if os.path.exists(file):
            with open(file, "rb") as f:
                models[name] = pickle.load(f)

    if os.path.exists("lstm_reg.h5"):
        models["LSTM"] = load_model("lstm_reg.h5", compile=False)

    return models

# ==========================================
# 4. FEATURE ENGINEERING
# ==========================================
def engineer_features(df):
    df = df.copy()

    df["RSI"] = RSIIndicator(df["Close"], 14).rsi()
    macd = MACD(df["Close"])
    df["MACD_Diff"] = macd.macd_diff()
    df["SMA_50"] = SMAIndicator(df["Close"], 50).sma_indicator()
    df["Dist_from_SMA"] = df["Close"] - df["SMA_50"]

    bb = BollingerBands(df["Close"])
    df["BB_Width"] = bb.bollinger_wband()

    df["Smooth_Price"] = df["Close"].rolling(10).mean()
    df["Target_Class"] = (df["Smooth_Price"].shift(-1) > df["Smooth_Price"]).astype(int)
    df["Target_Reg"] = df["Close"].shift(-1)

    df.dropna(inplace=True)
    return df

# ==========================================
# 5. SIDEBAR
# ==========================================
st.sidebar.header("🛠️ Control Panel")

with st.sidebar.expander("📂 Data Configuration", expanded=True):
    data_source = st.selectbox("Data Source", ["Live Ticker", "Upload CSV"])

    if data_source == "Live Ticker":
        ticker = st.selectbox("Asset", ["^IXIC", "AAPL", "NVDA", "BTC-USD", "GC=F"])
        start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))

        if st.button("Fetch Market Data"):
            df = yf.download(ticker, start=start_date, progress=False)
            st.session_state["data"] = df
            st.session_state["ticker"] = ticker

    else:
        uploaded = st.file_uploader("Upload CSV", type="csv")
        if uploaded:
            df = pd.read_csv(uploaded, index_col=0, parse_dates=True)
            st.session_state["data"] = df
            st.session_state["ticker"] = "Custom CSV"

with st.sidebar.expander("🧠 Model Configuration", expanded=True):
    module = st.radio("Analysis Type", ["Classification (Trend)", "Regression (Price)"])
    test_size_pct = st.slider("Test Size (%)", 10, 50, 20, 5)

# ==========================================
# 6. MAIN LOGIC
# ==========================================
if "data" in st.session_state:
    df_raw = st.session_state["data"]
    df = engineer_features(df_raw)

    split = int(len(df) * (1 - test_size_pct / 100))
    train, test = df.iloc[:split], df.iloc[split:]

    st.metric("Current Price", f"{df['Close'].iloc[-1]:.2f}")

    feats_class = ["RSI", "MACD_Diff", "Dist_from_SMA", "BB_Width"]
    feats_reg = ["Close", "RSI", "MACD_Diff", "Dist_from_SMA", "BB_Width"]

    # ======================================
    # CLASSIFICATION
    # ======================================
    if module == "Classification (Trend)":
        st.header("🔮 Trend Prediction")
        models = load_class_models()
        choice = st.selectbox("Model", [k for k in models if "scaler" not in k])

        if st.button("Run Analysis"):
            X_test = test[feats_class]
            y_test = test["Target_Class"]

            if choice in ["KNN", "SVM"]:
                scaler = models.get("knn_scaler")
                if scaler:
                    X_test = scaler.transform(X_test)

            if choice == "LSTM":
                scaler = models.get("lstm_scaler")
                X_scaled = scaler.transform(X_test)
                X_test = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
                preds = (models[choice].predict(X_test) > 0.5).astype(int).flatten()
            else:
                preds = models[choice].predict(X_test)

            acc = accuracy_score(y_test, preds)
            st.success(f"Accuracy: {acc:.2%}")

            cm = confusion_matrix(y_test, preds)
            fig = ff.create_annotated_heatmap(
                cm,
                x=["Down", "Up"],
                y=["Down", "Up"],
                colorscale="Viridis",
            )
            st.plotly_chart(fig, use_container_width=True)

    # ======================================
    # REGRESSION
    # ======================================
    else:
        st.header("💲 Price Forecasting")
        models = load_reg_models()
        choice = st.selectbox("Model", [k for k in models if k != "scaler"])

        if st.button("Generate Forecast"):
            X_test = test[feats_reg]
            y_test = test["Target_Reg"]

            if choice != "Random Forest":
                scaler = models.get("scaler")
                if scaler:
                    X_test = scaler.transform(X_test)

            if choice == "LSTM":
                X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
                preds = models[choice].predict(X_test).flatten()
            else:
                preds = models[choice].predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, preds))
            st.success(f"RMSE: {rmse:.4f}")

else:
    st.info("👈 Load data from the sidebar to begin.")
