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
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, confusion_matrix
from tensorflow.keras.models import load_model

# ==========================================
# 1. PAGE CONFIGURATION & THEME ENGINE
# ==========================================
st.set_page_config(
    page_title="Market Intelligence Dashboard",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# --- 🎨 DARK MODE FINANCIAL PALETTE ---
primary_color = "#00CC96"
danger_color = "#EF553B"
background_color = "#0E1117"
card_background = "#161B22"
text_color = "#FAFAFA"
secondary_text = "#8B949E"

# --- CUSTOM CSS INJECTION ---
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
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }}

    div[data-testid="stMetricLabel"] > div {{
        color: {secondary_text} !important;
        font-size: 0.9rem !important;
        font-weight: 500;
    }}

    div[data-testid="stMetricValue"] > div {{
        color: {text_color} !important;
        font-size: 1.8rem !important;
        font-weight: 700;
    }}

    h1, h2, h3 {{
        color: {text_color} !important;
        font-family: 'Inter', sans-serif;
    }}

    p {{
        color: {secondary_text};
    }}

    .js-plotly-plot .plotly .main-svg {{
        background-color: rgba(0,0,0,0) !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📈 Market Intelligence Dashboard")
st.markdown("**Advanced Predictive Analytics & Forecasting**")
st.markdown("---")

# ==========================================
# 2. MODEL LOADING
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

    for name, filename in file_map.items():
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                models[name] = pickle.load(f)

    if os.path.exists("lstm_classifier_model.keras"):
        models["LSTM"] = load_model("lstm_classifier_model.keras", compile=False)
    elif os.path.exists("lstm_model.h5"):
        models["LSTM"] = load_model("lstm_model.h5", compile=False)

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

    for name, filename in file_map.items():
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                models[name] = pickle.load(f)

    if os.path.exists("lstm_reg.h5"):
        models["LSTM"] = load_model("lstm_reg.h5", compile=False)

    return models

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
def engineer_features(df):
    df = df.copy()

    df["RSI"] = RSIIndicator(close=df["Close"], window=14).rsi()
    macd = MACD(close=df["Close"])
    df["MACD_Diff"] = macd.macd_diff()
    df["SMA_50"] = SMAIndicator(close=df["Close"], window=50).sma_indicator()
    df["Dist_from_SMA"] = df["Close"] - df["SMA_50"]
    bb = BollingerBands(close=df["Close"], window=20)
    df["BB_Width"] = bb.bollinger_wband()

    df["Smooth_Price"] = df["Close"].rolling(window=10).mean()
    df["Target_Class"] = (df["Smooth_Price"].shift(-1) > df["Smooth_Price"]).astype(int)
    df["Target_Reg"] = df["Close"].shift(-1)

    df.dropna(inplace=True)
    return df

# ==========================================
# 4. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("🛠️ Control Panel")

with st.sidebar.expander("📂 Data Configuration", expanded=True):
    data_source = st.selectbox("Data Source:", ["Live Ticker", "Upload CSV"])
    df_input = None

    if data_source == "Live Ticker":
        ticker = st.selectbox("Select Asset:", ["^IXIC", "AAPL", "NVDA", "BTC-USD", "GC=F"], index=0)
        start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))

        if st.button("Fetch Market Data", use_container_width=True):
            with st.spinner("Connecting to Global Markets..."):
                df_input = yf.download(ticker, start=start_date, progress=False)

                if isinstance(df_input.columns, pd.MultiIndex):
                    df_input = df_input.xs(ticker, level=1, axis=1)

                st.session_state["data"] = df_input
                st.session_state["ticker"] = ticker

    elif data_source == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded:
            df_input = pd.read_csv(uploaded, index_col=0, parse_dates=True)
            st.session_state["data"] = df_input
            st.session_state["ticker"] = "Custom CSV"

with st.sidebar.expander("🧠 Model Configuration", expanded=True):
    module = st.radio("Select Analysis Type:", ["Classification (Trend)", "Regression (Price)"])
    st.markdown("### Test Set Split")
    test_size_pct = st.slider("Percentage of data to test:", 10, 50, 20, 5)

# ==========================================
# 5. MAIN DASHBOARD LOGIC
# ==========================================
if "data" in st.session_state:
    df_raw = st.session_state["data"]

    try:
        df_clean = engineer_features(df_raw)
    except Exception as e:
        st.error(f"Data Error: {e}")
        st.stop()

    split_idx = int(len(df_clean) * (1 - test_size_pct / 100))
    train_data = df_clean.iloc[:split_idx]
    test_data = df_clean.iloc[split_idx:]

    current_price = df_clean["Close"].iloc[-1]
    prev_price = df_clean["Close"].iloc[-2]
    delta = current_price - prev_price

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Asset", st.session_state.get("ticker", "Unknown"))
    col2.metric("Current Price", f"{current_price:,.2f}", f"{delta:+.2f}")
    col3.metric("Total History", f"{len(df_clean):,} Days")
    col4.metric("Test Data Size", f"{len(test_data):,} Days")

    st.markdown("---")

else:
    st.info("👈 Please initialize the dashboard via the sidebar.")
    st.markdown(
        f"""
        ### Welcome to Market Intelligence
        <p style='color:{secondary_text}'>
        1. Enter a Ticker (e.g., ^IXIC, AAPL)<br>
        2. Click <b>Fetch Market Data</b><br>
        3. Select your Analysis Module
        </p>
        """,
        unsafe_allow_html=True
    )
