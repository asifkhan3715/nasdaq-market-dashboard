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
# 1. PAGE CONFIGURATION & THEME
# ==========================================
st.set_page_config(
    page_title="Market Intelligence Dashboard",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Dark Mode Palette
primary_color = "#00CC96"
danger_color = "#EF553B"
background_color = "#0E1117"
card_background = "#161B22"
text_color = "#FAFAFA"
secondary_text = "#8B949E"

st.markdown(f"""
    <style>
    .stApp {{ background-color: {background_color}; color: {text_color}; }}
    div[data-testid="stMetric"] {{
        background-color: {card_background};
        border: 1px solid #30363D;
        padding: 20px;
        border-radius: 10px;
    }}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. SESSION STATE INITIALIZATION
# ==========================================
# This ensures data stays visible even when you click expanders/tabs
if 'data' not in st.session_state: st.session_state['data'] = None
if 'class_results' not in st.session_state: st.session_state['class_results'] = None
if 'reg_results' not in st.session_state: st.session_state['reg_results'] = None

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
@st.cache_resource
def load_class_models():
    models = {}
    if os.path.exists('knn_scaler.pkl'):
        with open('knn_scaler.pkl', 'rb') as f: models['knn_scaler'] = pickle.load(f)
    if os.path.exists('lstm_scaler.pkl'):
        with open('lstm_scaler.pkl', 'rb') as f: models['lstm_scaler'] = pickle.load(f)
    
    file_map = {'Random Forest': 'rf_classifier_model.pkl', 'XGBoost': 'xgb_classifier_model.pkl', 
                'KNN': 'knn_classifier_model.pkl', 'SVM': 'svm_classifier_model.pkl'}
    
    for name, filename in file_map.items():
        if os.path.exists(filename):
            with open(filename, 'rb') as f: models[name] = pickle.load(f)
    
    if os.path.exists('lstm_classifier_model.keras'):
        models['LSTM'] = load_model('lstm_classifier_model.keras', compile=False)
    return models

@st.cache_resource
def load_reg_models():
    models = {}
    if os.path.exists('scaler_reg.pkl'):
        with open('scaler_reg.pkl', 'rb') as f: models['scaler'] = pickle.load(f)
    file_map = {'Linear Regression': 'linear_reg.pkl', 'Random Forest': 'rf_reg.pkl', 'SVR': 'svr_reg.pkl'}
    for name, filename in file_map.items():
        if os.path.exists(filename):
            with open(filename, 'rb') as f: models[name] = pickle.load(f)
    if os.path.exists('lstm_reg.h5'):
        models['LSTM'] = load_model('lstm_reg.h5', compile=False)
    return models

def engineer_features(df):
    df = df.copy()
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    macd = MACD(close=df['Close'])
    df['MACD_Diff'] = macd.macd_diff()
    df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
    df['Dist_from_SMA'] = df['Close'] - df['SMA_50']
    df['BB_Width'] = BollingerBands(close=df['Close'], window=20).bollinger_wband()
    df['Smooth_Price'] = df['Close'].rolling(window=10).mean()
    df['Target_Class'] = (df['Smooth_Price'].shift(-1) > df['Smooth_Price']).astype(int)
    df['Target_Reg'] = df['Close'].shift(-1)
    return df.dropna()

# ==========================================
# 4. SIDEBAR & DATA FETCHING
# ==========================================
st.sidebar.header("🛠️ Control Panel")
with st.sidebar.expander("📂 Data Configuration", expanded=True):
    data_source = st.selectbox("Data Source:", ["Live Ticker", "Upload CSV"])
    if data_source == "Live Ticker":
        ticker = st.selectbox("Select Asset:", ["^IXIC", "AAPL", "NVDA", "BTC-USD", "GC=F"])
        start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
        if st.button("Fetch Market Data", use_container_width=True):
            df_input = yf.download(ticker, start=start_date, progress=False)
            if isinstance(df_input.columns, pd.MultiIndex): df_input = df_input.xs(ticker, level=1, axis=1)
            st.session_state['data'] = df_input
            st.session_state['ticker'] = ticker
            # Reset results when new data is loaded
            st.session_state['class_results'] = None
            st.session_state['reg_results'] = None

module = st.sidebar.radio("Select Analysis Type:", ["Classification (Trend)", "Regression (Price)"])
test_size_pct = st.sidebar.slider("Test Set Size %:", 10, 50, 20)

# ==========================================
# 5. MAIN DASHBOARD
# ==========================================
st.title("📈 Market Intelligence Dashboard")

if st.session_state['data'] is not None:
    df_clean = engineer_features(st.session_state['data'])
    split_idx = int(len(df_clean) * (1 - test_size_pct/100))
    train_data, test_data = df_clean.iloc[:split_idx], df_clean.iloc[split_idx:]

    # KPI Row
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Price", f"{df_clean['Close'].iloc[-1]:,.2f}")
    c2.metric("History Size", f"{len(df_clean)} Days")
    c3.metric("Test Period", f"{len(test_data)} Days")

    # --- CLASSIFICATION MODULE ---
    if module == "Classification (Trend)":
        st.header("🔮 Trend Prediction")
        models = load_class_models()
        choice = st.selectbox("Choose AI Model:", [k for k in models.keys() if 'scaler' not in k])
        
        if st.button("🚀 Run Analysis", type="primary"):
            with st.spinner("Processing..."):
                model = models[choice]
                X_test, y_test = test_data[['RSI', 'MACD_Diff', 'Dist_from_SMA', 'BB_Width']], test_data['Target_Class']
                
                # Scaling Logic
                if choice in ['KNN', 'SVM']:
                    scaler = models.get('knn_scaler')
                    if scaler: X_test = scaler.transform(X_test)
                elif choice == 'LSTM':
                    scaler = models.get('lstm_scaler')
                    if scaler: 
                        X_scaled = scaler.transform(X_test)
                        X_test = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

                # Inference
                preds = model.predict(X_test)
                if choice == 'LSTM': preds = (preds > 0.5).astype(int).flatten()
                
                # SAVE TO STATE
                st.session_state['class_results'] = {
                    'preds': preds, 
                    'y_test': y_test, 
                    'acc': accuracy_score(y_test, preds),
                    'model_name': choice
                }

        # PERSISTENT DISPLAY
        if st.session_state['class_results']:
            res = st.session_state['class_results']
            st.success(f"Model Accuracy ({res['model_name']}): {res['acc']:.2%}")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(res['y_test'], res['preds'])
                fig_cm = ff.create_annotated_heatmap(cm, x=['Pred Down', 'Pred Up'], y=['Actual Down', 'Actual Up'], colorscale='Viridis')
                st.plotly_chart(fig_cm, use_container_width=True)

            with col_b:
                st.subheader("Signal Map (Last 60 Days)")
                recent = test_data.tail(60).copy()
                recent['Pred'] = res['preds'][-60:]
                recent['Actual'] = res['y_test'].values[-60:]
                
                fig_sig = go.Figure()
                fig_sig.add_trace(go.Scatter(x=recent.index, y=recent['Close'], line=dict(color='#4B5563')))
                # Add logic for markers (Correct/False) here...
                st.plotly_chart(fig_sig, use_container_width=True)

            with st.expander("📄 View Detailed Classification Report"):
                report = classification_report(res['y_test'], res['preds'], output_dict=True)
                st.table(pd.DataFrame(report).transpose())

    # --- REGRESSION MODULE ---
    else:
        st.header("💲 Price Forecasting")
        reg_models = load_reg_models()
        choice = st.selectbox("Choose AI Model:", [k for k in reg_models.keys() if 'scaler' not in k])

        if st.button("🚀 Generate Forecast", type="primary"):
            model = reg_models[choice]
            feats = ['Close', 'RSI', 'MACD_Diff', 'Dist_from_SMA', 'BB_Width']
            X_test, y_test = test_data[feats], test_data['Target_Reg']
            
            if choice != 'Random Forest':
                scaler = reg_models.get('scaler')
                if scaler: X_test = scaler.transform(X_test)
            
            preds = model.predict(X_test).flatten()
            st.session_state['reg_results'] = {'preds': preds, 'y_test': y_test, 'rmse': np.sqrt(mean_squared_error(y_test, preds))}

        if st.session_state['reg_results']:
            res = st.session_state['reg_results']
            st.metric("Forecast RMSE", f"{res['rmse']:.4f}")
            
            fig_p = px.line(x=res['y_test'].index, y=res['y_test'], title="Actual vs Forecast")
            fig_p.add_scatter(x=res['y_test'].index, y=res['preds'], name="Predicted", line=dict(dash='dash'))
            st.plotly_chart(fig_p, use_container_width=True)

else:
    st.info("Please use the sidebar to load data.")
