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
primary_color = "#00CC96"      # Mint Green (Success/Up)
danger_color = "#EF553B"       # Red (Down/Danger)
background_color = "#0E1117"   # Deep Dark Background
card_background = "#161B22"    # Card Surface
text_color = "#FAFAFA"         # Primary Text
secondary_text = "#8B949E"     # Subtitles

# --- CUSTOM CSS INJECTION ---
st.markdown(f"""
    <style>
    /* Global Styles */
    .stApp {{
        background-color: {background_color};
        color: {text_color};
    }}
    
    /* Metric Cards - Professional Look */
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
    
    /* Headers & Text */
    h1, h2, h3 {{
        color: {text_color} !important;
        font-family: 'Inter', sans-serif;
    }}
    p {{
        color: {secondary_text};
    }}
    
    /* Plots Background */
    .js-plotly-plot .plotly .main-svg {{
        background-color: rgba(0,0,0,0) !important;
    }}
    </style>
    """, unsafe_allow_html=True)

st.title("📈 Market Intelligence Dashboard")
st.markdown(f"**Advanced Predictive Analytics & Forecasting**")
st.markdown("---")

# ==========================================
# 2. MODEL LOADING (UPDATED: FLAT DIRECTORY)
# ==========================================
@st.cache_resource
def load_class_models():
    models = {}
    # No path prefix needed since files are in the same directory
    
    if os.path.exists('knn_scaler.pkl'):
        with open('knn_scaler.pkl', 'rb') as f: models['knn_scaler'] = pickle.load(f)
    if os.path.exists('lstm_scaler.pkl'):
        with open('lstm_scaler.pkl', 'rb') as f: models['lstm_scaler'] = pickle.load(f)

    file_map = {
        'Random Forest': 'rf_classifier_model.pkl',
        'XGBoost': 'xgb_classifier_model.pkl',
        'KNN': 'knn_classifier_model.pkl',
        'SVM': 'svm_classifier_model.pkl'
    }
    for name, filename in file_map.items():
        if os.path.exists(filename):
            with open(filename, 'rb') as f: models[name] = pickle.load(f)
    
    if os.path.exists('lstm_classifier_model.keras'):
        models['LSTM'] = load_model('lstm_classifier_model.keras', compile=False)
    elif os.path.exists('lstm_model.h5'):
        models['LSTM'] = load_model('lstm_model.h5', compile=False)
        
    return models

@st.cache_resource
def load_reg_models():
    models = {}
    
    if os.path.exists('scaler_reg.pkl'):
        with open('scaler_reg.pkl', 'rb') as f: models['scaler'] = pickle.load(f)

    file_map = {
        'Linear Regression': 'linear_reg.pkl',
        'Random Forest': 'rf_reg.pkl',
        'SVR': 'svr_reg.pkl'
    }
    for name, filename in file_map.items():
        if os.path.exists(filename):
            with open(filename, 'rb') as f: models[name] = pickle.load(f)
    
    if os.path.exists('lstm_reg.h5'):
        models['LSTM'] = load_model('lstm_reg.h5', compile=False)
        
    return models

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
def engineer_features(df):
    df = df.copy()
    # Indicators
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    macd = MACD(close=df['Close'])
    df['MACD_Diff'] = macd.macd_diff()
    df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
    df['Dist_from_SMA'] = df['Close'] - df['SMA_50']
    bb = BollingerBands(close=df['Close'], window=20)
    df['BB_Width'] = bb.bollinger_wband()
    
    # Targets
    df['Smooth_Price'] = df['Close'].rolling(window=10).mean()
    df['Target_Class'] = (df['Smooth_Price'].shift(-1) > df['Smooth_Price']).astype(int)
    df['Target_Reg'] = df['Close'].shift(-1)
    
    df.dropna(inplace=True)
    return df

# ==========================================
# 4. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("🛠️ Control Panel")

# A. Data Source
with st.sidebar.expander("📂 Data Configuration", expanded=True):
    data_source = st.selectbox("Data Source:", ["Live Ticker", "Upload CSV"])
    df_input = None

    if data_source == "Live Ticker":
        # Editable Selectbox for quick access
        ticker = st.selectbox("Select Asset:", ["^IXIC", "AAPL", "NVDA", "BTC-USD", "GC=F"], index=0)
        start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
        if st.button("Fetch Market Data", use_container_width=True):
            with st.spinner("Connecting to Global Markets..."):
                df_input = yf.download(ticker, start=start_date, progress=False)
                if isinstance(df_input.columns, pd.MultiIndex):
                    df_input = df_input.xs(ticker, level=1, axis=1)
                st.session_state['data'] = df_input
                st.session_state['ticker'] = ticker

    elif data_source == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded:
            df_input = pd.read_csv(uploaded, index_col=0, parse_dates=True)
            st.session_state['data'] = df_input
            st.session_state['ticker'] = "Custom CSV"

# B. Model Settings
with st.sidebar.expander("🧠 Model Configuration", expanded=True):
    module = st.radio("Select Analysis Type:", ["Classification (Trend)", "Regression (Price)"])
    st.markdown("### Test Set Split")
    test_size_pct = st.slider("Percentage of data to test:", 10, 50, 20, 5)

# ==========================================
# 5. MAIN DASHBOARD LOGIC
# ==========================================
if 'data' in st.session_state:
    df_raw = st.session_state['data']
    
    try:
        df_clean = engineer_features(df_raw)
    except Exception as e:
        st.error(f"Data Error: {e}")
        st.stop()

    # --- DATA SPLIT & OVERVIEW ---
    split_idx = int(len(df_clean) * (1 - test_size_pct/100))
    train_data = df_clean.iloc[:split_idx]
    test_data = df_clean.iloc[split_idx:]
    
    current_price = df_clean['Close'].iloc[-1]
    prev_price = df_clean['Close'].iloc[-2]
    delta = current_price - prev_price
    
    # METRICS
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Asset", st.session_state.get('ticker', 'Unknown'))
    col2.metric("Current Price", f"{current_price:,.2f}", f"{delta:+.2f}")
    col3.metric("Total History", f"{len(df_clean):,} Days")
    col4.metric("Test Data Size", f"{len(test_data):,} Days")

    st.markdown("---")

    # --- INTERACTIVE DATA OVERVIEW PLOT ---
    st.subheader("📉 Market Overview & Data Split")
    fig_split = go.Figure()
    fig_split.add_trace(go.Scatter(x=train_data.index, y=train_data['Close'], 
                                   mode='lines', name='Training Context', 
                                   line=dict(color='#4B5563', width=1)))
    fig_split.add_trace(go.Scatter(x=test_data.index, y=test_data['Close'], 
                                   mode='lines', name='Testing Data', 
                                   line=dict(color=primary_color, width=2)))
    
    fig_split.update_layout(
        plot_bgcolor=background_color,
        paper_bgcolor=background_color,
        font_color=text_color,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='#30363D'),
        hovermode="x unified",
        margin=dict(l=0, r=0, t=30, b=0),
        height=400,
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center")
    )
    st.plotly_chart(fig_split, use_container_width=True)

    # ==========================================
    # MODULES
    # ==========================================
    feats_class = ['RSI', 'MACD_Diff', 'Dist_from_SMA', 'BB_Width']
    feats_reg = ['Close', 'RSI', 'MACD_Diff', 'Dist_from_SMA', 'BB_Width']

    # ---------------- CLASSIFICATION ----------------
    if module == "Classification (Trend)":
        st.header("🔮 Trend Prediction Analysis")
        models = load_class_models()
        choice = st.selectbox("Choose AI Model:", [k for k in models.keys() if 'scaler' not in k])
        
        if st.button("🚀 Run Analysis", type="primary"):
            with st.spinner(f"Running {choice}..."):
                model = models[choice]
                X_test = test_data[feats_class]
                y_test = test_data['Target_Class']
                
                # Scaling
                if choice in ['KNN', 'SVM']:
                    scaler = models.get('knn_scaler')
                    if scaler: X_test = scaler.transform(X_test)
                elif choice == 'LSTM':
                    scaler = models.get('lstm_scaler')
                    if scaler: 
                        X_scaled = scaler.transform(X_test)
                        X_test = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
                
                # Predict
                if choice == 'LSTM':
                    preds_prob = model.predict(X_test)
                    preds = (preds_prob > 0.5).astype(int).flatten()
                else:
                    preds = model.predict(X_test)
                
                # Accuracy Metric
                acc = accuracy_score(y_test, preds)
                st.success(f"Model Accuracy on Unseen Data: {acc:.2%}")

                # --- PLOT 1: Confusion Matrix Heatmap ---
                c1, c2 = st.columns(2)
                
                with c1:
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test, preds)
                    # Use text heatmap
                    z = cm
                    x = ['Predicted Down', 'Predicted Up']
                    y = ['Actual Down', 'Actual Up']
                    
                    fig_cm = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Viridis')
                    fig_cm.update_layout(paper_bgcolor=background_color, font={'color':text_color})
                    st.plotly_chart(fig_cm, use_container_width=True)

               # --- PLOT 2: Market Signal Overlay (Replaces Bar Chart) ---
                with c2:
                    st.subheader("Model Signal Map (Last 60 Days)")
                    
                    # Prepare data for the last 60 days
                    lookback = 60
                    recent_test = test_data.tail(lookback).copy()
                    recent_preds = preds[-lookback:]
                    recent_actual = y_test.values[-lookback:]
                    
                    recent_test['Pred'] = recent_preds
                    recent_test['Actual'] = recent_actual
                    
                    # Define Signal Logic
                    # 1 = Up, 0 = Down
                    conditions = [
                        (recent_test['Pred'] == 1) & (recent_test['Actual'] == 1), # Correct Up
                        (recent_test['Pred'] == 0) & (recent_test['Actual'] == 0), # Correct Down
                        (recent_test['Pred'] != recent_test['Actual'])             # Mismatch
                    ]
                    choices = ['Correct Up-Trend', 'Correct Down-Trend', 'False Signal']
                    recent_test['Signal'] = np.select(conditions, choices, default='Neutral')

                    # Create Figure
                    fig_signal = go.Figure()

                    # 1. The Price Line
                    fig_signal.add_trace(go.Scatter(
                        x=recent_test.index, y=recent_test['Close'],
                        mode='lines',
                        line=dict(color='#4B5563', width=1),
                        name='Price',
                        showlegend=False
                    ))

                    # 2. Add Markers for Signals
                    colors = {'Correct Up-Trend': '#00CC96', 'Correct Down-Trend': '#EF553B', 'False Signal': '#8B949E'}
                    symbols = {'Correct Up-Trend': 'triangle-up', 'Correct Down-Trend': 'triangle-down', 'False Signal': 'x'}

                    for s_type in choices:
                        mask = recent_test['Signal'] == s_type
                        fig_signal.add_trace(go.Scatter(
                            x=recent_test.index[mask],
                            y=recent_test['Close'][mask],
                            mode='markers',
                            name=s_type,
                            marker=dict(
                                color=colors[s_type],
                                size=10,
                                symbol=symbols[s_type],
                                line=dict(width=1, color='white')
                            )
                        ))

                    fig_signal.update_layout(
                        plot_bgcolor=background_color,
                        paper_bgcolor=background_color,
                        font_color=text_color,
                        margin=dict(l=0, r=0, t=30, b=0),
                        height=400,
                        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
                        xaxis=dict(showgrid=False),
                        yaxis=dict(gridcolor='#30363D', title="Price")
                    )
                    
                    st.plotly_chart(fig_signal, use_container_width=True)

                # --- DETAIL: Classification Report ---
                with st.expander("📄 View Detailed Classification Report"):
                    report = classification_report(y_test, preds, output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"))

    # ---------------- REGRESSION ----------------
    elif module == "Regression (Price)":
        st.header("💲 Price Forecasting Analysis")
        models = load_reg_models()
        choice = st.selectbox("Choose AI Model:", [k for k in models.keys() if 'scaler' not in k])
        
        if st.button("🚀 Generate Forecast", type="primary"):
            with st.spinner(f"Forecasting with {choice}..."):
                model = models[choice]
                X_test = test_data[feats_reg]
                y_test = test_data['Target_Reg']
                
                if choice != 'Random Forest':
                    scaler = models.get('scaler')
                    if scaler: X_test = scaler.transform(X_test)
                
                if choice == 'LSTM':
                    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
                    preds = model.predict(X_test).flatten()
                else:
                    preds = model.predict(X_test)
                
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                st.success(f"Model RMSE (Error Margin): {rmse:.4f}")

                # --- PLOT 1: Interactive Time Series ---
                st.subheader("Forecast vs Actual Market Price")
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=y_test.index, y=y_test, name='Actual Price', 
                                              line=dict(color='#888888', width=2)))
                fig_pred.add_trace(go.Scatter(x=y_test.index, y=preds, name='Predicted Price', 
                                              line=dict(color=primary_color, width=2, dash='dash')))
                
                fig_pred.update_layout(
                    hovermode="x unified",
                    plot_bgcolor=background_color,
                    paper_bgcolor=background_color,
                    font_color=text_color,
                    xaxis=dict(showgrid=False, rangeslider=dict(visible=True)), # Added Range Slider
                    yaxis=dict(gridcolor='#30363D')
                )
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # --- PLOT 2: Distribution Analysis (KDE Style) ---
                st.subheader("Price Distribution Analysis (KDE)")
                
                # Calculate KDE using Scipy
                kde_actual = stats.gaussian_kde(y_test)
                kde_pred = stats.gaussian_kde(preds)
                x_range = np.linspace(min(min(y_test), min(preds)), max(max(y_test), max(preds)), 200)
                
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Scatter(x=x_range, y=kde_actual(x_range), name='Actual Dist',
                                              fill='tozeroy', line=dict(color='#888888')))
                fig_dist.add_trace(go.Scatter(x=x_range, y=kde_pred(x_range), name='Predicted Dist',
                                              fill='tozeroy', line=dict(color=primary_color), opacity=0.6))
                
                fig_dist.update_layout(
                    plot_bgcolor=background_color,
                    paper_bgcolor=background_color,
                    font_color=text_color,
                    xaxis_title="Price",
                    yaxis_title="Density"
                )
                st.plotly_chart(fig_dist, use_container_width=True)

else:
    st.info("👈 Please initialize the dashboard via the sidebar.")
    st.markdown(f"""
    ### Welcome to Market Intelligence
    <p style='color:{secondary_text}'>
    1. Enter a Ticker (e.g., ^IXIC, AAPL)<br>
    2. Click <b>Fetch Market Data</b><br>
    3. Select your Analysis Module
    </p>
    """, unsafe_allow_html=True)
