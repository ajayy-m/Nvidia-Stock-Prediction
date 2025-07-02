# Run using: streamlit run app.py
import streamlit as st
st.set_page_config(layout="wide")

# --- Imports ---
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

# --- Sidebar ---
refresh_interval = st.sidebar.slider("Auto-Refresh Interval (sec)", 10, 300, 60)
st_autorefresh(interval=refresh_interval * 1000, key="refresh")

# --- Title ---
st.title("NVIDIA Stock Price Prediction ")
st.caption("Using Linear Regression on real-time data from Yahoo Finance")

# --- Data Fetching ---
def fetch_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)
    df = yf.download('NVDA', start=start_date, end=end_date)
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    return df

# --- Model Training ---
def train_and_predict(df):
    features = ['Close', 'SMA_10', 'SMA_30']
    X = df[features]
    y = df['Target']
    split_index = int(len(df) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return df.index[split_index:], y_test, y_pred

# --- Main Execution ---
df = fetch_data()
dates, actual, predicted = train_and_predict(df)

# --- Latest Table ---
st.subheader("Latest NVIDIA Data")
st.dataframe(df.tail(5), use_container_width=True)

# --- Live Animated Plot using Plotly ---
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=dates,
    y=actual,
    mode='lines+markers',
    name='Actual',
    line=dict(color='cyan')
))
fig.add_trace(go.Scatter(
    x=dates,
    y=predicted,
    mode='lines+markers',
    name='Predicted',
    line=dict(color='orange')
))

fig.update_layout(
    title="NVIDIA Stock Price Prediction ",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    template="plotly_white",
    height=500,
    transition=dict(duration=500, easing='cubic-in-out'),
    uirevision='static'
)

st.plotly_chart(fig, use_container_width=True)

# --- Live Price Metric ---
latest = df['Close'].iloc[-1].item()
prev = df['Close'].iloc[-6].item()  # Approx. 5 intervals ago
delta = latest - prev
delta_pct = (delta / prev) * 100

col1, col2, col3 = st.columns(3) 
col1.metric("Live Price", f"${latest:.2f}", delta=f"{delta_pct:.2f}%")
col2.metric(" Score", f"{r2_score(actual, predicted):.4f}")
col3.metric(" RMSE", f"{np.sqrt(mean_squared_error(actual, predicted)):.2f}")
