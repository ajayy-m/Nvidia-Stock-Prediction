import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta
import time

# Function to fetch and prepare data
def fetch_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)
    df = yf.download('NVDA', start=start_date, end=end_date)
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    return df

# Function to train model and predict
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

    return df.index[split_index:], y_test, y_pred, model

# Streamlit UI
st.set_page_config(layout="wide")
st.title("🔮 NVIDIA Stock Price Prediction (Live)")
st.caption("Using Linear Regression on real-time data from Yahoo Finance")

refresh_interval = st.slider("Auto-Refresh Interval (seconds)", 10, 300, 60)

# Live Updating
placeholder = st.empty()

while True:
    with placeholder.container():
        df = fetch_data()
        dates, actual, predicted, model = train_and_predict(df)

        st.subheader("📈 Latest Data & Prediction")
        st.write(df.tail(5))

        # Plotting
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(dates, actual, label='Actual')
        ax.plot(dates, predicted, label='Predicted')
        ax.set_title('NVIDIA Stock Price Prediction')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid()
        st.pyplot(fig)

        # Metrics
        r2 = r2_score(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        st.metric("R² Score", f"{r2:.4f}")
        st.metric("RMSE", f"{rmse:.2f}")

    st.info(f"Auto-refreshing in {refresh_interval} seconds...")
    time.sleep(refresh_interval)
    placeholder.empty()
