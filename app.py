import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

import streamlit as st

# ✅ Must be first Streamlit command
st.set_page_config(layout="wide")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

# Sidebar for refresh interval
refresh_interval = st.sidebar.slider("⏱️ Auto-Refresh Interval (sec)", 10, 300, 60)
st_autorefresh(interval=refresh_interval * 1000, key="refresh")

# Title and caption
st.title("🔮 NVIDIA Stock Price Prediction (Live)")
st.caption("Using Linear Regression on real-time data from Yahoo Finance")

# --- Data fetching ---
def fetch_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)
    df = yf.download('NVDA', start=start_date, end=end_date)
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    return df

# --- ML training ---
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

# --- Main execution ---
df = fetch_data()
dates, actual, predicted = train_and_predict(df)

st.subheader("📊 Latest NVIDIA Data")
st.write(df.tail(5))

# --- Plotting ---
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(dates, actual, label='Actual Price')
ax.plot(dates, predicted, label='Predicted Price')
ax.set_title('NVIDIA Stock Price Prediction (Linear Regression)')
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# --- Metrics ---
r2 = r2_score(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
st.metric("R² Score", f"{r2:.4f}")
st.metric("RMSE", f"{rmse:.2f}")
