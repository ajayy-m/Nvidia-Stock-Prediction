# 📈 NVIDIA Stock Price Prediction (Live)

Welcome to the **NVIDIA Stock Price Prediction** web app — a real-time stock forecasting tool using historical data and a basic machine learning model.


![Streamlit](https://img.shields.io/badge/Built%20With-Streamlit-red)

> **Repository**: [ajayy-m/Nvidia-Stock-Prediction](https://github.com/ajayy-m/Nvidia-Stock-Prediction)  
> **Author**: [@ajayy-m](https://github.com/ajayy-m)


## 🧠 Overview

This app fetches live stock data for **NVIDIA (NVDA)** using Yahoo Finance and applies **Linear Regression** to predict the next closing price. Key features include:

- Real-time auto-refreshing with custom intervals.
- SMA indicators (10 & 30 days).
- Animated actual vs predicted prices chart.
- Evaluation metrics like **R² Score** and **RMSE**.
- Clean, white-themed interface for readability.



## 🚀 Demo

> Run locally with Streamlit:

```bash
streamlit run app.py
````



## 🔧 Features

* 📉 Real-time **stock data** via `yfinance`
* 📊 **Plotly** charts for interactive visualization
* 🧮 **Linear Regression** via `scikit-learn`
* ⏱ Auto-refresh slider for live predictions
* 📑 Clean tabular view of latest prices
* 📈 Performance metrics (R² and RMSE)



## 🛠 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/ajayy-m/Nvidia-Stock-Prediction.git
cd Nvidia-Stock-Prediction
```

### 2. Install Dependencies

> It's recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, here’s a list you can use:

```txt
streamlit
yfinance
pandas
numpy
plotly
scikit-learn
streamlit-autorefresh
```

### 3. Run the App

```bash
streamlit run app.py
```



## 📂 Project Structure

```bash
.
├── app.py              # Main Streamlit app
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies (optional)
```


## 📈 Model Used

* **Model:** Linear Regression (`sklearn.linear_model`)
* **Features:** Closing Price, 10-day SMA, 30-day SMA
* **Target:** Next-day closing price


## 💡 Future Enhancements

* Add LSTM / XGBoost models for better prediction
* Enable model selection in the UI
* Save predictions and compare across sessions
* Add news sentiment analysis for context




## 🙌 Acknowledgements

* [Yahoo Finance](https://finance.yahoo.com) for stock data
* [Streamlit](https://streamlit.io) for web UI
* [Scikit-learn](https://scikit-learn.org) for modeling
* [Plotly](https://plotly.com) for visualization

---

> Built with ❤️ by [ajayy-m](https://github.com/ajayy-m)
