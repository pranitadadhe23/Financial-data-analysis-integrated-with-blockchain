import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Set Streamlit page title
st.title("📈 Stock Price Prediction App")

# Sidebar - User Input
st.sidebar.header("Enter Stock Details")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT)", "AAPL").upper()
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"))


# Download stock data
st.sidebar.write("Fetching data... Please wait.")
data = yf.download(ticker, start=start_date, end=end_date)

if data.empty:
    st.error("No data found! Check the ticker symbol or date range.")
    st.stop()

# Display the data
st.subheader(f"Stock Data for {ticker}")
st.write(data.tail())

# Calculate Moving Averages
data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()
data['Daily Return'] = data['Close'].pct_change()
data['Volatility'] = data['Daily Return'].rolling(window=30).std()

# Drop NaN values
data = data.dropna()

# Plot Closing Price and Moving Averages
st.subheader("📊 Stock Price and Moving Averages")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data.index, data['Close'], label="Close Price", color="blue")
ax.plot(data.index, data['MA50'], label="50-Day MA", color="red")
ax.plot(data.index, data['MA200'], label="200-Day MA", color="green")
ax.set_title(f"{ticker} Stock Price")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# Plot Daily Returns
st.subheader("📈 Daily Returns")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data.index, data['Daily Return'], label="Daily Return", color="orange")
ax.set_xlabel("Date")
ax.set_ylabel("Daily Return")
ax.legend()
st.pyplot(fig)

# Plot Volatility
st.subheader("📉 Volatility (Risk)")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data.index, data['Volatility'], label="Volatility", color="green")
ax.set_xlabel("Date")
ax.set_ylabel("Volatility")
ax.legend()
st.pyplot(fig)

# Prepare data for Machine Learning
st.subheader("🤖 Predicting Stock Price Movement")

data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
data = data[:-1]  # Drop last row (no target)

features = ['Close', 'MA50', 'MA200', 'Daily Return', 'Volatility']
X = data[features]
y = data['Target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Display Model Performance
st.write("### Model Performance")
st.write(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Show Classification Report
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Feature Importance
st.subheader("🔍 Feature Importance")
feature_importance = pd.Series(model.feature_importances_, index=features)
fig, ax = plt.subplots()
feature_importance.sort_values(ascending=False).plot(kind="bar", ax=ax)
st.pyplot(fig)

st.write("""
### 💡 Key Insights:
- The model predicts if the stock price will rise or fall.
- Feature importance shows which factors (e.g., Moving Averages, Volatility) influence predictions.
- **Disclaimer**: This is an educational tool, not a financial advisory model.
""")
