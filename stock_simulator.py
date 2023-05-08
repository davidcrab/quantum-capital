import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
import streamlit as st

# Function to fetch stock data
def get_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data

# Function to compute technical indicators
def calculate_indicators(data):
    data['RSI'] = RSIIndicator(data['Close']).rsi()
    data['SMA50'] = SMAIndicator(data['Close'], window=50).sma_indicator()
    return data

# Function to simulate buying and selling stocks
def simulate_trading(data, rsi_buy_threshold, rsi_sell_threshold):
    portfolio = {}
    cash = 10000  # Initial investment
    buy_signals = (data['Close'] > data['SMA50']) & (data['RSI'] < rsi_buy_threshold)
    sell_signals = data['RSI'] > rsi_sell_threshold

    for idx in data.index:
        buy_signal = buy_signals.loc[idx]
        sell_signal = sell_signals.loc[idx]
        stock_price = data.loc[idx, 'Close']

        # Sell stocks if the RSI indicates a sell signal
        if sell_signal and portfolio:
            cash += sum([shares * stock_price for shares in portfolio.values()])
            portfolio.clear()

        # Buy stocks if the RSI indicates a buy signal
        if buy_signal and not portfolio:
            shares_to_buy = 0.02 * cash // stock_price
            if shares_to_buy > 0:
                cash -= shares_to_buy * stock_price
                portfolio[idx] = shares_to_buy

    # Final value of the portfolio
    final_stock_price = data.iloc[-1]['Close']
    portfolio_value = sum([shares * final_stock_price for shares in portfolio.values()]) + cash
    return portfolio, portfolio_value

st.title("Stock Trading Simulator")

# Input parameters
ticker = st.text_input("Enter stock ticker:", "AAPL")
start_date = st.text_input("Start date:", "2015-01-01")
end_date = st.text_input("End date:", "2023-01-01")
rsi_buy_threshold = st.slider("RSI buy threshold:", 1, 100, 30)
rsi_sell_threshold = st.slider("RSI sell threshold:", 1, 100, 70)

# Fetch data and calculate indicators
data = get_stock_data(ticker, start_date, end_date)
data = calculate_indicators(data)

# Simulate trading
portfolio, portfolio_value = simulate_trading(data, rsi_buy_threshold, rsi_sell_threshold)

# Display results
st.write(f"Final portfolio value: {portfolio_value:.2f}")
st.write("Stocks in the portfolio:")
for date, shares in portfolio.items():
    st.write(f"Date: {date}, Shares: {shares}")

# Plot stock data
st.line_chart(data[['Close', 'SMA50']])