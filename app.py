import os

# Set up the Alpaca API
os.environ['APCA_API_BASE_URL'] = 'https://paper-api.alpaca.markets'
os.environ['APCA_API_KEY_ID'] = 'PKBSV0834I61DKR0R5ZS'
os.environ['APCA_API_SECRET_KEY'] = '5cOPW5UZ8ezN0ht4MYPVCb6N5cZ2oayg6v2HMCNk'

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from alpaca_trade_api.rest import REST, TimeFrame
import streamlit as st
import plotly.graph_objs as go
import talib as ta


api = REST()

# Define a function to fetch historical data
def get_historical_data(symbol, start_date, end_date):
    df = api.get_bars(
        symbol,
        TimeFrame.Day,
        start=pd.Timestamp(start_date, tz='America/New_York').isoformat(),
        end=pd.Timestamp(end_date, tz='America/New_York').isoformat(),
    ).df

    df.index = df.index.tz_convert('America/New_York')
    return df

# Define a function to calculate moving averages
def calculate_sma(df, short_window, long_window):
    df['short_sma'] = df['close'].rolling(window=short_window).mean()
    df['long_sma'] = df['close'].rolling(window=long_window).mean()
    return df

# Define a function to implement the SMA crossover strategy
def sma_crossover(df):
    df['signal'] = np.where(df['short_sma'] > df['long_sma'], 1, 0)
    return df

# Streamlit app layout
st.title("Simple Moving Average Crossover Strategy")

symbol = st.text_input("Enter the stock symbol (e.g., AAPL):", "AAPL")
start_date = st.date_input("Start date:", value=pd.to_datetime("2022-01-01"))
end_date = st.date_input("End date:", value=pd.to_datetime("2022-12-31"))
short_window = st.slider("Short-term moving average window:", 1, 100, 50)
long_window = st.slider("Long-term moving average window:", 1, 200, 200)

if st.button("Analyze"):
    df = get_historical_data(symbol, start_date, end_date)
    df = calculate_sma(df, short_window, long_window)
    df = sma_crossover(df)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name=symbol))
    fig.add_trace(go.Scatter(x=df.index, y=df['short_sma'], mode='lines', name=f'{short_window} Day SMA'))
    fig.add_trace(go.Scatter(x=df.index, y=df['long_sma'], mode='lines', name=f'{long_window} Day SMA'))

    fig.update_layout(
        title=f'{symbol} Price with Simple Moving Average Crossover Strategy',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x',
    )

    st.plotly_chart(fig)

bars = api.get_bars("SPY", TimeFrame.Day, "2021-06-01", "2021-10-01").df
# bars
bars['30_Day_SMA'] = ta.SMA(bars['close'], timeperiod=30)

# plotly imports
import plotly.graph_objects as go
import plotly.express as px

# SPY bar data candlestick plot
candlestick_fig = go.Figure(data=[go.Candlestick(x=bars.index,
                open=bars['open'],
                high=bars['high'],
                low=bars['low'],
                close=bars['close'])])

# creating a line plot for our sma
sma_fig = px.line(x=bars.index, y=bars['30_Day_SMA'])

# adding both plots onto one chart
fig = go.Figure(data=candlestick_fig.data + sma_fig.data)

# Simple Moving Average with TA-Lib
st.header("Simple Moving Average with TA-Lib")
st.plotly_chart(fig)

bars['upper_band'], bars['middle_band'], bars['lower_band'] = ta.BBANDS(bars['close'], timeperiod =30)

# creating a line plot for our sma
upper_line_fig = px.line(x=bars.index, y=bars['upper_band'])
# creating a line plot for our sma
lower_line_fig = px.line(x=bars.index, y=bars['lower_band'])

# adding both plots onto one chart
fig = go.Figure(data=candlestick_fig.data + sma_fig.data + upper_line_fig.data + lower_line_fig.data)

# Bollinger Bands with TA-Lib
st.header("Bollinger Bands with TA-Lib")
st.plotly_chart(fig)

import backtrader as bt

def run_backtest(strategy, symbols, start, end, timeframe=TimeFrame.Day, cash=10000):
    '''params:
        strategy: the strategy you wish to backtest, an instance of backtrader.Strategy
        symbols: the symbol (str) or list of symbols List[str] you wish to backtest on
        start: start date of backtest in format 'YYYY-MM-DD'
        end: end date of backtest in format: 'YYYY-MM-DD'
        timeframe: the timeframe the strategy trades on (size of bars) -
                   1 min: TimeFrame.Minute, 1 day: TimeFrame.Day, 5 min: TimeFrame(5, TimeFrameUnit.Minute)
        cash: the starting cash of backtest
    '''

    # initialize backtrader broker
    cerebro = bt.Cerebro(stdstats=True)
    cerebro.broker.setcash(cash)

    # add strategy
    cerebro.addstrategy(strategy)

    # add analytics
    # cerebro.addobserver(bt.observers.Value)
    # cerebro.addobserver(bt.observers.BuySell)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='mysharpe')
    
    # historical data request
    if type(symbols) == str:
        symbol = symbols
        alpaca_data = api.get_bars(symbol, timeframe, start, end,  adjustment='all').df
        data = bt.feeds.PandasData(dataname=alpaca_data, name=symbol)
        cerebro.adddata(data)
    elif type(symbols) == list or type(symbols) == set:
        for symbol in symbols:
            alpaca_data = api.get_bars(symbol, timeframe, start, end, adjustment='all').df
            data = bt.feeds.PandasData(dataname=alpaca_data, name=symbol)
            cerebro.adddata(data)

    # run
    initial_portfolio_value = cerebro.broker.getvalue()
    print(f'Starting Portfolio Value: {initial_portfolio_value}')
    results = cerebro.run()
    final_portfolio_value = cerebro.broker.getvalue()
    print(f'Final Portfolio Value: {final_portfolio_value} ---> Return: {(final_portfolio_value/initial_portfolio_value - 1)*100}%')

    strat = results[0]
    print('Sharpe Ratio:', strat.analyzers.mysharpe.get_analysis()['sharperatio'])
    # cerebro.plot(iplot= False)

    # st.plotly_chart(cerebro.plot(iplot= False))
    
    # plot results
    st.write('Backtest Results')
    st.write('----------------')
    st.write(f'Starting Portfolio Value: {initial_portfolio_value}')
    st.write(f'Final Portfolio Value: {final_portfolio_value} ---> Return: {(final_portfolio_value/initial_portfolio_value - 1)*100}%')
    st.write(f'Sharpe Ratio: {strat.analyzers.mysharpe.get_analysis()["sharperatio"]}')

class SmaCross(bt.Strategy):
  # list of parameters which are configurable for the strategy
    params = dict(
        pfast=13,  # period for the fast moving average
        pslow=25   # period for the slow moving average
    )

    def __init__(self):
        sma1 = bt.ind.SMA(period=self.p.pfast)  # fast moving average
        sma2 = bt.ind.SMA(period=self.p.pslow)  # slow moving average
        self.crossover = bt.ind.CrossOver(sma1, sma2)  # crossover signal
  
    def next(self):
        if not self.position and self.crossover > 0:  # not in the market
            self.buy()
        elif self.position and self.crossover < 0:  # in the market & cross to the downside
            self.close()  # close long position

run_backtest(SmaCross, 'AAPL', '2022-01-01', '2023-05-07', TimeFrame.Day, 10000)