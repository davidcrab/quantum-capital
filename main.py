import backtrader as bt
import pandas as pd
import yfinance as yf

# Create a custom strategy class
class MovingAverageRSIStrategy(bt.Strategy):
    params = (
        ('sma_period', 50),
        ('rsi_buy_threshold', 30),
        ('rsi_sell_threshold', 70),
    )

    def __init__(self):
        self.sma = bt.indicators.SimpleMovingAverage(
            self.data.close,
            period=self.params.sma_period
        )
        self.rsi = bt.indicators.RSI(self.data.close)

    def next(self):
        if not self.position:
            if self.data.close > self.sma and self.rsi < self.params.rsi_buy_threshold:
                self.buy()
        else:
            if self.rsi > self.params.rsi_sell_threshold:
                self.sell()


# Download stock data
ticker = 'AAPL'
start_date = '2015-01-01'
end_date = '2023-01-01'
data = yf.download(ticker, start=start_date, end=end_date)

# Create a Data Feed
data_feed = bt.feeds.PandasData(dataname=data)

# Initialize the backtesting engine
cerebro = bt.Cerebro()

# Add the data feed and strategy
cerebro.adddata(data_feed)
cerebro.addstrategy(MovingAverageRSIStrategy)

# Set the initial cash for the portfolio
cerebro.broker.setcash(10000.0)

# Print out the starting conditions
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

# Run the backtest
cerebro.run()

# Print out the final conditions
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
