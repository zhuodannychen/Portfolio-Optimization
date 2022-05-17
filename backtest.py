import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


TICKERS = ['QQQ', 'SPY', 'TQQQ', 'TSLA']
start_date = '2018-03-01'
end_date = '2022-05-01'

hist_prices = yf.download(tickers=TICKERS, start=start_date, end=end_date)['Adj Close']
hist_prices.dropna(axis=0)
hist_return = (hist_prices / hist_prices.shift())
hist_return.dropna(axis=0)

print(hist_return)
