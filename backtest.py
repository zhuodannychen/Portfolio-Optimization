import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import sys

from equal_weight import equal_weight


TICKERS = ['AAPL',       # Apple
          'KO',         # Coca-Cola
          'DIS',        # Disney
          'XOM',        # Exxon Mobil
          'JPM',        # JPMorgan Chase
          'MCD',        # McDonald's
          'WMT']         # Walmart

TICKERS = ['QQQ', 'TQQQ', 'TMF']
# TICKERS = ['SPY', 'QQQ']

start_date = '2015-01-01'
end_date = '2017-12-31'

hist_prices = yf.download(tickers=TICKERS, start=start_date, end=end_date)['Adj Close']
hist_prices.dropna(axis=0)
hist_return = np.log(hist_prices / hist_prices.shift())
hist_return.dropna(axis=0)

hist_mean = hist_return.mean(axis=0).to_frame()
hist_mean.columns = ['mu']
hist_cov = hist_return.cov()
hist_corr = hist_return.corr()
print(hist_mean.T)
print(hist_cov)
# print(hist_corr)
print(len(hist_return))

n_portfolios = 5000
portfolio_returns = []
portfolio_stds = []

for i in range(n_portfolios):
    weights = np.random.rand(len(TICKERS))
    weights = weights / sum(weights)
    portfolio_return = np.dot(weights.T, hist_mean.values) * 250 # annualize data; ~250 trading days in a year
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(hist_cov, weights)) * 250)
    portfolio_returns.append(portfolio_return)
    portfolio_stds.append(portfolio_std)

equally_weighted_weights = np.array(equal_weight(TICKERS))
equally_weighted_return = np.dot(equally_weighted_weights.T, hist_mean.values) * 250
equally_weighted_std = np.sqrt(np.dot(equally_weighted_weights.T, np.dot(hist_cov, equally_weighted_weights)) * 250)
print(equally_weighted_std, equally_weighted_return)


plt.scatter(portfolio_stds, portfolio_returns, marker='o', s=3)
plt.title('Volatility vs Returns for Randomly Generated Portfolios')
plt.xlabel('Volatility')
plt.ylabel('Returns')
plt.show()





