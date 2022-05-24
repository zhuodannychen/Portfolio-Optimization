import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from equal_weight import equal_weight
from equal_weight import minimum_variance
from equal_weight import max_sharpe


TICKERS = ['AAPL',       # Apple
          'KO',         # Coca-Cola
          'DIS',        # Disney
          'XOM',        # Exxon Mobil
          'JPM',        # JPMorgan Chase
          'MCD',        # McDonald's
          'WMT']         # Walmart

# TICKERS = ['QQQ', 'TQQQ', 'TMF']

# TICKERS = [
  # "XOM", "SHW", "JPM", "AEP", "UNH", "AMZN", 
  # "KO", "BA", "AMT", "DD", "TSN", "SLG"
# ]

TOTAL_BALANCE = 10000
start_date = '2015-01-01'
end_date = '2017-12-31'
# start_date = '2011-09-13'
# end_date = '2022-01-09'


# Retrieve historical prices and calculate returns
hist_prices = yf.download(tickers=TICKERS, start=start_date, end=end_date)['Adj Close']
hist_prices.dropna(axis=0)
hist_return = np.log(hist_prices / hist_prices.shift())
hist_return.dropna(axis=0)

# Calculating mean (expected returns), covariance (expected volatility), and correlation
hist_mean = hist_return.mean(axis=0).to_frame()
hist_mean.columns = ['mu']
hist_cov = hist_return.cov()
hist_corr = hist_return.corr()
print(hist_mean.T)
print(hist_cov)


# calculate portfolio returns, volatility, and sharpe ratio
def portfolio_performance(weights, mean, covariance):
    portfolio_return = np.dot(weights.T, mean.values) * 253 # annualize data; ~250 trading days in a year
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)) * 253)
    sharpe_ratio = portfolio_return / portfolio_std
    return portfolio_return[0], portfolio_std, sharpe_ratio[0]
    

# simulate randomized portfolios
n_portfolios = 5000
portfolio_returns = []
portfolio_stds = []

for i in range(n_portfolios):
    weights = np.random.rand(len(TICKERS))
    weights = weights / sum(weights)
    portfolio_return, portfolio_std, sharpe_ratio = portfolio_performance(weights, hist_mean, hist_cov)

    portfolio_returns.append(portfolio_return)
    portfolio_stds.append(portfolio_std)

#------------ Optimized portfolios ------------------#

#----------- Equally Weighted Portfolio -------------#
equally_weighted_weights = np.array(equal_weight(TICKERS))
equally_weighted_return, equally_weighted_std, equally_weighted_sharpe_ratio = portfolio_performance(equally_weighted_weights, hist_mean, hist_cov)
print('---------- Equally Weighted Portfolio ----------')
print('Weights:', equally_weighted_weights)
print('Return:', equally_weighted_return)
print('Volatility:', equally_weighted_std)
print('Sharpe Ratio:', equally_weighted_sharpe_ratio)

print()

#----------- Global Minimum Variance Portfolio ------#
gmv_weights = np.array(minimum_variance(hist_return))
gmv_return, gmv_std, gmv_sharpe_ratio = portfolio_performance(gmv_weights, hist_mean, hist_cov)
print('---------- Global Minimum Variance ----------')
print('Weights:', gmv_weights)
print('Return:', gmv_return)
print('Volatility:', gmv_std)
print('Sharpe Ratio:', gmv_sharpe_ratio)

print()
#----------- Max Sharpe Portfolio ------#
max_sharpe_weights = np.array(max_sharpe(hist_return))
max_sharpe_return, max_sharpe_std, max_sharpe_sharpe_ratio = portfolio_performance(max_sharpe_weights, hist_mean, hist_cov)
print('---------- Max Sharpe Ratio ----------')
print('Weights:', max_sharpe_weights)
print('Return:', max_sharpe_return)
print('Volatility:', max_sharpe_std)
print('Sharpe Ratio:', max_sharpe_sharpe_ratio)


# Print out portfolio value over time
date_range = []
equally_weighted_portfolio = []
gmv_portfolio = []
max_sharpe_portfolio = []
FINAL_BALANCE = 0

for index, day in hist_prices.iterrows():
    equal_weighted_port_val = 0
    gmv_port_val = 0
    max_sharpe_port_val = 0
    for i, asset in enumerate(TICKERS):
        equal_weighted_port_val += equally_weighted_weights[i] * TOTAL_BALANCE / hist_prices[asset].iloc[1] * day[asset]
        gmv_port_val += gmv_weights[i] * TOTAL_BALANCE / hist_prices[asset].iloc[1] * day[asset]
        max_sharpe_port_val += max_sharpe_weights[i] * TOTAL_BALANCE / hist_prices[asset].iloc[1] * day[asset]

    date_range.append(index)
    equally_weighted_portfolio.append(equal_weighted_port_val)
    gmv_portfolio.append(gmv_port_val)
    max_sharpe_portfolio.append(max_sharpe_port_val)
    # FINAL_BALANCE = port_val

# print(TOTAL_BALANCE, FINAL_BALANCE)
# print('Percent Change:', (FINAL_BALANCE - TOTAL_BALANCE) / TOTAL_BALANCE)

plt.plot(date_range, equally_weighted_portfolio, label='equally weighted portfolio')
plt.plot(date_range, gmv_portfolio, label='global min variance portfolio')
plt.plot(date_range, max_sharpe_portfolio, label='max sharpe portfolio')

# SPY for comparison
spy_prices = yf.download(tickers='SPY', start=start_date, end=end_date)['Adj Close']
spy_prices.dropna(axis=0)
spy_portfolio = []

for price in spy_prices:
    spy_port_val = TOTAL_BALANCE / spy_prices[0] * price
    spy_portfolio.append(spy_port_val)
# print(spy_portfolio[:10], spy_portfolio[-6:])
plt.plot(date_range, spy_portfolio, label='SPY')
plt.legend()
plt.show()



# Display portfolios
# plt.scatter(portfolio_stds, portfolio_returns, marker='o', s=3)
# plt.plot(equally_weighted_std, equally_weighted_return, 'or')
# plt.plot(gmv_std, gmv_return, 'or')
# plt.plot(max_sharpe_std, max_sharpe_return, 'or')
# plt.title('Volatility vs Returns for Randomly Generated Portfolios')
# plt.xlabel('Volatility')
# plt.ylabel('Returns')
# plt.show()


