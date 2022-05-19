import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from equal_weight import equal_weight


TICKERS = ['AAPL',       # Apple
          'KO',         # Coca-Cola
          'DIS',        # Disney
          'XOM',        # Exxon Mobil
          'JPM',        # JPMorgan Chase
          'MCD',        # McDonald's
          'WMT']         # Walmart

TICKERS = ['QQQ', 'TQQQ', 'TMF']
TOTAL_BALANCE = 10000
start_date = '2015-01-01'
end_date = '2017-12-31'

# Retrieve historical prices and calculate returns
hist_prices = yf.download(tickers=TICKERS, start=start_date, end=end_date)['Adj Close']
hist_prices.dropna(axis=0)
hist_return = np.log(hist_prices / hist_prices.shift())
hist_return.dropna(axis=0)

# Calculating mean, covariance, and correlation
hist_mean = hist_return.mean(axis=0).to_frame()
hist_mean.columns = ['mu']
hist_cov = hist_return.cov()
hist_corr = hist_return.corr()
print(hist_mean.T)
print(hist_cov)


# calculate portfolio returns and volatility
def calc_returns_and_std(weights, mean, covariance):
    portfolio_return = np.dot(weights.T, mean.values) * 250 # annualize data; ~250 trading days in a year
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)) * 250)
    return portfolio_return[0], portfolio_std
    

# simulate randomized portfolios
n_portfolios = 1000
portfolio_returns = []
portfolio_stds = []

for i in range(n_portfolios):
    weights = np.random.rand(len(TICKERS))
    weights = weights / sum(weights)
    portfolio_return, portfolio_std = calc_returns_and_std(weights, hist_mean, hist_cov)

    portfolio_returns.append(portfolio_return)
    portfolio_stds.append(portfolio_std)

# Optimized portfolios
# Equally weighted portfolio
equally_weighted_weights = np.array(equal_weight(TICKERS))
equally_weighted_return, equally_weighted_std = calc_returns_and_std(equally_weighted_weights, hist_mean, hist_cov)
print('Equally weighted portfolio volatility and returns:', equally_weighted_std, equally_weighted_return)
print('Sharpe Ratio:', equally_weighted_return / equally_weighted_std)

# Print out portfolio value over time
date_range = []
equally_weighted_portfolio = []
FINAL_BALANCE = 0
for index, day in hist_prices.iterrows():
    port_val = 0
    for i, asset in enumerate(TICKERS):
        port_val += equally_weighted_weights[i] * TOTAL_BALANCE / hist_prices[asset].iloc[1] * day[asset]
    date_range.append(index)
    equally_weighted_portfolio.append(port_val)
    FINAL_BALANCE = port_val

print(TOTAL_BALANCE, FINAL_BALANCE)
print('Percent Change:', (FINAL_BALANCE - TOTAL_BALANCE) / TOTAL_BALANCE)
plt.plot(date_range, equally_weighted_portfolio)
# plt.show()




# Display portfolios
plt.scatter(portfolio_stds, portfolio_returns, marker='o', s=3)
plt.plot(equally_weighted_std, equally_weighted_return, 'or')
plt.title('Volatility vs Returns for Randomly Generated Portfolios')
plt.xlabel('Volatility')
plt.ylabel('Returns')
# plt.show()

