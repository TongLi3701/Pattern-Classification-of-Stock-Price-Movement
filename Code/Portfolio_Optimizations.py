import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import scipy.optimize as sco
from scipy.interpolate import splev, splrep


symbols = ['ADM.L', 'AZN.L', 'BNZL.L', 'CCH.L', 'CPG.L']
# data = pd.DataFrame()
# normalised_data = pd.DataFrame()
#
# start = datetime.datetime(2018,1,1)
# end = datetime.datetime(2018,12,31)
#
# for symbol in symbols:
#     data[symbol] = web.DataReader(symbol, "yahoo", start, end)["Adj Close"]
#
# data = data.fillna(method='ffill')
# data.to_excel('../Dataset/portfolio_data.xlsx')
data = pd.read_excel('../Dataset/portfolio_data.xlsx')
data.index = data['Date'].tolist()
data.pop('Date')

normalised_data = pd.DataFrame()
for symbol in symbols:
    normalised_data[symbol] = (data[symbol] - data[symbol].mean()) / data[symbol].mean()


#normalised_data.plot(grid = True)
#plt.show()

#Number of stocks
stock_num = len(symbols)

# Number of trading days.
trading_days = len(data.index)

# Invest 1 then how much return you can get.
log_returns = np.log(data / data.shift(1))

rets = log_returns

# log_returns.hist(bins=50, figsize=(12, 9))
# plt.show()

weights = np.random.random(stock_num)
weights /= np.sum(weights)

# Implement a Monte Carlo simulation to generate random portfolio weight
# vectors on a larger scale. For every simulated allocation, we record the
# resulting expected portfolio return and variance.
portfolio_returns = []
portfolio_volatilities = []
for p in range (2500):
      weights = np.random.random(stock_num)
      weights /= np.sum(weights)
      portfolio_returns.append(np.sum(rets.mean() * weights) * trading_days)
      portfolio_volatilities.append(np.sqrt(np.dot(weights.T,
                        np.dot(rets.cov() * trading_days, weights))))

portfolio_returns = np.array(portfolio_returns)
portfolio_volatilities = np.array(portfolio_volatilities)

plt.figure(figsize=(9, 5))
plt.scatter(portfolio_volatilities, portfolio_returns, c=portfolio_returns / portfolio_volatilities, marker='o')
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')

def statistics(weights):
    #根据权重，计算资产组合收益率/波动率/夏普率。
    #输入参数
    #==========
    #weights : array-like 权重数组
    #权重为股票组合中不同股票的权重
    #返回值
    #=======
    #pret : float
    #      投资组合收益率
    #pvol : float
    #      投资组合波动率
    #pret / pvol : float
    #    夏普率，为组合收益率除以波动率，此处不涉及无风险收益率资产
    #

    weights = np.array(weights)
    pret = np.sum(rets.mean() * weights) * trading_days
    pvol = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * trading_days, weights)))
    return np.array([pret, pvol, pret / pvol])

def min_func_sharpe(weights):
    return -statistics(weights)[2]

def min_func_variance(weights):
    return statistics(weights)[1] ** 2

cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
bnds = tuple((0, 1) for x in range(stock_num))

opts = sco.minimize(min_func_sharpe, stock_num * [1./ stock_num,], method='SLSQP',bounds=bnds, constraints=cons)
print(opts['x'].round(3))

print(statistics(opts['x']).round(3))

optv = sco.minimize(min_func_variance, stock_num * [1. / stock_num,], method='SLSQP',bounds=bnds, constraints=cons)
print(optv['x'].round(3))
print(statistics(optv['x']).round(3))

# Efficient Frontier
# Evenly spaced list from 0.0 to 0.25, 30 pieces.
target_returns = np.linspace(0.05, 0.18, 30)
target_volatilities = []

# Return the volatility
def min_func_port(weights):
    return statistics(weights)[1]

for tret in target_returns:
    # Cons: equal to the return, sum equals to 1
    cons = ({'type': 'eq', 'fun': lambda x:  statistics(x)[0] - tret},
            {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
    res = sco.minimize(min_func_port, stock_num * [1. / stock_num,], method='SLSQP',bounds=bnds, constraints=cons)
    target_volatilities.append(res['fun'])

target_volatilities = np.array(target_volatilities)

#画散点图
plt.figure()
#圆点为随机资产组合
plt.scatter(portfolio_volatilities, portfolio_returns,
            c=portfolio_returns / portfolio_volatilities, marker='o')
#叉叉为有效边界
plt.scatter(target_volatilities, target_returns,
            c=target_returns / target_volatilities, marker='x')
#红星为夏普率最大值的资产组合
plt.plot(statistics(opts['x'])[1], statistics(opts['x'])[0],
         'r*', markersize=15.0)
#黄星为最小方差的资产组合
plt.plot(statistics(optv['x'])[1], statistics(optv['x'])[0],
         'y*', markersize=15.0)
            # minimum variance portfolio
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.show()

