# Author : Tong Li
# Date : 2019/3/10
#
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.svm import SVR
import pandas_datareader as pdr
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def get_signal(r):
    if r['Predicted Next Close'] > r['Next Day Open']:
        return 0
    else:
        return 1

def get_ret(r):
    if r['Signal'] == 1:
        return ((r['Next Day Close'] - r['Next Day Open'])/r['Next Day Open']) * 100
    else:
        return 0

def get_stats(strategy,s, n=252):
    print(strategy)
    s = s.dropna()
    wins = len(s[s>0])
    losses = len(s[s<0])
    evens = len(s[s==0])
    mean_w = round(s[s>0].mean(), 3)
    mean_l = round(s[s<0].mean(), 3)
    win_r = round(wins/losses, 3)
    mean_trd = round(s.mean(), 3)
    # sd = round(np.std(s), 3)
    max_l = round(s.min(), 3)
    max_w = round(s.max(), 3)
    sharpe_r = round((s.mean()/np.std(s))*np.sqrt(n), 4)
    cnt = len(s)
    print('Trades:', cnt,
          '\nWins:', wins,
          '\nLosses:', losses,
          '\nBreakeven:', evens,
          '\nWin/Loss Ratio', win_r,
          '\nMean Win:', mean_w,
          '\nMean Loss:', mean_l,
          '\nMean', mean_trd,
          # '\nStd Dev:', sd,
          '\nMax Loss:', max_l,
          '\nMax Win:', max_w,
          '\nSharpe Ratio:', sharpe_r)
    print()



start_date = pd.to_datetime('2016-01-01')
stop_date = pd.to_datetime('2018-12-31')

stock = pdr.data.get_data_yahoo('EVR.L', start_date, stop_date)
# stock_close = stock['Close']
# # stock_close.plot(grid = True,title = 'EVR.L')
#
#
# for i in range(1, 21, 1):
#     stock.loc[:,'Close Minus ' + str(i)] = stock['Close'].shift(i)
#
# # three years stock price for EVR.L, everyday Date contains the close prices for the past 20 days.
# stock20 = stock[[x for x in stock.columns if 'Close Minus' in x or x == 'Close']].iloc[20:,]
#
# stock20 = stock20.iloc[:,::-1]
#
# train_test_split = 485
#
# x_train = stock20[:train_test_split]
# y_train = stock20['Close'].shift(-1)[:train_test_split]
# x_test = stock20[train_test_split:]
# y_test = stock20['Close'].shift(-1)[train_test_split:]
#
# clf = SVR(kernel='linear')
# svr_model = clf.fit(x_train, y_train)
# preds = svr_model.predict(x_test)
# tf = pd.DataFrame(list(zip(y_test, preds)), columns=['Next Day Close', 'Predicted Next Close'], index=y_test.index)
#
# cdc = stock[['Close']].iloc[train_test_split:]
# ndo = stock[['Open']].iloc[train_test_split:].shift(-1)
# tf1 = pd.merge(tf, cdc, left_index=True, right_index=True)
# tf2 = pd.merge(tf1, ndo, left_index=True, right_index=True)
# tf2.columns = ['Next Day Close', 'Predicted Next Close', 'Current Day Close', 'Next Day Open']
#
#
# tf2 = tf2.assign(Signal = tf2.apply(get_signal, axis=1))
# tf2 = tf2.assign(PnL = tf2.apply(get_ret, axis=1))
# tf2['Next Day Close'].plot()
# tf2['Predicted Next Close'].plot()
#
# plt.show()


# get_stats('Predicted Return',tf2['PnL'])

# daily returns
# daily_return = ((stock['Close'].iloc[train_test_split:] - stock['Close'].iloc[train_test_split:].shift(1))/stock['Close'].iloc[train_test_split:].shift(1))*100
# # intra day returns
# intra_return = (stock['Close'].iloc[train_test_split:] - stock['Open'].iloc[train_test_split:])/stock['Open'].iloc[train_test_split:] * 100
# # overnight returns
# overnight_return = ((stock['Open'].iloc[train_test_split:] - stock['Close'].iloc[train_test_split:].shift(1))/stock['Close'].iloc[train_test_split:].shift(1))*100

# get_stats('Daily Return',daily_return)
# get_stats('Intra Return',intra_return)
# get_stats('Overnight Return',overnight_return)

def dtw_dist(x, y):
    distance, path = fastdtw(x, y, dist=euclidean)
    return distance

tseries = []
tlen = 5
for i in range(tlen, len(stock), tlen):
    pctc = stock['Close'].iloc[i-tlen:i].pct_change()[1:].values * 100
    res = stock['Close'].iloc[i-tlen:i+1].pct_change()[-1] * 100
    tseries.append((pctc, res))


dist_pairs = []
for i in range(len(tseries)):
    for j in range(len(tseries)):
        dist = dtw_dist(tseries[i][0], tseries[j][0])
        dist_pairs.append((i,j,dist,tseries[i][1], tseries[j][1]))

dist_frame = pd.DataFrame(dist_pairs, columns=['A','B','Dist', 'A Ret', 'B Ret'])



sf = dist_frame[dist_frame['Dist']>0].sort_values(['A','B']).reset_index(drop=1)

sfe = sf[sf['A']<sf['B']]

#winf = sfe[(sfe['Dist']<=20)&(sfe['A Ret']>0)]
winf = sfe[(sfe['A Ret']>0)]
print(winf)


excluded = {}
return_list = []
def get_returns(r):
    if excluded.get(r['A']) is None:
        return_list.append(r['B Ret'])
        if r['B Ret'] < 0:
            excluded.update({r['A']:1})

winf.apply(get_returns, axis=1)
get_stats('DTW',pd.Series(return_list))



