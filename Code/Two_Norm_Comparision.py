import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import csv
import datetime

def AverageNum(num):
    nSum = 0
    for i in range(len(num)):
        nSum += num[i]
    return nSum / len(num)

start = datetime.datetime(2018,1,1)
end = datetime.datetime(2018,12,31)

Tesco = web.DataReader('TSCO.L', "yahoo", start, end)["Adj Close"]
EXPN = web.DataReader('EXPN.L', "yahoo", start, end)["Adj Close"]
RBS = web.DataReader('RBS.L',"yahoo", start, end)["Adj Close"]


# Tesco = (Tesco - Tesco.shift(-1) )/ Tesco * 100
# EXPN = (EXPN - EXPN.shift(-1) )/ EXPN * 100
# RBS = (RBS - RBS.shift(-1) )/ RBS * 100
# Tesco = (Tesco / AverageNum(Tesco) - 1) * 100
# EXPN = (EXPN / AverageNum(EXPN) - 1) * 100
# RBS = (RBS / AverageNum(RBS) - 1) * 100
# Tesco = (Tesco - AverageNum(Tesco) / Tesco[0] * 100
Tesco = (Tesco - Tesco[0]) / Tesco[0] * 100
EXPN = (EXPN - EXPN[0]) / EXPN[0] * 100
RBS = (RBS - RBS[0]) / RBS[0] * 100


# stock = pd.DataFrame({'TSCO.L': (stock - stock[0]) / stock[0] * 100 })
stock = pd.DataFrame({'TSCO.L': Tesco,'EXPN.L': EXPN,'RBS.L': RBS})
stock = stock.fillna(method='ffill')

#
fig = stock.plot(grid = True, title = 'Three stocks - 2018 (Normalisation Method 1)')
# fig.set_ylabel('Percentage')
# fig = fig.get_figure()
# file_path = '../Graph/'
# fig.savefig(file_path + 'Three stocks - 2018 (Normalisation Method 1)')
plt.show()