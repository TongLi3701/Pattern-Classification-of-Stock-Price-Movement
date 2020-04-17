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

start = datetime.datetime(2018,10,1)
end = datetime.datetime(2018,12,31)

with open('../Dataset/Dynamic_Pattern_FTSE100_Symbols.csv', 'r') as f:
    reader = csv.reader(f)
    symbolList = list(reader)

dataList = []
for symbol in symbolList[0]:
    dataList.append(web.DataReader(symbol, "yahoo", start, end)["Adj Close"])
    print(symbol)


for i in range (len(dataList)):
    dataList[i] = dataList[i] / AverageNum(dataList[i])

stocks = pd.DataFrame({symbolList[0][i]: (dataList[i] - 1.0) * 100 for i in range(len(symbolList[0]))})


# fill the missing cell by using the previous value.
stocks = stocks.fillna(method='ffill')

wekaData = pd.DataFrame(stocks.transpose())
wekaData = wekaData.reset_index()
wekaData.to_csv("../Dataset/FTSE100_Norm1_2018_Season4.csv",index = False)
#stocks.plot(grid = True)
#plt.show()