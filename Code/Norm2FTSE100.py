import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import csv
import datetime

start = datetime.datetime(2017,1,1)
end = datetime.datetime(2018,1,1)

with open('FTSE100_Symbols.csv', 'r') as f:
    reader = csv.reader(f)
    symbolList = list(reader)

dataList = []

for symbol in symbolList[0]:
    dataList.append(web.DataReader(symbol, "yahoo", start, end)["Adj Close"])

stocks = pd.DataFrame({symbolList[0][i]: (dataList[i] - dataList[i][0]) / dataList[i][0] for i in range(len(symbolList[0]))})
stocks = stocks.fillna(method='ffill')
wekaData = pd.DataFrame(stocks.transpose())
wekaData = wekaData.reset_index()
wekaData.to_csv("FTSE100_Data_Norm2.csv",index = False)