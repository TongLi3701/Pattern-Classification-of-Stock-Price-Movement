import json
from datetime import datetime
import datetime as dt
from twitterscraper import query_tweets

if __name__ == '__main__':
	start = dt.date(2016, 7, 1)
	end = dt.date(2017,12,31)
	jsonfile = open("../Dataset/Twitter_Data/Decreasing.json","w")
	list_of_tweets = query_tweets('Barclays OR BT Group OR Centrica OR GlaxoSmithKline OR IBM OR WPP OR United Utilities',limit = 50000,begindate=start,enddate=end)
	list_of_json = [] # create empty list to savBerkeley groue multiple tweets which is type of json(dictionary)

	for tweets in list_of_tweets:  # for looping
		tweets.timestamp = datetime.strftime(tweets.timestamp, '%Y-%m-%d %H:%M:%S')
		list_of_json.append(vars(tweets))

	json.dump(list_of_json, jsonfile)