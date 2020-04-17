import codecs, json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid', context='talk', palette='Dark2')

stock_name = 'Decreasing'

with codecs.open('../Dataset/Twitter_Data/'+ stock_name + '.json', 'r', 'utf-8') as f:
    tweets = json.load(f, encoding='utf-8')

list_tweets = [list(elem.values()) for elem in tweets]
list_columns = list(tweets[0].keys())
tweets_df = pd.DataFrame(list_tweets, columns=list_columns)

from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
sia = SIA()
results = []

for text in tweets_df['text']:
    pol_score = sia.polarity_scores(text)
    pol_score['tweet'] = text
    results.append(pol_score)

df = pd.DataFrame.from_records(results)


df['label'] = 0
df.loc[df['compound'] > 0.2, 'label'] = 1
df.loc[df['compound'] < -0.2, 'label'] = -1
fig, ax = plt.subplots()

counts = df.label.value_counts(normalize=True) * 100

sns.barplot(x=counts.index, y=counts, ax=ax).set_title('Sentiment Analysis: ' + stock_name)

ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
ax.set_ylabel("Percentage")
plt.show()