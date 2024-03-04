import snscrape.modules.twitter as sntwitter
import pandas as pd
# Creating list to append tweet DATA to
tweets_list2 = []
import yaml
import re
import sys


params = yaml.safe_load(open("params.yaml"))["prepare"]
limits = params["limits"]


# Using TwitterSearchScraper to scrape DATA and append tweets to list
for i, tweet in enumerate(
        sntwitter.TwitterSearchScraper('COVID Vaccine since:2021-01-01 until:2021-05-31').get_items()):
    if i > limits:
        break
    tweets_list2.append([tweet.rawContent])

# Creating a dataframe from the tweets list above
df = pd.DataFrame(tweets_list2, columns=['rawContent'])
df.to_csv("data/scrape/Tweets.csv")