import os
import json
import shutil
import numpy as np
import pandas as pd
from DataLoader import load_data, clean_data, preprocess
from LDA import getWordFreq, computeCoherence, LDA, LDA_OT, assign_topics_to_tweets
from utils import plotCoherence, plot_top_words, plot_tweets_per_day, plot_results
from gensim.models import ldaseqmodel
from gensim.models.ldamodel import LdaModel

def main():
    if not os.path.isfile("preprocessed_tweets.csv"):
        data_path = "../data/rivm/"
        tweets = load_data(data_path)

        tweets = clean_data(tweets)

        tweets = preprocess(tweets)
        tweets.to_csv('preprocessed_tweets.csv', index = False)
    else:
        tweets = pd.read_csv('preprocessed_tweets.csv', converters={"tokenized_tweet": lambda x: x.strip("[]").split(", ")})

    if not os.path.exists("tweets_lda"):
        tweets_lda = LDA(tweets, k=4)
        tweets_lda.save('tweets_lda')
    else:
        tweets_lda = LdaModel.load('tweets_lda')

    #print(assign_topics_to_tweets(tweets, tweets_lda))

    if not os.path.isfile("tweets_with_topics.csv"):
        tweets_with_topics = assign_topics_to_tweets(tweets, tweets_lda)
        tweets_with_topics.to_csv('tweets_with_topics.csv', index = False)
    else:
        tweets_with_topics = pd.read_csv('tweets_with_topics.csv', converters={"tokenized_tweet": lambda x: x.strip("[]").split(", ")})

    #print(tweets_with_topics.loc[tweets_with_topics['date'] == 'Feb 25', 'topic_0'].mean())
    dates = tweets_with_topics.date.unique()
    tweets_per_day = tweets_with_topics.date.value_counts(sort = False)
    ordered_tweets_per_day = []
    for day in dates:
        ordered_tweets_per_day.append(tweets_per_day[day])
    #plot_tweets_per_day(dates, tweets_per_day)

    topic_0 = []
    topic_1 = []
    topic_2 = []
    topic_3 = []

    topic_over_time = []
    for date in dates:
        topic_0 = tweets_with_topics.loc[tweets_with_topics['date'] == date, 'topic_0'].mean()
        topic_1 = tweets_with_topics.loc[tweets_with_topics['date'] == date, 'topic_1'].mean()
        topic_2 = tweets_with_topics.loc[tweets_with_topics['date'] == date, 'topic_2'].mean()
        topic_3 = tweets_with_topics.loc[tweets_with_topics['date'] == date, 'topic_3'].mean()
        sanity_check = topic_0 + topic_1 + topic_2 + topic_3
        topic_over_time.append([date, topic_0, topic_1, topic_2, topic_3, sanity_check])

    tot = pd.DataFrame(topic_over_time, columns=['date', 'topic_0', 'topic_1', 'topic_2', 'topic_3', 'sum'])
    tot.to_csv('tot.csv', index = False)

    plot_results(tot)

if __name__ == '__main__':
    main()

