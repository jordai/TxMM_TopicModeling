import os
import json
import shutil
import pandas as pd
from DataLoader import load_data, clean_data, preprocess
from LDA import getWordFreq, computeCoherence, LDA, LDA_OT, assign_topics_to_tweets
from utils import plotCoherence, plot_top_words
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



if __name__ == '__main__':
    main()

