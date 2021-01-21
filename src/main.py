import os
import json
import shutil
import numpy as np
import pandas as pd
from DataLoader import load_data, clean_data, preprocess
from LDA import getWordFreq, computeCoherence, LDA, assign_topics_to_tweets, perform_tot
from utils import plotCoherence, plot_top_words, plot_tweets_per_day, plot_results


def main():
    if not os.path.isfile("preprocessed_tweets.csv"):
        data_path = "../data/rivm/"
        tweets = load_data(data_path)

        tweets = clean_data(tweets)

        tweets = preprocess(tweets)
        tweets.to_csv('preprocessed_tweets.csv', index = False)
    else:
        tweets = pd.read_csv('preprocessed_tweets.csv', converters={"tokenized_tweet": lambda x: x.strip("[]").split(", ")})

    retest_coherence = False
    if retest_coherence:
        LDA_coherence = computeCoherence(tweets, k = 26)
        plotCoherence(LDA_coherence)

    if not os.path.exists("tweets_lda"):
        tweets_lda = LDA(tweets, k=4)
        plot_top_words(tweets_lda, k = 4)
        tweets_lda.save('tweets_lda')
    else:
        tweets_lda = LdaModel.load('tweets_lda')
        plot_top_words(tweets_lda, k = 4)

    if not os.path.isfile("tweets_with_topics.csv"):
        tweets_with_topics = assign_topics_to_tweets(tweets, tweets_lda)
        tweets_with_topics.to_csv('tweets_with_topics.csv', index = False)
    else:
        tweets_with_topics = pd.read_csv('tweets_with_topics.csv', converters={"tokenized_tweet": lambda x: x.strip("[]").split(", ")})

    tot = perform_tot(tweets_with_topics)

    plot_results(tot)

if __name__ == '__main__':
    main()

