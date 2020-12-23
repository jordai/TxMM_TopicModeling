import os
import json
import shutil
import pandas as pd
from DataLoader import load_data, clean_data, preprocess
from LDA import getWordFreq, computeCoherence, LDA
from utils import plotCoherence, plot_top_words

def main():
    if not os.path.isfile("preprocessed_tweets.csv"):
        data_path = "../data/rivm/"
        tweets = load_data(data_path)

        tweets = clean_data(tweets)

        tweets = preprocess(tweets)
        tweets.to_csv('preprocessed_tweets.csv', index = False)
    else:
        tweets = pd.read_csv('preprocessed_tweets.csv', converters={"tokenized_tweet": lambda x: x.strip("[]").split(", ")})

    freqlist = getWordFreq(tweets)
    print(freqlist.most_common(10))
    #tweet_coherence = computeCoherence(tweets)
    #plotCoherence(tweet_coherence)
    for k in range(1,10):
        tweets_lda = LDA(tweets, k)
        plot_top_words(tweets_lda, k=k)



if __name__ == '__main__':
    main()

