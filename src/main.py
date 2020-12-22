import os
import json
import shutil
import pandas as pd
from DataLoader import load_data, clean_data, preprocess
from LDA import getWordFreq

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
    print(freqlist.most_common(500))

if __name__ == '__main__':
    main()

