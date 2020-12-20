import os
import json
import shutil
import pandas as pd
from DataLoader import load_data, clean_data, preprocess

def main():
    
    data_path = "../data/rivm/"
    tweets = load_data(data_path)

    tweets = clean_data(tweets)

    tweets = preprocess(tweets)
    tweets.to_csv('preprocessed_tweets.csv')

if __name__ == '__main__':
    main()

