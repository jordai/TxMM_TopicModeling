import os
import json
import shutil
import pandas as pd
from DataLoader import load_data, clean_data, preprocess
from LDA import getWordFreq, computeCoherence, LDA, LDA_OT
from utils import plotCoherence, plot_top_words
from gensim.models import ldaseqmodel

def main():
    if not os.path.isfile("preprocessed_tweets.csv"):
        data_path = "../data/rivm/"
        tweets = load_data(data_path)

        tweets = clean_data(tweets)

        tweets = preprocess(tweets)
        tweets.to_csv('preprocessed_tweets.csv', index = False)
    else:
        tweets = pd.read_csv('preprocessed_tweets.csv', converters={"tokenized_tweet": lambda x: x.strip("[]").split(", ")})
    
    if not os.path.isfile("lda_over_time"):
        month_counts = tweets['month'].value_counts().to_dict()
        time_slices = []
        time_slices.append(month_counts['Feb'])
        time_slices.append(month_counts['Mar'])
        time_slices.append(month_counts['Apr'])
        time_slices.append(month_counts['May'])

        tweets_lda = LDA(tweets, k=4)
        print("LDA done")
        lda_ot = LDA_OT(tweets_lda, tweets, time_slices, k = 4)
        print("Finished TOT")
        lda_ot.save('lda_over_time')
    else:
        lda_ot = ldaseqmodel.LdaSeqModel.load('lda_over_time')
    
    print(lda_ot.print_topic_times(0, top_terms=10))
    #freqlist = getWordFreq(tweets)
    #print(freqlist.most_common(10))
    #tweet_coherence = computeCoherence(tweets)
    #plotCoherence(tweet_coherence)
    #for k in range(1,10):
    #    print("performing LDA with {} topics".format(k))
    #    tweets_lda = LDA(tweets, k)
    #    plot_top_words(tweets_lda, k=k)


if __name__ == '__main__':
    main()

