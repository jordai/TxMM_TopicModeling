import pandas as pd
from nltk import download, FreqDist
from multiprocessing import Pool, cpu_count


def getWordFreq(tweets):
    pool = Pool(cpu_count())
    out = pool.map(FreqDist, tweets.tokenized_tweet.to_list())
    pool.close()

    word_dist = FreqDist()
    for dist in out:
        word_dist.update(dist)
    return word_dist
