import pandas as pd
from nltk import download, FreqDist
from multiprocessing import Pool, cpu_count
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel

def getWordFreq(tweets):
    pool = Pool(cpu_count())
    out = pool.map(FreqDist, tweets.tokenized_tweet.to_list())
    pool.close()

    word_dist = FreqDist()
    for dist in out:
        word_dist.update(dist)
    return word_dist

def computeCoherence(tweets):
    tweets_dictionary = Dictionary(tweets.tokenized_tweet)

    # build the corpus i.e. vectors with the number of occurence of each word per tweet
    tweets_corpus = [tweets_dictionary.doc2bow(tweet) for tweet in tweets.tokenized_tweet]

    # compute coherence
    tweets_coherence = []
    for nb_topics in range(1,25):
        print("calculating coherence for {} topics".format(nb_topics))
        lda = LdaModel(tweets_corpus, num_topics = nb_topics, id2word = tweets_dictionary, passes=10)
        cohm = CoherenceModel(model=lda, corpus=tweets_corpus, dictionary=tweets_dictionary, coherence='u_mass')
        coh = cohm.get_coherence()
        tweets_coherence.append(coh)
    return tweets_coherence

def LDA(tweets, k = 1):
    tweets_dictionary = Dictionary(tweets.tokenized_tweet)
    tweets_corpus = [tweets_dictionary.doc2bow(tweet) for tweet in tweets.tokenized_tweet]
    return LdaModel(tweets_corpus, num_topics = k, id2word = tweets_dictionary, passes=10)
    
