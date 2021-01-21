import pandas as pd
from nltk import download, FreqDist
from multiprocessing import Pool, cpu_count
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel
from gensim.models import ldaseqmodel

def getWordFreq(tweets):
    pool = Pool(cpu_count())
    out = pool.map(FreqDist, tweets.tokenized_tweet.to_list())
    pool.close()

    word_dist = FreqDist()
    for dist in out:
        word_dist.update(dist)
    return word_dist

def computeCoherence(tweets, k = 25):
    tweets_dictionary = Dictionary(tweets.tokenized_tweet)

    # build the corpus i.e. vectors with the number of occurence of each word per tweet
    tweets_corpus = [tweets_dictionary.doc2bow(tweet) for tweet in tweets.tokenized_tweet]

    # compute coherence
    tweets_coherence = []
    for nb_topics in range(1,k):
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

def assign_topics_to_tweets(tweets, tweets_lda):
    tweets_dictionary = Dictionary(tweets.tokenized_tweet)
    tweets_corpus = [tweets_dictionary.doc2bow(tweet) for tweet in tweets.tokenized_tweet]
    topic_0 = []
    topic_1 = []
    topic_2 = []
    topic_3 = []
    for text in tweets['tokenized_tweet']:
        bow = tweets_dictionary.doc2bow(text)
        topic_distribution = dict(tweets_lda.get_document_topics(bow))
        if 0 in topic_distribution:
            topic_0.append(topic_distribution[0])
        else:
            topic_0.append(0)
        if 1 in topic_distribution:
            topic_1.append(topic_distribution[1])
        else:
            topic_1.append(0)
        if 2 in topic_distribution:
            topic_2.append(topic_distribution[2])
        else:
            topic_2.append(0)
        if 3 in topic_distribution:
            topic_3.append(topic_distribution[3])
        else:
            topic_3.append(0)
    tweets['topic_0'] = topic_0
    tweets['topic_1'] = topic_1
    tweets['topic_2'] = topic_2
    tweets['topic_3'] = topic_3
    return tweets
    
def perform_tot(tweets_with_topics):
    dates = tweets_with_topics.date.unique()
    tweets_per_day = tweets_with_topics.date.value_counts(sort = False)
    ordered_tweets_per_day = []
    for day in dates:
        ordered_tweets_per_day.append(tweets_per_day[day])

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
    return tot
