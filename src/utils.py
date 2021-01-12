import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math

def plotCoherence(tweets_coherence):

    plt.figure(figsize=(10,5))
    plt.plot(range(1,len(tweets_coherence)+1),tweets_coherence)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.savefig('coherence.png')

def plot_top_words(lda, k, nb_words=10):
    top_words = [[word for word,_ in lda.show_topic(topic_id, topn=50)] for topic_id in range(lda.num_topics)]
    top_betas = [[beta for _,beta in lda.show_topic(topic_id, topn=50)] for topic_id in range(lda.num_topics)]

    gs  = gridspec.GridSpec(round(math.sqrt(k))+1,round(math.sqrt(k))+1)
    gs.update(wspace=0.5, hspace=0.5)
    plt.figure(figsize=(20,15))
    for i in range(k):
        ax = plt.subplot(gs[i])
        plt.barh(range(nb_words), top_betas[i][:nb_words], align='center',color='blue', ecolor='black')
        ax.invert_yaxis()
        ax.set_yticks(range(nb_words))
        ax.set_yticklabels(top_words[i][:nb_words])
        plt.title("Topic "+str(i))
    plt.savefig('LDA_output_with_{}_topics.png'.format(k))


def plot_tweets_per_day(dates, tweets_per_day):
    
    plt.xticks(rotation = 90)
    plt.plot(dates, tweets_per_day)
    plt.title("Tweets Per Day related to COVID-19")
    plt.ylabel("Tweet Count")
    plt.savefig('tweets_per_day.png')

def plot_results(tot):
    tot = tot.drop(['sum'], axis = 1)
    plt.figure
    tot.plot(x = 'date')
    plt.savefig('test_result_plot.png')
