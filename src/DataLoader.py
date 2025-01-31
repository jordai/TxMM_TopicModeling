import os
import json
import shutil
import re
import pandas as pd
from nltk import download, FreqDist
from nltk.tokenize import TweetTokenizer
from nltk import ngrams
from nltk.corpus import stopwords

def fix_json(dir_path):
    i = 0
    for subdir, dirs, files in os.walk(dir_path):
        for file in files:
            with open(subdir+"/"+file) as f:
                if not f.name.endswith('Store'):
                    contents = ['[\n']
                    for line in f.readlines():
                        contents.append(line + ',')
                    contents[-1] = contents[-1][:-1]
                    contents.append(']')
                    with open(subdir+'/'+file+'.json', 'w') as new_file:
                        new_file.writelines(contents)
                    new_file.close()

def load_data(dir_path):
    tweet_list = []
    for subdir, dirs, files in os.walk(dir_path):
        for file in files:
            if not file.endswith("Store"):
                with open(subdir+"/"+file, encoding="utf8", errors='ignore') as f:
                    data = json.load(f)
                    for tweet in data:
                        tweet_id = tweet["id"]
                        username = tweet["user"]["screen_name"]

                        # Extracting month + day
                        date_regex = '[a-zA-Z]{3}\s+[0-9]{2}'
                        date = re.search(date_regex, tweet["created_at"])
                        if date is not None:
                            tweet_date = date[0]
                            month = date[0][0:3]
                            day = int(date[0][-2:])
                        # else:
                        #     tweet_date = ""
                        if tweet["truncated"]:
                            tweet_text = tweet["extended_tweet"]["full_text"]
                        else:
                            tweet_text = tweet['text']

                        tweet_hashtags = []
                        for hashtag in tweet['entities']['hashtags']:
                            tweet_hashtags.append(hashtag['text'])

                        tweet_tags = []
                        for tag in tweet['entities']['user_mentions']:
                            tweet_tags.append(tag['name'])

                        tweet_list.append([tweet_id, month, day,  username, tweet_date, tweet_text, tweet_hashtags, tweet_tags])
    tweets = pd.DataFrame(data=tweet_list, columns=["id","month", "day", "username", "date", "tweet", "hashtags", "tags"])
    tweets['month'] = pd.Categorical(tweets['month'], ["Feb", "Mar", "Apr", "May"])
    tweets.sort_values(["month", "day"], ascending = [True, True], inplace=True)
    return tweets

def clean_data(tweets, spam_threshold = 200):
    print("Size of dataset before cleaning: {}".format(tweets.shape))

    # Find Retweets
    tweets['is_retweet'] = tweets.tweet.apply(isRetweet)
    print("Amount of retweets: {}".format( sum(tweets.is_retweet)  ))

    # Find Duplicates
    tweets['is_duplicate'] = tweets.duplicated(subset='tweet',keep='first')
    print("Amount of duplicates: {}".format( sum(tweets.is_duplicate) ))

    # Find Spam accounts:
    counts = tweets.username.value_counts()
    counts = counts[counts > spam_threshold]
    counts_list = list(counts.to_dict())
    tweets['is_from_spammer'] = tweets['username'].isin(counts_list)
    print("Amount of spam tweets: {}, tweeted by {} accounts".format( sum(tweets.is_from_spammer), len(counts_list) ))

    # Delete RTs, duplicates:
    tweets['to_keep'] = ~ (tweets.is_retweet | tweets.is_duplicate | tweets.is_from_spammer)
    tweets = tweets[tweets.to_keep]

    tweets.tweet = tweets.tweet.map(replace_username)
    tweets.tweet = tweets.tweet.map(replace_hashtag)
    tweets.tweet = tweets.tweet.map(replace_link)
    tweets.tweet = tweets.tweet.map(exclude_emojis)
    print("Dataset size after cleanup: {}".format(tweets.shape))

    return tweets.drop(['is_retweet','is_duplicate','is_from_spammer','to_keep'], axis = 1)

def preprocess(tweets):
    tokenizer = TweetTokenizer(preserve_case=True, reduce_len=True, strip_handles= False)
    tweets['tokenized_tweet'] = tweets['tweet'].apply(tokenizer.tokenize)
    
    # Dutch Stopwords
    nl_stoplist = stopwords.words('dutch')
    nl_stoplist.append('hashtag')
    nl_stoplist.append('link')
    nl_stoplist.append('username')
    nl_stoplist.append('corona')
    nl_stoplist.append('covid')
    nl_stoplist.append('coronavirus')
    # English stopwords
    en_stoplist = stopwords.words('english')
    en_stoplist.remove("who")
    
    stop = set( en_stoplist + nl_stoplist )
    tweets['tokenized_tweet'] = tweets['tokenized_tweet'].apply(lambda x: [word.lower() for word in x if word.lower() not in stop and len(word)>2])
    return tweets

def isRetweet(string):
    return bool(re.search("^RT ", string))

def replace_username(string):
    return re.sub('@(\w){1,15}','USERNAME',string)

def replace_hashtag(string):
    return re.sub('#(\w)*','HASHTAG',string)

def replace_link(string):
    return re.sub('(https:)?//t.co/\w*', 'LINK',string)

def exclude_emojis(string):
    return re.sub('\W', ' ', string)
    