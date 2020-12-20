import os
import json
import shutil
import re
import pandas as pd

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
                        tweet_date = tweet["created_at"]

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

                        tweet_list.append([tweet_id, username, tweet_date, tweet_text, tweet_hashtags, tweet_tags])
    tweets = pd.DataFrame(data=tweet_list, columns=["id", "username", "date", "tweet", "hashtags", "tags"])
    return tweets

def clean_data(tweets):
    print("Size of dataset before cleaning: {}".format(tweets.shape))

    # Find Retweets
    tweets['is_retweet'] = tweets.tweet.apply(isRetweet)
    print("Amount of retweets: {}".format( sum(tweets.is_retweet)  ))

    # Find Duplicates
    tweets['is_duplicate'] = tweets.duplicated(subset='tweet',keep='first')
    print("Amount of duplicates: {}".format( sum(tweets.is_duplicate) ))

    tweets['to_keep'] = ~ (tweets.is_retweet | tweets.is_duplicate)

    # Count most frequent usernames:
    counts = tweets.username.value_counts()
    #print(counts.nlargest(20))

    # Delete RTs, duplicates:
    tweets = tweets[tweets.to_keep]
    print("Dataset size after cleanup: {}".format(tweets.shape))

    return tweets

def preprocess(tweets):
    return NotImplementedError

def isRetweet(string):
    return bool(re.search("^RT ", string))


    