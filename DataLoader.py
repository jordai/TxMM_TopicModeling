import os
import json
import shutil
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
                        tweet_list.append([tweet_id, tweet_date, tweet_text, tweet_hashtags, tweet_tags])
    tweets = pd.DataFrame(data=tweet_list, columns=["id", "date", "tweet", "hashtags", "tags"])
    return tweets