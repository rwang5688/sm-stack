import sys
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split


def clean_data(dataframe, column):
    """cleans the data by remove urls from any tweets, remove usernames ("@etc"), emojis, and remove all numbers"""
    
    dataframe[column] = dataframe[column].apply(lambda tweet: re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", tweet))
    dataframe[column] = dataframe[column].apply(lambda tweet: re.sub('@[^\s]+','',tweet))
    dataframe[column] = dataframe[column].replace(to_replace=r'\d+', value='', regex = True)
    # Removing punctuation:
    dataframe[column] = dataframe[column].str.replace('[^A-Za-z0-9 ]+','')
    # removing emojis 
    emoji_pattern = re.compile("["
          u"\U0001F600-\U0001F64F"  # emoticons
          u"\U0001F300-\U0001F5FF"  # symbols & pictographs
          u"\U0001F680-\U0001F6FF"  # transport & map symbols
          u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "]+", flags=re.UNICODE)
    dataframe[column] = dataframe[column].apply(lambda tweet: emoji_pattern.sub(r'', tweet)) # no emoji
    # removing RT from tweets
    dataframe[column] = dataframe[column].apply(lambda tweet: tweet.replace('RT', ''))
    return dataframe



def split_data(dataframe):
    """splits the dataframe into train and test dataframes"""
    train_df, val_df = train_test_split(dataframe, test_size=0.10)
    return train_df, val_df


