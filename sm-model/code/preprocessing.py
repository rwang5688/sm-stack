import sys
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split

from tqdm.auto import tqdm
import torch

import torchmetrics
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

import pytorch_lightning as pl
from torchmetrics.functional import accuracy, auroc
from torchmetrics.functional import f1_score
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from pylab import rcParams
from matplotlib import rc
import re


def clean_data(dataframe, column):
    """cleans the data by remove urls from any tweets, remove usernames ("@etc"), emojis, and remove all numbers"""
    
    print(f"clean_data_input: dataframe: {dataframe.head(10)}")
    print(f"clean_data_input: column: {column}")
    
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
    
    print(f"clean_data_output: return_val: {dataframe.head(10)}")
          
    return dataframe



def split_data(dataframe):
    """splits the dataframe into train and test dataframes"""
    print(f'split_data_input: dataframe: {dataframe.head(10)}')
    
    train_df, val_df = train_test_split(dataframe, test_size=0.10)
    
    print(f'split_data_output: train_df: {train_df.head(10)}')
    print(f'split_data_output: val_df: {val_df.head(10)}') 
    
    return train_df, val_df


def to_int(dataframe):
    
    print(f'to_int_input: dataframe: {dataframe.head(10)}')
    # remove NA valus and duplicates from rows 
    dataframe.drop_duplicates(subset='tweet',inplace=True)
    dataframe = dataframe.dropna()
    
    #convert columns to int 
    dataframe.loc[:,'general criticsm']=dataframe.loc[:,'general criticsm'].astype(int)
    dataframe.loc[:,'disability shaming']=dataframe.loc[:,'disability shaming'].astype(int)
    dataframe.loc[:,'racial prejudice']=dataframe.loc[:,'racial prejudice'].astype(int)
    dataframe.loc[:,'sexism']=dataframe.loc[:,'sexism'].astype(int)
    dataframe.loc[:,'lgbtq+ phobia']=dataframe.loc[:,'lgbtq+ phobia'].astype(int)
    
    
    print(f'to_int_output: dataframe: {dataframe.head(10)}')
    return dataframe