import argparse
import os 
import json
#%pip install torch
#%pip install watermark
#%pip install transformers
#%pip install --upgrade pytorch-lightning
#%pip install colored
import sys
import pandas as pd
import numpy as np
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

# import predefined functions
from preprocessing import clean_data, split_data, to_int

from create_model import (
    label_cols, create_tokenizer, create_data_module, 
    warmup_and_totaltraining_steps, train_model, create_model, predict_labels
)




def model(df):
    """Generate a simple model"""
    
    # make list of labeled columns
    LABEL_COLUMNS = label_cols(df)
    
    print('label_cols completed!')
    print("="*100)
    
    # create BERT tokenizer
    tokenizer = create_tokenizer()
    
    print('create_tokenizer completed!')
    print("="*100)
    
    # create data module 
    data_module = create_data_module(train_df, val_df, tokenizer)
    
    print('create_data_module completed!')
    print("="*100)

    # train model 
    warmup_steps, total_training_steps = warmup_and_totaltraining_steps(train_df)
    
    print('warmup_and_totaltraining_steps completed!')
    print('='*100)
    
#     train_model(LABEL_COLUMNS, warmup_steps, total_training_steps, data_module)
          
#     print('train_model completed!')
#     print('='*100)

    trainer = train_model(LABEL_COLUMNS, warmup_steps, total_training_steps, data_module)
    
    print('train_model completed!')
    print('='*100)

    model = create_model(LABEL_COLUMNS, trainer)
    
    print('create_model completed!')
    print('='*100)

    return model


def _load_training_data(base_dir):
    """Load Twitter training data"""
    # preprocess data
    #change this line from np.load to read_csv

    df = pd.read_csv(os.path.join(base_dir, 'multi_label_new.csv'))
    
    df = clean_data(df, 'tweet')
    
    print('clean_data completed!')
    print('='*100)

    df = to_int(df)
    
    print('to_int completed!')
    print('='*50)
    
    # split data
    train_df, val_df = split_data(df)
    
    print('split_data completed!')
    print('='*100)
    
    return train_df, val_df


# def _load_testing_data(base_dir):
#     """Load MNIST testing data"""
#     x_test = np.load(os.path.join(base_dir, 'eval_data.npy'))
#     y_test = np.load(os.path.join(base_dir, 'eval_labels.npy'))
#     return x_test, y_test


def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = _parse_args()

    train_df, val_df = _load_training_data(args.train)
    # eval_data  = _load_testing_data(args.train)

    multibert_classifier = model(train_df)

    if args.current_host == args.hosts[0]:
        # save model to an S3 directory with version number '00000001' in Tensorflow SavedModel Format
        # To export the model as h5 format use model.save('my_model.h5')
#         multibert_classifier.save(os.path.join(args.sm_model_dir, '000000001'))
        torch.save(multibert_classifier.state_dict(), os.path.join(args.sm_model_dir, '000000001'))

