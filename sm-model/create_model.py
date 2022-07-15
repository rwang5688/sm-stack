!pip install torch
!pip install watermark
!pip install transformers
!pip install --upgrade pytorch-lightning
!pip install colored

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
from preprocessing import clean_data
from preprocessing import split_data


#load cased-BERT model and tokenizer 

def label_cols(dataframe):
    LABEL_COLUMNS = dataframe.columns.tolist()[2:]
    return LABEL_COLUMNS

LABEL_COLUMNS = label_cols(df)



def create_tokenizer():
    model = 'bert-base-cased'
    BERT_MODEL_NAME = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    return tokenizer 

tokenizer = create_tokenizer()

class TweetsDataset(Dataset):
    def __init__(
    self,
    data: pd.DataFrame,
    tokenizer: BertTokenizer,
    max_token_len: int = 128
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        comment_text = data_row.tweet
        labels = data_row[LABEL_COLUMNS]
        encoding = self.tokenizer.encode_plus(
          comment_text,
          add_special_tokens=True,
          max_length=self.max_token_len,
          return_token_type_ids=False,
          padding="max_length",
          truncation=True,
          return_attention_mask=True,
          return_tensors='pt',
        )
        return dict(
          comment_text=comment_text,
          input_ids=encoding["input_ids"].flatten(),
          attention_mask=encoding["attention_mask"].flatten(),
          labels=torch.FloatTensor(labels)
        )

class TweetsDataModule(pl.LightningDataModule):
    def __init__(self, train_df, test_df, tokenizer, batch_size=8, max_token_len=128):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
    def setup(self, stage=None):
        self.train_dataset = TweetsDataset(
          self.train_df,
          self.tokenizer,
          self.max_token_len
        )
        self.test_dataset = TweetsDataset(
          self.test_df,
          self.tokenizer,
          self.max_token_len
        )
    def train_dataloader(self):
        return DataLoader(
          self.train_dataset,
          batch_size=self.batch_size,
          shuffle=True,
          num_workers=8
        )
    def val_dataloader(self):
        return DataLoader(
          self.test_dataset,
          batch_size=self.batch_size,
          num_workers=8
        )
    def test_dataloader(self):
        return DataLoader(
          self.test_dataset,
          batch_size=self.batch_size,
          num_workers=8
        )


# TweetsDataModule encapsulates all data loading logic and returns the necessary data loaders. Let’s create an instance of our data module:


def create_data_module():
    MAX_TOKEN_COUNT = 60
    N_EPOCHS = 4
    BATCH_SIZE = 32
    data_module = TweetsDataModule(
      train_df,
      val_df,
      tokenizer,
      batch_size=BATCH_SIZE,
      max_token_len=MAX_TOKEN_COUNT
    )
    return data_module

data_module = create_data_module()


#Our model will use a pre-trained BertModel and a linear layer to convert the BERT representation to a classification task. We’ll pack everything in a LightningModule:


class TweetTagger(pl.LightningModule):
    def __init__(self, n_classes: int, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.BCELoss()
    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss
    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss
    def training_epoch_end(self, outputs):
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)
                labels = torch.stack(labels).int()
                predictions = torch.stack(predictions)
            for i, name in enumerate(LABEL_COLUMNS):
                class_roc_auc = auroc(predictions[:, i], labels[:, i])
                self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        scheduler = get_linear_schedule_with_warmup(
          optimizer,
          num_warmup_steps=self.n_warmup_steps,
          num_training_steps=self.n_training_steps
        )
        return dict(
          optimizer=optimizer,
          lr_scheduler=dict(
            scheduler=scheduler,
            interval='step'
          )
        )
    
        
# We simulate 100 training steps and tell the scheduler to warm up for the first 20. The learning rate grows to the initial fixed value of 0.001 during the warm-up and then goes down (linearly) to 0.

# To use the scheduler, we need to calculate the number of training and warm-up steps. The number of training steps per epoch is equal to number of training examples / batch size. The number of total training steps is training steps per epoch * number of epochs:


def warmup_and_totaltraining_steps(dataframe):
    N_EPOCHS = 4
    BATCH_SIZE = 32
    steps_per_epoch=len(dataframe) // BATCH_SIZE
    total_training_steps = steps_per_epoch * N_EPOCHS

    # We’ll use a fifth of the training steps for a warm-up:

    warmup_steps = total_training_steps // 5
    return warmup_steps, total_training_steps

warmup_steps, total_training_steps = warmup_and_totaltraining_steps(train_df)


# TRAINING



def train_model(LABEL_COLUMNS, warmup_steps, total_training_steps, data_module):

    
    model = TweetTagger(
      n_classes=len(LABEL_COLUMNS),
      n_warmup_steps=warmup_steps,
      n_training_steps=total_training_steps
    )

    # Training 

    #The beauty of PyTorch Lightning is that you can build a standard pipeline that you like and train (almost?) every model you might imagine. I prefer to use at least 3 components.

    #Checkpointing that saves the best model (based on validation loss):

    checkpoint_callback = ModelCheckpoint(
      dirpath="checkpoints",
      filename="best-checkpoint",
      save_top_k=1,
      verbose=True,
      monitor="val_loss",
      mode="min"
    )

    # Log the progress in TensorBoard:

    logger = TensorBoardLogger("lightning_logs", name="tweets")

    # And early stopping triggers when the loss hasn’t improved for the last 2 epochs (you might want to remove/reconsider this when training on real-world projects):

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)


    # We can start the training process:

    trainer = pl.Trainer(
      checkpoint_callback=checkpoint_callback,
      callbacks=[early_stopping_callback],
      max_epochs=N_EPOCHS,
      gpus=1,
      progress_bar_refresh_rate=30
    )

    return trainer.fit(model, data_module)

train_model(LABEL_COLUMNS, warmup_steps, total_training_steps, data_module)


def create_model(LABEL_COLUMNS):
    trained_model = TweetTagger.load_from_checkpoint(
    trainer.checkpoint_callback.best_model_path,
    n_classes=len(LABEL_COLUMNS)
    )
    return trained_model

trained_model = create_model(LABEL_COLUMNS)


# We put our model into “eval” mode, and we’re ready to make some predictions. Here’s the prediction on a sample (totally fictional) comment:

def predict_labels(tokenizer, test_comment):
    encoding = tokenizer.encode_plus(
      test_comment,
      add_special_tokens=True,
      max_length=512,
      return_token_type_ids=False,
      padding="max_length",
      return_attention_mask=True,
      return_tensors='pt',
    )
    _, test_prediction = trained_model(encoding["input_ids"], encoding["attention_mask"])
    test_prediction = test_prediction.flatten().numpy()
    for label, prediction in zip(LABEL_COLUMNS, test_prediction):
        print(f"{label}: {prediction}")

    














    
    


    


    
    




