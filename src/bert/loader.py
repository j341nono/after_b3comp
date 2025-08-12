import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel
from transformers.tokenization_bert_japanese import BertJapaneseTokenizer


class CreateDataset(Dataset):
    """
    DataFrameを下記のitemを保持するDatasetに変換。
    text(原文)、input_ids(tokenizeされた文章)、attention_mask、labels(ラベル)
    """

    def __init__(self, data, tokenizer, max_token_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        text = data_row[TEXT_COLUMN]
        labels = data_row[LABEL_COLUMN]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return dict(
            text=text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.tensor(labels)
        )

class CreateDataModule(pl.LightningDataModule):
    """
    DataFrameからモデリング時に使用するDataModuleを作成
    """
    def __init__(self, train_df, valid_df, test_df, batch_size=16, max_token_len=512, 
                 pretrained_model='cl-tohoku/bert-base-japanese-char-whole-word-masking'):
        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_model)

    def setup(self):
        self.train_dataset = CreateDataset(self.train_df, self.tokenizer, self.max_token_len)
        self.vaild_dataset = CreateDataset(self.valid_df, self.tokenizer, self.max_token_len)
        self.test_dataset = CreateDataset(self.test_df, self.tokenizer, self.max_token_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count())

    def val_dataloader(self):
        return DataLoader(self.vaild_dataset, batch_size=self.batch_size, num_workers=os.cpu_count())

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=os.cpu_count())


if __name__ == "__main__":
    print()