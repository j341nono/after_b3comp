import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel
from transformers.tokenization_bert_japanese import BertJapaneseTokenizer
import json


class CreateDataset(Dataset):
    def __init__(self, data, tokenizer, max_token_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_raw = self.data[index]
        text = data_raw.get("sentence")
        labels = data_raw.get("writer_sentiment")

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
    def __init__(self, train_path: str, valid_path: str, test_path: str, batch_size=16, max_token_len=512, 
                 pretrained_model="cl-tohoku/bert-base-japanese-char-whole-word-masking"):
        super().__init__()
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_model)
        self.save_hyperparameters()

    def load_json(self, path: str):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(f))
        return data

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            train_data = self.load_json(self.train_path)
            valid_data = self.load_json(self.valid_path)
            self.train_dataset = CreateDataset(train_data, self.tokenizer, self.max_token_len)
            self.vaild_dataset = CreateDataset(valid_data, self.tokenizer, self.max_token_len)
        if stage == "test" or stage is None:
            test_data = self.load_json(self.test_path)
            self.test_dataset = CreateDataset(test_data, self.tokenizer, self.max_token_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count())

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=os.cpu_count())

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=os.cpu_count())


if __name__ == "__main__":
    print()