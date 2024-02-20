import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Optional

class ImbalancedDataset(Dataset):
    def __init__(self, data, tokenizer, max_token_len) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        text = self.data.iloc[index]["text"]
        encoded_text = self.tokenizer(
            text,
            padding="max_length",
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_token_len,
            return_tensors="pt"
        )
        label = self.data.iloc[index]["label"]
        return {"encoded_text": {"input_ids": encoded_text["input_ids"].flatten(),
                                 "attention_mask": encoded_text["attention_mask"].flatten(),
                                 "token_type_ids": encoded_text["token_type_ids"].flatten()}, 
                "label": label}
    
class ImbalancedDataModule(pl.LightningDataModule):
    def __init__(self, 
                 data_path, 
                 tokenizer_url, 
                 batch_size, 
                 max_token_len,
                 num_workers=os.cpu_count()) -> None:
        super().__init__()
        self.data_path = data_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_url)
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        data = pd.read_csv(self.data_path,
                           encoding = "utf-8",
                           engine = "python",
                           header = 0)[:100]
        self.num_labels = data["label"].nunique()
        # split the whole data into train/val/test (6/2/2)
        train_data, val_data, test_data = np.split(data.sample(frac=1, random_state=0), 
                                                   [int(.6*len(data)), 
                                                    int(.8*len(data))])
        self.train_set = ImbalancedDataset(train_data, tokenizer=self.tokenizer, max_token_len=self.max_token_len)
        self.val_set = ImbalancedDataset(val_data, tokenizer=self.tokenizer, max_token_len=self.max_token_len)
        self.test_set = ImbalancedDataset(test_data, tokenizer=self.tokenizer, max_token_len=self.max_token_len)
    
    def train_dataloader(self):
        return DataLoader(self.train_set, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          shuffle=False)