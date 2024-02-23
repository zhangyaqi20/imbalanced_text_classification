import math
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from transformers import AutoTokenizer
from typing import Optional
from utils.check_resampled_dataloader import check_resampled_dataloader

class ImbalancedDataset(Dataset):
    def __init__(self, data, tokenizer, max_token_len, label_col="label") -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.label_col = label_col

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
        label = self.data.iloc[index][self.label_col]
        return {"index": index,
                "encoded_text": {"input_ids": encoded_text["input_ids"].flatten(),
                                 "attention_mask": encoded_text["attention_mask"].flatten(),
                                 "token_type_ids": encoded_text["token_type_ids"].flatten()}, 
                "label": label}
    
class ImbalancedDataModule(pl.LightningDataModule):
    def __init__(self, 
                 data_path,
                 train_filename,
                 tokenizer_url,
                 val_filename=None,
                 test_filename=None, 
                 label_col="label",
                 batch_size=32, 
                 max_token_len=128,
                 num_workers=os.cpu_count(),
                 sampling_percentage=None) -> None:
        super().__init__()
        self.data_path = data_path
        self.train_filename = train_filename
        self.val_filename = val_filename
        self.test_filename = test_filename
        self.label_col = label_col
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_url)
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.num_workers = num_workers
        self.sampling_percentage = sampling_percentage

    def setup(self, stage: Optional[str] = None):
        # Get data splits.
        print("------ Dataset Statistics ------")
        print("Raw data reading from CSV files:")
        print(f"| Split\t | Size\t | Label Counts")
        if self.test_filename:
            test_data = self._read_from_csv(self.data_path + self.test_filename)
            self._print_data_info(test_data, "test", self.label_col)
            if self.val_filename:
                val_data = self._read_from_csv(self.data_path + self.val_filename)
                train_data = self._read_from_csv(self.data_path + self.train_filename)
                self._print_data_info(train_data, "train", self.label_col)
                self._print_data_info(val_data, "val", self.label_col)
            else:
                train_val_data = self._read_from_csv(self.data_path + self.train_filename)
                self._print_data_info(train_val_data, "train_val", self.label_col)
                train_data, val_data = train_test_split(train_val_data, test_size=0.2, random_state=0, stratify=train_val_data[self.label_col])
        else:
            data = self._read_from_csv(self.data_path + self.train_filename)
            self._print_data_info(data, "all", self.label_col)
            train_val_data, test_data = train_test_split(data, test_size=0.2, random_state=0, stratify=data[self.label_col])
            train_data, val_data = train_test_split(train_val_data, test_size=0.2, random_state=0, stratify=train_val_data[self.label_col])

        print("After split (used for training):")
        print(f"| Split\t | Size\t | Label Counts")
        self._print_data_info(train_data, "train", self.label_col)
        self._print_data_info(val_data, "val", self.label_col)
        self._print_data_info(test_data, "test", self.label_col)

        train_data = train_data.reset_index()
        if self.sampling_percentage is not None:
            sample_weights, num_samples = self._get_oversampling_weights(train_data, ratio=self.sampling_percentage)
            print(f"Resampling with sampling_percentage = {self.sampling_percentage} => training set has {num_samples} samples.")
            self.sampler = WeightedRandomSampler(weights=sample_weights,
                                                    num_samples=num_samples,
                                                    replacement=True)
        self.train_set = ImbalancedDataset(train_data, tokenizer=self.tokenizer, max_token_len=self.max_token_len, label_col=self.label_col)
        val_data = val_data.reset_index()
        self.val_set = ImbalancedDataset(val_data, tokenizer=self.tokenizer, max_token_len=self.max_token_len, label_col=self.label_col)
        test_data = test_data.reset_index()
        self.test_set = ImbalancedDataset(test_data, tokenizer=self.tokenizer, max_token_len=self.max_token_len, label_col=self.label_col)
    
    def train_dataloader(self):
        if self.sampling_percentage is not None:
            train_dataloader_resampled = DataLoader(self.train_set, 
                                                    batch_size=self.batch_size, 
                                                    num_workers=self.num_workers, 
                                                    sampler=self.sampler,
                                                    shuffle=False
                                                    )
            check_resampled_dataloader(train_dataloader_resampled)
            return train_dataloader_resampled
        else:
            return DataLoader(self.train_set, 
                            batch_size=self.batch_size, 
                            num_workers=self.num_workers, 
                            shuffle=True
                            )

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
    
    def _read_from_csv(self, data_name):
        if "tsv" in data_name:
            data = pd.read_csv(data_name,
                               sep='\t',
                               encoding = "utf-8",
                               engine = "python",
                               header = 0)
        elif "csv" in data_name:
            data = pd.read_csv(data_name,
                            encoding = "utf-8",
                            engine = "python",
                            header = 0)
        else:
            raise NotImplementedError("Given data file type is not supported yet.")
        return data
    
    def _print_data_info(self, data, split, label_col):
        label_counts = data[label_col].value_counts().to_dict()
        output = f"| {split}\t | {len(data)} |"
        for label in sorted(label_counts.keys()):
            output += f"\t{label}: {label_counts[label]}, "
            output += "{:.1%}".format(label_counts[label]/len(data))
        print(output)

    def _get_oversampling_weights(self, data, ratio):
        label_counts = data[self.label_col].value_counts().to_dict()

        # num_minority_class = (len(data) - label_counts[1]) * ratio
        # print(num_minority_class)
        # num_minority_class_per_sample = math.ceil(num_minority_class / label_counts[1])
        # num_per_sample = [1 if y == 0 else num_minority_class_per_sample for y in data[self.label_col]]
        # num_after_sampling = sum(num_per_sample)
        # return num_per_sample, num_after_sampling

        class_weights = [1.0 / label_counts[label] for label in range(len(label_counts))]
        sample_weights = [class_weights[y] for y in data[self.label_col]]
        num_after_sampling = int(ratio * len(data)) + len(data)
        return sample_weights, num_after_sampling