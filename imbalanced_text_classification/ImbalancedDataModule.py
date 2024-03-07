import math
import numpy as np
import nlpaug.augmenter.word.context_word_embs as nawcwe
import nlpaug.augmenter.word.synonym as nawsyn
import os
import re
import pandas as pd
import pytorch_lightning as pl
import torch
from sampler.ModifiedRandomSampler import ModifiedRandomSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from transformers import AutoTokenizer
from typing import Optional
from utils.utils import check_dataloader_label_counts
from augmenter.AbusiveLexiconAugmenter import AbusiveLexiconAugmenter
from augmenter.augmentation_src.external_data.external_datamap import category2data_label, data_name2label_col

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
                 data_name,
                 data_path,
                 train_filename,
                 tokenizer_url,
                 val_filename=None,
                 test_filename=None, 
                 label_col="label",
                 batch_size=32, 
                 max_token_len=128,
                 num_workers=os.cpu_count(),
                 sampling_weightedRS_percentage=None,
                 sampling_modifiedRS_mode=None,
                 sampling_modifiedRS_rho=None,
                 augmentation_rho=None,
                 augmentation_src=None,
                 augmentation_percentage=None,
                 augmentation_top_k=None,
                 augmentation_categories=None,
                 preprocessing=False) -> None:
        super().__init__()
        self.data_name = data_name
        self.data_path = data_path
        self.train_filename = train_filename
        self.val_filename = val_filename
        self.test_filename = test_filename
        self.label_col = label_col
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_url)
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.num_workers = num_workers
        self.sampling_weightedRS_percentage = sampling_weightedRS_percentage
        self.sampling_modifiedRS_mode = sampling_modifiedRS_mode
        self.sampling_modifiedRS_rho = sampling_modifiedRS_rho
        self.augmentation_rho = augmentation_rho
        self.augmentation_src = augmentation_src
        self.augmentation_percentage = augmentation_percentage
        self.augmentation_top_k = augmentation_top_k
        self.augmentation_categories = augmentation_categories
        self.preprocessing = preprocessing

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def setup(self, stage: Optional[str] = None):
        # Get data splits.
        
        if not self.train_set and not self.val_set and not self.test_set:
            print("Calling ImbalancedDataModule.setup() for train...")
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
                    train_data, val_data = train_test_split(train_val_data, test_size=0.25, random_state=0, stratify=train_val_data[self.label_col])
            else:
                data = self._read_from_csv(self.data_path + self.train_filename)
                self._print_data_info(data, "all", self.label_col)
                train_val_data, test_data = train_test_split(data, test_size=0.2, random_state=0, stratify=data[self.label_col])
                train_data, val_data = train_test_split(train_val_data, test_size=0.25, random_state=0, stratify=train_val_data[self.label_col])
            print("After splitting for training:")
            print(f"| Split\t | Size\t | Label Counts")
            self._print_data_info(train_data, "train", self.label_col)
            self._print_data_info(val_data, "val", self.label_col)
            self._print_data_info(test_data, "test", self.label_col)

            train_data = train_data.reset_index()
            if self.augmentation_rho is not None:
                train_data = self.augmentation(train_data)
                train_data = train_data.reset_index()
            if self.preprocessing:
                train_data = self.data_preprocessing(train_data)
            self.train_set = ImbalancedDataset(train_data, tokenizer=self.tokenizer, max_token_len=self.max_token_len, label_col=self.label_col)
            self.sampler = None
            if self.sampling_weightedRS_percentage is not None:
                sample_weights, num_samples = self._get_oversampling_weights(train_data, ratio=self.sampling_weightedRS_percentage)
                print(f"Resampling with sampling_weightedRS_percentage = {self.sampling_weightedRS_percentage} => training set has {num_samples} samples.")
                self.sampler = WeightedRandomSampler(weights=sample_weights,
                                                        num_samples=num_samples,
                                                        replacement=True)
            if self.sampling_modifiedRS_rho is not None:
                self.sampler = ModifiedRandomSampler(dataset=self.train_set,
                                                     rho_target=self.sampling_modifiedRS_rho,
                                                     mode=self.sampling_modifiedRS_mode)
            val_data = val_data.reset_index()
            self.val_set = ImbalancedDataset(val_data, tokenizer=self.tokenizer, max_token_len=self.max_token_len, label_col=self.label_col)
            test_data = test_data.reset_index()
            self.test_set = ImbalancedDataset(test_data, tokenizer=self.tokenizer, max_token_len=self.max_token_len, label_col=self.label_col)
        else: 
            print("Calling ImbalancedDataModule.setup() for validation/test...")
    
    def train_dataloader(self):
        if self.sampler is not None:
            train_dataloader_resampled = DataLoader(self.train_set, 
                                                    batch_size=self.batch_size, 
                                                    num_workers=self.num_workers, 
                                                    sampler=self.sampler,
                                                    shuffle=False
                                                    )
            label_counts = check_dataloader_label_counts(train_dataloader_resampled)
            print(f"Label counts after resampling: {label_counts}")
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
    
    def augmentation(self, train_data):
        # Augment the training data to a imbalance rate rho
        # - Configure aug object
        # augmentation_src = "Bert" # "WordNet"/"Bert"/"AbusiveLexicon/"ExternalDatal"
        if self.augmentation_src == "ExternalData":
            augmented_train_data_name = f"{self.data_path}/{self.augmentation_src}Aug-rho={self.augmentation_rho}_train_data.csv"
        else:
            augmented_train_data_name = f"{self.data_path}/{self.augmentation_src}Aug-rho={self.augmentation_rho}-aug_p={self.augmentation_percentage}-top_k={self.augmentation_top_k}_train_data.csv"
        if os.path.isfile(augmented_train_data_name):
            print(f"Loading existing augmented train data from {augmented_train_data_name}")
            train_augmented = self._read_from_csv(augmented_train_data_name)
        else:
            # Label count
            label_counts = dict(train_data[self.label_col].value_counts())
            label2indices = dict()
            for label in label_counts.keys():
                label2indices[label] = train_data.index[train_data[self.label_col] == label].tolist()
            print(f"Original label counts: {label_counts}")
            max_class = max(label_counts, key=label_counts.get)
            num_samples_other_classes = math.ceil(label_counts[max_class] / self.augmentation_rho)
            labels_to_augment = [label for label, count in label_counts.items() if count < num_samples_other_classes]
            print(f"Labels {labels_to_augment} need to be augmented")

            if self.augmentation_src == "ExternalData": # TODO test
                print(f"Augmenting train data with {self.augmentation_src} with target rho={self.augmentation_rho} ...")
                aug_label2external_data = dict()
                print(f"Categories to augment = {self.augmentation_categories}")
                for aug_cat in self.augmentation_categories:
                    data_labels_to_replace = category2data_label[aug_cat]
                    print(f"- Check {aug_cat} with {data_labels_to_replace}")
                    aug_label = data_labels_to_replace[self.data_name]
                    if aug_label in labels_to_augment:
                        print(f"- Start concatenating external data of label {aug_label}")
                        external_data_aug_concat = pd.DataFrame()
                        for external_data_name, external_data_label in data_labels_to_replace.items():
                            if external_data_name != self.data_name:
                                train_file_path = f"data/{external_data_name}/train_clean.csv"
                                if os.path.isfile(train_file_path):
                                    external_data = self._read_from_csv(train_file_path)
                                else:
                                    all_file_path = f"data/{external_data_name}/data_clean.csv"
                                    external_data = self._read_from_csv(all_file_path)
                                external_data_aug = external_data[external_data[data_name2label_col[external_data_name]] == external_data_label]
                                external_data_aug_concat = pd.concat([external_data_aug_concat, external_data_aug], ignore_index=True)
                                print(f" -- Concatenated label {external_data_label} of {external_data_name}")
                        aug_label2external_data[aug_label] = external_data_aug_concat
                
                augmented_data = {"text": [], self.label_col: []}
                for aug_label in aug_label2external_data.keys():
                    num_aug = num_samples_other_classes - label_counts[aug_label]
                    external_data_aug = aug_label2external_data[aug_label]
                    augmented_indices = torch.randint(0, len(external_data_aug), (num_aug,)).tolist()
                    augmented_data["text"] += external_data_aug.iloc[augmented_indices].text.values.tolist()
                    augmented_data[self.label_col] += [aug_label] * len(external_data_aug.iloc[augmented_indices]) # Need to use label of current dataset
                    assert len(augmented_data["text"]) == len(augmented_data[self.label_col])

            else:
                aug_batch_size = 32
                if self.augmentation_src == 'Bert' or self.augmentation_src == 'WordNet': 
                    print(f"Augmenting train data with {self.augmentation_src}" 
                        f"(aug_p={self.augmentation_percentage} top_k={self.augmentation_top_k}) with target rho={self.augmentation_rho} ...")
                    if self.augmentation_src == 'WordNet':
                        aug = nawsyn.SynonymAug(aug_src='wordnet', aug_p=self.augmentation_percentage)
                    elif self.augmentation_src == 'Bert':
                        aug = nawcwe.ContextualWordEmbsAug(
                                model_path="GroNLP/hateBERT",
                                model_type='bert',
                                action='substitute',
                                top_k=self.augmentation_top_k,
                                aug_p=self.augmentation_percentage,
                                device="cpu",
                                batch_size=aug_batch_size
                            )
                    #  - Find out which texts to augment
                    augmented_indices = []
                    for label, indices in label2indices.items():
                        print(f"\nCheck label {label}")
                        if label in labels_to_augment: # only class with less samples need to be augmented
                            print(f"Augmenting from {label_counts[label]} to {num_samples_other_classes}")
                            num_aug = num_samples_other_classes - label_counts[label] # the augmented class = original samples + num_aug augmented samples
                            # randomly choose num_aug samples for augmentation
                            augmented_indices_index = torch.randint(0, len(indices), (num_aug,)).tolist()
                            augmented_indices_of_label = torch.tensor(indices)[augmented_indices_index].tolist()
                            augmented_indices += augmented_indices_of_label
                    print(f"In total need to augment {len(augmented_indices)} samples.")

                elif self.augmentation_src == "AbusiveLexicon":
                    print(f"Augmenting train data with abuisve lexicon" 
                        f"(aug_p={self.augmentation_percentage} top_k={self.augmentation_top_k}) with target rho={self.augmentation_rho} ...")
                    aug = AbusiveLexiconAugmenter(aug_p=self.augmentation_percentage, top_k=self.augmentation_top_k)
                    #  - Find out which texts to augment
                    augmented_indices = []
                    for label, indices in label2indices.items():
                        print(f"\nCheck label {label}")
                        if label in labels_to_augment: # only class with less samples need to be augmented
                            print(f"Augmenting from {label_counts[label]} to {num_samples_other_classes}")
                            num_aug = num_samples_other_classes - label_counts[label] # the augmented class = original samples + num_aug augmented samples
                            # findout which indices contain the words in abusive lexicon
                            indices_containing_abusive_words = [index for index in indices if aug.check_if_text_contain_lexicon_words(train_data.loc[index, "text"])]
                            # randomly choose num_aug samples for augmentation
                            augmented_indices_index = torch.randint(0, len(indices_containing_abusive_words), (num_aug,)).tolist()
                            augmented_indices_of_label = torch.tensor(indices_containing_abusive_words)[augmented_indices_index].tolist()
                            augmented_indices += augmented_indices_of_label
                    print(f"In total need to augment {len(augmented_indices)} samples.")
                
                else:
                    raise NotImplementedError(f"Required augmentation source {self.augmentation_src} is not supported.")
            
                # Apply augmenter
                augmented_data = {"text": [], self.label_col: []}
                for i in range(0, len(augmented_indices), aug_batch_size):
                    indices = augmented_indices[i:i+aug_batch_size]
                    texts = [train_data.loc[index, "text"] for index in indices]
                    labels = [train_data.loc[index, self.label_col] for index in indices]
                    augmented_data["text"] += aug.augment(data=texts, n=1)
                    augmented_data[self.label_col] += labels

            df_augmented_data = pd.DataFrame(augmented_data)
            train_augmented = pd.concat([train_data, df_augmented_data], ignore_index=True)
            train_augmented.to_csv(augmented_train_data_name, index=False)

        print(f"After augmentation: {dict(train_augmented[self.label_col].value_counts())}")
        return train_augmented
    
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
    
    def data_preprocessing(self, train_data):
        print(f"Conduting preprocessing ...")
        train_data["text"] = train_data.apply(lambda row: self._preprocessing(row.text), axis=1)
        return train_data

    def _preprocessing(self, text):
        # text = ("Alex URL Brosas <URL> (URL) another someemail@gmail.com idiot @user 😩😩🙌🏽💃🏽SHE"
        #         "#ALDUBKSGoesToUS  https://t.co/14G7hFwVQm" 
        #         "<MENTION_1> <MENTION_2> Talk about NYers like that MENTION3🙌🏽🙌🏽 AFTER you've survived 9/11.")
        # output: "alex  brosas   another someemail@gmail.com idiot she#aldubksgoestous    talk about nyers like that  after you've survived 9/11."
        text = text.lower()
        # Remove URLS: "httpxxxx", "URL", 
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"\S*url\S*", "", text)
        # # Remove user mention: "@user_name", "@user", "MENTION3", "<MENTION_3>"
        text = re.sub(r" @\S+", "", text)
        text = re.sub(r"<mention_[0-9]+>", "", text)
        text = re.sub(r"mention[0-9]+", "", text)
        # # Remove emojis
        emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        return text