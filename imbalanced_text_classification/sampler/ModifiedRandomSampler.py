import torch
from collections import defaultdict
import math
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from typing import Iterator

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ModifiedRandomSampler(Sampler[int]):
    r"""Random Over/Undersampler.

    Args:
        dataset: dataset to sample from
        rho_target (float): The target imbalance ratio to achieve: num_samples_max_class / num_samples_min_class
        mode (float): "oversampling" or "undersampling"
        resampling_per_epoch (bool): Whether to resample each epoch
    """

    def __init__(self, 
                 dataset, 
                 rho_target: float,
                 mode: str
                 ) -> None:
        self.dataset = dataset
        self.rho_target = rho_target
        self.mode = mode

        self.label2indices = defaultdict(lambda: [])
        for item in self.dataset:
            self.label2indices[item["label"]].append(item["index"])
        self.label_counts = {label: len(indices) for label, indices in self.label2indices.items()}
        max_class = max(self.label_counts, key=self.label_counts.get)
        min_class = min(self.label_counts, key=self.label_counts.get)

        if self.mode == "oversampling":
            self.num_samples_other_classes = math.ceil(self.label_counts[max_class] / self.rho_target)
            self.pivot_class = max_class
        elif self.mode == "undersampling":
            self.num_samples_other_classes = math.ceil(self.label_counts[min_class] * self.rho_target)
            self.pivot_class = min_class
        num_classes = len(self.label_counts)
        self.len_resampled_data = (num_classes - 1) * self.num_samples_other_classes + self.label_counts[self.pivot_class]
        print(f"Original label_counts: {self.label_counts}")
        print(f"Current imbalance rho = {self.label_counts[max_class] / self.label_counts[min_class]}")
        print(f"Random {self.mode} with sampling_modifiedRS_rho = {self.rho_target} => training set has {self.len_resampled_data} samples.")
        
        self.resampled_indices = []

    def __resampling_classes(self):
        self.resampled_indices = torch.tensor([], dtype=int)
        for label, indices in self.label2indices.items():
            if (label != self.pivot_class and # if not the maximum/minimum class
                ((self.mode == "oversampling" and len(indices) < self.num_samples_other_classes) # if has less/more samples than required
                 or (self.mode == "undersampling" and len(indices) > self.num_samples_other_classes))):
                sample_index_indexes = torch.randint(0, len(indices), (self.num_samples_other_classes,)).tolist()
                resampled_indices_label = torch.tensor(indices)[sample_index_indexes]
                self.resampled_indices = torch.cat((self.resampled_indices, resampled_indices_label))
            else:
                self.resampled_indices = torch.cat((self.resampled_indices, torch.tensor(indices)))
        indexes_shuffle = torch.randperm(self.resampled_indices.shape[0])
        self.resampled_indices = self.resampled_indices[indexes_shuffle].tolist()

    def __iter__(self) -> Iterator[int]:
        self.__resampling_classes()
        return iter(self.resampled_indices)

    def __len__(self) -> int:
        return self.len_resampled_data
    
# # Sampler Test:
# class MyDataset(Dataset):
#     def __init__(self):
#         self.data = [f"text{i}" for i in range(10)]
#         self.labels = [0,0,0,1,0,1,0,0,0,0,0]
        
#     def __getitem__(self, index):
#         return {
#             "index": index,
#             "text": self.data[index],
#             "label": self.labels[index]
#         }
    
#     def __len__(self):
#         return len(self.data)
    
# dataset = MyDataset()
# loader = DataLoader(dataset, batch_size=5)
# for x in loader:
#     print(x)
    
# loader = DataLoader(dataset, batch_size=5, sampler=ModifiedRandomSampler(dataset, 1, "undersampling"))

# # works
# for x in loader:
#     print(x)
    
# print(len(loader))