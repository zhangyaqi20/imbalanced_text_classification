import torch
from collections import defaultdict
import math
from torch.utils.data.sampler import Sampler
from typing import Iterator

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
                 mode: str,
                 resampling_per_epoch: bool = True
                 ) -> None:
        self.dataset = dataset
        self.rho_target = rho_target # 
        self.mode = mode
        self.resampling_per_epoch = resampling_per_epoch
        self.len_resampled_data = self.__compute_resampled_dataset_length()
        print(f"Resampling with sampling_modifiedRS_rho = {self.rho_target} => training set has {self.len_resampled_data} samples.")
        self.resampled_indices = None

    def __resampling_minority_classes(self):
        self.resampled_indices = []
        print(f"Original label_counts: {self.label_counts}")
        print(f"target rho = {self.rho_target}")
        if self.mode == "oversampling":
            max_class = max(self.label_counts, key=self.label_counts.get)
            num_samples_other_classes = math.ceil(self.label_counts[max_class] / self.rho_target)
            pivot_class = max_class
        else:
            min_class = min(self.label_counts, key=self.label_counts.get)
            num_samples_other_classes = math.ceil(self.label_counts[min_class] * self.rho_target)
            pivot_class = min_class

        for label, indices in self.label2indices.items():
            if label != pivot_class:
                sample_index_indexes = torch.randint(0, len(indices), (num_samples_other_classes,)).tolist()
                resampled_indices_label = torch.tensor(indices)[sample_index_indexes].tolist()
                self.resampled_indices += resampled_indices_label
            else:
                self.resampled_indices += indices
        self.resampled_indices = torch.tensor(self.resampled_indices)
        indexes_shuffle = torch.randperm(self.resampled_indices.shape[0])
        self.resampled_indices = self.resampled_indices[indexes_shuffle].tolist()

    def __iter__(self) -> Iterator[int]:
        if self.resampled_indices is None or self.resampling_per_epoch:
            self.__resampling_minority_classes()
        return iter(self.resampled_indices)

    def __len__(self) -> int:
        return self.len_resampled_data
    
    def __compute_resampled_dataset_length(self):
        self.label2indices = defaultdict(lambda: [])
        for item in self.dataset:
            self.label2indices[item["label"]].append(item["index"])
        self.label_counts = {label: len(indices) for label, indices in self.label2indices.items()}
        num_classes = len(self.label_counts)
        
        if self.mode == "oversampling":
            max_class = max(self.label_counts, key=self.label_counts.get)
            len_resampled_dataset = (num_classes - 1) * (self.label_counts[max_class] / self.rho_target) + self.label_counts[max_class]
        elif self.mode == "undersampling":
            min_class = min(self.label_counts, key=self.label_counts.get)
            len_resampled_dataset = (num_classes - 1) * (self.label_counts[min_class] * self.rho_target) + self.label_counts[min_class]

        return len_resampled_dataset
