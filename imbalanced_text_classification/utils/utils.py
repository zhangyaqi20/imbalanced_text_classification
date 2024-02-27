# https://towardsdatascience.com/demystifying-pytorchs-weightedrandomsampler-by-example-a68aceccb452
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def check_dataloader_label_counts(dl):
    labels = []
    for i, batch in enumerate(dl):
        labels += batch["label"].tolist()
    label_counts = dict(Counter(labels))
    return label_counts