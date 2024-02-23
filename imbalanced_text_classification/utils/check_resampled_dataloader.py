# https://towardsdatascience.com/demystifying-pytorchs-weightedrandomsampler-by-example-a68aceccb452
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def check_resampled_dataloader(dl):
    idxs_seen = []
    classes_seen = []

    for i, batch in enumerate(dl):
        idxs_seen += batch["index"].tolist()
        classes_seen += batch["label"].tolist()

    sample_counts = dict(Counter(idxs_seen)) # TODO
    class_counts = dict(Counter(classes_seen))

    print(f"Label counts after resampling: {class_counts}")