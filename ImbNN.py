import math
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torchmetrics import MetricCollection, F1Score
from transformers import AutoModelForSequenceClassification

class ImbNN(pl.LightningModule):
    def __init__(self, model_url, learning_rate, weight_decay, num_labels):
        super().__init__()
        self.classifier = AutoModelForSequenceClassification.from_pretrained(model_url)
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.num_labels = num_labels

        self.train_losses = []
        self.val_losses = []

        metrics = MetricCollection({
            "f1_macro": F1Score(task="binary", num_labels=self.num_labels, average="macro")
        })
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

        self.train_set_resampled = dict()

    def forward(self, x):
        y_hat = self.classifier(**x).logits
        return y_hat

    def training_step(self, batch, batch_idx):
        loss, probs, preds, targets = self._common_step(batch, batch_idx)
        self.train_losses.append(loss)
        self.train_metrics.update(preds=preds, target=targets)
        
        batch_resampled = self._update_batch(batch, probs)
        if "label" in self.train_set_resampled.keys():
            self.train_set_resampled["label"] = torch.cat((self.train_set_resampled["label"], batch_resampled["label"]))
            encoded_text_resampled = dict()
            for k, v in self.train_set_resampled["encoded_text"].items():
                encoded_text_resampled[k] = torch.cat((self.train_set_resampled["encoded_text"][k], batch_resampled["encoded_text"][k]))
            self.train_set_resampled["encoded_text"] = encoded_text_resampled
        else:
            self.train_set_resampled = batch_resampled
        
    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_losses).mean()
        self.log_dict({"train_loss": avg_loss}, on_step=False, on_epoch=True, prog_bar=True)
        self.train_losses = []
        metrics = self.train_metrics.compute()
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        self.train_metrics.reset()
        
    def validation_step(self, batch, batch_idx):
        loss, _, preds, targets = self._common_step(batch, batch_idx)
        self.val_losses.append(loss)
        self.val_metrics.update(preds=preds, target=targets)
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_losses).mean()
        self.log_dict({"val_loss": avg_loss}, on_step=False, on_epoch=True, prog_bar=True)
        self.val_losses = []
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        _, _, preds, targets = self._common_step(batch, batch_idx)
        self.test_metrics.update(preds=preds, target=targets)
    
    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        self.test_metrics.reset()

    def _common_step(self, batch, batch_idx):
        x = batch["encoded_text"]
        y = batch["label"]
        y_hat = self.forward(x)
        probs = y_hat
        loss = self.loss_fn(y_hat.view(-1, self.num_labels), y.view(-1))
        y_hat = torch.argmax(y_hat, dim=1)
        return loss, probs, y_hat, y
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _update_batch(self, batch, probs):
        labels = batch.get("label")
        p = nn.Softmax(dim=1)(probs)
        pt = torch.Tensor([p[i, 1] if y_i == 1 else 1 - p[i, 1] for i, y_i in enumerate(labels)])
        hard_idxs = torch.nonzero(pt <= 0.5).squeeze(dim=1).tolist()
        pos_idxs = torch.nonzero(labels == 1).squeeze(dim=1).tolist()
        idxs_to_resample = [idx for idx in hard_idxs if idx in pos_idxs]
        idxs_to_keep = [idx for idx in range(len(labels)) if idx not in idxs_to_resample]

        samples_to_resample = dict()
        samples_to_keep = dict()
        encoded_text = batch.get("encoded_text")
        encoded_text_to_resample = dict()
        encoded_text_to_keep = dict()
        for k, v in encoded_text.items():
            encoded_text_to_resample[k] = v[idxs_to_resample]
            encoded_text_to_keep[k] = v[idxs_to_keep]
        samples_to_resample["encoded_text"] = encoded_text_to_resample
        samples_to_keep["encoded_text"] = encoded_text_to_keep
        samples_to_resample["label"] = labels[idxs_to_resample]
        samples_to_keep["label"] = labels[idxs_to_keep]

        if len(idxs_to_resample) > 0:
            # num_resample = (len(labels) - len(idxs_to_resample)) * 2
            num_resample_per_sample = math.ceil((len(labels) - len(idxs_to_resample))/len(idxs_to_resample))
            batch_resampled = dict()
            # oversample hard samples - labels
            batch_resampled["label"] = samples_to_resample["label"].repeat(num_resample_per_sample)
            # concatenate easy samples - labels
            batch_resampled["label"] = torch.cat((batch_resampled["label"], samples_to_keep["label"]))
            encoded_text_resampled = dict()
            for k, v in encoded_text_to_resample.items():
                # oversample hard samples - encoded_text
                encoded_text_resampled[k] = v.repeat(num_resample_per_sample, 1)
                # concatenate easy samples - encoded_text
                encoded_text_resampled[k] = torch.cat((encoded_text_resampled[k], encoded_text_to_keep[k]))
            batch_resampled["encoded_text"] = encoded_text_resampled
        else:
            batch_resampled = samples_to_keep
        return batch_resampled

