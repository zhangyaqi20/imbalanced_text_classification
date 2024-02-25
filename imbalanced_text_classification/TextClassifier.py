import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from collections import Counter
from torchmetrics import MetricCollection
from torchmetrics.classification import Precision, Recall, F1Score, Accuracy, AveragePrecision
from transformers import AutoModelForSequenceClassification
from loss.FocalLoss import FocalLoss
from loss.Loss import Loss

class TextClassifier(pl.LightningModule):
    def __init__(self, 
                 model_url, 
                 learning_rate,
                 weight_decay,
                 num_classes, 
                 device: list,
                 loss: str,
                 wce_alpha: float = None,
                 fl_gamma: float = None
                 ):
        super().__init__()
        self.classifier = AutoModelForSequenceClassification.from_pretrained(model_url, num_labels=num_classes)

        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        if loss == Loss.CE_Loss:
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss == Loss.Weighted_CE_Loss:
            if self.num_classes == 2:
                self.weights = torch.FloatTensor([1 - wce_alpha, wce_alpha]).cuda(device=device[0])
            else:
                raise NotImplementedError
            self.loss_fn = nn.CrossEntropyLoss(weight=self.weights)
        elif loss == Loss.Focal_Loss:
            self.loss_fn = FocalLoss(gamma=fl_gamma)
        elif loss == Loss.Weighted_Focal_Loss:
            if self.num_classes == 2:
                self.weights = torch.FloatTensor([1 - wce_alpha, wce_alpha]).cuda(device=device[0])
            else:
                raise NotImplementedError
            self.loss_fn = FocalLoss(gamma=fl_gamma, alpha=self.weights)

        self.train_losses = []
        self.val_losses = []

        metrics = MetricCollection({
            "accuracy": Accuracy(task="multiclass", num_classes=self.num_classes),
            "auprc": AveragePrecision(task="multiclass", num_classes=self.num_classes),
            "precision_macro": Precision(task="multiclass", num_classes=self.num_classes, average="macro"),
            "precision_weighted": Precision(task="multiclass", num_classes=self.num_classes, average="weighted"),
            "recall_macro": Recall(task="multiclass", num_classes=self.num_classes, average="macro"),
            "recall_weighted": Recall(task="multiclass", num_classes=self.num_classes, average="weighted"),
            "f1_macro": F1Score(task="multiclass", num_classes=self.num_classes, average="macro"),
            "f1_weighted": F1Score(task="multiclass", num_classes=self.num_classes, average="weighted"),
            # "f1_per_label": F1Score(task="multiclass", num_classes=self.num_classes, average="none"), # cannot be logged as single metric
        })
        
        self.train_metrics = metrics.clone(prefix="train_")
        self.train_f1_per_label = F1Score(task="multiclass", num_classes=self.num_classes, average="none")
        self.val_metrics = metrics.clone(prefix="val_")
        self.val_f1_per_label = F1Score(task="multiclass", num_classes=self.num_classes, average="none")
        self.test_metrics = metrics.clone(prefix="test_")
        self.test_f1_per_label = F1Score(task="multiclass", num_classes=self.num_classes, average="none")

    def forward(self, x):
        outputs = self.classifier(**x).logits
        return outputs

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self._common_step(batch, batch_idx)
        self.train_losses.append(loss)
        self.train_metrics.update(preds=preds, target=targets)
        self.train_f1_per_label.update(preds=preds, target=targets)
        return loss
        
    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_losses).mean()
        self.log_dict({"train_loss": avg_loss}, on_step=False, on_epoch=True, prog_bar=True)
        self.train_losses = []
        # Metrics overall
        metrics = self.train_metrics.compute()
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.train_metrics.reset()
        # Metrics per label
        f1_per_label = self.train_f1_per_label.compute()
        for label, score in enumerate(f1_per_label):
            self.log_dict({f"train_f1_per_label_{label}": score}, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.train_f1_per_label.reset()
        return avg_loss
        
    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self._common_step(batch, batch_idx)
        self.val_losses.append(loss)
        self.val_metrics.update(preds=preds, target=targets)
        self.val_f1_per_label.update(preds=preds, target=targets)
        return loss
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_losses).mean()
        self.log_dict({"val_loss": avg_loss}, on_step=False, on_epoch=True, prog_bar=True)
        self.val_losses = []
        # Metrics overall
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_metrics.reset()
        # Metrics per label
        f1_per_label = self.val_f1_per_label.compute()
        for label, score in enumerate(f1_per_label):
            self.log_dict({f"val_f1_per_label_{label}": score}, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_f1_per_label.reset()
        return avg_loss

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self._common_step(batch, batch_idx)
        self.test_metrics.update(preds=preds, target=targets)
        self.test_f1_per_label.update(preds=preds, target=targets)
        return loss
    
    def on_test_epoch_end(self):
        # Metrics overall
        metrics = self.test_metrics.compute()
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.test_metrics.reset()
        # Metrics per label
        f1_per_label = self.test_f1_per_label.compute()
        for label, score in enumerate(f1_per_label):
            self.log_dict({f"test_f1_per_label_{label}": score}, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.test_f1_per_label.reset()

    def _common_step(self, batch, batch_idx):
        x = batch["encoded_text"]
        y = batch["label"]
        logits = self.forward(x)
        loss = self.loss_fn(input=logits, target=y)
        return loss, logits, y
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
