import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from torchmetrics.classification import Precision, Recall, F1Score
from torchmetrics.classification import AveragePrecision
from transformers import AutoModelForSequenceClassification

class TextClassifier(pl.LightningModule):
    def __init__(self, 
                 model_url, 
                 learning_rate,
                 weight_decay,
                 num_labels, 
                 loss):
        super().__init__()
        self.classifier = AutoModelForSequenceClassification.from_pretrained(model_url)
        self.softmax = nn.Softmax(dim=1)

        if loss == "CE_loss":
            self.loss_fn = nn.CrossEntropyLoss()
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.num_labels = num_labels

        self.train_losses = []
        self.val_losses = []

        metrics = MetricCollection({
            "auroc": AveragePrecision(task="binary", num_labels=self.num_labels, average="micro"),
            "precision_macro": Precision(task="binary", num_labels=self.num_labels, average="macro"),
            "precision_micro": Precision(task="binary", num_labels=self.num_labels, average="micro"),
            "recall_macro": Recall(task="binary", num_labels=self.num_labels, average="macro"),
            "recall_micro": Recall(task="binary", num_labels=self.num_labels, average="micro"),
            "f1_macro": F1Score(task="binary", num_labels=self.num_labels, average="macro"),
            "f1_micro": F1Score(task="binary", num_labels=self.num_labels, average="micro"),
            "f1_per_label": F1Score(task="binary", num_labels=self.num_labels, average="none"),
        })
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(self, x):
        outputs = self.classifier(**x).logits
        return outputs

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self._common_step(batch, batch_idx)
        self.train_losses.append(loss)
        self.train_metrics.update(preds=preds, target=targets)
        
    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_losses).mean()
        self.log_dict({"train_loss": avg_loss}, on_step=False, on_epoch=True, prog_bar=True)
        self.train_losses = []
        metrics = self.train_metrics.compute()
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        self.train_metrics.reset()
        
    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self._common_step(batch, batch_idx)
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
        _, preds, targets = self._common_step(batch, batch_idx)
        self.test_metrics.update(preds=preds, target=targets)
    
    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        self.test_metrics.reset()

    def _common_step(self, batch, batch_idx):
        x = batch["encoded_text"]
        y = batch["label"]
        outputs = self.forward(x)
        loss = self.loss_fn(input=outputs, target=y)
        y_prob = self.softmax(outputs)[:,1].squeeze(-1)
        return loss, y_prob, y
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
