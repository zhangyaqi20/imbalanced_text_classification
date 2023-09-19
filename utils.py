import pytorch_lightning as pl

class UpdateTrainData(pl.Callback):
  def __init__(self, reload_dataloaders_every_n_epochs=1):
    self.reload_dataloaders_every_n_epochs = reload_dataloaders_every_n_epochs

  def on_train_epoch_end(self, trainer, model):
    if trainer.current_epoch % self.reload_dataloaders_every_n_epochs == 0:
        trainer.datamodule.train_set = [{"encoded_text": {"input_ids": model.train_set_resampled["encoded_text"]["input_ids"][idx], 
                                  "attention_mask": model.train_set_resampled["encoded_text"]["attention_mask"][idx]},
                          "label": label} for idx, label in enumerate(model.train_set_resampled['label'])]
        model.train_set_resampled = dict()