import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from ImbDataModule import ImbDataModule
from ImbNN import ImbNN
from utils import UpdateTrainData

def main():
    pl.seed_everything(42)

    data_name = "FHC"
    data_path = f"data/{data_name}.csv"
    model_url = "bert_base_uncased"
    datamodule = ImbDataModule(
        data_path=data_path,
        tokenizer_url=model_url,
        batch_size=8,
        num_workers=0
    )

    datamodule.setup(stage="fit")
    model = ImbNN(
        model_url=model_url,
        learning_rate=1e-5,
        weight_decay=0.01,
        num_labels=datamodule.num_labels
    )

    ckpt_filename = "-{epoch:02d}-{val_f1_macro:.2f}"
    ckpt_path = "logs/class_imbalance/model_checkpoints"
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        dirpath=ckpt_path,
        filename=data_name+ckpt_filename,
        monitor="val_f1_macro",
        mode="max",
        every_n_epochs=1
    )
    logger = CSVLogger("logs", name="class_imbalance")
    reload_dataloaders_every_n_epochs = 1
    update_traindata_callback = UpdateTrainData(reload_dataloaders_every_n_epochs)
    trainer = pl.Trainer(
        accelerator="auto", devices="auto", strategy="auto",
        precision=16,
        max_epochs=5,
        num_sanity_val_steps=0,
        # overfit_batches=1,
        callbacks=[checkpoint_callback, update_traindata_callback],
        logger=logger,
        reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.validate(datamodule=datamodule, ckpt_path="best")
    datamodule.setup(stage="test")
    trainer.test(datamodule=datamodule, ckpt_path="best")

if __name__ == "__main__":
    main()