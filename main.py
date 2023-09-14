import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from ImbDataModule import ImbDataModule
from ImbNN import ImbNN

def main():
    pl.seed_everything(42)

    data_name = "TROS"
    data_path = f"data/{data_name}.csv"
    datamodule = ImbDataModule(
        data_path=data_path,
        tokenizer_url="bert-base-uncased",
        batch_size=8,
        num_workers=0
    )

    datamodule.setup(stage="fit")
    model = ImbNN(
        model_url="bert-base-uncased",
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
        filename=ckpt_filename,
        monitor="val_f1_macro",
        mode="max",
        every_n_epochs=1
    )
    logger = CSVLogger("logs", name="class_imbalance")
    trainer = pl.Trainer(
        max_epochs=10,
        num_sanity_val_steps=0,
        # overfit_batches=1,
        callbacks=[checkpoint_callback],
        logger=logger
    )

    trainer.fit(model=model, datamodule=datamodule)

if __name__ == "__main__":
    main()