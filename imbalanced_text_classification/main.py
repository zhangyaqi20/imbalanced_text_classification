import mlflow
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from ImbalancedDataModule import ImbalancedDataModule
from TextClassifier import TextClassifier

def main():
    data_name = "us-election-2020"
    mlflow.log_param("data_name", data_name)
    mlflow_tracking_uri = "file:///mounts/Users/cisintern/zhangyaq/imbalanced_text_classification/mlruns"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.log_param("mlflow_tracking_uri", mlflow_tracking_uri)
    experiment_name = "imbalalanced_text_classification"
    mlflow.set_experiment(experiment_name)
    mlflow.log_param("mlflow_experiment_name", experiment_name)
    mlflow.end_run()
    mlflow.start_run(run_name=f"baseline-{data_name}")
    mlflow_experiment_id = mlflow.active_run().info.experiment_id
    mlflow.log_param("mlflow_experiment_id", mlflow_experiment_id)
    mlflow_run_id = mlflow.active_run().info.run_id
    mlflow.log_param("mlflow_run_id", mlflow_run_id)

    pl_seed = 42
    mlflow.log_param("seed", pl_seed)
    pl.seed_everything(pl_seed)

    data_type = "bin"
    mlflow.log_param("data_type", data_type)
    num_classes = 2
    mlflow.log_param("num_classes", num_classes)
    data_path = f"data/{data_name}/"
    mlflow.log_param("data_path", data_path)
    train_filename = "train_clean.csv"
    mlflow.log_param("train_filename", train_filename)
    test_filename = "test_clean.csv"
    mlflow.log_param("test_filename", test_filename)
    tokenizer_url = "bert-base-uncased"
    mlflow.log_param("tokenizer_url", tokenizer_url)
    batch_size = 8
    mlflow.log_param("batch_size", batch_size)
    max_token_len = 128
    mlflow.log_param("max_token_len", max_token_len)
    datamodule = ImbalancedDataModule(
        data_path=data_path,
        train_filename=train_filename,
        test_filename=test_filename,
        tokenizer_url=tokenizer_url,
        batch_size=batch_size,
        max_token_len=max_token_len
    )

    model_url = "bert-base-uncased"
    mlflow.log_param("model_url", model_url)
    learning_rate = 1e-5
    mlflow.log_param("learning_rate", learning_rate)
    weight_decay = 0.01
    mlflow.log_param("weight_decay", weight_decay)
    loss = "CE_loss"
    mlflow.log_param("loss", loss)
    model = TextClassifier(
        model_url="bert-base-uncased",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_classes=num_classes,
        loss=loss
    )
    
    ckpt_filename = "-{epoch:02d}-{val_f1_macro:.2f}"
    ckpt_dirpath = f"./mlruns/{mlflow_experiment_id}/{mlflow_run_id}/artifacts/model_checkpoints"
    mlflow.log_param("checkpoint_dirpath", ckpt_dirpath)
    checkpoint_monitor="val_f1_macro"
    mlflow.log_param("checkpoint_monitor", checkpoint_monitor)
    checkpoint_mode="max"
    mlflow.log_param("checkpoint_mode", checkpoint_mode)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=False,
        dirpath=ckpt_dirpath,
        filename=data_name + "-" + ckpt_filename,
        monitor=checkpoint_monitor,
        mode=checkpoint_mode
    )

    early_stopping_monitor="val_f1_macro"
    mlflow.log_param("early_stopping_monitor", early_stopping_monitor)
    min_delta=1e-5
    mlflow.log_param("early_stopping_min_delta", min_delta)
    patience=3
    mlflow.log_param("early_stopping_patience", patience)
    early_stopping_mode="max"
    mlflow.log_param("early_stopping_mode", early_stopping_mode)
    early_stopping_callback = EarlyStopping(
        verbose=True,
        monitor=early_stopping_monitor,
        min_delta=min_delta,
        patience=patience,
        mode=early_stopping_mode
    )
    
    mlflow_logger = MLFlowLogger(experiment_name=experiment_name, 
                                 tracking_uri=mlflow_tracking_uri,
                                 run_id=mlflow_run_id)

    max_epochs=10
    mlflow.log_param("max_epochs", max_epochs)
    num_sanity_val_steps=0
    mlflow.log_param("num_sanity_val_steps", num_sanity_val_steps)
    num_devices=1
    mlflow.log_param("num_devices", num_devices)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        num_sanity_val_steps=num_sanity_val_steps,
        devices=num_devices,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=mlflow_logger
    )

    resume_training = False
    test_only = False
    if not test_only:
        if not resume_training:
            trainer.fit(model=model, datamodule=datamodule)
    trainer.validate(datamodule=datamodule, ckpt_path="best")
    trainer.test(datamodule=datamodule, ckpt_path="best")
    mlflow.end_run()

if __name__ == "__main__":
    main()