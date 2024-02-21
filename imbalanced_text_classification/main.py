import mlflow
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from ImbalancedDataModule import ImbalancedDataModule
from TextClassifier import TextClassifier

def main():
    mlflow_tracking_uri = "./logs"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    experiment_name = "imbalalanced_text_classification"
    experiment = mlflow.set_experiment(experiment_name)
    experiment_id = experiment.experiment_id
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        pl_seed = 42
        mlflow.log_param("seed", pl_seed)
        pl.seed_everything(pl_seed)

        data_name = "us-election-2020"
        mlflow.log_param("data_name", data_name)
        data_type = "bin"
        mlflow.log_param("data_type", data_type)
        num_labels = 2
        mlflow.log_param("num_labels", num_labels)
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
            num_labels=num_labels,
            loss=loss
        )
        
        ckpt_filename = "-{epoch:02d}-{val_f1_macro:.2f}"
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            save_last=True,
            dirpath=mlflow_tracking_uri + f"/{experiment_id}/{run_id}/artifacts/model_checkpoints",
            filename=data_name + "-" + ckpt_filename,
            monitor="val_f1_macro",
            mode="max"
        )

        monitor="val_f1_macro"
        mlflow.log_param("monitor", monitor)
        min_delta=1e-5
        mlflow.log_param("min_delta", min_delta)
        patience=3
        mlflow.log_param("patience", patience)
        mode="max"
        mlflow.log_param("mode", mode)
        early_stopping_callback = EarlyStopping(
            verbose=True,
            monitor="val_f1_macro",
            min_delta=1e-5,
            patience=3,
            mode="max"
        )
        
        mlflow_logger = MLFlowLogger(experiment_name=experiment_name, tracking_uri=mlflow_tracking_uri)
        mlflow_logger._run_id = run_id

        max_epochs=10
        mlflow.log_param("max_epochs", max_epochs)
        num_devices=1
        mlflow.log_param("num_devices", num_devices)
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            num_sanity_val_steps=0,
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

if __name__ == "__main__":
    main()