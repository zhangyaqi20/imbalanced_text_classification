import mlflow
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from imbalanced_text_classification.ImbalancedDataModule import ImbalancedDataModule
from imbalanced_text_classification.TextClassifier import TextClassifier

def main():
    mlflow_tracking_uri = "./logs"
    mlflow.set_training_uri(mlflow_tracking_uri)
    experiment_name = "text_classification_imbalalanced"
    experiment = mlflow.set_experiment(experiment_name)
    experiment_id = experiment.experiment_id
    with mlflow.start_run(experiment_id=experiment_id) as run:
        pl_seed = 42
        mlflow.log_param("seed", pl_seed)
        pl.seed_everything(pl_seed)

        data_name = "TROS"
        mlflow.log_param("data_name", data_name)
        data_path = f"data/{data_name}.csv"
        mlflow.log_param("data_path", data_path)
        tokenizer_url = "bert-base-uncased"
        mlflow.log_param("tokenizer_url", tokenizer_url)
        batch_size = 8
        mlflow.log_param("batch_size", batch_size)
        max_token_len = 128
        mlflow.log_param("max_token_len", max_token_len)
        datamodule = ImbalancedDataModule(
            data_path=data_path,
            tokenizer_url=tokenizer_url,
            batch_size=batch_size,
            max_token_len=max_token_len
        )
        datamodule.setup(stage="fit")

        model_url = "bert-base-uncased"
        mlflow.log_param("model_url", model_url)
        learning_rate = 1e-5,
        mlflow.log_param("learning_rate", learning_rate)
        weight_decay = 0.0
        mlflow.log_param("weight_decay", weight_decay)
        loss = "CE_loss"
        mlflow.log_param("loss", loss)
        model = TextClassifier(
            model_url="bert-base-uncased",
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_labels=datamodule.num_labels,
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
        
        run_id = run.info.run_id
        mlflow_logger = MLFlowLogger(experiment_name=experiment_name, tracking_uri=mlflow_tracking_uri)
        mlflow_logger._run_id = run_id

        max_epochs=2
        mlflow.log_param("max_epochs", max_epochs)
        num_devices=1
        mlflow.log_param("num_devices", num_devices)
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            num_sanity_val_steps=0,
            devices=num_devices,
            callbacks=[checkpoint_callback],
            logger=mlflow_logger
        )

        resume_training = False
        test_only = False
        if not test_only:
            if not resume_training:
                trainer.fit(model=model, datamodule=datamodule)
        datamodule.setup(stage="validate")
        trainer.validate(datamodule=datamodule, ckpt_path="best")
        datamodule.setup(stage="test")
        trainer.test(datamodule=datamodule, ckpt_path="best")

if __name__ == "__main__":
    main()