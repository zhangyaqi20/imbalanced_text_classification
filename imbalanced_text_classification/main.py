import argparse
import mlflow
import optuna
import pytorch_lightning as pl
import torch
from enum import Enum
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from ImbalancedDataModule import ImbalancedDataModule
from TextClassifier import TextClassifier
from callbacks.PyTorchLightningPruningCallback import PyTorchLightningPruningCallback
from callbacks.OptunaChampionCallback import optuna_champion_callback
from loss.Loss import Loss

# DataModule
TOKENIZER_URL = "bert-base-uncased"
MAX_TOKEN_LEN = 128
BATCH_SIZE = 64
# Model
MODEL_URL = "bert-base-uncased"
LEARNING_RATE = 5e-5 # baseline: 5e-5
WEIGHT_DECAY = 0.01 # baseline: 0.01
# Trainer
NUM_SANITY_VAL_STEPS = 0
ACCELERATOR = "gpu"
MONITOR = "val_f1_macro"
# MlFlow logger
MLFLOW_TRACKING_URI = "file:///mounts/Users/cisintern/zhangyaq/imbalanced_text_classification/mlruns"

class Variant(Enum):
    Baseline = "baseline"
    Sampling_ModifiedRS = "sampling_modifiedRS"
    Sampling_WeightedRS_Combi = "sampling_weightedRS_combi"
    Sampling_WeightedRS_Oversampling = "sampling_weightedRS_oversampling"
    Augmentation_WordNet = "augmentation_wordnet"
    Augmentation_Bert = "augmentation_bert"
    WCE = "wce"
    FL = "fl"
    WFL = "wfl"
    TH = "th"
    DL = "dl"

def main(args):
    variant = Variant(args.variant)
    if variant == Variant.Baseline or variant == Variant.TH or variant == Variant.DL:
        objective(args)
    else:
        pruner = optuna.pruners.NopPruner() # Disable pruning to run all(..) epochs (optuna.pruners.MedianPruner())
        # Configure search space
        if variant == Variant.FL:
            search_space = {"fl_gamma": args.fl_gamma_search_space}
        elif variant == Variant.WCE:
            if args.data_type == "bin":
                search_space = {"wce_alpha": args.wce_alpha_search_space_bin}
            else:
                wce_alpha_search_space_multi = [args.wce_alpha_search_space_multi,] # The perfect class weights = 1/N_c
                # Generate random class weights (probabilities)
                probs_pool = torch.tensor(range(1,10)) * 0.1
                for i in range(args.wce_multi_trial_nums - len(wce_alpha_search_space_multi)):
                    generator_i = torch.Generator()
                    generator_i.manual_seed(i)
                    a = torch.randint(0, 9, (args.num_classes,), generator=generator_i).tolist()
                    wce_alpha = probs_pool[a].tolist()
                    wce_alpha_search_space_multi.append(wce_alpha)
                assert len(wce_alpha_search_space_multi) == args.wce_multi_trial_nums
                setattr(args, 'wce_alpha_search_space_multi', wce_alpha_search_space_multi)
                search_space = {"wce_alpha_index": range(args.wce_multi_trial_nums)}
                print(f"The final wce_alpha_search_space_multi is {args.wce_alpha_search_space_multi}")
        elif variant == Variant.WFL:
            if args.data_type == "bin":
                search_space = {"wce_alpha": args.wce_alpha_search_space_bin,
                                "fl_gamma": args.fl_gamma_search_space}
            else:
                wce_alpha_search_space_multi = [args.wce_alpha_search_space_multi,# The perfect class weights = 1/N_c
                                                [1.0]*args.num_classes] # pure FL
                # Generate random class weights (probabilities)
                probs_pool = torch.tensor(range(1,10)) * 0.1
                for i in range(args.wce_multi_trial_nums - len(wce_alpha_search_space_multi)):
                    generator_i = torch.Generator()
                    generator_i.manual_seed(i)
                    a = torch.randint(0, 9, (args.num_classes,), generator=generator_i).tolist()
                    wce_alpha = probs_pool[a].tolist()
                    wce_alpha_search_space_multi.append(wce_alpha)
                assert len(wce_alpha_search_space_multi) == args.wce_multi_trial_nums
                setattr(args, 'wce_alpha_search_space_multi', wce_alpha_search_space_multi)
                search_space = {"wce_alpha_index": range(args.wce_multi_trial_nums),
                                "fl_gamma": args.fl_gamma_search_space}
                print(f"The final wce_alpha_search_space_multi is {args.wce_alpha_search_space_multi}")
        elif variant == Variant.Sampling_ModifiedRS:
            search_space = {"sampling_modifiedRS_rho": args.sampling_modifiedRS_rho_search_space}
        elif variant == Variant.Sampling_WeightedRS_Combi or variant == Variant.Sampling_WeightedRS_Oversampling:
            search_space = {"sampling_weightedRS_percentage": args.sampling_weightedRS_percentage_search_space}
        elif variant == Variant.Augmentation_WordNet:
            search_space = {"augmentation_rho": args.augmentation_rho_search_space,
                            "augmentation_percentage": args.augmentation_percentage_search_space}
        elif variant == Variant.Augmentation_Bert:
            search_space = {"augmentation_rho": args.augmentation_rho_search_space,
                            "augmentation_percentage": args.augmentation_percentage_search_space,
                            "augmentation_top_k": args.augmentation_top_k_search_space}
        sampler = optuna.samplers.GridSampler(search_space)
        study = optuna.create_study(study_name=f"{args.data_name}_{variant.value}", direction="maximize", pruner=pruner, sampler=sampler)
        study.optimize(lambda trial: objective(trial=trial, args=args), gc_after_trial=True, callbacks=[optuna_champion_callback])
        print(f"OPTUNA - Number of finished trials: {len(study.trials)}")
        print(f"Best trial:")
        best_trial = study.best_trial
        print(f"  {MONITOR}: {best_trial.value}")
        print(f"  Params:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

def objective(args, trial: optuna.trial.Trial=None) -> float:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow_experiment_name = f"imbalalanced_text_classification_{args.data_name}"
    print(mlflow_experiment_name)
    mlflow.set_experiment(experiment_name=mlflow_experiment_name)
    variant = Variant(args.variant)
    run_name = f"{variant.value}-{args.data_name}-{args.data_type}"
    if trial is not None:
        run_name += f"-Trial-{trial.number}"
    with mlflow.start_run(run_name=run_name) as run:
        print(f"\nMLFlow run_name = {run_name}")
        pl.seed_everything(args.pl_seed, workers=True)
        mlflow_experiment_id_trial = run.info.experiment_id # The same with the parent run id
        mlflow_run_id_trial = run.info.run_id
        mlflow.log_params(
            params={
                # DataModule
                "train_filename": args.train_filename,
                "val_filename": args.val_filename,
                "test_filename": args.test_filename,
                "num_classes": args.num_classes,
                "label_col": args.label_col,
                "tokenizer_url": TOKENIZER_URL,
                "max_token_len": MAX_TOKEN_LEN,
                "batch_size": BATCH_SIZE,
                # Model
                "pl_seed": args.pl_seed,
                "model_url": MODEL_URL,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                # Trainer
                "max_epochs": args.max_epochs,
                "num_sanity_val_steps": NUM_SANITY_VAL_STEPS,
                "accelerator": ACCELERATOR,
                "devices": args.using_gpus,
            }
        )
        # Baseline setting
        loss = Loss.CE_Loss
        sampling_modifiedRS_rho = None
        sampling_modifiedRS_mode = None
        sampling_weightedRS_percentage = None
        augmentation_rho = None
        augmentation_src = None
        augmentation_percentage = None
        augmentation_top_k = None
        fl_gamma = None
        wce_alpha = None
        adjusting_th = None
        if variant == Variant.Baseline:
            print(f"MLflow Saved Baseline Log in './mlruns/{mlflow_experiment_id_trial}/{mlflow_run_id_trial}'")
        else:
            if variant == Variant.TH:
                print(f"MLflow Saved Thresholding Log in './mlruns/{mlflow_experiment_id_trial}/{mlflow_run_id_trial}'")
                adjusting_th = True
            elif variant == Variant.DL:
                print(f"MLflow Saved Dice Loss Log in './mlruns/{mlflow_experiment_id_trial}/{mlflow_run_id_trial}'")
                loss = Loss.Dice_Loss
            else:
                print(f"MLflow Saved Child Search Trial {trial.number} Log in './mlruns/{mlflow_experiment_id_trial}/{mlflow_run_id_trial}'")
                if variant == Variant.Sampling_ModifiedRS:
                    sampling_modifiedRS_rho = trial.suggest_float("sampling_modifiedRS_rho", 1.0, 20.0)
                    sampling_modifiedRS_mode = args.sampling_modifiedRS_mode
                    mlflow.log_params(
                        params={
                            "sampling_modifiedRS_rho_search_space": args.sampling_modifiedRS_rho_search_space,
                            "sampling_modifiedRS_mode": sampling_modifiedRS_mode
                            }
                        )
                elif variant == Variant.Sampling_WeightedRS_Combi or variant == Variant.Sampling_WeightedRS_Oversampling:
                    sampling_weightedRS_percentage = trial.suggest_float("sampling_weightedRS_percentage", -0.5, 10.0)
                    mlflow.log_param("sampling_weightedRS_percentage_search_space", args.sampling_weightedRS_percentage_search_space)
                elif variant == Variant.FL:
                    loss = Loss.Focal_Loss
                    fl_gamma = trial.suggest_float("fl_gamma", 0.0, 5.0)
                    mlflow.log_param("fl_gamma_search_space", args.fl_gamma_search_space)
                elif variant == Variant.WCE:
                    loss = Loss.Weighted_CE_Loss
                    if args.data_type == "bin":
                        wce_alpha = trial.suggest_float("wce_alpha", 0.0, 1.0)
                        mlflow.log_param("wce_alpha_search_space_bin", args.wce_alpha_search_space_bin)
                    else:
                        wce_alpha_index = trial.suggest_int("wce_alpha_index", 0, (args.wce_multi_trial_nums-1))
                        wce_alpha = args.wce_alpha_search_space_multi[wce_alpha_index]
                        mlflow.log_param("wce_alpha_search_space_multi", args.wce_alpha_search_space_multi)
                elif variant == Variant.WFL:
                    loss = Loss.Weighted_Focal_Loss
                    fl_gamma = trial.suggest_float("fl_gamma", 0.0, 5.0)
                    mlflow.log_param("fl_gamma_search_space", args.fl_gamma_search_space)
                    if args.data_type == "bin":
                        wce_alpha = trial.suggest_float("wce_alpha", 0.0, 1.0)
                        mlflow.log_param("wce_alpha_search_space_bin", args.wce_alpha_search_space_bin)
                    else:
                        wce_alpha_index = trial.suggest_int("wce_alpha_index", 0, (args.wce_multi_trial_nums-1))
                        wce_alpha = args.wce_alpha_search_space_multi[wce_alpha_index]
                        mlflow.log_param("wce_alpha_search_space_multi", args.wce_alpha_search_space_multi)
                elif variant == Variant.Augmentation_Bert or variant == Variant.Augmentation_WordNet:
                    augmentation_rho = trial.suggest_float("augmentation_rho", 1.0, 20.0)
                    augmentation_src = args.augmentation_src
                    augmentation_percentage = trial.suggest_float("augmentation_percentage", 0.0, 1.0)
                    mlflow.log_params(
                        params={
                            "augmentation_rho_search_space": args.augmentation_rho_search_space,
                            "augmentation_percentage_search_space": args.augmentation_percentage_search_space
                            }
                        )
                    if variant == Variant.Augmentation_Bert:
                        augmentation_top_k = trial.suggest_int("augmentation_top_k", 1, 50)
                        mlflow.log_params(
                        params={
                            "augmentation_top_k_search_space": args.augmentation_top_k_search_space
                            }
                        )
                else:
                    raise NotImplementedError("Not supported vairant, please choose one from fl, sampling_weightedRS, baseline")
            
        hparams = {
            "sampling_modifiedRS_mode": sampling_modifiedRS_mode,
            "sampling_modifiedRS_rho": sampling_modifiedRS_rho,
            "sampling_weightedRS_percentage": sampling_weightedRS_percentage,
            "augmentation_rho": augmentation_rho,
            "augmentation_src": augmentation_src,
            "augmentation_percentage": augmentation_percentage,
            "augmentation_top_k": augmentation_top_k,
            "wce_alpha": wce_alpha,
            "fl_gamma": fl_gamma,
            "loss": loss,
            "adjusting_th": adjusting_th
        }
        mlflow.log_params(hparams)

        if variant.value == "sampling_modifiedRS" and sampling_modifiedRS_mode is not None:
            variant_log = variant.value + "_" + sampling_modifiedRS_mode
        else:
            variant_log = variant.value
        mlflow.set_tags(
            tags={
                "data_name": args.data_name,
                "data_type": args.data_type,
                "variant": variant_log,
            }
        )
        datamodule = ImbalancedDataModule(
            data_path=f"data/{args.data_name}/",
            train_filename=args.train_filename,
            val_filename=args.val_filename,
            test_filename=args.test_filename,
            label_col=args.label_col,
            tokenizer_url=TOKENIZER_URL,
            batch_size=BATCH_SIZE,
            max_token_len=MAX_TOKEN_LEN,
            sampling_modifiedRS_mode=sampling_modifiedRS_mode,
            sampling_modifiedRS_rho=sampling_modifiedRS_rho,
            sampling_weightedRS_percentage=sampling_weightedRS_percentage,
            augmentation_rho=augmentation_rho,
            augmentation_src=augmentation_src,
            augmentation_percentage=augmentation_percentage,
            augmentation_top_k=augmentation_top_k
        )
        model = TextClassifier(
            model_url=MODEL_URL,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            num_classes=args.num_classes,
            loss=loss,
            wce_alpha=wce_alpha,
            fl_gamma=fl_gamma,
            device=args.using_gpus,
            adjusting_th=adjusting_th
        )
        
        # checkpoint callback:
        ckpt_filename_prefix = args.data_name + "-" + variant_log
        if variant != Variant.Baseline and variant != Variant.TH and variant != Variant.DL:
            ckpt_filename_prefix += f"-Trial_{trial.number}"
            if wce_alpha is not None:
                ckpt_filename_prefix += f"-wce_alpha={wce_alpha}"
            if fl_gamma is not None:
                ckpt_filename_prefix += f"-fl_gamma={fl_gamma}"
            if sampling_modifiedRS_rho is not None:
                ckpt_filename_prefix += f"-{sampling_modifiedRS_mode}-sampling_modifiedRS_rho={sampling_modifiedRS_rho}"
            if sampling_weightedRS_percentage is not None:
                ckpt_filename_prefix += f"-sampling_weightedRS_percentage={sampling_weightedRS_percentage}"
        ckpt_filename_prefix += f"-seed{args.pl_seed}"
        ckpt_filename = "-{epoch:02d}-{val_f1_macro:.2f}"
        ckpt_dirpath = f"./mlruns/{mlflow_experiment_id_trial}/{mlflow_run_id_trial}/artifacts/model_checkpoints"
        ckpt_mode = "max"
        mlflow.log_params(
            params={
                "checkpoint_dirpath": ckpt_dirpath,
                "checkpoint_monitor": MONITOR,
                "checkpoint_mode": ckpt_mode
            }
        )
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            dirpath=ckpt_dirpath,
            filename=ckpt_filename_prefix+ckpt_filename,
            monitor=MONITOR,
            mode=ckpt_mode
        )
        # early_stopping callback:
        min_delta = 1e-5
        patience = 3
        early_stopping_mode = "max"
        mlflow.log_params(
            params = {
                "early_stopping_monitor": MONITOR,
                "early_stopping_min_delta": min_delta,
                "early_stopping_patience": patience,
                "early_stopping_mode": early_stopping_mode
                }
        )
        early_stopping_callback = EarlyStopping(
            monitor=MONITOR,
            min_delta=min_delta,
            patience=patience,
            mode=early_stopping_mode
        )
        # MLFlow Logger
        mlflow_logger = MLFlowLogger(experiment_name=mlflow_experiment_name, 
                                    tracking_uri=MLFLOW_TRACKING_URI,
                                    run_id=mlflow_run_id_trial)
        if variant != Variant.Baseline and variant != Variant.TH and variant != Variant.DL:
            print(f"Runing optuna for hyperparameter search:\nParmas: {hparams}")
            # Optuna tuning callback
            optuna_pruning_callback = PyTorchLightningPruningCallback(trial, monitor=MONITOR)
            # Trainer
            trainer = pl.Trainer(
                max_epochs=args.max_epochs,
                num_sanity_val_steps=NUM_SANITY_VAL_STEPS,
                accelerator=ACCELERATOR,
                devices=args.using_gpus,
                callbacks=[checkpoint_callback, 
                        early_stopping_callback,
                        optuna_pruning_callback],
                logger=mlflow_logger,
                deterministic=True
            )

            trainer.fit(model, datamodule=datamodule)
            trainer.validate(model, datamodule=datamodule, ckpt_path="best") # Need to explicitly call validate, otherwise will only take the last val score as the best
            best_monitor_score = trainer.callback_metrics[MONITOR].item()
            trainer.test(model, datamodule=datamodule, ckpt_path="best")
            print(f"END running Trial {trial.number}.\n\n")
        else:
            print(f"Runing {variant.value} model:\nParmas: {hparams}")
            trainer = pl.Trainer(
                max_epochs=args.max_epochs,
                num_sanity_val_steps=NUM_SANITY_VAL_STEPS,
                accelerator=ACCELERATOR,
                devices=args.using_gpus,
                callbacks=[checkpoint_callback, 
                        early_stopping_callback],
                logger=mlflow_logger,
                deterministic=True
            )

            trainer.fit(model, datamodule=datamodule)
            trainer.validate(model, datamodule=datamodule, ckpt_path="best") # Need to explicitly call validate, otherwise will only take the last val score as the best
            best_monitor_score = trainer.callback_metrics[MONITOR].item()
            trainer.test(model, datamodule=datamodule, ckpt_path="best")
            print(f"END running {variant.value}.\n\n")
        
    return best_monitor_score

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Imbalanced Text Classification with PyTorch Lightning')
    parser.add_argument('--data_name', type=str, help='Data name.', required=True)
    parser.add_argument('--data_type', type=str, default="bin", help='Data type: "bin" or "multi".')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes.')
    parser.add_argument('--label_col', type=str, default="label", help='Name of label column.')
    parser.add_argument('--train_filename', type=str, help='Name of train data.', required=True)
    parser.add_argument('--val_filename', type=str, default=None, help='Name of validation data.')
    parser.add_argument('--test_filename', type=str, default=None, help='Name of test data.')
    parser.add_argument('--variant', type=str, help='Variant of comparison: "baseline", "sampling_modifiedRS", "sampling_weightedRS_combi", "sampling_weightedRS_oversampling", "augmentation_wordnet", "augmentation_bert", "fl", "wce", "wfl", "th", "dl".', 
                        choices=["baseline", "sampling_modifiedRS", "sampling_weightedRS_combi", "sampling_weightedRS_oversampling", "augmentation_wordnet", "augmentation_bert", "fl", "wce", "wfl", "th", "dl"], required=True)
    parser.add_argument('--wce_alpha_search_space_bin', nargs="*", type=float, help="alpha for the class 1 in weighted cross entropy.", default=[0.1, 0.25, 0.75, 0.9, 0.99])
    parser.add_argument('--wce_alpha_search_space_multi', nargs="*", type=float, help="Must provide the perfect class weights. Will generate random weights later.")
    parser.add_argument('--wce_multi_trial_nums', type=int, help="Generate how many groups of random class weights for multi-class sets (plus the perfect one).", default=7)
    parser.add_argument('--fl_gamma_search_space', nargs="*", type=float, default=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
    parser.add_argument('--sampling_modifiedRS_mode', type=str, help="Use Modified RS to 'oversamping' or 'undersampling'.", default="oversampling", choices=["oversampling", "undersampling"])
    parser.add_argument('--sampling_modifiedRS_rho_search_space', nargs="*", type=float, help="The target imbalance rate rho for a random over/undersampler (>=1).", default=[1.0, 1.2, 1.5, 2.0, 3, 5])
    parser.add_argument('--augmentation_rho_search_space', nargs="*", type=float, help="The target imbalance rate rho for augmentation (>=1).", default=[1.0, 1.2, 1.5, 2.0, 3, 5])
    parser.add_argument('--augmentation_src', type=str, help='Source of augmentation: "WordNet", "Bert".', choices=["WordNet", "Bert"])
    parser.add_argument('--augmentation_percentage_search_space', nargs="*", type=float, help="How much percentage of tokens in a sentence to augment.", default=[0.1, 0.3, 0.5])
    parser.add_argument('--augmentation_top_k_search_space', nargs="*", type=int, help="How many top k candidates to consider for bertAug.", default=[1, 3, 5])
    parser.add_argument('--sampling_weightedRS_percentage_search_space', nargs="*", type=float, help="The sampling for weighted random sampler (-1, inf). If combi version: [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75]. Otherwise can be large.", default=[-0.5, -0.25, 0.0, 0.25, 0.5, 0.75])
    parser.add_argument('--using_gpus', nargs="*", type=int, default=[0])
    parser.add_argument('--pl_seed', type=int, help='Seed for Lightning.')
    parser.add_argument('--max_epochs', type=int, default=10, help='Number of epochs.')

    args = parser.parse_args()
    main(args)