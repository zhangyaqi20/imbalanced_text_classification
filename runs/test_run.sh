#!/usr/bin/env bash
python imbalanced_text_classification/main.py --data_name us-election-2020 --train_filename train_clean.csv --test_filename test_clean.csv --variant th --pl_seed 42 --using_gpus 4 &> ./outputs/us-election-2020_th_seed42_output.txt &
