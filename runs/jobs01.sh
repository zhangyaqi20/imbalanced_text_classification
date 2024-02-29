# !/usr/bin/env bash
# data_names = ['bretschneider-th-main', 'bretschneider-th-school' 'cmsb-tsd', 'gao-2018-fhc', 'gibert-2018-shs', 'twitter-hate-speech-tsa', 'us-election-2020', 'waseem-and-hovy-2016']
# multi_class_data_names = ['ami', 'davidson-thon', 'founta-2018-thas']
# variant: "baseline", "sampling_modifiedRS", "sampling_weightedRS", "fl", "wce", "wfl", "th", "dl"

# python imbalanced_text_classification/main.py --data_name ami --data_type multi --num_classes 5 --label_col label_multi --train_filename train_clean.csv --test_filename test_clean.csv --variant baseline --pl_seed 0 --using_gpus 0 &> ./outputs/ami_baseline_seed0_output.txt &
# python imbalanced_text_classification/main.py --data_name ami --data_type multi --num_classes 5 --label_col label_multi --train_filename train_clean.csv --test_filename test_clean.csv --variant baseline --pl_seed 21 --using_gpus 1 &> ./outputs/ami_baseline_seed21_output.txt &
# python imbalanced_text_classification/main.py --data_name ami --data_type multi --num_classes 5 --label_col label_multi --train_filename train_clean.csv --test_filename test_clean.csv --variant baseline --pl_seed 42 --using_gpus 2 &> ./outputs/ami_baseline_seed42_output.txt &

# python imbalanced_text_classification/main.py --data_name bretschneider-th-main --train_filename main_data_clean.csv --variant sampling_modifiedRS --sampling_modifiedRS_mode undersampling --sampling_modifiedRS_rho_search_space 1.0 1.2 1.5 2.0 3 5 7.5 10 --pl_seed 0 --using_gpus 0 &> ./outputs/bretschneider-th-main_sampling_modifiedRUS_seed0_output.txt &
# python imbalanced_text_classification/main.py --data_name bretschneider-th-main --train_filename main_data_clean.csv --variant sampling_modifiedRS --sampling_modifiedRS_mode undersampling --sampling_modifiedRS_rho_search_space 1.0 1.2 1.5 2.0 3 5 7.5 10 --pl_seed 21 --using_gpus 1 &> ./outputs/bretschneider-th-main_sampling_modifiedRUS_seed21_output.txt &
# python imbalanced_text_classification/main.py --data_name bretschneider-th-main --train_filename main_data_clean.csv --variant sampling_modifiedRS --sampling_modifiedRS_mode undersampling --sampling_modifiedRS_rho_search_space 1.0 1.2 1.5 2.0 3 5 7.5 10 --pl_seed 42 --using_gpus 2 &> ./outputs/bretschneider-th-main_sampling_modifiedRUS_seed42_output.txt &

# python imbalanced_text_classification/main.py --data_name bretschneider-th-school --train_filename school_data_clean.csv --variant sampling_modifiedRS --sampling_modifiedRS_mode undersampling --sampling_modifiedRS_rho_search_space 1.0 1.2 1.5 2.0 3 5 7.5 --pl_seed 0 --using_gpus 0 &> ./outputs/bretschneider-th-school_sampling_modifiedRUS_seed0_output.txt &
# python imbalanced_text_classification/main.py --data_name bretschneider-th-school --train_filename school_data_clean.csv --variant sampling_modifiedRS --sampling_modifiedRS_mode undersampling --sampling_modifiedRS_rho_search_space 1.0 1.2 1.5 2.0 3 5 7.5 --pl_seed 21 --using_gpus 1 &> ./outputs/bretschneider-th-school_sampling_modifiedRUS_seed21_output.txt &
# python imbalanced_text_classification/main.py --data_name bretschneider-th-school --train_filename school_data_clean.csv --variant sampling_modifiedRS --sampling_modifiedRS_mode undersampling --sampling_modifiedRS_rho_search_space 1.0 1.2 1.5 2.0 3 5 7.5 --pl_seed 42 --using_gpus 2 &> ./outputs/bretschneider-th-school_sampling_modifiedRUS_seed42_output.txt &



# python imbalanced_text_classification/main.py --data_name gao-2018-fhc --train_filename data_clean.csv --variant baseline --pl_seed 0 --using_gpus 0 &> ./outputs/gao-2018-fhc_baseline_seed0_output.txt &
# python imbalanced_text_classification/main.py --data_name gao-2018-fhc --train_filename data_clean.csv --variant baseline --pl_seed 21 --using_gpus 1 &> ./outputs/gao-2018-fhc_baseline_seed21_output.txt &
# python imbalanced_text_classification/main.py --data_name gao-2018-fhc --train_filename data_clean.csv --variant baseline --pl_seed 42 --using_gpus 2 &> ./outputs/gao-2018-fhc_baseline_seed42_output.txt &





# python imbalanced_text_classification/main.py --data_name us-election-2020 --train_filename train_clean.csv --test_filename test_clean.csv --sampling_modifiedRS_mode undersampling --variant sampling_modifiedRS --pl_seed 0 --using_gpus 1 &> ./outputs/us-election-2020_sampling_modifiedRUS_seed0_output.txt &
# python imbalanced_text_classification/main.py --data_name us-election-2020 --train_filename train_clean.csv --test_filename test_clean.csv --sampling_modifiedRS_mode undersampling --variant sampling_modifiedRS --pl_seed 21 --using_gpus 1 &> ./outputs/us-election-2020_sampling_modifiedRUS_seed21_output.txt &
# python imbalanced_text_classification/main.py --data_name us-election-2020 --train_filename train_clean.csv --test_filename test_clean.csv --sampling_modifiedRS_mode undersampling --variant sampling_modifiedRS --pl_seed 42 --using_gpus 3 &> ./outputs/us-election-2020_sampling_modifiedRUS_seed42_output.txt &
# python imbalanced_text_classification/main.py --data_name us-election-2020 --train_filename train_clean.csv --test_filename test_clean.csv --variant wce --wce_alpha_search_space_bin 0.99 --pl_seed 0 --using_gpus 3 &> ./outputs/us-election-2020_wce3_seed0_output.txt &
# python imbalanced_text_classification/main.py --data_name us-election-2020 --train_filename train_clean.csv --test_filename test_clean.csv --variant wce --wce_alpha_search_space_bin 0.99 --pl_seed 21 --using_gpus 4 &> ./outputs/us-election-2020_wce3_seed21_output.txt &
# python imbalanced_text_classification/main.py --data_name us-election-2020 --train_filename train_clean.csv --test_filename test_clean.csv --variant wce --wce_alpha_search_space_bin 0.99 --pl_seed 42 --using_gpus 5 &> ./outputs/us-election-2020_wce3_seed42_output.txt &


# python imbalanced_text_classification/main.py --data_name waseem-and-hovy-2016 --train_filename data_clean.csv --sampling_modifiedRS_mode undersampling --variant sampling_modifiedRS --sampling_modifiedRS_rho_search_space 1.0 1.2 1.5 2.0 2.5 --pl_seed 0 --using_gpus 1 &> ./outputs/waseem-and-hovy-2016_sampling_modifiedRUS_seed0_output.txt &
# python imbalanced_text_classification/main.py --data_name waseem-and-hovy-2016 --train_filename data_clean.csv --sampling_modifiedRS_mode undersampling --variant sampling_modifiedRS --sampling_modifiedRS_rho_search_space 1.0 1.2 1.5 2.0 2.5 --pl_seed 21 --using_gpus 1 &> ./outputs/waseem-and-hovy-2016_sampling_modifiedRUS_seed21_output.txt &
# python imbalanced_text_classification/main.py --data_name waseem-and-hovy-2016 --train_filename data_clean.csv --sampling_modifiedRS_mode undersampling --variant sampling_modifiedRS --sampling_modifiedRS_rho_search_space 1.0 1.2 1.5 2.0 2.5 --pl_seed 42 --using_gpus 3 &> ./outputs/waseem-and-hovy-2016_sampling_modifiedRUS_seed42_output.txt &
