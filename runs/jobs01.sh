# !/usr/bin/env bash
# data_names = ['bretschneider-th-main', 'bretschneider-th-school' 'cmsb-tsd', 'gao-2018-fhc', 'gibert-2018-shs', 'twitter-hate-speech-tsa', 'us-election-2020', 'waseem-and-hovy-2016']
# multi_class_data_names = ['ami', 'davidson-thon', 'founta-2018-thas']
# variant: "baseline", "sampling_modifiedRS", "sampling_weightedRS", "fl", "wce", "wfl", "th", "dl"


# python imbalanced_text_classification/main.py --data_name ami --data_type multi --num_classes 5 --label_col label_multi --train_filename train_clean.csv --test_filename test_clean.csv --variant sampling_modifiedRS --sampling_modifiedRS_mode oversampling --sampling_modifiedRS_rho_search_space 1.0 1.2 1.5 2.0 3 5 7.5 --pl_seed 0 --using_gpus 0 &> ./outputs/ami_sampling_modifiedROS_seed0_output.txt &
# python imbalanced_text_classification/main.py --data_name ami --data_type multi --num_classes 5 --label_col label_multi --train_filename train_clean.csv --test_filename test_clean.csv --variant sampling_modifiedRS --sampling_modifiedRS_mode oversampling --sampling_modifiedRS_rho_search_space 1.0 1.2 1.5 2.0 3 5 7.5 --pl_seed 21 --using_gpus 1 &> ./outputs/ami_sampling_modifiedROS_seed21_output.txt &
# python imbalanced_text_classification/main.py --data_name ami --data_type multi --num_classes 5 --label_col label_multi --train_filename train_clean.csv --test_filename test_clean.csv --variant sampling_modifiedRS --sampling_modifiedRS_mode oversampling --sampling_modifiedRS_rho_search_space 1.0 1.2 1.5 2.0 3 5 7.5 --pl_seed 42 --using_gpus 2 &> ./outputs/ami_sampling_modifiedROS_seed42_output.txt &
# python imbalanced_text_classification/main.py --data_name ami --data_type multi --num_classes 5 --label_col label_multi --train_filename train_clean.csv --test_filename test_clean.csv --variant sampling_modifiedRS --sampling_modifiedRS_mode undersampling --sampling_modifiedRS_rho_search_space 1.0 1.2 1.5 2.0 3 5 7.5 --pl_seed 0 --using_gpus 0 &> ./outputs/ami_sampling_modifiedRUS_seed0_output.txt &
# python imbalanced_text_classification/main.py --data_name ami --data_type multi --num_classes 5 --label_col label_multi --train_filename train_clean.csv --test_filename test_clean.csv --variant sampling_modifiedRS --sampling_modifiedRS_mode undersampling --sampling_modifiedRS_rho_search_space 1.0 1.2 1.5 2.0 3 5 7.5 --pl_seed 21 --using_gpus 1 &> ./outputs/ami_sampling_modifiedRUS_seed21_output.txt &
# python imbalanced_text_classification/main.py --data_name ami --data_type multi --num_classes 5 --label_col label_multi --train_filename train_clean.csv --test_filename test_clean.csv --variant sampling_modifiedRS --sampling_modifiedRS_mode undersampling --sampling_modifiedRS_rho_search_space 1.0 1.2 1.5 2.0 3 5 7.5 --pl_seed 42 --using_gpus 2 &> ./outputs/ami_sampling_modifiedRUS_seed42_output.txt &

# python imbalanced_text_classification/main.py --data_name ami --data_type multi --num_classes 5 --label_col label_multi --train_filename train_clean.csv --test_filename test_clean.csv --variant wfl --wce_alpha_search_space_multi 0.1 0.7 0.9 0.4 1.4 --wce_multi_trial_nums 5 --pl_seed 42 --using_gpus 2 &> ./outputs/ami_wfl_seed42_output.txt &





# python imbalanced_text_classification/main.py --data_name cmsb-tsd --train_filename data_clean.csv --variant sampling_modifiedRS --sampling_modifiedRS_mode undersampling --pl_seed 0 --using_gpus 2 &> ./outputs/cmsb-tsd_sampling_modifiedRUS_seed0_output.txt &
# python imbalanced_text_classification/main.py --data_name cmsb-tsd --train_filename data_clean.csv --variant sampling_modifiedRS --sampling_modifiedRS_mode undersampling --pl_seed 21 --using_gpus 3 &> ./outputs/cmsb-tsd_sampling_modifiedRUS_seed21_output.txt &
# python imbalanced_text_classification/main.py --data_name cmsb-tsd --train_filename data_clean.csv --variant sampling_modifiedRS --sampling_modifiedRS_mode undersampling --pl_seed 42 --using_gpus 4 &> ./outputs/cmsb-tsd_sampling_modifiedRUS_seed42_output.txt &



# python imbalanced_text_classification/main.py --data_name us-election-2020 --train_filename train_clean.csv --test_filename test_clean.csv --sampling_modifiedRS_mode undersampling --variant sampling_modifiedRS --pl_seed 0 --using_gpus 1 &> ./outputs/us-election-2020_sampling_modifiedRUS_seed0_output.txt &
# python imbalanced_text_classification/main.py --data_name us-election-2020 --train_filename train_clean.csv --test_filename test_clean.csv --sampling_modifiedRS_mode undersampling --variant sampling_modifiedRS --pl_seed 21 --using_gpus 1 &> ./outputs/us-election-2020_sampling_modifiedRUS_seed21_output.txt &
# python imbalanced_text_classification/main.py --data_name us-election-2020 --train_filename train_clean.csv --test_filename test_clean.csv --sampling_modifiedRS_mode undersampling --variant sampling_modifiedRS --pl_seed 42 --using_gpus 3 &> ./outputs/us-election-2020_sampling_modifiedRUS_seed42_output.txt &
# python imbalanced_text_classification/main.py --data_name us-election-2020 --train_filename train_clean.csv --test_filename test_clean.csv --variant wce --wce_alpha_search_space_bin 0.99 --pl_seed 0 --using_gpus 3 &> ./outputs/us-election-2020_wce3_seed0_output.txt &
# python imbalanced_text_classification/main.py --data_name us-election-2020 --train_filename train_clean.csv --test_filename test_clean.csv --variant wce --wce_alpha_search_space_bin 0.99 --pl_seed 21 --using_gpus 4 &> ./outputs/us-election-2020_wce3_seed21_output.txt &
# python imbalanced_text_classification/main.py --data_name us-election-2020 --train_filename train_clean.csv --test_filename test_clean.csv --variant wce --wce_alpha_search_space_bin 0.99 --pl_seed 42 --using_gpus 5 &> ./outputs/us-election-2020_wce3_seed42_output.txt &




# python imbalanced_text_classification/main.py --data_name waseem-and-hovy-2016 --train_filename data_clean.csv --sampling_modifiedRS_mode undersampling --variant sampling_modifiedRS --sampling_modifiedRS_rho_search_space 1.0 1.2 1.5 2.0 2.5 --pl_seed 0 --using_gpus 1 &> ./outputs/waseem-and-hovy-2016_sampling_modifiedRUS_seed0_output.txt &
# python imbalanced_text_classification/main.py --data_name waseem-and-hovy-2016 --train_filename data_clean.csv --sampling_modifiedRS_mode undersampling --variant sampling_modifiedRS --sampling_modifiedRS_rho_search_space 1.0 1.2 1.5 2.0 2.5 --pl_seed 21 --using_gpus 1 &> ./outputs/waseem-and-hovy-2016_sampling_modifiedRUS_seed21_output.txt &
# python imbalanced_text_classification/main.py --data_name waseem-and-hovy-2016 --train_filename data_clean.csv --sampling_modifiedRS_mode undersampling --variant sampling_modifiedRS --sampling_modifiedRS_rho_search_space 1.0 1.2 1.5 2.0 2.5 --pl_seed 42 --using_gpus 3 &> ./outputs/waseem-and-hovy-2016_sampling_modifiedRUS_seed42_output.txt &
