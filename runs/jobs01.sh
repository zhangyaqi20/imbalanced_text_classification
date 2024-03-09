# !/usr/bin/env bash
# data_names = ['bretschneider-th-main', 'bretschneider-th-school' 'cmsb-tsd', 'gao-2018-fhc', 'gibert-2018-shs', 'twitter-hate-speech-tsa', 'us-election-2020', 'waseem-and-hovy-2016']
# multi_class_data_names = ['ami', 'davidson-thon', 'founta-2018-thas']
# variant: "baseline", "sampling_modifiedRS", "sampling_weightedRS", "fl", "wce", "wfl", "th", "dl"







# python imbalanced_text_classification/main.py --data_name civil-comments-40k --train_filename data_clean.csv --variant baseline --pl_seed 0 --using_gpus 0 &> ./outputs/civil-comments-40k_baseline_seed0_output.txt &
# python imbalanced_text_classification/main.py --data_name civil-comments-40k --train_filename data_clean.csv --variant baseline --pl_seed 21 --using_gpus 1 &> ./outputs/civil-comments-40k_baseline_seed21_output.txt &
# python imbalanced_text_classification/main.py --data_name civil-comments-40k --train_filename data_clean.csv --variant baseline --pl_seed 42 --using_gpus 2 &> ./outputs/civil-comments-40k_baseline_seed42_output.txt &
# python imbalanced_text_classification/main.py --data_name civil-comments-40k --train_filename data_clean.csv --variant th --pl_seed 0 --using_gpus 3 &> ./outputs/civil-comments-40k_th_seed0_output.txt &
# python imbalanced_text_classification/main.py --data_name civil-comments-40k --train_filename data_clean.csv --variant th --pl_seed 21 --using_gpus 4 &> ./outputs/civil-comments-40k_th_seed21_output.txt &
# python imbalanced_text_classification/main.py --data_name civil-comments-40k --train_filename data_clean.csv --variant th --pl_seed 42 --using_gpus 5 &> ./outputs/civil-comments-40k_th_seed42_output.txt &
# python imbalanced_text_classification/main.py --data_name civil-comments-40k --train_filename data_clean.csv --variant sampling_modifiedRS --sampling_modifiedRS_mode oversampling --sampling_modifiedRS_rho_search_space 2.0 3 5 7.5 --pl_seed 0 --using_gpus 6 &> ./outputs/civil-comments-40k_sampling_modifiedROS_seed0_output.txt &
# python imbalanced_text_classification/main.py --data_name civil-comments-40k --train_filename data_clean.csv --variant sampling_modifiedRS --sampling_modifiedRS_mode oversampling --sampling_modifiedRS_rho_search_space 2.0 3 5 7.5 --pl_seed 21 --using_gpus 7 &> ./outputs/civil-comments-40k_sampling_modifiedROS_seed21_output.txt &
# python imbalanced_text_classification/main.py --data_name civil-comments-40k --train_filename data_clean.csv --variant sampling_modifiedRS --sampling_modifiedRS_mode oversampling --sampling_modifiedRS_rho_search_space 2.0 3 5 7.5 --pl_seed 42 --using_gpus 0 &> ./outputs/civil-comments-40k_sampling_modifiedROS_seed42_output.txt &
# python imbalanced_text_classification/main.py --data_name civil-comments-40k --train_filename data_clean.csv --variant sampling_modifiedRS --sampling_modifiedRS_mode undersampling --sampling_modifiedRS_rho_search_space 2.0 3 5 7.5 --pl_seed 0 --using_gpus 6 &> ./outputs/civil-comments-40k_sampling_modifiedRUS_seed0_output.txt &
# python imbalanced_text_classification/main.py --data_name civil-comments-40k --train_filename data_clean.csv --variant sampling_modifiedRS --sampling_modifiedRS_mode undersampling --sampling_modifiedRS_rho_search_space 2.0 3 5 7.5 --pl_seed 21 --using_gpus 7 &> ./outputs/civil-comments-40k_sampling_modifiedRUS_seed21_output.txt &
# python imbalanced_text_classification/main.py --data_name civil-comments-40k --train_filename data_clean.csv --variant sampling_modifiedRS --sampling_modifiedRS_mode undersampling --sampling_modifiedRS_rho_search_space 2.0 3 5 7.5 --pl_seed 42 --using_gpus 0 &> ./outputs/civil-comments-40k_sampling_modifiedRUS_seed42_output.txt &
# python imbalanced_text_classification/main.py --data_name civil-comments-40k --train_filename data_clean.csv --variant fl --pl_seed 0 --using_gpus 1 &> ./outputs/civil-comments-40k_fl_seed0_output.txt &
# python imbalanced_text_classification/main.py --data_name civil-comments-40k --train_filename data_clean.csv --variant fl --pl_seed 21 --using_gpus 2 &> ./outputs/civil-comments-40k_fl_seed21_output.txt &
# python imbalanced_text_classification/main.py --data_name civil-comments-40k --train_filename data_clean.csv --variant fl --pl_seed 42 --using_gpus 4 &> ./outputs/civil-comments-40k_fl_seed42_output.txt &
# python imbalanced_text_classification/main.py --data_name civil-comments-40k --train_filename data_clean.csv --variant augmentation_external_data --augmentation_src ExternalData --augmentation_rho_search_space 2.0 3.0 5.0 7.5 --augmentation_categories abusive_offensive_toxic --preprocessing True --pl_seed 0 --using_gpus 3 &> ./outputs/civil-comments-40k_augmentation_external-preprocessing_data_seed0_output.txt &
# python imbalanced_text_classification/main.py --data_name civil-comments-40k --train_filename data_clean.csv --augmentation_src AbusiveLexicon --variant augmentation_abusive_lexicon --augmentation_rho_search_space 2.0 3 5 7.5 --pl_seed 0 --using_gpus 5 &> ./outputs/civil-comments-40k_augmentation_abusive_lexicon_seed0_output.txt &





# python imbalanced_text_classification/main.py --data_name civil-comments-40k --train_filename data_clean.csv --variant augmentation_external_data --augmentation_src ExternalData --augmentation_rho_search_space 2.0 3.0 5.0 7.5 --augmentation_categories abusive_offensive_toxic --preprocessing True --pl_seed 21 --using_gpus 3 &> ./outputs/civil-comments-40k_augmentation_external-preprocessing_data_seed21_output.txt &
# python imbalanced_text_classification/main.py --data_name civil-comments-40k --train_filename data_clean.csv --variant augmentation_external_data --augmentation_src ExternalData --augmentation_rho_search_space 2.0 3.0 5.0 7.5 --augmentation_categories abusive_offensive_toxic --preprocessing True --pl_seed 42 --using_gpus 4 &> ./outputs/civil-comments-40k_augmentation_external-preprocessing_data_seed42_output.txt &

# python imbalanced_text_classification/main.py --data_name civil-comments-40k --train_filename data_clean.csv --augmentation_src AbusiveLexicon --variant augmentation_abusive_lexicon --augmentation_rho_search_space 2.0 3 5 7.5 --pl_seed 21 --using_gpus 3 &> ./outputs/civil-comments-40k_augmentation_abusive_lexicon_seed21_output.txt &
# python imbalanced_text_classification/main.py --data_name civil-comments-40k --train_filename data_clean.csv --augmentation_src AbusiveLexicon --variant augmentation_abusive_lexicon --augmentation_rho_search_space 2.0 3 5 7.5 --pl_seed 42 --using_gpus 5 &> ./outputs/civil-comments-40k_augmentation_abusive_lexicon_seed42_output.txt &
