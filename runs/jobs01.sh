# !/usr/bin/env bash
# data_names = ['bretschneider-th-main', 'bretschneider-th-school' 'cmsb-tsd', 'gao-2018-fhc', 'gibert-2018-shs', 'twitter-hate-speech-tsa', 'us-election-2020', 'waseem-and-hovy-2016']
# multi_class_data_names = ['ami', 'davidson-thon', 'founta-2018-thas']
# variant: "baseline", "sampling_modifiedRS", "sampling_weightedRS", "fl", "wce", "wfl", "th", "dl"







# # python imbalanced_text_classification/main.py --data_name twitter-hate-speech-tsa --train_filename data_clean.csv --augmentation_src Bert --variant augmentation_bert --augmentation_rho_search_space 7.5 --pl_seed 21 --using_gpus 2 &> ./outputs/twitter-hate-speech-tsa_augmentation_bert2_seed21_output.txt &
# # python imbalanced_text_classification/main.py --data_name twitter-hate-speech-tsa --train_filename data_clean.csv --augmentation_src Bert --variant augmentation_bert --augmentation_rho_search_space 7.5 --pl_seed 42 --using_gpus 4 &> ./outputs/twitter-hate-speech-tsa_augmentation_bert2_seed42_output.txt &



# python imbalanced_text_classification/main.py --data_name ami --data_type multi --num_classes 5 --label_col label_multi --train_filename train_clean.csv --test_filename test_clean.csv --variant wce --wce_alpha_search_space_multi 0.1 0.1 0.1 0.2 0.9 --wce_multi_trial_nums 5 --pl_seed 0 --using_gpus 1 &> ./outputs/ami_wce2_seed0_output.txt &
# python imbalanced_text_classification/main.py --data_name ami --data_type multi --num_classes 5 --label_col label_multi --train_filename train_clean.csv --test_filename test_clean.csv --variant wce --wce_alpha_search_space_multi 0.1 0.1 0.1 0.2 0.9 --wce_multi_trial_nums 5 --pl_seed 21 --using_gpus 2 &> ./outputs/ami_wce2_seed21_output.txt &
# python imbalanced_text_classification/main.py --data_name ami --data_type multi --num_classes 5 --label_col label_multi --train_filename train_clean.csv --test_filename test_clean.csv --variant wce --wce_alpha_search_space_multi 0.1 0.1 0.1 0.2 0.9 --wce_multi_trial_nums 5 --pl_seed 42 --using_gpus 3 &> ./outputs/ami_wce2_seed42_output.txt &

# python imbalanced_text_classification/main.py --data_name ami --data_type multi --num_classes 5 --label_col label_multi --train_filename train_clean.csv --test_filename test_clean.csv --variant wfl --wce_alpha_search_space_multi 0.1 0.1 0.1 0.2 0.9 --wce_multi_trial_nums 5 --pl_seed 0 --using_gpus 1 &> ./outputs/ami_wfl2_seed0_output.txt &
# python imbalanced_text_classification/main.py --data_name ami --data_type multi --num_classes 5 --label_col label_multi --train_filename train_clean.csv --test_filename test_clean.csv --variant wfl --wce_alpha_search_space_multi 0.1 0.1 0.1 0.2 0.9 --wce_multi_trial_nums 5 --pl_seed 21 --using_gpus 2 &> ./outputs/ami_wfl2_seed21_output.txt &
# python imbalanced_text_classification/main.py --data_name ami --data_type multi --num_classes 5 --label_col label_multi --train_filename train_clean.csv --test_filename test_clean.csv --variant wfl --wce_alpha_search_space_multi 0.1 0.1 0.1 0.2 0.9 --wce_multi_trial_nums 5 --pl_seed 42 --using_gpus 3 &> ./outputs/ami_wfl2_seed42_output.txt &


python imbalanced_text_classification/main.py --data_name civil-comments --train_filename data_clean.csv --augmentation_src Bert --variant augmentation_bert --augmentation_rho_search_space 5 7.5 --pl_seed 21 --using_gpus 5 &> ./outputs/civil-comments_augmentation_bert_seed21_output.txt &
python imbalanced_text_classification/main.py --data_name civil-comments --train_filename data_clean.csv --augmentation_src Bert --variant augmentation_bert --augmentation_rho_search_space 5 7.5 --pl_seed 42 --using_gpus 6 &> ./outputs/civil-comments_augmentation_bert_seed42_output.txt &












# python imbalanced_text_classification/main.py --data_name cmsb-tsd --train_filename data_clean.csv --variant sampling_weightedRS_oversampling --sampling_weightedRS_percentage_search_space 1.0 2.0 --pl_seed 0 --using_gpus 6 &> ./outputs/cmsb-tsd_sampling_weightedRS_oversampling_seed0_output.txt &
# python imbalanced_text_classification/main.py --data_name cmsb-tsd --train_filename data_clean.csv --variant sampling_weightedRS_oversampling --sampling_weightedRS_percentage_search_space 1.0 2.0 --pl_seed 21 --using_gpus 0 &> ./outputs/cmsb-tsd_sampling_weightedRS_oversampling_seed21_output.txt &
# python imbalanced_text_classification/main.py --data_name cmsb-tsd --train_filename data_clean.csv --variant sampling_weightedRS_oversampling --sampling_weightedRS_percentage_search_space 1.0 2.0 --pl_seed 42 --using_gpus 1 &> ./outputs/cmsb-tsd_sampling_weightedRS_oversampling_seed42_output.txt &

# python imbalanced_text_classification/main.py --data_name founta-2018-thas --data_type multi --num_classes 4 --label_col label_multi --train_filename data_clean.csv --variant sampling_weightedRS_oversampling --sampling_weightedRS_percentage_search_space 2.0 2.5 --pl_seed 0 --using_gpus 6 &> ./outputs/founta-2018-thas_sampling_weightedRS_oversampling_seed0_output.txt &
# python imbalanced_text_classification/main.py --data_name founta-2018-thas --data_type multi --num_classes 4 --label_col label_multi --train_filename data_clean.csv --variant sampling_weightedRS_oversampling --sampling_weightedRS_percentage_search_space 2.0 2.5 --pl_seed 21 --using_gpus 7 &> ./outputs/founta-2018-thas_sampling_weightedRS_oversampling_seed21_output.txt &
# python imbalanced_text_classification/main.py --data_name founta-2018-thas --data_type multi --num_classes 4 --label_col label_multi --train_filename data_clean.csv --variant sampling_weightedRS_oversampling --sampling_weightedRS_percentage_search_space 2.0 2.5 --pl_seed 42 --using_gpus 0 &> ./outputs/founta-2018-thas_sampling_weightedRS_oversampling_seed42_output.txt &



# Running:

# python imbalanced_text_classification/main.py --data_name civil-comments --train_filename data_clean.csv --augmentation_src AbusiveLexicon --variant augmentation_abusive_lexicon --augmentation_rho_search_space 1.0 2.0 3 5 7.5 --pl_seed 0 --using_gpus 2 &> ./outputs/civil-comments_augmentation_abusive_lexicon_seed0_output.txt &


