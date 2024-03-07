category2data_label = {"hate": {'twitter-hate-speech-tsa': 1, 
                                'gibert-2018-shs': 1,
                                'founta-2018-thas': 3,
                                'davidson-thon': 0},
                        "non-hate": {'twitter-hate-speech-tsa': 0,
                                        'gibert-2018-shs': 0,
                                        'davidson-thon': 2},
                        "abusive_offensive_toxic": {'founta-2018-thas': 2,
                                                'davidson-thon': 1,
                                                'civil-comments': 1},
                        "sexism": {'cmsb-tsd': 1,
                                    'ami': 0, 'ami': 1, 'ami': 2, 'ami': 3, 'ami': 4}
                        }

bin_data_names = ['twitter-hate-speech-tsa', 'civil-comments', 'gibert-2018-shs', 'us-election-2020', 'cmsb-tsd']
multi_class_data_names = ['founta-2018-thas', 'ami', 'davidson-thon']
data_name2label_col = dict()
for data_name in bin_data_names:
    data_name2label_col[data_name] = "label"
for data_name in multi_class_data_names:
    data_name2label_col[data_name] = "label_multi"