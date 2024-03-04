category2data_label = {"hate": {'bretschneider-th-main': 1,
                                'twitter-hate-speech-tsa': 1, 
                                'bretschneider-th-school': 1,
                                'gibert-2018-shs': 1,
                                'us-election-2020': 1,
                                'founta-2018-thas': 3,
                                'davidson-thon': 0},
                        "non-hate": {'bretschneider-th-main': 0,
                                        'twitter-hate-speech-tsa': 0, 
                                        'bretschneider-th-school': 0,
                                        'gibert-2018-shs': 0,
                                        'us-election-2020': 0,
                                        'davidson-thon': 2},
                        "abusive_offensive": {'founta-2018-thas': 2,
                                                'davidson-thon': 1},
                        "racism_sexism": {'cmsb-tsd': 1,
                                          'waseem-and-hovy-2016': 1,
                                          'ami': 0, 'ami': 1, 'ami': 2, 'ami': 3, 'ami': 4}
                                          }

bin_data_names = ['bretschneider-th-main', 'twitter-hate-speech-tsa', 'bretschneider-th-school', 'gibert-2018-shs', 'us-election-2020', 'cmsb-tsd', 'waseem-and-hovy-2016']
multi_class_data_names = ['founta-2018-thas', 'ami', 'davidson-thon']
data_name2label_col = dict()
for data_name in bin_data_names:
    data_name2label_col[data_name] = "label"
for data_name in multi_class_data_names:
    data_name2label_col[data_name] = "label_multi"