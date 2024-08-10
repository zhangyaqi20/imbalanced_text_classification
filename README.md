Code for paper [A Study of the Class Imbalance Problem in Abusive Language Detection](https://aclanthology.org/2024.woah-1.4/) published at the 8th Workshop on Online Abuse and Harms (WOAH 2024).

**Project Description**
-----

Abusive language detection has drawn increasing interest in recent years. However, a less systematically explored obstacle is label imbalance, i.e., the amount of abusive data is much lower than non-abusive data, leading to performance issues. The aim of this work is to conduct a comprehensive comparative study of popular methods for addressing the class imbalance issue. 

We explore 10 well-known approaches on 8 datasets with distinct characteristics: binary or multi-class, moderately or largely imbalanced, focusing on various types of abuse, etc. Additionally, we propose two novel methods specialized for abuse detection: AbusiveLexiconAug and ExternalDataAug, which enrich the training data using abusive lexicons and external abusive datasets, respectively. 

We conclude that: 

- Our AbusiveLexiconAug approach, random oversampling, and focal loss are the most versatile methods on various datasets

- Focal loss tends to yield peak model performance

- Oversampling and focal loss provide promising results for binary datasets and small multi-class sets, while undersampling and weighted cross-entropy are more suitable for large multi-class sets

- most methods are sensitive to hyperparameters, yet our suggested choice of hyperparameters provides a good starting point.

**Notes**
----

The current code version can be easily extended in terms of other datasets, imbalanced learning methods, loss functions etc.
