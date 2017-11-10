# Binary Rare Events Classifier

There exists many applications of machine learning to a binary classification problem.  Examples include:

+ Customer churn
+ Fraud detection
+ Customer conversion

In many instances, the binary classes are severely imbalanced with the event class occurring in less than 5% of total cases.  This situation presents multiple overlapping challenges:

1. The algorithm may be biased toward classifying everything as a non-event.
2. Accuracy can be >90% even while many of the events are mis-categorized.
3. Algorithms that utilize the entire dataset can perform poorly.

Two sampling approaches for imbalanced data were proposed by Chen, Liaw, and Breiman specifically for Random Forests:
http://statistics.berkeley.edu/sites/default/files/tech-reports/666.pdf

The binary rare events classifier provides methods for sampling and classifying imbalanced data.  These methods include:
1. sample_abundant_data - Calculate the number of rare cases and create k samples of similar size from the abundant cases.  Sample size is determined from a +/- tolerance coefficient.  The k samples are re-created from the original data each time the sample is drawn.
2. bootstrap_sample - Calculate k bootstrap samples of size n from both the rare and abundant cases.
3. get_majority_vote - Classify data based on the k samples and calculate a majority vote classification.
4. get_quality_metrics - Calculate accuracy, precision, recall, AUC from the majority vote classifier.
