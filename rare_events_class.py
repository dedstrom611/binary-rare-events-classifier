import pandas as pd
import numpy as np
import random
import os
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC, OneClassSVM
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.discrete.discrete_model import Logit

class ImbalancedClassifier(object):
    ''' A set of tools to help with classification problems for imbalanced data.'''

    def __init__(self):
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test
        dep_variable=self.dep_variable

    def sample_abundant_data(self, tolerance=0.20):
        '''Create a sample from the abundant class of a binary dependent variable.

        INPUTS:
        df (dataframe) - A pandas dataframe containing the set of features for modeling.
        y_df (dataframe) - A pandas dataframe containing the dependent variable for which to produce the sample.
        dep_variable (str) - The dataframe column representing the dependent variable, stored as 0/1 boolean values.
        tolerance (float) - A tolerance factor for the number of samples to produce.  The resulting sample will be
        between 1 +/- tolerance of the rare events.

        RETURNS:
        X_tr (dataframe) - A new dataframe containing all instances where dep_variable == 1 and the sampled rows
        where dep_variable == 0.
        y_tr (Pandas data series) - A new Pandas data series of the response variable based on the sample.
        '''
        df = pd.merge(self.X_train, pd.DataFrame(self.y_train, columns=[self.dep_variable], index=self.y_train.index),\
        how='inner', left_index=True, right_index=True)
        y_1 = df.loc[df[self.dep_variable] == 1]

        sample_pct = random.uniform(1 - tolerance, 1 + tolerance)
        sample_size = int(np.sum(df[self.dep_variable]) * sample_pct)

        samp = df.loc[df[self.dep_variable] == 0].sample(n=sample_size)

        new_x = pd.concat([samp, y_1], axis=0)
        new_y = new_x.pop(dep_variable)
        return new_x, new_y

    def bootstrap_sample(self, tolerance=0.20):
        '''Create bootstrap samples from the majority and minority class of a data frame.  The resulting
        samples are used in balanced random forests (http://statistics.berkeley.edu/sites/default/files/tech-reports/666.pdf)
        and gradient boosting algorithms.

        INPUTS:
        df (dataframe) - A pandas dataframe containing the set of features for modeling.
        y_df (dataframe) - A pandas dataframe containing the dependent variable for which to produce the sample.
        dep_variable (str) - The dataframe column representing the dependent variable, stored as 0/1 boolean values.
        tolerance (float) - A tolerance factor for the number of samples to produce.  The resulting sample will be
        between 1 +/- tolerance of the rare events.

        RETURNS:
        X_tr (dataframe) - A new dataframe containing all instances where dep_variable == 1 and the sampled rows
        where dep_variable == 0.
        y_tr (Pandas data series) - A new Pandas data series of the response variable based on the sample.
        '''
        df = pd.merge(self.X_train, pd.DataFrame(self.y_train, columns=[self.dep_variable], index=self.y_train.index),\
        how='inner', left_index=True, right_index=True)

        sample_pct = random.uniform(1 - tolerance, 1 + tolerance)
        sample_size = int(np.sum(df[self.dep_variable]) * sample_pct)

        samp0 = df.loc[df[self.dep_variable] == 0].sample(n=sample_size, replace=True)
        samp1 = df.loc[df[self.dep_variable] == 1].sample(n=sample_size, replace=True)

        new_x = pd.concat([samp0, samp1], axis=0)
        new_y = new_x.pop(dep_variable)
        return new_x, new_y

    def get_combined_proba(models, X_train, X_test, y_train, y_test, sample_method='abundant', ksamples=15):
        '''sample_method = 'abundant' or 'bootstrap'
        '''
        sample_methods={'abundant': sample_abundant_data, 'bootstrap': bootstrap_sample}
        predictions = []
        for k in range(ksamples):
            Xt, Yt = sample_methods[sample_method](X_train, y_train, dep_variable='N188')
            for model in models.values():
                model.fit(Xt, Yt)
                p = model.predict_proba(X_test)
                predictions.append(p)

        return predictions

    def get_majority_vote(models, modname='Default Model',sample_method='abundant', ksamples=15, print_results=True):
        ''' Fit k models using either abundant of bootstrap samples and create classification predictions.
        Use a majority vote of the k samples to produce a final classification prediction.

        INPUTS:

        models (dict) - A dictionary of model class instantiation references.
        E.g. if gb=GradientBoostingClassifier, lr=LogisticRegression, then models = {'gb': gb, 'lr': lr}
        modname (str) - A string for printing the algorithm name if quality metrics are printed.
        sample_method (str) - 'abundant' or 'bootstrap', the preferred sample method to use.
        k_samples (int) - The number of samples to take from the training dataset
        print_results (bool) - A boolean of whether or not quality metrics should be printed for final results.

        RETURNS:

        roc_auc (float) - The ROC AUC (area under the receiver/operator characteristic)
        votes (Numpy Array) - An array of the binary predictions for y_test
        probs (Numpy Array) - An array of the probabilities of positive prediction for y_test
        '''
        sample_methods={'abundant': sample_abundant_data, 'bootstrap': bootstrap_sample}
        predictions = []
        probs = np.zeros(len(y_test))
        for k in range(ksamples):
            Xt, Yt = sample_methods[sample_method](self.X_train, self.y_train, dep_variable=self.dep_variable)
            for model in models.values():
                model.fit(Xt, Yt)
                p = model.predict(X_test)
                predictions.append(p)
                prob = model.predict_proba(self.X_test)[:,1]
                probs = np.sum([probs, prob], axis=0)

        votes = np.array([1 if sum(x) > (len(predictions) / 2) else 0 for x in zip(*predictions)])
        probs = probs / len(predictions)
        accuracy = accuracy_score(self.y_test, votes)
        precision = precision_score(self.y_test, votes)
        cm = confusion_matrix(self.y_test, votes)
        recall = recall_score(self.y_test, votes)
        fpr, tpr, thresholds = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)

        if print_results is True:
            model = 'combined'
            self.print_quality_metrics(model, modname, self.X_test, self.y_test)

        return roc_auc, votes, probs

    def print_quality_metrics(model, model_name):
        '''Print basic quality metrics for a given model, including:
        confusion matrix, AUC, accuracy, precision and recall.

        INPUTS:
        model - The instantiated class name of the model used (e.g. gb, lr, rf)
        model_name (str) - The text description of the model for use in printing.

        RETURNS:
        None
        '''
        accuracy = accuracy_score(self.y_test, model.predict(self.X_test))
        precision = precision_score(self.y_test, model.predict(self.X_test))
        recall = recall_score(self.y_test, model.predict(self.X_test))
        fpr, tpr, thresholds = roc_curve(self.y_test, model.predict(self.X_test))
        roc_auc = auc(fpr, tpr)
        cm = confusion_matrix(self.y_test, model.predict(self.X_test))
        print ("Model Name: Accuracy\tPrecision\tRecall\tAUC")
        print('{0}: {1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}'.format(model_name, accuracy, precision, recall, roc_auc))
        print ("Confusion Matrix")
        print (cm)

if __name__ == '__main__':
        dep = 'N189'

        X_train = pd.read_pickle('data/X_train.pkl')
        X_test = pd.read_pickle('data/X_test.pkl')
        y_train = pd.read_pickle('data/y_train.pkl')
        y_test = pd.read_pickle('data/y_test.pkl')

        gb = GradientBoostingClassifier(learning_rate=0.005, n_estimators=500,\
        max_features='sqrt', max_depth=5)

        models = {'Gradient Boosting': model1}
        model1 = {'gb': gb}

        for modname, model in models.items():
            roc_auc, votes, probs = get_majority_vote(model, X_train, X_test, y_train, y_test,\
            modname=modname, dep_variable=dep, sample_method='abundant', ksamples=k_samp)

            fpr, tpr, thresholds = roc_curve(y_test, probs)
