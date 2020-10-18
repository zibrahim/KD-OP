import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from Models.Metrics import performance_metrics
from Models.Utils import get_distribution, get_distribution_scalars

class XGBoostClassifier():
    def __init__(self, X, y,outcome, grouping, saved_model = None):

        self.predicted_probabilities = pd.DataFrame()
        self.X = X
        self.y = y.astype(int)
        self.outcome = outcome
        self.grouping = grouping
        configs = json.load(open('Configuration.json', 'r'))
        self.output_path = configs['paths']['xgboost_output_path']

        class_distributions = [get_distribution_scalars(y.astype(int))]

        class_weights = class_distributions[0][0] / class_distributions[0][1]

        self.model = xgb.XGBClassifier(scale_pos_weight=class_weights,
                                 learning_rate=0.007,
                                 n_estimators=100,
                                 gamma=0,
                                 min_child_weight=2,
                                 subsample=1,
                                 eval_metric='error')

        if saved_model != None:
            self.model =  xgb.Booster({'nthread' : 4})  # init model
            (self.model).load_model(saved_model)

    def save_model( self, filename ):
        self.model.save_model(filename)

    def fit(self, label, sample_weights):
        x_columns = ((self.X.columns).tolist())
        X = self.X[x_columns]
        X.reset_index()
        y = self.y
        y.reset_index()

        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        scores = cross_validate(self.model.fit(X,y, sample_weight=sample_weights), X, y, scoring=['f1_macro', 'precision_macro',
                                                                    'recall_macro'], cv=cv, n_jobs=-4)

        print(label+'Mean F1 Macro:', np.mean(scores['test_f1_macro']), 'Mean Precision Macro: ',
              np.mean(scores['test_precision_macro']), 'mean Recall Macro' ,
              np.mean(scores['test_recall_macro']))

        #self.model.fit(X,y,sample_weight=sample_weights)
        #return predicted_Y, predicted_thredholds, predicted_IDs, self.model.feature_importances_


    def predict( self, holdout_X, holdout_y):

        x_columns = ((holdout_X.columns).tolist())
        #x_columns.remove(self.grouping)

        holdout_X = holdout_X[x_columns]
        holdout_X.reset_index()

        yhat = (self.model).predict_proba(holdout_X)[:, 1]
        precision_rt, recall_rt, thresholds = precision_recall_curve(holdout_y, yhat)
        fscore = (2 * precision_rt * recall_rt) / (precision_rt + recall_rt)

        ix = np.argmax(fscore)
        best_threshold = thresholds[ix]
        y_pred_binary = (yhat > thresholds[ix]).astype('int32')

        return y_pred_binary, best_threshold, precision_rt, recall_rt, yhat


    def plot_pr( self, precision, recall, label):
        pr_auc =  auc(recall, precision)
        plt.figure(figsize=(10, 10))
        plt.plot(recall, precision, linewidth=5, label='PR-AUC = %0.3f' % pr_auc)
        plt.plot([0, 1], [1, 0], linewidth=5)

        plt.xlim([-0.01, 1])
        plt.ylim([0, 1.01])
        plt.legend(loc='lower right')
        plt.title(self.outcome+' Precision Recall Curive-'+label)
        plt.ylabel('Precision')
        plt.xlabel('Recall')

        plt.savefig(self.output_path+self.outcome+label+"precision_recall_auc.pdf", bbox_inches='tight')

    def plot_feature_importance( self ,colnames):
        plt.figure(figsize=(10, 10))
        plt.bar(range(len((self.model).feature_importances_)), (self.model).feature_importances_)
        plt.savefig(self.output_path + self.outcome  + "xgbprecision_recall_auc.pdf", bbox_inches='tight')

    def output_performance ( self, true_class, pred_y ) :
        perf_df = pd.DataFrame()
        perf_dict = performance_metrics(true_class, pred_y)
        perf_df = perf_df.append(perf_dict, ignore_index=True)
        perf_df.to_csv(self.output_path + "xgboostperformancemetrics" + self.outcome + ".csv", index=False)
