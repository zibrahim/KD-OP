import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.metrics import auc, precision_recall_curve, f1_score
from Models.Utils import stratified_group_k_fold, get_distribution, get_distribution_scalars
from Models.Metrics import performance_metrics

class XGBoostClassifier():

    def __init__(self, X, y,outcome, grouping):

        self.predicted_probabilities = pd.DataFrame()
        self.X = X
        self.y = y.astype(int)
        self.outcome = outcome
        self.grouping = grouping
        class_distributions = [get_distribution_scalars(y.astype(int))]

        class_weights = class_distributions[0][0] / class_distributions[0][1]

        self.model = xgb.XGBClassifier(scale_pos_weight=class_weights,
                                 learning_rate=0.007,
                                 n_estimators=100,
                                 gamma=0,
                                 max_depth=4,
                                 min_child_weight=2,
                                 subsample=1,
                                 eval_metric='error')


    def fit(self, label, groups):

        x_columns = ((self.X.columns).tolist())
        X = self.X[x_columns]
        X.reset_index()
        #groups = np.array(self.X[self.grouping])

        distrs = [get_distribution(self.y)]
        index = ['Entire set']

        prs = []
        aucs = []
        mean_recall = np.linspace(0, 1, 100)
        threshold_indices = []
        i = 0

        plt.figure(figsize=(10, 10))

        stats_df = pd.DataFrame()

        false_X = []
        false_Y = []
        false_IDs = []

        predicted_Y = []
        predicted_thredholds = []
        predicted_IDs = []


        for fold_ind, (training_ind, testing_ind) in enumerate(stratified_group_k_fold(self.y, groups, k=10)) : #CROSS-VALIDATION
            #Train

                training_groups = groups.iloc[training_ind], groups.iloc[testing_ind]
                y_grab = self.y
                y_grab.reset_index()
                training_y, testing_y = y_grab.iloc[training_ind], y_grab.iloc[testing_ind]
                training_X, testing_X = X.iloc[training_ind], X.iloc[testing_ind]
                #testing_X.drop(self.grouping, inplace=True)
                eval_set = [(training_X, training_y), (testing_X, testing_y)]

                self.model.fit(training_X, training_y, early_stopping_rounds=10,
                               eval_metric=["error","logloss"], eval_set=eval_set)

                testing_ids = groups[testing_ind]

                # Train, predict and Plot
                #self.model.fit(testing_X, testing_y)
                y_pred_rt = self.model.predict_proba(testing_X)[:, 1]

                precision, recall, thresholds = precision_recall_curve(testing_y, y_pred_rt)
                fscore = (2 * precision * recall) / (precision + recall)
                ix = np.argmax(fscore)
                threshold_indices.append(ix)
                threshold = thresholds[ix]
                y_pred_binary = (y_pred_rt > thresholds[ix]).astype('int32')

                #Get all predictions
                for w in range(0, len(testing_y)):
                    predicted_Y.append(y_pred_rt[w])
                    predicted_IDs.append(testing_ids.iloc[w])
                    predicted_thredholds.append(threshold)

                #Get false negatives:
                for w in range(0,len(testing_y)):
                    if (testing_y.iloc[w] ==1 and y_pred_binary[w] == 0) or (testing_y.iloc[w] ==0 and y_pred_binary[w] ==1):
                        false_Y.append(testing_y.iloc[w])
                        false_X.append(testing_X.iloc[w])
                        false_IDs.append(testing_ids.iloc[w])

                prs.append(np.interp(mean_recall, precision, recall))
                pr_auc = auc(recall, precision)
                aucs.append(pr_auc)
                i += 1

                stats_df = stats_df.append(performance_metrics(testing_y, y_pred_binary, y_pred_rt), ignore_index=True)

                # add to the distribution dataframe, for verification purposes
                distrs.append(get_distribution(training_y))

                index.append(f'training set - fold {fold_ind}')
                distrs.append(get_distribution(testing_y))
                index.append(f'testing set - fold {fold_ind}')

        plt.plot([0, 1], [1, 0], linestyle='--', lw=3, label='Luck', alpha=.8)

        mean_precision = np.mean(prs, axis=0)

        mean_auc = auc(mean_recall, mean_precision)
        plt.plot(mean_precision, mean_recall, color='navy',
             label=r' Mean AUCPR = %0.3f' % mean_auc,
             lw=4)

        plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Recall', fontsize=20)
        plt.ylabel('Precision', fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=20)

        stats_path = "Run/Stats/"
        prediction_path = "Run/XGBoost/"

        plt.legend(prop={'size' : 10}, loc=4)
        plt.savefig(prediction_path+"ROCXGB"+label+self.outcome+".pdf", bbox_inches='tight')

        stats_df.to_csv(stats_path + label+ self.outcome  + "XGB.csv", index=False)

        #f = plt.figure()

        #rf_shap_values = shap.KernelExplainer(self.model.predict, testing_X)
        #shap.summary_plot(rf_shap_values, testing_X)

        #f.savefig(prediction_path+"SHAP"+self.outcome+experiment_number+".pdf", bbox_inches='tight')

        return predicted_Y, predicted_thredholds, predicted_IDs, self.model.feature_importances_


    def predict( self, holdout_X, holdout_y):

        x_columns = ((holdout_X.columns).tolist())
        #x_columns.remove(self.grouping)

        holdout_X = holdout_X[x_columns]
        holdout_X.reset_index()

        yhat = self.model.predict_proba(holdout_X)[:, 1]
        precision, recall, thresholds = precision_recall_curve(holdout_y, yhat)
        fscore = (2 * precision * recall) / (precision + recall)

        ix = np.argmax(fscore)

        y_pred_binary = (yhat > thresholds[ix]).astype('int32')

        plt.figure(figsize=(10, 10))

        plt.plot([0,1], [1, 0], linestyle='--', label='No Skill')
        plt.plot(recall, precision, marker='.', label='Logistic')
        plt.scatter(recall[ix], recall[ix], marker='o', color='black', label='Best')
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()



        stats_path = "Run/Stats/"
        prediction_path = "Run/XGBoost/"

        plt.legend(prop={'size' : 10}, loc=4)
        plt.savefig(prediction_path+"HOLOUT"+self.outcome+".pdf", bbox_inches='tight')

        #return y_pred_rt, y_pred_binary, threshold

