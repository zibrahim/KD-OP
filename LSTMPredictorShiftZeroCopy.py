import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

import Models
from Models.LSTMAutoEncoder.LSTMAutoEncoder import LSTMAutoEncoder
from Models.LSTMAutoEncoder.Utils import process_data, flatten
from ProcessResults.VisualisePopulation import DecisionMaker
from Models.Metrics import performance_metrics
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from pylab import rcParams

from keras.callbacks import ModelCheckpoint, TensorBoard

from sklearn.metrics import precision_recall_curve, auc, roc_curve
from numpy.random import seed
seed(7)

rcParams['figure.figsize'] = 8, 6
LABELS = ["0","1"]

from Utils.Data import scale, impute

def main():
    configs = json.load(open('Configuration.json', 'r'))

    grouping = configs['data']['grouping']
    dynamic_features = configs['data']['dynamic_columns']

    outcomes = configs['data']['classification_outcome']
    lookback = configs['data']['batch_size']
    timeseries_path = configs['paths']['data_path']
    autoencoder_path = configs['paths']['autoencoder_path']

    ##read, impute and scale dataset
    non_smotedtime_series = pd.read_csv(timeseries_path + "TimeSeriesAggregatedUpto0.csv")
    non_smotedtime_series[dynamic_features] = impute(non_smotedtime_series, dynamic_features)
    normalized_timeseries = scale(non_smotedtime_series, dynamic_features)
    normalized_timeseries.insert(0, grouping, non_smotedtime_series[grouping])

    ##start working per outcome
    for outcome in outcomes:
        decision_maker = DecisionMaker()

        X_train_y0, X_valid_y0, X_valid, y_valid, X_test, y_test,  timesteps, n_features =\
            process_data(normalized_timeseries, non_smotedtime_series, outcome, grouping, non_smotedtime_series[grouping], lookback)

        epochs = 100

        autoencoder = LSTMAutoEncoder(configs['model']['name'] + outcome, outcome, timesteps, n_features)
        autoencoder.summary()

        cp = ModelCheckpoint(filepath="lstm_autoencoder_classifier.h5",
                             save_best_only=True,
                             verbose=0)

        tb = TensorBoard(log_dir='./logs',
                         histogram_freq=0,
                         write_graph=True,
                         write_images=True)

        autoencoder.fit(X_train_y0, X_train_y0, epochs,lookback,X_valid_y0,X_valid_y0,2)
        ####LSTM autoencoder

        autoencoder.plot_history()
        valid_x_predictions = autoencoder.predict(X_valid)

        mse = np.mean(np.power(flatten(X_valid) - flatten(valid_x_predictions), 2), axis=1)
        error_df = pd.DataFrame({'Reconstruction_error' : mse,
                                 'True_class' : y_valid.tolist()})

        precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df.True_class, error_df.Reconstruction_error)

        fscore = (2 * precision_rt * recall_rt) / (precision_rt + recall_rt)

        ix = np.argmax(fscore)
        best_threshold = threshold_rt[ix]
        # print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], fscore[ix]))
        pred_y = (error_df.Reconstruction_error > best_threshold).astype('int32')

        perf_df = pd.DataFrame()
        perf_dict = performance_metrics(error_df.True_class, pred_y, error_df.Reconstruction_error )
        perf_df = perf_df.append(perf_dict, ignore_index=True)
        perf_df.to_csv(autoencoder_path+"performancemetrics"+outcome+".csv", index=False)

        test_x_predictions = autoencoder.predict(X_test)
        mse = np.mean(np.power(flatten(X_test) - flatten(test_x_predictions), 2), axis=1)

        error_df = pd.DataFrame({'Reconstruction_error' : mse,
                                 'True_class' : y_test.tolist()})

        plt.figure(figsize=(10, 10))

        groups = error_df.groupby('True_class')
        fig, ax = plt.subplots()

        for name, group in groups :
            ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
                    label="1" if name == 1 else "0")
        ax.hlines(threshold_rt[ix], ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
        ax.legend()
        plt.title("Reconstruction error for different classes")
        plt.ylabel("Reconstruction error")
        plt.xlabel("Data point index")
        plt.savefig(autoencoder_path+outcome+"Reconstructionerror.pdf", bbox_inches='tight')


        false_pos_rate, true_pos_rate, thresholds = roc_curve(error_df.True_class, error_df.Reconstruction_error)
        roc_auc = auc(false_pos_rate, true_pos_rate, )

        plt.figure(figsize=(10, 10))

        plt.plot(false_pos_rate, true_pos_rate, linewidth=5, label='AUC = %0.3f' % roc_auc)
        plt.plot([0, 1], [0, 1], linewidth=5)


        plt.xlim([-0.01, 1])
        plt.ylim([0, 1.01])
        plt.legend(loc='lower right')
        plt.title('Receiver operating characteristic curve (ROC)')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(autoencoder_path+outcome+"roc.pdf", bbox_inches='tight')

        pr_auc =  auc(recall_rt, precision_rt)

        plt.figure(figsize=(10, 10))

        plt.plot(recall_rt, precision_rt, linewidth=5, label='PR-AUC = %0.3f' % pr_auc)
        plt.plot([0, 1], [1, 0], linewidth=5)


        plt.xlim([-0.01, 1])
        plt.ylim([0, 1.01])
        plt.legend(loc='lower right')
        plt.title('Precision Recall Curive')
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.savefig(autoencoder_path+outcome+"precision_recall_auc.pdf", bbox_inches='tight')
if __name__ == '__main__':
    main()
