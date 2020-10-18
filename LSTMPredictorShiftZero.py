import os
import json

from Models.LSTMAutoEncoder.LSTMAutoEncoder import LSTMAutoEncoder
from Models.LSTMAutoEncoder.Utils import process_data, flatten
from ProcessResults.VisualisePopulation import DecisionMaker
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from pylab import rcParams

from keras.callbacks import ModelCheckpoint, TensorBoard

from os import makedirs

from numpy.random import seed

from Models.Utils import get_train_test_split

seed(7)

rcParams['figure.figsize'] = 8, 6
LABELS = ["0","1"]

from Utils.Data import scale, impute

def main():
    configs = json.load(open('Configuration.json', 'r'))
    epochs = configs['training']['epochs']
    grouping = configs['data']['grouping']
    dynamic_features = configs['data']['dynamic_columns']

    outcomes = configs['data']['classification_outcome']
    lookback = configs['data']['batch_size']
    timeseries_path = configs['paths']['data_path']
    saved_models_path = configs['paths']['saved_models_path']

    ##read, impute and scale dataset
    non_smotedtime_series = pd.read_csv(timeseries_path + "TimeSeriesAggregatedUpto0.csv")
    non_smotedtime_series[dynamic_features] = impute(non_smotedtime_series, dynamic_features)
    normalized_timeseries = scale(non_smotedtime_series, dynamic_features)
    normalized_timeseries.insert(0, grouping, non_smotedtime_series[grouping])

    ##start working per outcome
    for outcome in outcomes:
        decision_maker = DecisionMaker()
        fold_ind, train_ind, test_ind = get_train_test_split(non_smotedtime_series[outcome].astype(int),
                                                             non_smotedtime_series[grouping])

        X_train_y0, X_valid_y0, X_valid, y_valid, X_test, y_test, timesteps,\
        n_features = \
            process_data(normalized_timeseries, non_smotedtime_series, outcome, grouping, lookback,
                         train_ind, test_ind)

        autoencoder = LSTMAutoEncoder(configs['model']['name'] + outcome, outcome, timesteps, n_features)
        autoencoder.summary()

        autoencoder.fit(X_train_y0, X_train_y0, epochs,lookback,X_valid_y0,X_valid_y0,2)

        ###save model
        filename = saved_models_path+ configs['model']['name'] + outcome+ '.h5'
        autoencoder.save_model(filename)

        ####LSTM autoencoder
        autoencoder.plot_history()
        test_x_predictions = autoencoder.predict(X_test)
        mse = np.mean(np.power(flatten(X_test) - flatten(test_x_predictions), 2), axis=1)

        test_error_df = pd.DataFrame({'Reconstruction_error' : mse,
                                 'True_class' : y_test.tolist()})

        pred_y, best_threshold, precision_rt, recall_rt= \
            autoencoder.predict_binary(test_error_df.True_class, test_error_df.Reconstruction_error)

        autoencoder.output_performance(test_error_df.True_class, test_error_df.Reconstruction_error,pred_y)
        autoencoder.plot_reconstruction_error(test_error_df, best_threshold)
        autoencoder.plot_roc(test_error_df)
        autoencoder.plot_pr(precision_rt, recall_rt)

if __name__ == '__main__':
    main()
