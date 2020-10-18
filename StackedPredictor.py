import os
import json

from sklearn.metrics import auc

from Models.LSTMAutoEncoder.LSTMAutoEncoder import LSTMAutoEncoder
from Models.LSTMAutoEncoder.Utils import process_data, lstm_flatten
from ProcessResults.ClassificationReport import ClassificationReport
from Utils.Data import flatten, scale, impute
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

from pylab import rcParams
from Models.Utils import class_weights, class_counts
import numpy as np
np.seterr(divide='ignore')

from numpy.random import seed

from Models.Utils import get_train_test_split, generate_aggregates
from Models.XGBoost.XGBoost import XGBoostClassifier
import os.path

seed(7)

rcParams['figure.figsize'] = 8, 6
LABELS = ["0", "1"]



def main () :
    configs = json.load(open('Configuration.json', 'r'))
    epochs = configs['training']['epochs']
    grouping = configs['data']['grouping']
    dynamic_features = configs['data']['dynamic_columns']
    static_features = configs['data']['static_columns']

    outcomes = configs['data']['classification_outcome']
    lookback = configs['data']['batch_size']
    timeseries_path = configs['paths']['data_path']
    autoencoder_models_path = configs['paths']['autoencoder_models_path']
    test_data_path = configs['paths']['test_data_path']

    ##read, impute and scale dataset
    non_smotedtime_series = pd.read_csv(timeseries_path + "TimeSeriesAggregatedUpto0.csv")
    non_smotedtime_series[dynamic_features] = impute(non_smotedtime_series, dynamic_features)
    normalized_timeseries = scale(non_smotedtime_series, dynamic_features)
    normalized_timeseries.insert(0, grouping, non_smotedtime_series[grouping])

    #intialise classification report which will house results of all outcomes
    classification_report = ClassificationReport()

    #save lstm performance for comparison with final outcome
    lstm_praucs = []
    ##start working per outcome
    for outcome in outcomes :
        fold_ind, train_ind, test_ind = get_train_test_split(non_smotedtime_series[outcome].astype(int),
                                                             non_smotedtime_series[grouping])

        ##Load LSTM models if they exist, otherwise train new models and save them
        autoencoder_filename = autoencoder_models_path + configs['model']['name'] + outcome + '.h5'
        X_train, X_train_y0, X_valid_y0, X_valid, y_valid, X_test, y_test, timesteps, \
        n_features = \
            process_data(normalized_timeseries, non_smotedtime_series, outcome, grouping, lookback,
                         train_ind, test_ind)
        if ("3D" not in outcome) :
            if os.path.isfile(autoencoder_filename):
                print(" Autoencoder trained model exists for oucome", outcome,"file:" , autoencoder_filename)
                autoencoder = LSTMAutoEncoder(configs['model']['name'] + outcome, outcome,
                                              timesteps, n_features,saved_model = autoencoder_filename)
                autoencoder.summary()


            else :
                print("Autencoder trained model does not exist for outcome", outcome, "file:", autoencoder_filename)
                autoencoder = LSTMAutoEncoder(configs['model']['name'] + outcome, outcome, timesteps, n_features)
                autoencoder.summary()

                autoencoder.fit(X_train_y0, X_train_y0, epochs, lookback, X_valid_y0, X_valid_y0, 2)
                autoencoder.plot_history()

            train_x_predictions = autoencoder.predict(X_train)
            mse_train = np.mean(np.power(lstm_flatten(X_train) - lstm_flatten(train_x_predictions), 2), axis=1)

            test_x_predictions = autoencoder.predict(X_test)

            mse_test = np.mean(np.power(lstm_flatten(X_test) - lstm_flatten(test_x_predictions), 2), axis=1)

            test_error_df = pd.DataFrame({'Reconstruction_error' : mse_test,
                                          'True_class' : y_test.tolist()})

            pred_y, best_threshold, precision_rt, recall_rt = \
                  autoencoder.predict_binary(test_error_df.True_class, test_error_df.Reconstruction_error)

            autoencoder.output_performance(test_error_df.True_class, pred_y)
            autoencoder.plot_reconstruction_error(test_error_df, best_threshold)
            autoencoder.plot_roc(test_error_df)
            autoencoder.plot_pr(precision_rt, recall_rt)
            lstm_prauc = auc(recall_rt, precision_rt)
            lstm_praucs.append(lstm_prauc)
            #Feature Selector
            training_loc = train_ind[0]#+train_ind[1]
            training_ids = non_smotedtime_series.iloc[training_loc]
            training_ids = training_ids[grouping]

            testing_ids = non_smotedtime_series.iloc[test_ind[1]]
            testing_ids = testing_ids[grouping]

            flat_df ,timesteps= flatten (non_smotedtime_series, dynamic_features, grouping, static_features, outcome)
            temporal_features = set(flat_df.columns) - set(static_features)
            temporal_features = set(temporal_features) - set([outcome,grouping])

            X_train = flat_df.loc[flat_df[grouping].isin(training_ids)]
            y_train = X_train[outcome].astype(int)
            training_groups  = X_train[grouping]
            X_train_static = X_train[static_features]
            X_train_static[grouping] = training_groups
            X_train = X_train[temporal_features]

            X_test = flat_df.loc[flat_df[grouping].isin(testing_ids)]
            y_test = X_test[outcome].astype(int)
            testing_groups = X_test[grouping]
            X_test_static = X_test[static_features]
            X_test_static.loc[grouping] = testing_groups
            X_test= X_test[temporal_features]

            ########
            aggregate_df = generate_aggregates ( X_train, temporal_features, grouping, training_groups )

            static_aggregate_train_df = pd.concat([aggregate_df, X_train_static], axis=1,join='inner')
            static_aggregate_train_df = static_aggregate_train_df.loc[:, ~static_aggregate_train_df.columns.duplicated()]
            static_aggregate_train_df.drop(columns = [grouping], inplace=True, axis=1)
            static_aggregate_train_df['mse'] = mse_train


            aggregate_df_test = generate_aggregates ( X_test, temporal_features, grouping, testing_groups )
            static_aggregate_test_df = pd.concat([aggregate_df_test, X_test_static], axis=1,join='inner')
            static_aggregate_test_df = static_aggregate_test_df.loc[:, ~static_aggregate_test_df.columns.duplicated()]
            static_aggregate_test_df.drop(columns = [grouping], inplace=True,axis=1)
            static_aggregate_test_df['mse'] = mse_test


            static_aggregate_test_df.to_csv("static_aggretate.csv", index=False)
            static_baseline_classifier = XGBoostClassifier(static_aggregate_train_df,
                                                                  y_train, outcome, grouping)

            static_baseline_classifier.fit("aggregate_static", mse_train*100)

            y_pred_binary, best_threshold, precision_rt, recall_rt, yhat = \
                static_baseline_classifier.predict(static_aggregate_test_df, y_test)


            print(" CLASS WEIGHTS FOR Y ACTUAL: ", class_counts(y_test))
            print(" CLASS WEIGHTS FOR Y PREDICTE: ", class_counts(y_pred_binary))

            static_baseline_classifier.output_performance(y_test, y_pred_binary)
            static_baseline_classifier.plot_pr(precision_rt, recall_rt, "XGBoost Static")
            static_baseline_classifier.plot_feature_importance(static_aggregate_test_df.columns)

            to_write_for_plotting = static_aggregate_test_df
            to_write_for_plotting['outcome'] = y_test
            to_write_for_plotting.to_csv(test_data_path+outcome+".csv", index=False)

            #add to classification report

            classification_report.add_model_result(outcome,y_test, y_pred_binary, best_threshold,
                                                   precision_rt, recall_rt, yhat)


            #delete variables
            del static_aggregate_train_df
            del static_aggregate_test_df
            del X_train
            del X_train_y0
            del X_valid_y0
            del X_valid
            del y_valid
            del X_test
            del y_test
            del timesteps
            del train_x_predictions
            del test_x_predictions
            del test_error_df
    #risk_score_visualiser = Visualiser(normalized_timeseries, non_smotedtime_series,
     #                                  dynamic_features, static_features
      #                                 )
    #After fitting model to all outcomes, plot and get summary statistics
    classification_report.plot_distributions_vs_aucs()
    classification_report.plot_pr_auc()
    classification_report.plot_auc()
    classification_report.compare_lstim_xgboost(lstm_praucs)


if __name__ == '__main__' :
    main()
