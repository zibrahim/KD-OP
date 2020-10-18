import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

import Models
from Models.LSTMAutoEncoder.Utils import temporalize, flatten, scale, curve_shift
from ProcessResults.VisualisePopulation import DecisionMaker
from Models.Utils import generate_slopes, get_distribution_percentages

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
from pylab import rcParams


from keras import Input
from numpy import newaxis

from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import optimizers, Sequential
from keras.models import Model
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from keras.callbacks import ModelCheckpoint, TensorBoard


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, auc, roc_curve

from numpy.random import seed
seed(7)
from sklearn.model_selection import train_test_split

SEED = 123 #used to help randomly select the data points
DATA_SPLIT_PCT = 0.2

rcParams['figure.figsize'] = 8, 6
LABELS = ["Normal","Break"]

from Utils.Data import scale, impute

def main():
    configs = json.load(open('Configuration.json', 'r'))

    grouping = configs['data']['grouping']
    static_features = configs['data']['static_columns']
    dynamic_features = configs['data']['dynamic_columns']

    outcomes = configs['data']['classification_outcome']
    lookback = configs['data']['batch_size']
    timeseries_path = configs['paths']['data_path']

    ##read, impute and scale dataset


    ##start working per outcome
    for outcome in outcomes:
        time_series = pd.read_csv(timeseries_path + "SMOTEDTimeSeries/" + outcome + "StackedTimeSeries1Day.csv")

        time_series[dynamic_features] = impute(time_series, dynamic_features)
        normalised_series = scale(time_series, dynamic_features)
        normalised_series.insert(0, grouping, time_series[grouping])
        normalised_series.insert(len(normalised_series.columns), outcome, time_series[outcome])

        normalised_series = curve_shift(normalised_series, grouping, outcome, shift_by=lookback-1)

        decision_maker = DecisionMaker()

        #train/test and validation sets
        X_cols = (normalised_series.columns).tolist()
        X_cols.remove(outcome)
        X_cols.remove(grouping)

        input_X = normalised_series.loc[:, normalised_series.columns.isin(X_cols)].values  # converts the df to a numpy array
        input_y = normalised_series[outcome].values

        n_features = input_X.shape[1]  # number of features

        X, y = temporalize(X=input_X, y=input_y, lookback=lookback)


        X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.33,
                                                            random_state=SEED, stratify = y)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.33,
                                                              random_state=SEED,stratify = y_train)


        X_train = X_train.reshape(X_train.shape[0], lookback, n_features)
        X_valid = X_valid.reshape(X_valid.shape[0], lookback, n_features)
        X_test = X_test.reshape(X_test.shape[0], lookback, n_features)

        distrs_percents = [get_distribution_percentages((normalised_series[outcome]).astype(int))]
        scaler = StandardScaler().fit(flatten(X_train))

        a = flatten(X_train)
        print('colwise mean', np.mean(a, axis=0).round(6))
        print('colwise variance', np.var(a, axis=0))

        X_valid_scaled = LSTMAutoEncoder.Utils.scale(X_valid, scaler)
        X_test_scaled = LSTMAutoEncoder.Utils.scale(X_test, scaler)

        timesteps = X_train.shape[1]  # equal to the lookback
        n_features = X_train.shape[2]  # 59

        epochs = 100
        lr = 0.0001

        lstm_autoencoder = Sequential()
        # Encoder
        lstm_autoencoder.add(LSTM(32, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
        lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=False))
        lstm_autoencoder.add(RepeatVector(timesteps))
        # Decoder
        lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=True))
        lstm_autoencoder.add(LSTM(32, activation='relu', return_sequences=True))
        lstm_autoencoder.add(TimeDistributed(Dense(n_features)))

        lstm_autoencoder.summary()

        adam = optimizers.Adam(lr)
        lstm_autoencoder.compile(loss='mse', optimizer=adam)

        cp = ModelCheckpoint(filepath="lstm_autoencoder_classifier.h5",
                             save_best_only=True,
                             verbose=0)

        tb = TensorBoard(log_dir='./logs',
                         histogram_freq=0,
                         write_graph=True,
                         write_images=True)

        lstm_autoencoder_history = lstm_autoencoder.fit(X_train, X_train,
                                                        epochs=epochs,
                                                        batch_size=lookback,
                                                        validation_data=(X_valid, X_train),
                                                        verbose=2).history

        #print(distrs_percents)
        ####LSTM autoencoder

        plt.figure(figsize=(10, 10))
        plt.plot(lstm_autoencoder_history['loss'], linewidth=2, label='Train')
        plt.plot(lstm_autoencoder_history['val_loss'], linewidth=2, label='Valid')
        plt.legend(loc='upper right')
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.savefig("LossOverEpochsSMOTE.pdf", bbox_inches='tight')

        plt.figure(figsize=(10, 10))

        valid_x_predictions = lstm_autoencoder.predict(X_valid_scaled)
        mse = np.mean(np.power(flatten(X_valid_scaled) - flatten(valid_x_predictions), 2), axis=1)

        error_df = pd.DataFrame({'Reconstruction_error' : mse,
                                 'True_class' : y_valid.tolist()})

        precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df.True_class,
                                                                       error_df.Reconstruction_error)
        plt.plot(threshold_rt, precision_rt[1 :], label="Precision", linewidth=5)
        plt.plot(threshold_rt, recall_rt[1 :], label="Recall", linewidth=5)
        plt.title('Precision and recall for different threshold values')
        plt.xlabel('Threshold')
        plt.ylabel('Precision/Recall')
        plt.legend()
        plt.savefig(outcome+"ThresholdSMOTE.pdf", bbox_inches='tight')

        test_x_predictions = lstm_autoencoder.predict(X_test_scaled)
        mse = np.mean(np.power(flatten(X_test_scaled) - flatten(test_x_predictions), 2), axis=1)

        error_df = pd.DataFrame({'Reconstruction_error' : mse,
                                 'True_class' : y_test.tolist()})

        plt.figure(figsize=(10, 10))

        threshold_fixed = 0.3
        groups = error_df.groupby('True_class')
        fig, ax = plt.subplots()

        for name, group in groups :
            ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
                    label="Break" if name == 1 else "Normal")
        ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
        ax.legend()
        plt.title("Reconstruction error for different classes")
        plt.ylabel("Reconstruction error")
        plt.xlabel("Data point index")
        plt.savefig(outcome+"ReconstructionerrorSMOTE.pdf", bbox_inches='tight')

        pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]
        conf_matrix = confusion_matrix(error_df.True_class, pred_y)

        plt.figure(figsize=(6, 6))
        sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
        plt.title("Confusion matrix")
        plt.ylabel('True class')
        plt.xlabel('Predicted class')
        plt.savefig(outcome+"ConfusionMatrixSMOTE.pdf", bbox_inches='tight')

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
        plt.savefig(outcome+"rocSMOTE.pdf", bbox_inches='tight')



        precision, recall, thresholds = precision_recall_curve(error_df.True_class, error_df.Reconstruction_error)
        pr_auc =  auc(recall, precision)

        plt.figure(figsize=(10, 10))

        plt.plot(recall, precision, linewidth=5, label='AUC = %0.3f' % pr_auc)
        plt.plot([0, 1], [0, 1], linewidth=5)


        plt.xlim([-0.01, 1])
        plt.ylim([0, 1.01])
        plt.legend(loc='lower right')
        plt.title('Receiver operating characteristic curve (ROC)')
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.savefig(outcome+"precision_recall_aucSMOTE.pdf", bbox_inches='tight')
if __name__ == '__main__':
    main()
