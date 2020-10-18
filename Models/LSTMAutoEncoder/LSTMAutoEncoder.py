import os
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.metrics import precision_recall_curve
from Models.Metrics import performance_metrics
from sklearn.metrics import auc, roc_curve

import json

from keras.layers import Dense, Activation, Dropout, LSTM, RepeatVector, TimeDistributed, Masking

import tensorflow as tf
from keras import optimizers, Sequential
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint



class LSTMAutoEncoder():
    def __init__(self, name, outcome, timesteps, n_features,saved_model = None):
        if saved_model == None:
            self.lstm_autoencoder = Sequential(name = name)
        # Encoder
            self.lstm_autoencoder.add(LSTM(32, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
            #self.lstm_autoencoder.add(LSTM(16, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
            self.lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=False))
            self.lstm_autoencoder.add(Dropout(0.5))

            self.lstm_autoencoder.add(RepeatVector(timesteps))
        # Decoder
            #self.lstm_autoencoder.add(LSTM(8, activation='relu', return_sequences=True))
            self.lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=True))
            self.lstm_autoencoder.add(Dropout(0.5))
            self.lstm_autoencoder.add(LSTM(32, activation='relu', return_sequences=True))

            self.lstm_autoencoder.add(TimeDistributed(Dense(n_features)))
            lr = 0.001
            adam = optimizers.Adam(lr)
            (self.lstm_autoencoder).compile(loss='mse', optimizer=adam, metrics=["mean_squared_error"])

            self.history = None

        else:
            self.lstm_autoencoder = load_model(saved_model)
            ##ZI: Fix how do I initialise history if model loaded from disk?

        self.outcome = outcome

        configs = json.load(open('Configuration.json', 'r'))
        self.output_path = configs['paths']['autoencoder_output_path']

    def summary( self ):
        self.lstm_autoencoder.summary()

    def save_model( self , filename):
        self.lstm_autoencoder.save(filename)
        print('>Saved %s' % filename)

    def fit(self, trainx, trainy,e, b,val_x, val_y,v):
        configs = json.load(open('Configuration.json', 'r'))
        autoencoder_models_path = configs['paths']['autoencoder_models_path']
        es = EarlyStopping(monitor='val_loss', mode='min',
                           verbose=1, patience=50, restore_best_weights=True)

        filename = autoencoder_models_path + configs['model']['name'] + self.outcome + '.h5'

        mc = ModelCheckpoint(filename, monitor='val_loss', save_best_only=True, verbose=1)

        history = self.lstm_autoencoder.fit(trainx, trainy,epochs = e, batch_size = b,
                          validation_data = (val_x,val_y), verbose = v,
                                            callbacks=[es,mc]).history
        self.history = history

        best_acc = max(self.history["val_loss"])
        _, eval_acc =(self.lstm_autoencoder).evaluate(val_x, val_y, verbose=0)
        print(f"Best accuracy in training: {best_acc}. In evaluation: {eval_acc}\n")

    def plot_history( self ):
        plt.figure(figsize=(10, 10))
        plt.plot(self.history['loss'], linewidth=2, label='Train')
        plt.plot(self.history['val_loss'], linewidth=2, label='Valid')
        plt.legend(loc='upper right')
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.savefig(self.output_path +self.outcome+self.outcome+"LossOverEpochs.pdf", bbox_inches='tight')

    def predict( self , xval):
        predictions = self.lstm_autoencoder.predict(xval)
        return predictions

    def predict_binary( self, true_class, reconstruction_error):
        precision_rt, recall_rt, threshold_rt = precision_recall_curve(true_class,
                                                                       reconstruction_error)
        fscore = (2 * precision_rt * recall_rt) / (precision_rt + recall_rt)
        ix = np.argmax(fscore)
        best_threshold = threshold_rt[ix]
        # print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], fscore[ix]))
        pred_y = (reconstruction_error > best_threshold).astype('int32')
        return pred_y, best_threshold, precision_rt, recall_rt


    def output_performance(self, true_class,pred_y):

        perf_df = pd.DataFrame()
        perf_dict = performance_metrics(true_class, pred_y )
        perf_df = perf_df.append(perf_dict, ignore_index=True)
        perf_df.to_csv(self.output_path +"performancemetrics"+self.outcome+".csv", index=False)

    def plot_reconstruction_error(self, error_df, best_threshold):
        plt.figure(figsize=(10, 10))

        groups = error_df.groupby('True_class')
        fig, ax = plt.subplots()

        for name, group in groups :
            ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
                label="1" if name == 1 else "0")
        ax.hlines(best_threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
        ax.legend()
        plt.title("Reconstruction error for different classes")
        plt.ylabel("Reconstruction error")
        plt.xlabel("Data point index")
        plt.savefig(self.output_path  + self.outcome + "Reconstructionerror.pdf", bbox_inches='tight')


    def plot_roc(self, error_df):
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
        plt.savefig(self.output_path + self.outcome + "roc.pdf", bbox_inches='tight')

    def plot_pr( self, precision, recall ):

        pr_auc =  auc(recall, precision)
        plt.figure(figsize=(10, 10))
        plt.plot(recall, precision, linewidth=5, label='PR-AUC = %0.3f' % pr_auc)
        plt.plot([0, 1], [1, 0], linewidth=5)

        plt.xlim([-0.01, 1])
        plt.ylim([0, 1.01])
        plt.legend(loc='lower right')
        plt.title('Precision Recall Curive')
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.savefig(self.output_path+self.outcome+"precision_recall_auc.pdf", bbox_inches='tight')