import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

from Models.Utils import generate_slopes, generate_aggregates
from Utils.Data import flatten, scale


from Models.LSTMAutoEncoder.LSTMAutoEncoder import LSTMAutoEncoder
from Models.LSTMAutoEncoder.Utils import curve_shift, lstm_flatten


class Visualiser():

    def __init__(self, df_dynamic, df_full, dynamic_features, static_features):
        configs = json.load(open('Configuration.json', 'r'))
        lookback = configs['data']['batch_size']
        grouping = configs['data']['grouping']
        autoencoder_models_path = configs['paths']['autoencoder_models_path']
        outcomes = configs['data']['classification_outcome']

        self.risk_scores = pd.DataFrame()
        self.no_scores = 0
        for outcome in outcomes:
            X, y, timesteps, nfeatures = self.reshape_data_encoder(df_dynamic, df_full, outcome, grouping, lookback)
            X_xgb, y_xgb, = self.reshape_data_xgboost(df_full, dynamic_features,static_features, outcome, grouping)

            filename = autoencoder_models_path+ configs['model']['name'] + outcome+ '.h5'

            autoencoder = LSTMAutoEncoder(configs['model']['name'] + outcome, outcome,
                                          timesteps, nfeatures ,saved_model = filename)
            X_predictions = autoencoder.predict(X)
            mse= np.mean(np.power(lstm_flatten(X) - lstm_flatten(X_predictions), 2), axis=1)
            self.risk_scores[outcome+"_true"] =  y
            self.risk_scores[outcome+"_risk"] = pd.Series(mse) * 10
            this_col = self.risk_scores[outcome+"_risk"]
            max_val = this_col.max()
            self.risk_scores[outcome+"_risk"] = (self.risk_scores[outcome+"_risk"])/max_val

        self.risk_scores.to_csv("risk_scores.csv", index=False)
    def plot_risk_scores( self, target ):

        colnames = ((self.risk_scores).columns).tolist()
        this_outcome_columns = [x for x in colnames if
                                target in x and x.partition('_')[2] =="risk"]
        this_true_outcome_columns =[x for x in colnames if
                                target in x and x.partition('_')[2] =="true"]

        print(target+" has columns ")
        print(*this_outcome_columns)
        outcome_scores = self.risk_scores[this_outcome_columns]
        outcome_true = self.risk_scores[this_true_outcome_columns]

        plt.figure(figsize=(10, 5))

        for index, row in outcome_scores.iterrows() :
            true_row = outcome_true.iloc[index]
            max_true = true_row.max()
            if max_true == 1:
                plt.plot(row[1:], linewidth=2,linestyle='dashdot', color=(1, 0.77, 0.78))
            else:
                plt.plot(row[1:], linewidth=2,linestyle='dashdot', color=(0.66, 0.75, 0.88))

        #plt.plot(self.history['val_loss'], linewidth=2, label='Valid')
        #plt.legend(loc='upper right')
        #plt.title('Model loss')
        #plt.ylabel('Loss')
        #plt.xlabel('Epoch')
        plt.savefig(target+".pdf", bbox_inches='tight')

    def reshape_data_xgboost(self, full_series, dynamic_features, static_features, outcome, grouping):

        flat_df ,timesteps= flatten (full_series, dynamic_features, grouping, static_features, outcome)
        temporal_features = set(flat_df.columns) - set(static_features)
        temporal_features = set(temporal_features) - set([outcome,grouping])

        X_working = flat_df
        y_working = X_working[outcome].astype(int)
        training_groups  = X_working[grouping]
        X_working_static = X_working[static_features]
        X_working_static.loc[grouping] = training_groups
        X_working = X_working[temporal_features]
        X_working = scale(X_working, temporal_features)

        slopes_df = generate_slopes ( X_working, temporal_features, static_features,grouping, training_groups)
        aggregate_df = generate_aggregates ( X_working, temporal_features, grouping, training_groups )

        slopes_static_aggregate_df = pd.concat([slopes_df, X_working_static], axis=1,join='inner')
        slopes_static_aggregate_df = pd.concat([slopes_static_aggregate_df, aggregate_df], axis=1,join='inner')

        slopes_static_aggregate_df = slopes_static_aggregate_df.loc[:, ~slopes_static_aggregate_df.columns.duplicated()]
        slopes_static_aggregate_groups  = slopes_static_aggregate_df[grouping]
        slopes_static_aggregate_df.drop(columns = [grouping], inplace=True, axis=1)

        return slopes_static_aggregate_df, y_working

    def reshape_data_encoder(self, dynamic_series, full_series, outcome, grouping, lookback):

        dynamic_series.insert(len(dynamic_series.columns), outcome, full_series[outcome])
        dynamic_series[outcome] = dynamic_series[outcome].astype(int)

        X_cols = (dynamic_series.columns).tolist()

        input_X = dynamic_series.loc[:,
                  dynamic_series.columns.isin(X_cols)]  # converts the df to a numpy array
        input_y = dynamic_series[outcome].values
        n_features = input_X[X_cols].shape[1] - 2  # number of features

        X_working = input_X.copy()
        y_working = input_y.copy()

        aggregated_y_working = pd.DataFrame(X_working[grouping])
        aggregated_y_working[outcome] = y_working
        aggregated_y_working = aggregated_y_working.groupby(grouping).first()
        aggregated_y_working = aggregated_y_working[outcome].to_numpy()


        X_working = curve_shift(X_working, grouping, outcome, shift_by=lookback - 1)

        X_working = X_working.to_numpy()
        X_working = X_working.reshape(-1, lookback, n_features)

        timesteps = X_working.shape[1]  # equal to the lookback
        n_features = X_working.shape[2]  # 59
        X_autoencoder = X_working

        ##now on to xgboost input
        return X_autoencoder, aggregated_y_working, timesteps, n_features


