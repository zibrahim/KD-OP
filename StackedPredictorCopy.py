import os
import json
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from Models.XGBoost.XGBoost import XGBoostClassifier
from ProcessResults.VisualisePopulation import DecisionMaker
from Models.Utils import generate_slopes, get_distribution_percentages


def main():
    configs = json.load(open('Configuration.json', 'r'))

    grouping = configs['data']['grouping']
    static_features = configs['data']['static_columns']

    outcomes = (configs['data']['classification_outcome'])
    timeseries_path = configs['paths']['data_path']

    for outcome in outcomes:
        decision_maker = DecisionMaker()

        time_series_nosmote = pd.read_csv(timeseries_path + "NonSMOTEDTimeSeries/"+outcome+"FlatTimeSeries1Day.csv")
        time_series = pd.read_csv(timeseries_path + "SMOTEDTimeSeries/" + outcome + "FlatTimeSeries1Day.csv")

        X_cols = (time_series.columns).tolist()
        X_cols.remove(outcome)
        X_train, X_valid, y_train, y_valid = train_test_split(time_series[X_cols], time_series[outcome],
                                                            test_size = 0.33,
                                                            stratify = time_series[outcome],
                                                            random_state = 42)

        test_ids = set([x.partition('.')[0] for x in X_valid[grouping]])

        #print(test_ids.intersection(time_series_nosmote[grouping] ))
        X_test1 = time_series_nosmote.loc[time_series_nosmote[grouping].isin(test_ids)]
        distrs_percents = [get_distribution_percentages((time_series_nosmote[outcome]).astype(int))]
        print(distrs_percents)

        #####feature selector
        temporal_features = set(X_train.columns) - set(static_features)

        feature_selector = XGBoostClassifier(X_train[temporal_features], y_train,outcome, grouping)
        fs_y, fs_ths, fs_id, fs_fi = feature_selector.run_xgb("temporal")

        feature_selector.predict( X_valid[temporal_features], y_valid)

        decision_maker.add_classifier(outcome+"Tmp", fs_y, fs_ths, fs_id, fs_fi)

        featuredf = pd.DataFrame()

        temporal_features.remove(grouping)
        featuredf['features'] = list(temporal_features)
        featuredf['imp'] = fs_fi
        featuredf = featuredf[featuredf['imp']> 0]

        ########################################
        #baseline and static
        baseline_features = featuredf['features']

        baseline_features= set([x.partition('_')[0] for x in list(baseline_features)])

        baseline_features = [x+"_0" for x in list(baseline_features)]

        baseline_features.insert(0,grouping)
        baseline_static_features = baseline_features + static_features


        slopes_df = generate_slopes ( X_train, static_features, grouping)
        slopes_static_baseline_df = pd.concat([slopes_df, X_train[baseline_static_features]], axis=1,join='inner')

        slopes_static_baseline_df = slopes_static_baseline_df.loc[:, ~slopes_static_baseline_df.columns.duplicated()]

        slopes_df_test = generate_slopes ( X_valid, static_features, grouping)
        slopes_static_baseline_test_df = pd.concat([slopes_df_test, X_valid[baseline_static_features]], axis=1,join='inner')
        slopes_static_baseline_test_df = slopes_static_baseline_test_df.loc[:, ~slopes_static_baseline_test_df.columns.duplicated()]


        slopes_static_baseline_classifier = XGBoostClassifier(slopes_static_baseline_df, y_train, outcome, grouping)

        bs_y, bs_ths, bs_id, bs_fi = slopes_static_baseline_classifier.run_xgb("baseline_static_slope")
        slopes_static_baseline_classifier.predict( slopes_static_baseline_test_df, y_valid)
        tf.keras.backend.clear_session()

        decision_maker.add_classifier(outcome+"bss", bs_y, bs_ths, bs_id, bs_fi)

        ####LSTM autoencoder
        smoted_stacked_series = pd.read_csv(timeseries_path+"SMOTEDTimeSeries/"+outcome+"StackedTimeSeries1Day.csv")

if __name__ == '__main__':
    main()
