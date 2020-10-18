import os
import json

from ProcessResults.VisualisePopulation import Visualiser
import pandas as pd
from pylab import rcParams


from numpy.random import seed

import os.path

seed(7)

rcParams['figure.figsize'] = 8, 6
LABELS = ["0", "1"]

from Utils.Data import scale, impute


def main () :
    configs = json.load(open('Configuration.json', 'r'))
    grouping = configs['data']['grouping']
    dynamic_features = configs['data']['dynamic_columns']
    static_features = configs['data']['static_columns']

    targets = configs['data']['classification_target']
    timeseries_path = configs['paths']['data_path']

    ##read, impute and scale dataset
    non_smotedtime_series = pd.read_csv(timeseries_path + "TimeSeriesAggregatedUpto0.csv")
    non_smotedtime_series[dynamic_features] = impute(non_smotedtime_series, dynamic_features)
    normalized_timeseries = scale(non_smotedtime_series, dynamic_features)
    normalized_timeseries.insert(0, grouping, non_smotedtime_series[grouping])


    risk_score_visualiser = Visualiser(normalized_timeseries, non_smotedtime_series,
                                       dynamic_features, static_features)
    for target in targets:
            risk_score_visualiser.plot_risk_scores(target)


if __name__ == '__main__' :
    main()
