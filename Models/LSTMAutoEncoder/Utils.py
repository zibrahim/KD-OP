import numpy as np
import random
import pandas as pd
from collections import Counter, defaultdict

def process_data(dynamic_series, full_series, outcome, grouping, lookback,
                 training_ind, testing_ind):

    dynamic_series.insert(len(dynamic_series.columns), outcome, full_series[outcome])
    dynamic_series[outcome] = dynamic_series[outcome].astype(int)

    X_cols = (dynamic_series.columns).tolist()

    input_X = dynamic_series.loc[:,
              dynamic_series.columns.isin(X_cols)] # converts the df to a numpy array
    input_y = dynamic_series[outcome].values

    n_features = input_X[X_cols].shape[1] -2 # number of features

    X_train = input_X.iloc[training_ind[0]]
    y_train = input_y[training_ind[0]]
    aggregated_y_train = pd.DataFrame(X_train[grouping])
    aggregated_y_train[outcome] = y_train
    aggregated_y_train =aggregated_y_train.groupby(grouping).first()
    y_train1 = aggregated_y_train[outcome].to_numpy()

    X_valid = input_X.iloc[training_ind[1]]
    y_val = input_y[training_ind[1]]


    aggregated_y_val= pd.DataFrame(X_valid[grouping])
    aggregated_y_val[outcome] = y_val
    aggregated_y_val =aggregated_y_val.groupby(grouping).first()
    y_val1 = aggregated_y_val[outcome].to_numpy()

    X_test = input_X.iloc[testing_ind[1]]
    y_test = input_y[testing_ind[1]]

    aggregated_y_test= pd.DataFrame(X_test[grouping])
    aggregated_y_test[outcome] = y_test
    aggregated_y_test =aggregated_y_test.groupby(grouping).first()
    y_test1 = aggregated_y_test[outcome].to_numpy()

    X_train_y0 = pd.DataFrame(X_train[y_train == 0])

    X_train_y1 = X_train.iloc[y_train == 1]
    X_valid_y0 = X_valid.iloc[y_val == 0]
    X_valid_y1 = X_valid.iloc[y_val == 1]

    X_train_y0 = (curve_shift(X_train_y0, grouping, outcome, shift_by=lookback - 1)).to_numpy()
    X_train_y1 = (curve_shift(X_train_y1, grouping, outcome, shift_by=lookback - 1)).to_numpy()
    X_valid_y0 = (curve_shift(X_valid_y0, grouping, outcome, shift_by=lookback - 1)).to_numpy()
    X_valid_y1 = (curve_shift(X_valid_y1, grouping, outcome, shift_by=lookback - 1)).to_numpy()

    X_train = X_train.drop(grouping, axis=1)
    X_train  =X_train.drop(outcome, axis=1)
    X_train = X_train.to_numpy()
    X_train = X_train.reshape(-1, lookback, n_features)
    X_train_y0 = X_train_y0.reshape(-1, lookback, n_features)
    X_train_y1 = X_train_y1.reshape(-1, lookback, n_features)

    X_valid = X_valid.drop(grouping, axis=1)
    X_valid  =X_valid.drop(outcome, axis=1)
    X_valid = X_valid.to_numpy()
    X_valid = X_valid.reshape(-1, lookback, n_features)
    X_valid_y0 = X_valid_y0.reshape(-1, lookback, n_features)
    X_valid_y1 = X_valid_y1.reshape(-1, lookback, n_features)
    X_test = X_test.drop(outcome, axis=1)
    X_test = X_test.drop(grouping, axis=1)
    X_test = X_test.to_numpy()
    X_test = X_test.reshape(-1, lookback, n_features)


    timesteps = X_train_y0.shape[1]  # equal to the lookback
    n_features = X_train_y0.shape[2]  # 59


    return X_train, X_train_y0, X_valid_y0, X_valid, y_val1, X_test, y_test1, timesteps, n_features

def stratified_group_k_fold ( X, y, groups, k, seed=None) :
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda : np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups) :
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda : np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold ( y_counts, fold ) :
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num) :
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)

    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x : -np.std(x[1])) :
        best_fold = None
        min_eval = None
        for i in range(k) :
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval :
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k) :
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices


def temporalize(X, y, lookback):
    '''
    Inputs
    X         A 2D numpy array ordered by time of shape:
              (n_observations x n_features)
    y         A 1D numpy array with indexes aligned with
              X, i.e. y[i] should correspond to X[i].
              Shape: n_observations.
    lookback  The window size to look back in the past
              records. Shape: a scalar.
    Output
    output_X  A 3D numpy array of shape:
              ((n_observations-lookback-1) x lookback x
              n_features)
    output_y  A 1D array of shape:
              (n_observations-lookback-1), aligned with X.
    '''
    output_X = []
    output_y = []
    for i in range(len(X) - lookback - 1):
        t = []
        for j in range(1, lookback + 1):
            # Gather the past records upto the lookback period
            t.append(X[[(i + j + 1)], :])
        output_X.append(t)
        output_y.append(y[i + lookback + 1])
    return np.squeeze(np.array(output_X)), np.array(output_y)

def scale ( X, scaler ) :
    '''
    Scale 3D array.
    Inputs
    X            A 3D array for lstm, where the array is sample x timesteps x features.
    scaler       A scaler object, e.g., sklearn.preprocessing.StandardScaler, sklearn.preprocessing.normalize
    Output
    X            Scaled 3D array.
    '''
    for i in range(X.shape[0]) :
        X[i, :, :] = scaler.transform(X[i, :, :])

    return X


sign = lambda x : (1, -1)[x < 0]


def curve_shift ( df, grouping, outcome, shift_by ) :
    for patient_id in df[grouping]:
        if ((df.loc[df[grouping]==patient_id, outcome]).tolist())[0] ==1:
            patientFrame = df.loc[df[grouping]==patient_id, outcome]
            patientFrame.iloc[0:shift_by] = 0
            df.loc[df[grouping]==patient_id, outcome] = [x for x in patientFrame.values]
    df = df.drop(grouping, axis=1)
    df = df.drop(outcome, axis=1)
    return df

def lstm_flatten ( X ) :
    '''
    Flatten a 3D array.
    Input
    X            A 3D array for lstm, where the array is sample x timesteps x features.
    Output
    flattened_X  A 2D array, sample x features.
    '''
    flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
    for i in range(X.shape[0]) :
        flattened_X[i] = X[i, (X.shape[1] - 1), :]
    return (flattened_X)