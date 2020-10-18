from collections import Counter, defaultdict
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import f1_score

import pandas as pd
#from kmeans_smote import KMeansSMOTE
import numpy as np

import keras.backend as K

lstm_dict={}
lstm_dict['activation'] = ['sigmoid']
lstm_dict['batch'] = ['all']
lstm_dict['data'] = ['balanced', 'unchanged']
lstm_dict['dropout'] = [0.0, 0.2, 0.4, 0.6]
lstm_dict['layers'] = [1, 2]
lstm_dict['masking'] = [True]
lstm_dict['optimizer'] = ['RMSprop', 'Adagrad']
lstm_dict['outcome'] = ['ITUAdmission7Days', 'ITUAdmission14Days', 'ITUAdmission30Days',
                         'Mortality7Days','Mortality14Days','Mortality30Days']
lstm_dict['units'] = [1, 4, 8, 16, 32, 64, 128]

lstm_running_params = {}
lstm_running_params['monitor_checkpoint'] = ['val_matthews', 'max']
lstm_running_params['monitor_early_stopping'] = ['val_matthews', 'max']
lstm_running_params['early_stopping'] = True
lstm_running_params['save_checkpoint'] = True
lstm_running_params['seed'] = 42
lstm_running_params['n_splits'] = 10
lstm_running_params['n_epochs'] = 1000
lstm_running_params['patience'] = 200

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

def get_distribution ( y_vals ) :
    y_distr = Counter(y_vals)
    y_vals_sum = sum(y_distr.values())
    return [f'{y_distr[i] / y_vals_sum:.2%}' for i in range(np.max(y_vals) + 1)]


def get_distribution_scalars( y_vals ) :
    y_distr = Counter(y_vals)
    return [int(y_distr[i]) for i in range(np.max(y_vals) + 1)]

def get_distribution_percentages ( y_vals ) :
    y_distr = Counter(y_vals)
    y_vals_sum = sum(y_distr.values())
    return [(y_distr[i] / y_vals_sum) for i in range(np.max(y_vals) + 1)]

def generate_balanced_arrays(df, x_features, outcome, grouping, no_groups):
 df = df[:,not (df[grouping].isin(no_groups))]
 y_test = (df[outcome]).to_numpy()
 X_test = df[x_features].to_numpy()

 while True:
  positive = np.where(y_test==1)[0].tolist()
  negative = np.random.choice(np.where(y_test==0)[0].tolist(),size = len(positive), replace = False)
  balance = np.concatenate((positive, negative), axis=0)
  np.random.shuffle(balance)
  input = X_test.iloc[balance, :]
  target = y_test.iloc[balance]
  yield input, target

def class_weights(y):
    total = len(y)
    neg = np.count_nonzero(y == 0)
    pos = np.count_nonzero(y == 1)
    weight_for_0 = (1 / neg) * (total) / 2.0
    weight_for_1 = (1 / pos) * (total) / 2.0

    class_weight = {0 : weight_for_0, 1 : weight_for_1}

    return class_weight


def generate_trajectory_timeseries(df, baseline_columns, static_columns, timeseries_columns, id_col, outcome_columns):
    for i, j in zip(timeseries_columns, baseline_columns):
        df[i] = df[i] - df[j]

    new_df = df[timeseries_columns]
    new_df.insert(0, id_col, df[id_col])
    new_df[outcome_columns] = df[outcome_columns]
    new_df[static_columns] = df[static_columns]

    return new_df

def impute(df, impute_columns):

    imp = IterativeImputer(max_iter=10, random_state=0)
    imp.fit(df[impute_columns])
    df[impute_columns] = imp.transform(df[impute_columns])

    return df[impute_columns]

def scale(df, scale_columns):

    scaler = MinMaxScaler()
    normalized_df = pd.DataFrame(scaler.fit_transform(df[scale_columns]))
    normalized_df.columns = scale_columns

    return normalized_df


def flatten(time_series, dynamic_features, grouping, static_features, outcome_column):
    #First, create the structure of a flat DF
    newdf = time_series.copy()
    newdf.insert(0, grouping+'2', newdf[grouping])
    aggregated_df = newdf.groupby(grouping+'2').aggregate('first')

    timesteps = len(newdf)/len(aggregated_df)
    flat_df = pd.DataFrame()
    for id in aggregated_df[grouping].tolist():
        patient_dict = {}
        patient_chunk = time_series.loc[time_series[grouping] == id, dynamic_features]
        patient_dict.update({grouping: id})

        for timestep in range(0, int(timesteps)) :
            row_dictionary = {}
            for x in patient_chunk.columns :
                datum = patient_chunk.iloc[timestep]
                datum = datum.loc[x]
                if x != grouping:
                    row_dictionary[x+'_'+str(timestep) ] = datum

            patient_dict.update(row_dictionary)

        static_feature_values = time_series.loc[time_series[grouping] == id, static_features]
        patient_dict.update(zip(static_features, static_feature_values.iloc[0]))

        outcome_value = time_series.loc[time_series[grouping] == id, outcome_column]

        patient_dict.update({outcome_column: outcome_value.iloc[0]})
        flat_df = flat_df.append(patient_dict, ignore_index=True)

    flat_df.to_csv("flat.csv", index = False)
    return flat_df, timesteps

def smote(target_df, target_outcome, grouping):
    y = target_df[target_outcome]
    target_df.drop(target_outcome, axis=1, inplace=True)

    target_df[grouping] = [float((x.partition('_')[2])) for x in target_df[grouping]]
    #target_df.drop(grouping, axis=1, inplace=True)
    X = target_df

    target_columns = target_df.columns
    #target_columns = target_columns.insert(0, grouping)
    target_columns= target_columns.insert(len(target_columns), target_outcome)
    kmeans_smote = KMeansSMOTE(
        kmeans_args={
            'n_clusters' : 5
        },
        smote_args={
            'k_neighbors' : 10
        }
    )
    X_resampled, y_resampled = kmeans_smote.fit_sample(X, y)
    X_resampled = pd.DataFrame(X_resampled)
    y_resampled = pd.DataFrame(y_resampled)
    frames = [X_resampled, y_resampled]

    total_df = pd.concat(frames, axis=1)
    total_df.columns  = target_columns
    total_df[grouping] = ['p_'+str(x) for x in total_df[grouping]]

    return total_df

def unflatten(flat_df, grouping, static_features, outcome_column, timesteps):
        smoted_timeseries = pd.DataFrame()
        for patient in flat_df[grouping]:
            patient_row = flat_df.loc[flat_df[grouping]==patient]
            patient_row.columns = flat_df.columns
            for step in range(0, timesteps):
                matching_columns = [x for x in patient_row.columns if x.endswith('_'+str(step))]
                dynamic_slot = pd.DataFrame(patient_row[matching_columns])
                dynamic_slot.columns = [x.partition('_')[0] for x in list(dynamic_slot.columns) ]
                dynamic_slot[grouping] = patient
                static_slot = pd.DataFrame(patient_row[static_features])
                static_slot.columns = static_features
                static_slot[grouping] = patient
                static_slot[outcome_column] = patient_row[outcome_column]

                full_slot  = static_slot.merge(dynamic_slot, on=grouping)
                full_slot[grouping] = patient

                slot_dictionary = dict(zip((full_slot.columns).tolist(), (full_slot.values).tolist()[0]))
                smoted_timeseries = smoted_timeseries.append(slot_dictionary, ignore_index=True)

        return smoted_timeseries