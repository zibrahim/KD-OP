from collections import Counter, defaultdict
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from Utils.Dictionary import aggregation

SEED = 123 #used to help randomly select the data points

def get_train_test_split(outcome_col, grouping_col):
    fold_ind, training_ind, testing_ind= stratified_group_k_fold(outcome_col,
                                                                 grouping_col, 3, seed=SEED)

    return fold_ind, training_ind, testing_ind


def stratified_group_k_fold (y, groups, k, seed=None) :
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
    y_vals_sum = sum(y_distr.values())

    return [y_distr[i]/ y_vals_sum for i in range(np.max(y_vals) + 1)]

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


def class_counts(y):
    neg = np.count_nonzero(y == 0)
    pos = np.count_nonzero(y == 1)
    class_weight = {0 : neg, 1 : pos}
    return class_weight

def generate_aggregates(X, dynamic_columns, id_col, training_groups):
    agg_df = pd.DataFrame()
    agg_df.insert(0, id_col, training_groups)
    abstract_cols = [x.partition('_')[0] for x in dynamic_columns if int(x.partition('_')[2]) ==0]

    for col in abstract_cols:
        col_aggregate = aggregation[col]
        batch_columns = [x for x in X.columns.tolist() if x.partition('_')[0] == col]

        if col_aggregate =='min':
            new_col = X[batch_columns].min(axis=1)
            label="_min"
            agg_df[col + label] = new_col
        elif col_aggregate =='max':
            new_col = X[batch_columns].max(axis=1)
            label="_max"
            agg_df[col + label] = new_col
        elif col_aggregate=='min/max':
            new_col = X[batch_columns].max(axis=1)
            label="_max"
            agg_df[col + label] = new_col
            new_col = X[batch_columns].min(axis=1)
            label="_min"
            agg_df[col + label] = new_col
        else:
            new_col = X[batch_columns].mean(axis=1)
            label="_mean"
            agg_df[col + label] = new_col

    agg_df.to_csv("aggreagte.csv", index=False)
    return agg_df

def apply_func(df, col):
    return df.apply(aggregation[col], axis=1)

def impute(df, impute_columns):

    imp = IterativeImputer(max_iter=10, random_state=0)
    imp.fit(df[impute_columns])
    df[impute_columns] = imp.transform(df[impute_columns])

    return df[impute_columns]

def scale(df, scale_columns):

    scaler = MinMaxScaler()
    normalized_df = pd.DataFrame(scaler.fit_transform(df[scale_columns]))
    normalized_df.columns = scale_columns

    print(" in scaling, columns are:", scale_columns, len(scale_columns))
    return normalized_df

def smote(X, y):
    over = SMOTE(sampling_strategy=0.9)
    under = RandomUnderSampler(sampling_strategy=0.9)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    X, y = pipeline.fit_resample(X, y)
    return X, y