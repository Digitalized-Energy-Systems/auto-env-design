
import os

import pandas as pd

import pareto_front


def load_optuna_csv(path):
    """ Load the data from the csv file. """
    return pd.read_csv(os.path.join(path, 'trials.csv'))


def load_drl_csv(path):
    """ Load the data from the csv file. """
    try:
        return pd.read_csv(os.path.join(path, 'test_returns.csv'))
    except FileNotFoundError:
        # Enumerate over all subdirectories and sum up the test_returns.csv files
        test_returns = []
        for sub_path in os.listdir(path):
            try:
                test_returns.append(pd.read_csv(os.path.join(path, sub_path, 'test_returns.csv')))
            except FileNotFoundError:
                pass

        # Concatenate all dataframes
        return pd.concat(test_returns)


def load_optuna_data(optuna_path, metrics, allowed_error=None):
    print('Load data from:', optuna_path)
    optuna_df = load_optuna_csv(optuna_path)
    print('Number of samples:', len(optuna_df))

    default_metrics = ['values_' + str(m) for m in range(len(metrics))]
    try:
        optuna_df[metrics[0]] = optuna_df[default_metrics[0]]
        optuna_df[metrics[1]] = optuna_df[default_metrics[1]]
    except KeyError:
        pass

    # Remove NaN values and reset the index
    optuna_df = optuna_df[optuna_df.state == 'COMPLETE']
    optuna_df = optuna_df.dropna(subset=list(metrics))
    optuna_df = optuna_df.reset_index(drop=True)

    optuna_df = add_normalized_metrics_columns(optuna_df, metrics)

    optuna_df['non_dominated'] = pareto_front.compute_pareto_front(optuna_df[metrics])
    if allowed_error:
        optuna_df['fuzzy_dominated'] = pareto_front.compute_fuzzy_dominance(optuna_df[metrics], optuna_df['non_dominated'], allowed_error)

    # Normalize the data sampling probabilities
    sum_of_shares = optuna_df.params_normal_share + optuna_df.params_simbench_share + optuna_df.params_uniform_share
    optuna_df['params_normal_share'] = optuna_df.params_normal_share / sum_of_shares
    optuna_df['params_simbench_share'] = optuna_df.params_simbench_share / sum_of_shares
    optuna_df['params_uniform_share'] = optuna_df.params_uniform_share / sum_of_shares

    return optuna_df


def add_normalized_metrics_columns(df, metrics):
    for metric in metrics:
        df[metric + '_normalized'] = (df[metric] - df[metric].min()) / (
                df[metric].max() - df[metric].min())
    return df