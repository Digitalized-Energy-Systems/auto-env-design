""" Manual plot of the pareto front of an optuna experiment plus some other 
experiments for comparison. """

import os

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from get_best_design import add_not_dominated_column


def main(optuna_path: str, other_paths: tuple, show_all=True, store=True):
    # Load the data
    optuna_df = load_optuna_data(optuna_path)

    other_dfs = []
    for other_path in other_paths:
        other_dfs.append(load_drl_data(other_path))

    # Plot the data
    metrics = ('invalid share', 'error')
    default_metrics = ['values_' + str(m) for m in range(len(metrics))]

    optuna_df[metrics[0]] = optuna_df[default_metrics[0]]
    optuna_df[metrics[1]] = optuna_df[default_metrics[1]]

    # Remove NaN values
    optuna_df = optuna_df.dropna(subset=list(metrics))

    allowed_error = 0.1
    optuna_df = add_not_dominated_column(optuna_df, list(metrics), epsilon_relative=allowed_error)

    fig = plt.figure()
    # Plot Pareto front solutions
    for idx, row in enumerate(optuna_df[optuna_df.not_dominated].iterrows()):
        label = 'Non-dominated' if idx==0 else None
        plt.plot(row[1][metrics[0]], row[1][metrics[1]], 'ro', label=label)

    # Plot dominated solutions
    if show_all:
        for idx, row in enumerate(optuna_df[~optuna_df.not_dominated].iterrows()):
            label = 'Dominated' if idx==0 else None
            plt.plot(row[1][metrics[0]], row[1][metrics[1]], 'bo', label=label)

    # Plot other experiments as comparison
    for idx, other_df in enumerate(other_dfs):
        label = 'Manual' if idx==0 else None
        invalid_share = 1 - other_df['valid_share_possible'].mean()
        valid_regret = other_df['valid_regret'].mean()
        plt.plot(invalid_share, valid_regret, 'go', label=label)

    plt.xlabel(metrics[0])
    plt.ylabel(metrics[1])
    plt.legend()

    if store:
        plt.savefig('pareto_front_comparison.pdf', format='pdf')
    else:
        plt.show()
    plt.close(fig)


def load_optuna_data(path):
    """ Load the data from the csv file. """
    return pd.read_csv(os.path.join(path, 'trials.csv'))


def load_drl_data(path):
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


if __name__ == '__main__':
    main('data/test/', ('data/test/test/',))
