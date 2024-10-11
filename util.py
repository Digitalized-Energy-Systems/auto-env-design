
import os

import pandas as pd


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
