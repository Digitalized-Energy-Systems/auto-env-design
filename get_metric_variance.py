
import collections
import json
import os

import numpy as np
import pandas as pd

from util import load_drl_csv


def get_metric_variance(path: str):
    all_errors = collections.defaultdict(list)
    all_invalids = collections.defaultdict(list)

    for sub_path in os.listdir(path):
        full_path = os.path.join(path, sub_path)

        if not os.path.isdir(full_path):
            continue

        print(f'Processing {full_path}')

        try:
            df = load_drl_csv(full_path)
        except (FileNotFoundError, NotADirectoryError):
            continue

        mean_error = df['valid_regret'].mean()
        mean_invalids = 1 - df['valid_share_possible'].mean()

        # Open the env_hyperparams.json as string
        with open(os.path.join(full_path, 'env_hyperparams.json')) as f:
            env_hyperparams = str(json.load(f))

        all_errors[env_hyperparams].append(mean_error)
        all_invalids[env_hyperparams].append(mean_invalids)

    all_errors = list(all_errors.values())
    all_invalids = list(all_invalids.values())

    # Compute standard deviation per metric for all experiments
    std_error = [np.std(l) for l in all_errors]
    mean_std_error = np.mean([std for std in std_error if not np.isnan(std)])
    std_invalids = [np.std(l) for l in all_invalids]
    mean_std_invalids = np.mean([std for std in std_invalids if not np.isnan(std)])

    print(f'Mean std error: {mean_std_error}')
    print(f'Mean std invalid share: {mean_std_invalids}')
    print('standard deviations error:', (mean_std_invalids, mean_std_error))

    # Compute min and max per metric for all experiments
    min_error = np.min([np.mean(l) for l in all_errors if not np.isnan(np.mean(l))])
    max_error = np.max([np.mean(l) for l in all_errors if not np.isnan(np.mean(l))])
    min_invalids = np.min([np.mean(l) for l in all_invalids if not np.isnan(np.mean(l))])
    max_invalids = np.max([np.mean(l) for l in all_invalids if not np.isnan(np.mean(l))])

    # Compute relative std per metric for all experiments
    rel_std_error = mean_std_error / (max_error - min_error)
    rel_std_invalids = mean_std_invalids / (max_invalids - min_invalids)

    print(f'Relative std error: {rel_std_error}')
    print(f'Relative std invalid share: {rel_std_invalids}')

    # Compute std per metric only for non-dominated runs
    # TODO


if __name__ == '__main__':
    get_metric_variance('HPC/auto_env_design/data/20240906_renewable_multi/')
