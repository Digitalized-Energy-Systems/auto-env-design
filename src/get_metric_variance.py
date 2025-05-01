
import os

import numpy as np
import pandas as pd

from util import load_drl_csv


def get_metric_std_dev(path: str):
    all_errors = []
    all_invalids = []

    for sample_path in os.listdir(path):
        sample_errors = []
        sample_invalid_shares = []
        sample_path = os.path.join(path, sample_path)
        if not os.path.isdir(sample_path):
            continue

        # print(f'Processing {sample_path}')
        for run_path in os.listdir(sample_path):
            full_path = os.path.join(sample_path, run_path)

            if not os.path.isdir(full_path):
                continue

            try:
                df = load_drl_csv(full_path)
            except (FileNotFoundError, NotADirectoryError):
                continue

            sample_errors.append(df['valid_regret'].mean())
            sample_invalid_shares.append(1 - df['valid_share_possible'].mean())

        # Compute standard dev within the n runs with same hyperparams
        all_errors.append(np.std(sample_errors))
        all_invalids.append(np.std(sample_invalid_shares))

    mean_std_invalids = np.nanmean(all_invalids)
    mean_std_error = np.nanmean(all_errors)

    # print(f'Mean std error: {mean_std_error}')
    # print(f'Mean std invalid share: {mean_std_invalids}')
    # print('standard deviations error:', (mean_std_invalids, mean_std_error))

    # # Compute min and max per metric for all experiments
    # min_error = np.min([np.mean(l) for l in all_errors if not np.isnan(np.mean(l))])
    # max_error = np.max([np.mean(l) for l in all_errors if not np.isnan(np.mean(l))])
    # min_invalids = np.min([np.mean(l) for l in all_invalids if not np.isnan(np.mean(l))])
    # max_invalids = np.max([np.mean(l) for l in all_invalids if not np.isnan(np.mean(l))])

    # # Compute relative std per metric for all experiments
    # rel_std_error = mean_std_error / (max_error - min_error)
    # rel_std_invalids = mean_std_invalids / (max_invalids - min_invalids)

    # print(f'Relative std error: {rel_std_error}')
    # print(f'Relative std invalid share: {rel_std_invalids}')

    # Compute std per metric only for non-dominated runs
    # TODO

    return mean_std_invalids, mean_std_error


if __name__ == '__main__':
    get_metric_std_dev('HPC/auto_env_design/data/20240906_renewable_multi/')
