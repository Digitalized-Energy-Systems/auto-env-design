""" Manual plot of the pareto front of an optuna experiment plus some other 
experiments for comparison. """

import os

import matplotlib.pyplot as plt

import pareto_front
from util import load_optuna_csv, load_drl_csv


ALLOWED_ERROR = (0.0, 0.0004)


def main(optuna_path: str, other_paths: tuple, show_all=True, store=False):
    # Load the data
    optuna_df = load_optuna_csv(optuna_path)

    other_dfs = []
    for other_path in other_paths:
        other_dfs.append(load_drl_csv(other_path))

    # Plot the data
    metrics = ('invalid share', 'error')
    default_metrics = ['values_' + str(m) for m in range(len(metrics))]
    metrics = ['values_' + str(m) for m in range(len(metrics))]

    try:
        optuna_df[metrics[0]] = optuna_df[default_metrics[0]]
        optuna_df[metrics[1]] = optuna_df[default_metrics[1]]
    except KeyError:
        pass

    # Remove NaN values and reset the index
    optuna_df = optuna_df.dropna(subset=list(metrics))
    optuna_df = optuna_df.reset_index(drop=True)

    # print(optuna_df)

    # # Create columns of normalized metrics
    # normalized_metrics = [m + '_normalized' for m in metrics]
    # for metric, normalized_metric in zip(metrics, normalized_metrics):
    #     optuna_df[normalized_metric] = (optuna_df[metric] - optuna_df[metric].min()) / (
    #             optuna_df[metric].max() - optuna_df[metric].min())

    optuna_df['non_dominated'] = pareto_front.compute_pareto_front(optuna_df[metrics])
    optuna_df['fuzzy_dominated'] = pareto_front.compute_fuzzy_dominance(optuna_df[metrics], optuna_df['non_dominated'], ALLOWED_ERROR)

    fig = plt.figure()
    # Plot dominated solutions
    if show_all:
        for idx, row in enumerate(optuna_df[~optuna_df.non_dominated].iterrows()):
            label = 'Dominated' if idx==0 else None
            plt.plot(row[1][metrics[0]], row[1][metrics[1]], 'bo', label=label)

    # Plot Pareto front solutions
    for idx, row in enumerate(optuna_df[optuna_df.non_dominated].iterrows()):
        label = 'Non-dominated' if idx==0 else None
        plt.plot(row[1][metrics[0]], row[1][metrics[1]], 'ro', label=label)

    # Plot fuzzy dominated solutions
    for idx, row in enumerate(optuna_df[optuna_df.fuzzy_dominated].iterrows()):
        label = 'Fuzzy-dominated' if idx==0 else None
        plt.plot(row[1][metrics[0]], row[1][metrics[1]], 'yo', label=label)

    # Plot other experiments as comparison
    for idx, other_df in enumerate(other_dfs):
        label = 'Manual' if idx==0 else None
        invalid_share = 1 - other_df['valid_share_possible'].mean()
        valid_regret = other_df['valid_regret'].mean()
        plt.plot(invalid_share, valid_regret, 'go', label=label)

    plt.xlabel('Invalid share')
    plt.ylabel('Mean valid error')
    plt.legend()

    if store:
        plt.savefig(os.path.join(optuna_path, 'pareto_front_comparison.pdf'), format='pdf')
    else:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main('data/test/', ('data/test/test/',))
