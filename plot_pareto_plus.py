""" Manual plot of the pareto front of an optuna experiment plus some other 
experiments for comparison. """

import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import tikzplotlib

import get_best_design
import pareto_front
from util import load_optuna_csv, load_drl_csv


ALLOWED_ERROR = (0.047080512485089744, 0.02019430266310582) # qmarket
ALLOWED_ERROR = (0.03316069287974844, 0.0006718227701652069) # voltage
ALLOWED_ERROR = (0.013042655732193589, 0.4634594642907113)  # eco
ALLOWED_ERROR = (0.022824315312188172, 0.10965274847082523) # renewable
ALLOWED_ERROR = (0.01101716635261566, 0.5170811024214211) # load

def plot_pareto_plus(optuna_df: str | pd.DataFrame, baseline_paths: tuple, show_all=True, store=True):
    """ Plot pareto front plus some baseline experiments """
    metrics = ['values_Invalid share', 'values_Mean error']
    if isinstance(optuna_df, str):
        optuna_path = optuna_df
        optuna_df = load_optuna_data(optuna_path, metrics)

    baseline_dfs = load_baseline_data(baseline_paths)

    fig = plt.figure()
    # Plot dominated solutions
    if show_all:
        plot_dominated_points(optuna_df, metrics)

    # Plot fuzzy dominated solutions
    plot_fuzzy_dominated_points(optuna_df, metrics)

    # Plot Pareto front solutions
    plot_pareto_points(optuna_df, metrics)

    # Plot other experiments as comparison
    for idx, baseline_df in enumerate(baseline_dfs):
        label = 'Manual' if idx==0 else None
        invalid_share = 1 - baseline_df['valid_share_possible'].mean()
        valid_regret = baseline_df['valid_regret'].mean()
        plt.plot(invalid_share, valid_regret, 'go', label=label)

    plt.xlabel('Invalid share')
    plt.ylabel('Mean valid error')
    plt.legend()

    if store:
        plt.savefig(os.path.join(optuna_path, 'pareto_front_comparison.pdf'), format='pdf')
    else:
        plt.show()
    plt.close(fig)


def plot_annotated(optuna_df: str | pd.DataFrame, top_n: int=8, store_as_tiks=False):
    """ Plot pareto front, highlight areas of interest (constraint focus,
    objective focus, utopia), and annotate the areas with the the prevalent
    environment design decisions. """

    metrics = ['values_Invalid share', 'values_Mean error']
    if isinstance(optuna_df, str):
        optuna_df = load_optuna_data(optuna_df, metrics)

    # Plot pareto front data points
    plot_dominated_points(optuna_df, metrics)
    plot_pareto_points(optuna_df, metrics)

    ax = plt.gca()

    # Highlight and annotate areas of interest
    # Constraint focus: Vertical line to highlight the min top_n data points
    top_df, bottom_df = get_best_design.filter_by_constraints(optuna_df, top_n)
    plt.axvline(x=top_df[metrics[0]].max(), color='g', linestyle='--')
    x_position = top_df[metrics[0]].min() + (top_df[metrics[0]].max() - top_df[metrics[0]].min()) / 2
    y_position = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) / 2
    add_annotation_to_plot('Constraints:', x_position, y_position, top_df, bottom_df)

    # Objective focus: Horizontal line to highlight the min top_n data points
    top_df, bottom_df = get_best_design.filter_by_objective(optuna_df, top_n)
    plt.axhline(y=top_df[metrics[1]].max(), color='r', linestyle='--')
    x_position = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) / 2
    y_position = top_df[metrics[1]].min() + (top_df[metrics[1]].max() - top_df[metrics[1]].min()) / 2
    add_annotation_to_plot('Objective:', x_position, y_position, top_df, bottom_df)

    # Utopia focus: box in bottom left to highlight the min top_n data points
    top_df, bottom_df = get_best_design.filter_by_utopia(optuna_df, top_n)
    vertices = (
        (-100, top_df[metrics[1]].max()),  # Top left
        (top_df[metrics[0]].max(), top_df[metrics[1]].max()),  # Top right
        (top_df[metrics[0]].max(), -100)  # Bottom right
    )
    ax.add_patch(patches.Polygon(vertices, closed=False, edgecolor='blue', facecolor='none', linestyle='--'))
    x_position = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) / 10
    y_position = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) / 10
    add_annotation_to_plot('Utopia:', x_position, y_position, top_df, bottom_df)

    # TODO: Pareto annotation?! Where to place?!
    top_df, bottom_df = get_best_design.filter_by_pareto(optuna_df)
    x_position = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) / 2
    y_position = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) / 2
    add_annotation_to_plot('Pareto:', x_position, y_position, top_df, bottom_df)


    # General plotting stuff
    plt.xlabel('Invalid share')
    plt.ylabel('Mean valid error')
    plt.legend()

    if store_as_tiks:
        tikzplotlib.save("plot.tikz")
    else:
        plt.show()


def load_baseline_data(baseline_paths):
    baseline_dfs = []
    for other_path in baseline_paths:
        baseline_dfs.append(load_drl_csv(other_path))

    return baseline_dfs


def load_optuna_data(optuna_path, metrics):
    optuna_df = load_optuna_csv(optuna_path)

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
    optuna_df['fuzzy_dominated'] = pareto_front.compute_fuzzy_dominance(optuna_df[metrics], optuna_df['non_dominated'], ALLOWED_ERROR)

    return optuna_df


def add_normalized_metrics_columns(df, metrics):
    for metric in metrics:
        df[metric + '_normalized'] = (df[metric] - df[metric].min()) / (
                df[metric].max() - df[metric].min())
    return df


def plot_dominated_points(full_df, metrics):
    plot_data_points(full_df[~full_df.non_dominated], metrics, 'bo', 'Dominated')

def plot_fuzzy_dominated_points(full_df, metrics):
    plot_data_points(full_df[full_df.fuzzy_dominated], metrics, 'yo', 'Fuzzy-dominated')

def plot_pareto_points(full_df, metrics):
    plot_data_points(full_df[full_df.non_dominated], metrics, 'ro', 'Non-dominated')


def plot_data_points(df_to_plot, metrics, style='bo', label=None):
    """ Scatter plot of the data points """
    for idx, row in enumerate(df_to_plot.iterrows()):
        label = label if idx==0 else None
        plt.plot(row[1][metrics[0]], row[1][metrics[1]], style, label=label)


def add_annotation_to_plot(header: str, x_position: float, y_position: float, top_df, bottom_df):
    best_design_continuous, best_design_discrete = get_best_design.get_best_design(top_df, bottom_df)
    annotation = extract_significant_design_decisions_as_str(best_design_continuous)
    annotation += extract_significant_design_decisions_as_str(best_design_discrete)
    if annotation == '':
        annotation = 'No significant design decisions'
    else:
        annotation = header + '\n' + annotation
    plt.text(x_position, y_position, annotation, fontsize=8, color='k', ha='center', va='center')


def extract_significant_design_decisions_as_str(df):
    """ Extract the significant design decisions from the data frame """
    text = ''
    for idx, row in df.iterrows():
        significant = row['Significant']
        if significant:
            text += f'{row["Parameter"][7:]}\n'
    return text


if __name__ == '__main__':
    env = 'renewable'
    # plot_pareto_plus(f'data/20240906_{env}_multi', (f'data/comparison2/{env}_default_50k_09',f'data/comparison2/{env}_default_50k'))
    plot_annotated(f'HPC/auto_env_design/data/20241104_multi_GA/{env}_new_dist_pwl_1step/')