""" Manual plot of the pareto front of an optuna experiment plus some other
experiments for comparison. Used for Figs 2/3 of the paper.

Note: Minor adjustments in line 209 to improve plot readability.
"""

import copy
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

import get_best_design
from get_metric_variance import get_metric_std_dev
from util import load_optuna_csv, load_drl_csv, load_optuna_data


LINEWIDTH = 1.0


def plot_pareto_plus(optuna_df: str | pd.DataFrame, baseline_paths: tuple, show_all=True, store=True):
    """ Plot pareto front plus some baseline experiments """
    metrics = ['values_Invalid share', 'values_Mean error']
    if isinstance(optuna_df, str):
        optuna_path = optuna_df
        optuna_df = load_optuna_data(optuna_path, metrics)

    fig = plt.figure()
    # Plot dominated solutions
    if show_all:
        plot_dominated_points(optuna_df, metrics)

    # Plot fuzzy dominated solutions
    plot_fuzzy_dominated_points(optuna_df, metrics)

    # Plot Pareto front solutions
    plot_pareto_points(optuna_df, metrics)

    # Plot other experiments as comparison
    plot_baseline(baseline_paths)

    add_labels_and_legend_to_plot()

    if store:
        plt.savefig(os.path.join(optuna_path, 'pareto_front_comparison.pdf'), format='pdf', bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_baseline(baseline_paths):
    baseline_dfs = load_baseline_data(baseline_paths)
    # Plot baseline experiments as comparison
    for idx, baseline_df in enumerate(baseline_dfs):
        label = 'Manual Design' if idx==0 else None
        invalid_share = 1 - baseline_df['valid_share_possible'].mean()
        valid_regret = baseline_df['valid_regret'].mean()
        print('Baseline performance: ', (invalid_share, valid_regret,))
        plt.plot(invalid_share, valid_regret, 'go', label=label)


def load_baseline_data(baseline_paths):
    baseline_dfs = []
    for other_path in baseline_paths:
        baseline_dfs.append(load_drl_csv(other_path))

    return baseline_dfs


def plot_full_annotated_variant1(
        optuna_dfs: tuple[str, ...] | tuple[pd.DataFrame, ...],
        top_n: int=20, store_as=False):
    """ Throw all together and then split based on normalized metrics (Pareto may prefer some environments over others) """
    metrics = ['values_Invalid share', 'values_Mean error']
    if isinstance(optuna_dfs[0], str):
        for idx, optuna_path in enumerate(optuna_dfs):
            optuna_dfs[idx] = load_optuna_data(optuna_path, metrics)
            # Normalize the metrics for comparability (overwrite existing)
            for metric in metrics:
                optuna_dfs[idx][metric] = optuna_dfs[idx][metric + '_normalized']

    # Variant 1: Throw all together and then split based on normalized metrics (may prefer some environments over others)
    combined_df = pd.concat(optuna_dfs)
    combined_df = combined_df.reset_index(drop=True)
    print('Total number of data points: ', len(combined_df))
    plot_single_annotated(combined_df, top_n, store_as,
                          add_error_bar=False,
                          add_fuzzy_dominated=False)


def plot_full_annotated_variant2(
        optuna_dfs: tuple[str, ...] | tuple[pd.DataFrame, ...],
        top_n: int=20, store_as=False):
    """ Split each separately and then combine the annotations (all environments have the same influence regarding Pareto. TODO: Is this true?!)"""
    metrics = ['values_Invalid share', 'values_Mean error']
    if isinstance(optuna_dfs[0], str):
        optuna_paths = copy.deepcopy(optuna_dfs)
        for idx, optuna_path in enumerate(optuna_dfs):
            optuna_dfs[idx] = load_optuna_data(optuna_path, metrics)
            # Normalize the metrics for comparability
            for metric in metrics:
                optuna_dfs[idx][metric] = optuna_dfs[idx][metric + '_normalized']

    # TODO: Maybe different colors to differentiate them?! Or different markers?!

    # plot_dominated_points(pd.concat(optuna_dfs), metrics)
    # plot_pareto_points(pd.concat(optuna_dfs), metrics)

    ax = plt.gca()

    # Constraint focus
    top_dfs, bottom_dfs = [], []
    for df in optuna_dfs:
        top_df, bottom_df = get_best_design.filter_by_constraints(df, top_n)
        top_dfs.append(top_df)
        bottom_dfs.append(bottom_df)

    full_top_df = pd.concat(top_dfs)
    full_bottom_df = pd.concat(bottom_dfs)
    x_position = full_top_df[metrics[0]].min() + (full_top_df[metrics[0]].max() - full_top_df[metrics[0]].min()) / 2
    y_position = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) / 2
    add_annotation_to_plot('Constraints:', x_position, y_position, full_top_df, full_bottom_df)

    # Objective focus
    top_dfs, bottom_dfs = [], []
    for df in optuna_dfs:
        top_df, bottom_df = get_best_design.filter_by_objective(df, top_n)
        top_dfs.append(top_df)
        bottom_dfs.append(bottom_df)

    full_top_df = pd.concat(top_dfs)
    full_bottom_df = pd.concat(bottom_dfs)
    x_position = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) / 2
    y_position = full_top_df[metrics[1]].min() + (full_top_df[metrics[1]].max() - full_top_df[metrics[1]].min()) / 2   
    add_annotation_to_plot('Objective:', x_position, y_position, full_top_df, full_bottom_df)

    # Utopia focus
    top_dfs, bottom_dfs = [], []
    for df in optuna_dfs:
        top_df, bottom_df = get_best_design.filter_by_utopia(df, top_n)
        top_dfs.append(top_df)
        bottom_dfs.append(bottom_df)

    full_top_df = pd.concat(top_dfs)
    full_bottom_df = pd.concat(bottom_dfs)
    x_position = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) / 10
    y_position = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) / 10
    add_annotation_to_plot('Utopia:', x_position, y_position, full_top_df, full_bottom_df)

    # Pareto focus
    top_dfs, bottom_dfs = [], []
    for df in optuna_dfs:
        top_df, bottom_df = get_best_design.filter_by_pareto(df)
        top_dfs.append(top_df)
        bottom_dfs.append(bottom_df)

    full_top_df = pd.concat(top_dfs)
    full_bottom_df = pd.concat(bottom_dfs)
    x_position = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) / 2
    y_position = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) / 2
    add_annotation_to_plot('Pareto:', x_position, y_position, full_top_df, full_bottom_df)

    # Using these p-values is statistically more valid!!! (fishers method)
    # Otherwise, the envs with more non-dominated points get preferred
    p_values = get_best_design.compute_combined_p_values(top_dfs, bottom_dfs)
    print('Combined p-values for Pareto:', p_values)

    add_labels_and_legend_to_plot()

    if not store_as:
        plt.show()
    else:
        path = os.path.dirname(optuna_paths[0].rstrip('/'))
        name = os.path.join(path, 'combined_annotated_plot')
        print('Store to:', name)
        plt.savefig(name + '.' + store_as, format=store_as, bbox_inches="tight")

    plt.close()


def plot_single_annotated(optuna_df: str | pd.DataFrame, top_n: int=20,
                          store_as=None, baseline_paths: list[str]=None,
                          add_annotations=True, add_areas=True,
                          add_error_bar=True,
                          add_fuzzy_dominated=True, normalize_error=False):
    """ Plot pareto front, highlight areas of interest (constraint focus,
    objective focus, utopia), and annotate the areas with the the prevalent
    environment design decisions. """

    metrics = ['values_Invalid share', 'values_Mean error']
    allowed_error = None
    if add_fuzzy_dominated or add_error_bar:
        optuna_path = copy.deepcopy(optuna_df)
        invalid_share_std, valid_error_std = get_metric_std_dev(optuna_path)
        print('Standard devs:', (invalid_share_std, valid_error_std))
        allowed_error = (invalid_share_std, valid_error_std)

    if isinstance(optuna_df, str):
        optuna_path = copy.deepcopy(optuna_df)
        optuna_df = load_optuna_data(optuna_df, metrics, allowed_error)
    else:
        optuna_path = 'combined'

    plt.figure(figsize=(8, 4))
    ax = plt.gca()

    # Plot pareto front data points
    plot_dominated_points(optuna_df, metrics,
                          remove_y_bottom_range=None,  # set 0.2 to remove outliers in ['eco', 'load', 'qmarket'] 
                          remove_x_bottom_range=None)  # set 0.5 for 'qmarket' to remove outliers
    if add_fuzzy_dominated:
        plot_fuzzy_dominated_points(optuna_df, metrics)
    plot_pareto_points(optuna_df, metrics)

    if baseline_paths:
        plot_baseline(baseline_paths)
    if add_error_bar:
        add_std_dev_to_plot(invalid_share_std, valid_error_std)

    if add_annotations or add_areas:
        # Highlight and annotate areas of interest
        # Constraint focus: Vertical line to highlight the min top_n data points
        top_df, bottom_df = get_best_design.filter_by_constraints(optuna_df, top_n)
        plt.axvline(x=top_df[metrics[0]].max(), color='k', linestyle='--', linewidth=LINEWIDTH)
        x_position = top_df[metrics[0]].min() + (top_df[metrics[0]].max() - top_df[metrics[0]].min()) / 2
        y_position = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) / 2
        if add_annotations:
            add_annotation_to_plot('Constraints:', x_position, y_position, top_df, bottom_df)

        # Objective focus: Horizontal line to highlight the min top_n data points
        top_df, bottom_df = get_best_design.filter_by_objective(optuna_df, top_n)
        plt.axhline(y=top_df[metrics[1]].max(), color='k', linestyle='--', linewidth=LINEWIDTH)
        x_position = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) / 2
        y_position = top_df[metrics[1]].min() + (top_df[metrics[1]].max() - top_df[metrics[1]].min()) / 2
        if add_annotations:
            add_annotation_to_plot('Objective:', x_position, y_position, top_df, bottom_df)

        # Pareto focus: box in the middle to highlight the pareto data points
        top_df, bottom_df = get_best_design.filter_by_pareto(optuna_df)
        x_position = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) / 2
        y_position = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) / 2
        if add_annotations:
            add_annotation_to_plot('Pareto:', x_position, y_position, top_df, bottom_df)

        if add_fuzzy_dominated:
            # Fuzzy Pareto focus: box in the middle to highlight the pareto data points
            top_df, bottom_df = get_best_design.filter_by_fuzzy_pareto(optuna_df)
            x_position = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) / 2
            y_position = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.75
            if add_annotations:
                add_annotation_to_plot('Fuzzy Pareto:', x_position, y_position, top_df, bottom_df)

        # Utopia focus: box in bottom left to highlight the min top_n data points
        top_df, bottom_df = get_best_design.filter_by_utopia(optuna_df, top_n)
        vertices = (
            (ax.get_xlim()[0], top_df[metrics[1]].max()),  # Top left
            (top_df[metrics[0]].max(), top_df[metrics[1]].max()),  # Top right
            (top_df[metrics[0]].max(), ax.get_ylim()[0])  # Bottom right
        )
        ax.add_patch(patches.Polygon(vertices, closed=False, edgecolor='black', facecolor='none', linestyle='--', linewidth=LINEWIDTH))
        x_position = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) / 10
        y_position = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) / 10
        if add_annotations:
            add_annotation_to_plot('Utopia:', x_position, y_position, top_df, bottom_df)

    add_labels_and_legend_to_plot()

    if normalize_error:
        y_max = optuna_df[metrics[1]].max()
        # Add a formatter to normalize the y-axis labels
        def normalize_formatter(y, pos):
            return f"{y / y_max:.2f}"  # Normalize and format as a decimal

        # Apply the formatter to the y-axis
        plt.gca().yaxis.set_major_formatter(FuncFormatter(normalize_formatter))

        tick_positions = np.linspace(0, y_max, 6)  # 6 ticks, including 0 and y_max
        plt.yticks(tick_positions)


    if not store_as:
        plt.show()
    else:
        try:
            path = os.path.dirname(optuna_path.rstrip('/'))
            name = optuna_path.split('/')[-2] if isinstance(optuna_path, str) else 'plot'
            if add_annotations:
                name = os.path.join(path, name + '_annotated')
            else:
                name = os.path.join(path, name + '_plain')
        except IndexError:
            name = 'HPC/auto_env_design/data/20241128_multi_GA_reduced/combined_annotated_plot'
        print('Store to:', name)
        plt.savefig(f'{name}.{store_as}', format=store_as, bbox_inches="tight")

    plt.close()


def plot_dominated_points(full_df,
                          metrics,
                          remove_x_bottom_range: float=None,
                          remove_y_bottom_range: float=None,
                          **kwargs):
    overall_condition = ~full_df.non_dominated
    if remove_x_bottom_range:
        # Remove the bottom x percent of the data range to make the rest better visible
        metric_range = full_df[metrics[0]].max() - full_df[metrics[0]].min()
        boundary = full_df[metrics[0]].max() - metric_range * remove_x_bottom_range
        stay_condition = full_df[metrics[0]] < boundary
        overall_condition = stay_condition & overall_condition
    if remove_y_bottom_range:
        # Remove the bottom x percent of the data range to make the rest better visible
        metric_range = full_df[metrics[1]].max() - full_df[metrics[1]].min()
        boundary = full_df[metrics[1]].max() - metric_range * remove_y_bottom_range
        stay_condition = full_df[metrics[1]] < boundary
        overall_condition = stay_condition & overall_condition

    plot_data_points(full_df[overall_condition], metrics, 'bo', 'Dominated', **kwargs)

def plot_fuzzy_dominated_points(full_df, metrics, **kwargs):
    plot_data_points(full_df[full_df.fuzzy_dominated], metrics, 'yo', 'Fuzzy-dominated', **kwargs)

def plot_pareto_points(full_df, metrics, **kwargs):
    plot_data_points(full_df[full_df.non_dominated], metrics, 'ro', 'Non-dominated', **kwargs)


def plot_data_points(df_to_plot, metrics, style='bo', label=None, normalized=False):
    """ Scatter plot of the data points """
    if normalized:
        metrics = [m + '_normalized' for m in metrics]
    for idx, row in enumerate(df_to_plot.iterrows()):
        label = label if idx==0 else None
        plt.plot(row[1][metrics[0]], row[1][metrics[1]], style, label=label)


def add_annotation_to_plot(header: str, x_position: float, y_position: float, top_df, bottom_df):
    best_design_continuous, best_design_discrete = get_best_design.get_best_design(top_df, bottom_df)
    print('Significant share: ', get_best_design.compute_significance_share(
        best_design_continuous, best_design_discrete))
    annotation = extract_significant_design_decisions_as_str(best_design_continuous)
    annotation += extract_significant_design_decisions_as_str(best_design_discrete)
    if annotation == '':
        annotation = header + '\n' + 'No significant design decisions'
    else:
        annotation = header + '\n' + annotation[:-1]
    plt.text(x_position, y_position, annotation, fontsize=6, color='k', ha='center', va='center',
                 bbox=dict(facecolor='lightgrey', edgecolor='black', alpha=0.8, boxstyle='round,pad=1'))
    # plt.annotate(annotation,
    #              (x_position, y_position),
    #              textcoords="offset points",
    #              xytext=(10, 10),  # Offset position of the text
    #              ha='center',
    #              bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray", alpha=0.7),
    #              arrowprops=dict(arrowstyle="->", color='black'),
    #              fontsize=6
    # )


def extract_significant_design_decisions_as_str(df, significance_level=0.05) -> str:
    """ Extract the significant design decisions from the data frame """
    text = ''
    for idx, row in df.iterrows():
        if row['p-value'] < significance_level:
            if 'Mean' in row:
                high_or_low = '(high)' if row['Overutilization'] > 1 else '(low)'
                text += f'{row["Parameter"][7:]}={round(row["Mean"], 2)} {high_or_low}\n'
            else: 
                text += f'{row["Parameter"][7:]}={row["Most Used"]}\n'
    return text


def add_labels_and_legend_to_plot():
    """ Add general information to the plot """
    plt.xlabel('Invalid share')
    plt.ylabel('Mean valid error')
    plt.legend(loc='upper right')


def add_std_dev_to_plot(invalid_share_std, valid_error_std):
    """ Add annotation window that show standard deviations of both
    metrics. """

    x_pos = plt.xlim()[0] + 0.5 * (plt.xlim()[1] - plt.xlim()[0])
    y_pos = plt.ylim()[0] + 0.9 * (plt.ylim()[1] - plt.ylim()[0])

    plt.errorbar(
        x_pos,
        y_pos,
        xerr=invalid_share_std,
        yerr=valid_error_std,
        fmt='o',
        color='black',
        capsize=4,
        label="Mean std deviation"
    )


if __name__ == '__main__':
    envs = ['voltage', 'eco', 'renewable', 'load', 'qmarket']
    paths = [f'data/20241128_multi_GA_reduced/{env}/' for env in envs]
    for idx, env in enumerate(envs):
        base_paths = [f'data/20241203_baseline/{env}_{weight}' for weight in ['01',  '03', '05', '07', '09']] # , '03', '05', '07', '09'
        plot_single_annotated(paths[idx],
                              store_as='pdf',
                              baseline_paths=base_paths,
                              add_annotations=False,
                              add_areas=False,
                              add_fuzzy_dominated=False,
                              normalize_error=False)
    # plot_full_annotated_variant2(paths, store_as='pdf', top_n=20)
