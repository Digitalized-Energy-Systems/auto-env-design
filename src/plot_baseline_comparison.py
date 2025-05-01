"""
Create plots for final verification experiments. (Figure 6 in the paper)

"""

import os

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

from drl.util import plot_returns


def float_formatter(x, pos):
    return f'{x:.1f}'  # Format tick labels as floats with 1 decimal place


def plot_baseline_comparison(best_path, base_path, y_column='valid_regret', 
                             plot_std=True, rolling_window=2, store_as=None):

    base_labels = ['Auto Design', 'Manual Design']

    name = os.path.basename(os.path.normpath(best_path))

    plt.figure(figsize=(7, 3))
    ax = plt.gca()

    voltage_factor = 100000  # For better and more consistent scaling of the y axis

    for algo in ('sac', 'ddpg'):
        for path, base_label in zip([best_path, base_path], base_labels):
            path_ = path.replace('algo', algo)

            x_column = ''
            x_range = (0, None)
            all_x, all_y = plot_returns.get_all_xy_data_from(path_, x_column, [y_column], x_range)

            mean_y = np.mean(list(all_y.values()), axis=1).flatten()
            # Rescale y axis for better comparison (has no meaning anyway)
            mean_y = mean_y * voltage_factor if ('voltage' in path and y_column == 'valid_regret') else mean_y

            std_y = np.std(list(all_y.values()), axis=1).flatten()
            std_y = std_y * voltage_factor if ('voltage' in path and y_column == 'valid_regret') else std_y


            if y_column == 'valid_share_possible':
                # Data is stored inverted
                mean_y = 1 - mean_y

            # Assuming that all x values are the same anyway
            x = list(all_x.values())[0][0]

            if rolling_window:
                mean_y = np.convolve(mean_y, np.ones(rolling_window), 'valid') / rolling_window
                std_y = np.convolve(std_y, np.ones(rolling_window), 'valid') / rolling_window
                x = x[rolling_window - 1:]

            label = f'{base_label} {algo.upper()}'

            plt.plot(x, mean_y, label=label)
            if plot_std:
                plt.fill_between(x, mean_y - std_y, mean_y + std_y, alpha=0.1)

    plt.xlabel('Training Steps')

    if y_column == 'valid_regret':
        plt.ylabel('Mean valid error')
    elif y_column == 'valid_share_possible':
        plt.ylabel('Invalid share')


    plt.xticks(range(min(x), max(x)+1, 100000))  # Ticks every 100,000
    # ax.yaxis.set_major_formatter(FuncFormatter(float_formatter))

    plt.grid()
    plt.legend(loc="upper right")

    if store_as:
        # plt.tight_layout()
        name = f'figures/{name}_{y_column}.{store_as}'
        print('Store to', name)
        plt.savefig(name, format=store_as, bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    best_path = "data/20241223_best_design_500k/env_algo_best_alldata/"
    base_path = "data/20241223_best_design_500k/env_algo_base_alldata/"
    for env in ('voltage', 'eco'):
        for y_column in ['valid_regret', 'valid_share_possible']:
            plot_baseline_comparison(
                best_path.replace('/env', '/'+env),
                base_path.replace('/env', '/'+env),
                y_column=y_column,
                store_as="pdf"
            )
