""" Get a set of environment design decisions to directly start new experiments. """

import json

import numpy as np

import get_best_design
import util


def get_single_best_design(
    optuna_path: str,
    mean_of_top_n: int = 10,
    store: bool = True
    ):
    assert isinstance(optuna_path, str)

    metrics = ['values_Invalid share', 'values_Mean error']

    df = util.load_optuna_data(optuna_path, metrics, None)
    top_df, bottom_df = get_best_design.filter_by_utopia(df, mean_of_top_n)
    best_design_continuous, best_design_discrete = get_best_design.get_best_design(top_df, bottom_df)

    best_design = {}
    for row in best_design_continuous.iterrows():
        row = row[1]
        best_design[row['Parameter'].replace('params_', '')] = row['Mean']

    for row in best_design_discrete.iterrows():
        row = row[1]
        best_design[row['Parameter'].replace('params_', '')] = row['Most Used']

    hyperparameters = convert_to_hyperparameters(best_design)

    # Store to Json
    if store:  
        with open(f'{optuna_path}/best_design_n{mean_of_top_n}.json', 'w') as f:
            # Store to text file so that it can be loaded again with eval
            f.write(json.dumps(hyperparameters).replace('"', "'").replace('true', 'True').replace('false', 'False').replace('[', '(').replace(']', ')'))

    return hyperparameters


def convert_to_hyperparameters(best_design: dict) -> dict:
    """ opfgym requires a special format in some cases, .e.g. dicts in dicts """
    hyperparameters = {}
    hyperparameters['reward_function_params'] = {'reward_scaling': 'normalization'}
    hyperparameters['constraint_params'] = {}
    hyperparameters['sampling_params'] = {}

    hyperparameters['sampling_params']['data_probabilities'] = (
        # Cumulative probabilities
        best_design['simbench_share'],
        best_design['simbench_share'] + best_design['uniform_share'],
        1.0
    )

    for key, value in best_design.items():
        if isinstance(value, np.bool_):
            value = bool(value)
        if isinstance(value, float):
            value = round(value, 4)

        if key in ('invalid_penalty', 'valid_reward', 'penalty_weight'):
            hyperparameters['reward_function_params'][key] = value
        if 'objective_share' in key:  # Was renamed in the meantime
            hyperparameters['reward_function_params']['invalid_objective_share'] = value
        elif 'worst_case_violations' in key: # Renamed as well
            hyperparameters['constraint_params']['only_worst_case_violations'] = value
        elif 'penalty_index' in key:
            hyperparameters['constraint_params']['penalty_power'] = [0.5, 1.0, 2.0][value]
        elif key in ('noise_factor', 'interpolate_steps'):
            hyperparameters['sampling_params'][key] = value
        elif key == 'diff_action_space':
            hyperparameters['diff_action_step_size'] = False  # Removed in pre-study
        elif key in ('simbench_share', 'uniform_share', 'normal_share'):
            pass # handled separately
        else:
            # Can be directly used as hyperparameter
            hyperparameters[key] = value

    # After optimizing these on the validation data, we should test these only on the test data to prevent positive bias
    hyperparameters['evaluate_on'] = 'test'
    hyperparameters['random_validation_steps'] = True

    hyperparameters['reward_function'] = 'parameterized'

    return hyperparameters


if __name__ == '__main__':
    env_name = 'renewable'
    path = f'HPC/auto_env_design/data/20241128_multi_GA_reduced/{env_name}/'
    get_single_best_design(optuna_path=path, mean_of_top_n=5)