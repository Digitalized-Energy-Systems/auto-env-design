
"""
Define the optuna hyperparameter (env design) sampling and optimization.
Note: Contains multiple env design decisions that turned out to to be not relevant,
e.g. data augmentation methods like interpolate_steps or noise_factor.
See the OPF-Gym docs for more details on these parameters.

"""

import random

import numpy as np
import optuna

from drl.hp_tuning.optuna_interface import main


def custom_multi_obj_metric(eval_metrics):
    invalid_share = 1 - np.mean([m['valid_share_possible'] for m in eval_metrics])
    valid_regret = np.mean([m['valid_regret'] for m in eval_metrics])
    return invalid_share, valid_regret


def custom_multi_obj_metric_all_regret(eval_metrics):
    invalid_share = 1 - np.mean([m['valid_share_possible'] for m in eval_metrics])
    regret = np.mean([m['regret'] for m in eval_metrics])
    return invalid_share, regret


def opf_env_design_sampling(trial):
    # Define hyperparameters to optimize
    hps = dict()
    # hps['train_data'] = trial.suggest_categorical('train_data', ["mixed"])
    hps['add_res_obs'] = 'mixed'  # trial.suggest_categorical('add_res_obs', ['mixed'])
    hps['add_mean_obs'] = trial.suggest_categorical('add_mean_obs', [True, False])
    hps['autoscale_actions'] = trial.suggest_categorical('autoscale_actions', [True, False])
    hps['diff_objective'] = trial.suggest_categorical('diff_objective', [True, False])

    # Constraints
    hps['constraint_params'] = {}
    hps['worst_case_violations'] = trial.suggest_categorical('only_worst_case_violations', [True, False])
    hps['constraint_params']['only_worst_case_violations'] = hps['only_worst_case_violations']
    penalty_fct_options = [0.5, 1.0, 2.0]
    hps['penalty_index'] = trial.suggest_int('penalty_index', 0, 2)
    hps['constraint_params']['penalty_power'] = penalty_fct_options[hps['penalty_index']]

    # Addidional observations
    hps['add_voltage_magnitude'] = trial.suggest_categorical('add_voltage_magnitude', [True, False])
    hps['add_voltage_angle'] = trial.suggest_categorical('add_voltage_angle', [True, False])
    hps['add_line_loading'] = trial.suggest_categorical('add_line_loading', [True, False])
    hps['add_trafo_loading'] = trial.suggest_categorical('add_trafo_loading', [True, False])
    hps['add_ext_grid_power'] = trial.suggest_categorical('add_ext_grid_power', [True, False])
    # Add results observations in different format
    if hps['add_res_obs'] == 'mixed':
        hps['add_res_obs'] = []
        if hps['add_voltage_magnitude']:
            hps['add_res_obs'].append('voltage_magnitude')
        if hps['add_voltage_angle']:
            hps['add_res_obs'].append('voltage_angle')
        if hps['add_line_loading']:
            hps['add_res_obs'].append('line_loading')
        if hps['add_trafo_loading']:
            hps['add_res_obs'].append('trafo_loading')
        if hps['add_ext_grid_power']:
            hps['add_res_obs'].append('ext_grid_power')
        hps['add_n_obs_channels'] = len(hps['add_res_obs'])

    # Data distribution
    share_hps = ['simbench_share', 'normal_share', 'uniform_share']
    total = 0
    for idx, share in enumerate(share_hps):
        hps[share] = trial.suggest_float(share, 0, 1)
        total += hps[share]
    # Need to be constrained to sum of 1 -> Approximate Dirichlet Distribution
    for idx, share in enumerate(share_hps):
        hps[share] /= total
    simbench = hps['simbench_share']
    uniform = simbench + hps['uniform_share']
    normal = uniform + hps['normal_share']
    hps['sampling_params'] = dict()
    hps['sampling_params']['data_probabilities'] = (simbench, uniform, normal)
    hps['noise_factor'] = trial.suggest_float('noise_factor', 0, 0.2)
    hps['sampling_params']['noise_factor'] = hps['noise_factor']
    hps['sampling_params']['interpolate_steps'] = trial.suggest_categorical('interpolate_steps', [True, False])
    hps['interpolate_steps'] = hps['sampling_params']['interpolate_steps']
    hps['train_data'] = 'mixed'

    # Define episode / action
    # if hps['add_n_obs_channels'] > 1:  # Assumption at least two additional obs required
    #     hps['steps_per_episode'] = trial.suggest_int('steps_per_episode', 1, 5, step=2)
    # else:
    #     hps['steps_per_episode'] = trial.suggest_int('steps_per_episode', 1, 1, step=2)
    # hps['add_act_obs'] = True if hps['steps_per_episode'] > 1 else False

    # Diff action space only possible for multi-step
    # hps['diff_action_space'] = trial.suggest_categorical('diff_action_space', [True, False])
    # if hps['diff_action_space'] and hps['steps_per_episode']:
    #     hps['diff_action_step_size'] = 2 / hps['steps_per_episode']
    # else:
    #     hps['diff_action_step_size'] = None
    # hps['initial_action'] = trial.suggest_categorical('initial_action', ['center', 'random'])

    hps['clipped_action_penalty'] = trial.suggest_float('clipped_action_penalty', 0, 10)

    # Reward function params
    hps['penalty_weight'] = trial.suggest_float('penalty_weight', 0.01, 0.99)
    hps['valid_reward'] = trial.suggest_float('valid_reward', 0.0, 2.0)
    hps['invalid_penalty'] = trial.suggest_float('invalid_penalty', 0.0, 2.0)
    hps['invalid_objective_share'] = trial.suggest_float('invalid_objective_share', 0.0, 1.0)
    hps['reward_function_params'] = dict()
    hps['reward_function_params']['penalty_weight'] = hps['penalty_weight']
    hps['reward_function_params']['valid_reward'] = hps['valid_reward']
    hps['reward_function_params']['invalid_penalty'] = hps['invalid_penalty']
    hps['reward_function_params']['invalid_objective_share'] = hps['invalid_objective_share']
    hps['reward_function_params']['reward_scaling'] = 'normalization'

    return hps


# Multi-objective: optuna.samplers.NSGAIIISampler() or default (None)
main(
    sampler=optuna.samplers.NSGAIIISampler(),
    env_hp_sampling_method=opf_env_design_sampling,
    hp_sampling_method=None,
    metric=custom_multi_obj_metric, # (env_name='load'),
    storage=True,
    load_if_exists=True,
    directions=["minimize", "minimize"],
    # direction='minimize',
    n_seeds=3,
    # median=False,
    last_n_steps=4,
    pruner=optuna.pruners.PercentilePruner(50.0, n_startup_trials=999, n_warmup_steps=4)
)
