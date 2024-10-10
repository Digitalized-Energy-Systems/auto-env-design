import random

import numpy as np
import optuna

from drl.hp_tuning.optuna_interface import main


class custom_single_obj_metric():
    def __init__(self, env_name='default', constraint_preference=0.8):
        normalization = {
            'default': 1.0,
            'qmarket': 0.212,
            'voltage': 0.113,
            'load': 7.14,
            'eco': 24.21,
            'renewable': 7.3}
        self.normalization = normalization[env_name]
        self.constraint_preference = constraint_preference

    def __call__(self, eval_metrics):
        valid_share = np.mean([m['valid_share_possible'] for m in eval_metrics])
        invalid_share = 1 - valid_share
        # Only consider regret for valid solutions
        valid_regret = np.mean([m['valid_regret'] for m in eval_metrics])
        # Normalize regret with experience from old experiments (otherwise regret far too dominating) 
        valid_regret /= self.normalization
        # Downsize regret with valid_share to prevent that few valid solutions dominate everything
        # Essentially: A good regret is only worth something if the solution is also valid
        valid_regret *= valid_share
        # Compute final weighted metric
        return self.constraint_preference * invalid_share + (1 - self.constraint_preference) * valid_regret


def custom_multi_obj_metric(eval_metrics):
    invalid_share = 1 - np.mean([m['valid_share_possible'] for m in eval_metrics])
    valid_regret = np.mean([m['valid_regret'] for m in eval_metrics])
    return invalid_share, valid_regret


def opf_env_design_sampling(trial):
    # Define hyperparameters to optimize
    hps = dict()
    # hps['train_data'] = trial.suggest_categorical('train_data', ["mixed"])
    hps['add_res_obs'] = 'mixed'  # trial.suggest_categorical('add_res_obs', ['mixed'])
    hps['add_mean_obs'] = trial.suggest_categorical('add_mean_obs', [True, False])
    hps['add_act_obs'] = trial.suggest_categorical('add_act_obs', [True, False])
    hps['autoscale_actions'] = trial.suggest_categorical('autoscale_actions', [True, False])
    hps['diff_objective'] = trial.suggest_categorical('diff_objective', [True, False])
    hps['steps_per_episode'] = trial.suggest_int('steps_per_episode', 1, 5, step=2)

    # Diff action space only possible for multi-step
    hps['diff_action_space'] = trial.suggest_categorical('diff_action_space', [True, False])
    if hps['steps_per_episode'] == 1:
        hps['diff_action_space'] = False
    elif hps['diff_action_space']:
        hps['diff_action_step_size'] = 2 / hps['steps_per_episode']
    # hps['initial_action'] = trial.suggest_categorical('initial_action', ['center', 'random'])
    hps['only_worst_case_violations'] = trial.suggest_categorical('only_worst_case_violations', [True, False])

    # Penalty function
    hps['penalty_fct'] = trial.suggest_categorical('penalty_fct', ['linear', 'quadr', 'sqrt'])
    linear_penalty = 0
    quadr_penalty = 0
    sqrt_penalty = 0
    if hps['penalty_fct'] == 'linear':
        linear_penalty = 1
    elif hps['penalty_fct'] == 'quadr':
        quadr_penalty = 1
    elif hps['penalty_fct'] == 'sqrt':
        sqrt_penalty = 1
    # Simplification: Same setting for all constraints!
    hps['volt_pen_kwargs'] = {'linear_penalty': linear_penalty, 'quadr_penalty': quadr_penalty, 'sqrt_penalty': sqrt_penalty}
    hps['trafo_pen_kwargs'] = {'linear_penalty': linear_penalty, 'quadr_penalty': quadr_penalty, 'sqrt_penalty': sqrt_penalty}
    hps['line_pen_kwargs'] = {'linear_penalty': linear_penalty, 'quadr_penalty': quadr_penalty, 'sqrt_penalty': sqrt_penalty}
    hps['ext_grid_pen_kwargs'] = {'linear_penalty': linear_penalty, 'quadr_penalty': quadr_penalty, 'sqrt_penalty': sqrt_penalty}

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
    remaining_share = 1
    share_hps = ['simbench_share', 'uniform_share', 'normal_share']
    random.shuffle(share_hps)
    for idx, share in enumerate(share_hps):
        # Constraint: sum of shares must be 1
        if idx == 2:
            hps[share] = trial.suggest_float(share, remaining_share, remaining_share)
        else:
            hps[share] = trial.suggest_float(share, 0, remaining_share)
            remaining_share -= hps[share]
    simbench = hps['simbench_share']
    uniform = simbench + hps['uniform_share']
    normal = uniform + hps['normal_share']
    hps['sampling_kwargs'] = dict()
    hps['sampling_kwargs']['data_probabilities'] = (simbench, uniform, normal)
    hps['noise_factor'] = trial.suggest_float('noise_factor', 0, 0.2)
    hps['sampling_kwargs']['noise_factor'] = hps['noise_factor']

    hps['penalty_weight'] = trial.suggest_float('penalty_weight', 0.01, 0.99)
    hps['clipped_action_penalty'] = trial.suggest_float('clipped_action_penalty', 0, 10)

    # Reward function params
    hps['valid_reward'] = trial.suggest_float('valid_reward', 0.0, 2.0)
    hps['invalid_penalty'] = trial.suggest_float('invalid_penalty', 0.0, 2.0)
    hps['objective_share'] = trial.suggest_float('objective_share', 0.0, 1.0)
    hps['reward_function_params'] = dict()
    hps['reward_function_params']['valid_reward'] = hps['valid_reward']
    hps['reward_function_params']['invalid_penalty'] = hps['invalid_penalty']
    hps['reward_function_params']['objective_share'] = hps['objective_share']

    return hps


# Multi-objective: optuna.samplers.NSGAIIISampler() or default (None)
# Single_objective: optuna.samplers.GPSampler(n_startup_trials=10)
main(sampler=None,
        env_hp_sampling_method=opf_env_design_sampling,
        metric=custom_multi_obj_metric,
        storage=True,
        load_if_exists=True,
        directions=["minimize", "minimize"],
        # direction='minimize',
        n_seeds=3,
        mean=True,
        last_n_steps=4)
        # pruner=optuna.pruners.PercentilePruner(50.0, n_startup_trials=10, n_warmup_steps=1))
