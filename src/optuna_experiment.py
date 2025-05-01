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


def custom_multi_obj_metric_all_regret(eval_metrics):
    invalid_share = 1 - np.mean([m['valid_share_possible'] for m in eval_metrics])
    regret = np.mean([m['regret'] for m in eval_metrics])
    return invalid_share, regret


class custom_single_obj_metric_safe():
    def __init__(self,
                 env_name='default',
                 min_valid_share=0.999,
                 valid_offset=1):
        normalization = {  # Re-used parameters from objective normalization
            'default': 1.0,
            'qmarket': 0.212,
            'voltage': 0.113,
            'load': 7.14,
            'eco': 24.21,
            'renewable': 7.3}
        self.normalization = normalization[env_name]
        self.min_valid_share = min_valid_share
        self.valid_offset = valid_offset

    def __call__(self, eval_metrics):
        valid_share = np.mean([m['valid_share_possible'] for m in eval_metrics])
        invalid_share = 1 - valid_share
        if valid_share < self.min_valid_share:
            return invalid_share

        # Only consider regret if constraint satisfaction is high enough
        regret = np.mean([m['regret'] for m in eval_metrics])
        # Normalize regret with experience from old experiments (otherwise regret far too dominating or too weak)
        regret /= self.normalization

        return invalid_share + regret - self.valid_offset


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
    remaining_share = 1
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


def opf_env_design_sampling_safe(trial):
    assert False, 'Removed for now. Maybe in later publication again?!'
    # Focus only on the design decisions relevant for the cost function for safe RL
    hps = dict()

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

    # Reward function params
    # Valid reward not possible here because the cost function must not flip sign
    # hps['valid_reward'] = trial.suggest_float('valid_reward', 0.0, 2.0)
    hps['invalid_penalty'] = trial.suggest_float('invalid_penalty', 0.0, 3.0)
    # hps['objective_share'] = trial.suggest_float('objective_share', 0.0, 1.0)
    hps['reward_function_params'] = dict()
    # hps['reward_function_params']['valid_reward'] = hps['valid_reward']
    hps['reward_function_params']['invalid_penalty'] = hps['invalid_penalty']
    # hps['reward_function_params']['objective_share'] = hps['objective_share']

    return hps


def opf_algo_hp_sampling_safe(trial):
    # Sample HPs for the safe RL algorithm
    hps = dict()

    # USL-SAC-specific HPs
    hps['penalty_factor'] = trial.suggest_float('penalty_factor', 0.5, 10, log=True)
    hps['max_projection_iterations'] = trial.suggest_int('max_projection_iterations', 1, 20)
    hps['extra_cost_critic'] = trial.suggest_categorical('extra_cost_critic', [True, False])
    # hps['use_correction_optimizer'] = trial.suggest_categorical('use_correction_optimizer', [True, False])

    # Specific SAC HPs
    hps['entropy_learning_rate'] = trial.suggest_float('entropy_learning_rate', 0.00005, 0.01, log=True)
    hps['target_entropy'] = trial.suggest_int('target_entropy', -50, 0)

    # General HPs
    batch_sizes = [64, 128, 256, 512, 1024, 2048]
    batch_size_index = trial.suggest_int('batch_size_index', 0, 5)
    hps['batch_size'] = batch_sizes[batch_size_index]
    hps['start_train'] = int(hps['batch_size'] * 1.5)

    hps['actor_learning_rate'] = trial.suggest_float('actor_learning_rate', 0.00005, 0.01, log=True)
    hps['critic_learning_rate'] = trial.suggest_float('critic_learning_rate', 0.00005, 0.01, log=True)

    train_intervals = (
        {'train_interval': 1, 'train_steps': 2},
        {'train_interval': 1, 'train_steps': 1},
        {'train_interval': 2, 'train_steps': 1},
        {'train_interval': 3, 'train_steps': 1},
    )
    train_interval_index = trial.suggest_int('train_interval_index', 0, 3)
    hps.update(train_intervals[train_interval_index])

    actor_net_layers = trial.suggest_int('actor_net_layers', 1, 4)
    actor_net_neurons = [128, 256, 512]
    actor_net_neuron_index = trial.suggest_int('actor_net_size_index', 0, 2)
    hps['actor_fc_dims'] = [actor_net_neurons[actor_net_neuron_index]] * actor_net_layers

    critic_net_layers = trial.suggest_int('critic_net_layers', 1, 4)
    critic_net_neurons = [128, 256, 512]
    critic_net_neuron_index = trial.suggest_int('critic_net_size_index', 0, 2)
    hps['critic_fc_dims'] = [critic_net_neurons[critic_net_neuron_index]] * critic_net_layers

    return hps


# Multi-objective: optuna.samplers.NSGAIIISampler() or default (None)
# Single_objective: optuna.samplers.GPSampler(n_startup_trials=10)
main(sampler=optuna.samplers.NSGAIIISampler(),
        env_hp_sampling_method=opf_env_design_sampling,
        hp_sampling_method=None, # opf_algo_hp_sampling_safe,
        metric=custom_multi_obj_metric, # (env_name='load'),
        storage=True,
        load_if_exists=True,
        directions=["minimize", "minimize"],
        # direction='minimize',
        n_seeds=3,
        # median=False,
        last_n_steps=4,
        pruner=optuna.pruners.PercentilePruner(50.0, n_startup_trials=999, n_warmup_steps=4))
