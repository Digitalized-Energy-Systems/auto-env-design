
import os

import optuna
import numpy as np
import pandas as pd

import pareto_front


EPSILONS = {
    'load': (0.01101716635261566, 0.5170811024214211),
    'eco': (0.013042655732193589, 0.4634594642907113),
    'qmarket': (0.047080512485089744, 0.02019430266310582),
    'voltage': (0.03316069287974844, 0.0006718227701652069),
    'renewable': (0.022824315312188172, 0.10965274847082523),
    'default': (0.0, 0.0)
}


def main(paths, env_names, n_standard_devs=2.0):

    total_designs = []
    for idx, path in enumerate(paths):
        if path[-1] == '/':
            path = path[:-1]
        storage = f'sqlite:///{os.path.join(path, "storage.db")}'
        print(storage)
        study = optuna.study.load_study(study_name=None, storage=storage)
        metric_names = ["Invalid share", "Mean error"]        
        study.set_metric_names(metric_names)
        metric_names = ["values_" + m for m in metric_names]

        epsilon = EPSILONS.get(env_names[idx], 0)
        epsilon = np.array(epsilon) * n_standard_devs
        print(f'Using epsilon: {epsilon}')

        ## Compute the pareto front
        df = study.trials_dataframe()
        df = df[df.state == 'COMPLETE']
        df = df.dropna(subset=metric_names)
        df.reset_index(drop=True, inplace=True)
        # Normalize the dataframe
        # for metric_name in metric_names:
        #     df[metric_name] -= df[metric_name].min()
        #     df[metric_name] /= df[metric_name].max() - df[metric_name].min()
        # comparison = df[metric_names].to_numpy() <= df[metric_names].values[:, None]
        # TODO: Maybe also create some fuzzy pareto front where solutions are not dominated if some tolerance to dominating solution
        # dominated_mask = (comparison[:, :, 0] & comparison[:, :, 1]).any(axis=1) & (comparison.sum(axis=2) > 1).any(axis=1)
        df['not_dominated'] = pareto_front.compute_pareto_front(df[metric_names])
        df['fuzzy_dominated'] = pareto_front.compute_fuzzy_dominance(df[metric_names], df['not_dominated'], epsilon)
        df = df[df.fuzzy_dominated + df.not_dominated]
        print('N Non-dominated: ', len(df.index.values))

        pareto_design = get_pareto_design(df)
        pprint_pareto_design(pareto_design)

        total_designs.append(df)

    # Trow all environments together (what is the best env design over all problems?)
    print('Get overall pareto design')
    total_df = pd.concat(total_designs)

    total_pareto_design = get_pareto_design(total_df)
    pprint_pareto_design(total_pareto_design)


def get_pareto_design(df):
    pareto_design = []
    design_decisions = [c for c in df.columns if "params_" in c]
    for param in design_decisions:
        # Compute mean and std for continuous values
        if isinstance(df[param].iloc[0], float):
            # print(f'{param}: {df[param].mean()} +- {df[param].std()}')
            pareto_design.append((param, df[param].mean(), df[param].std()))
        else:
            # If boolean or string: return most used entry instead
            # print(f'{param}: {df[param].mode()[0]} ({df[param].value_counts().iloc[0]}/{len(df)})')
            pareto_design.append((param, df[param].mode()[0], df[param].value_counts().iloc[0]/len(df)))

    return pareto_design


def pprint_pareto_design(pareto_design):
    for design_decision in pareto_design:
        print(design_decision)
    print('---------------------------------')


if __name__ == '__main__':
    env_names = ('eco', 'load', 'renewable', 'voltage', 'qmarket')
    paths = [f'HPC/auto_env_design/data/20240906_{env_name}_multi' for env_name in env_names]

    main(paths, env_names)
