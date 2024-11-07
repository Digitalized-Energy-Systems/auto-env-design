
import os

import optuna
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency

import pareto_front

# One standard deviation of the two metrics invalid_share and mean_error
EPSILONS = {
    'load': (0.01101716635261566, 0.5170811024214211),
    'eco': (0.013042655732193589, 0.4634594642907113),
    'qmarket': (0.047080512485089744, 0.02019430266310582),
    'voltage': (0.03316069287974844, 0.0006718227701652069),
    'renewable': (0.022824315312188172, 0.10965274847082523),
    'default': (0.0, 0.0)
}
N_STANDARD_DEVS = 1.0

"""
With multi-objective optimization, we can't just pick the best design, as there 
are multiple objectives to consider. Variants defined here are:
- pareto: Look at all solutions on the pareto front
- fuzzy_pareto: Look at all solutions on the fuzzy pareto front (pareto front with epsilon error)
- constraints: Look at top x solutions regarding constraint satisfaction
- objective: Look at top x solutions regarding objective optimization
"""
VARIANTS = ('pareto, fuzzy_pareto, constraints, objective', 'utopia')
VARIANT = 'utopia'
LOOK_AT_TOP_N = 10


def main(paths, env_names):

    top_designs = []
    bottom_designs = []
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
        epsilon = np.array(epsilon) * N_STANDARD_DEVS
        print(f'Using epsilon: {epsilon}')

        ## Compute the pareto front
        df = study.trials_dataframe()
        df = df[df.state == 'COMPLETE']
        df = df.dropna(subset=metric_names)
        df.reset_index(drop=True, inplace=True)
        df['not_dominated'] = pareto_front.compute_pareto_front(df[metric_names])
        df['fuzzy_dominated'] = pareto_front.compute_fuzzy_dominance(df[metric_names], df['not_dominated'], epsilon)
        full_df = df.copy()

        if VARIANT == 'pareto':
            top_df, bottom_df = filter_by_pareto(df)
        elif VARIANT == 'fuzzy_pareto':
            top_df, bottom_df = filter_by_fuzzy_pareto(df)
        elif VARIANT == 'constraints':
            top_df, bottom_df = filter_by_constraints(df, LOOK_AT_TOP_N)
        elif VARIANT == 'objective':
            top_df, bottom_df = filter_by_objective(df, LOOK_AT_TOP_N)
        elif VARIANT == 'utopia':
            top_df, bottom_df = filter_by_utopia(df, LOOK_AT_TOP_N)
        else:
            raise ValueError(f'Unknown variant: {VARIANT}')


        best_design = get_best_design(top_df, bottom_df)
        pprint_best_design(best_design, env_names[idx])

        top_designs.append(top_df)
        bottom_designs.append(bottom_df)

    # Trow all environments together (what is the best env design over all problems?)
    print(f'Get overall best design regardign {VARIANT}')
    total_df = pd.concat(top_designs)

    total_best_design = get_best_design(total_df, pd.concat(bottom_designs))
    pprint_best_design(total_best_design, 'Overall best design')


def filter_by_pareto(df):
    top_df = df[df.not_dominated]
    bottom_df = df[~df.not_dominated]
    print('N Non-dominated: ', len(top_df.index.values))
    return top_df, bottom_df

def filter_by_fuzzy_pareto(df):
    flag = df.fuzzy_dominated + df.not_dominated
    top_df = df[flag]
    bottom_df = df[~flag]
    print('N Non-dominated: ', len(top_df.index.values))
    return top_df, bottom_df

def filter_by_constraints(df, best_n=10):
    df = df.sort_values(by=['values_Invalid share'], ascending=[True])
    return df.head(best_n), df.iloc[best_n:]

def filter_by_objective(df, best_n=10):
    df = df.sort_values(by=['values_Mean error'], ascending=[True])
    return df.head(best_n), df.iloc[best_n:]

def filter_by_utopia(df, best_n=10):
    """ Look at only solutions that are within the top n of both metrics after
    normalization."""
    df['norm_invalid_share'] = (df['values_Invalid share'] - df['values_Invalid share'].min()) / (df['values_Invalid share'].max() - df['values_Invalid share'].min())
    df['norm_mean_error'] = (df['values_Mean error'] - df['values_Mean error'].min()) / (df['values_Mean error'].max() - df['values_Mean error'].min())
    # This is manhattan distance -> better use euclidean?
    df['utopia_distance'] = df['norm_invalid_share'] + df['norm_mean_error']
    df = df.sort_values(by=['utopia_distance'], ascending=[True])
    return df.head(best_n), df.iloc[best_n:]


def get_best_design(df, bottom_df=None):
    print('Number of considered designs: ', len(df))
    best_design = []
    design_decisions = [c for c in df.columns if "params_" in c]
    for param in design_decisions:
        # Compute mean and std for continuous values
        if isinstance(df[param].iloc[0], float):
            # print(f'{param}: {df[param].mean()} +- {df[param].std()}')
            overutilization = df[param].mean() / bottom_df[param].mean()
            # Test for statistical significance (is the distribution different from the overall distribution?)
            t, p = ttest_ind(df[param], bottom_df[param], equal_var=False)  # assumption: unequal variance
            best_design.append((param, df[param].mean(), df[param].std(), overutilization, f'p-value:', p))
        else:
            # If boolean or string: return most used entry instead
            # print(f'{param}: {df[param].mode()[0]} ({df[param].value_counts().iloc[0]}/{len(df)})')
            most_used_category = df[param].mode()[0]
            share_of_most_used = df[param].value_counts().iloc[0]/len(df)
            share_overall = bottom_df[param].value_counts()[most_used_category]/len(bottom_df)
            overutilization = share_of_most_used / share_overall
            # Test for statistical significance
            contingency_table = pd.DataFrame({
                'Top Group': df[param].value_counts(),
                'Rest Group': bottom_df[param].value_counts()
            }).fillna(0)
            chi2, p_value, _, _ = chi2_contingency(contingency_table)
            best_design.append((param, most_used_category, share_of_most_used, overutilization, f'p-value:', p_value))

    return best_design


def pprint_best_design(best_design, name='Best design', min_overutilization=0.3):
    print('---------------------------------')
    print(name, ':')
    for design_decision in best_design:
        if design_decision[-1] < 0.05:
            print(design_decision)
        else:
            print(f'Skip: {design_decision[0]}')
    print('---------------------------------')


if __name__ == '__main__':
    env_names = ('eco', 'load', 'renewable', 'voltage', 'qmarket')
    paths = [f'HPC/auto_env_design/data/20240906_{env_name}_multi' for env_name in env_names]

    main(paths, env_names)
