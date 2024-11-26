
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
VARIANTS = ('pareto', 'fuzzy_pareto', 'constraints', 'objective', 'utopia')
VARIANT = 'fuzzy_pareto'
LOOK_AT_TOP_N = 10


def main(paths, env_names, variant, store=True):

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
        df['non_dominated'] = pareto_front.compute_pareto_front(df[metric_names])
        df['fuzzy_dominated'] = pareto_front.compute_fuzzy_dominance(df[metric_names], df['non_dominated'], epsilon)
        full_df = df.copy()

        print('Use evaluation: ', variant)
        if variant == 'pareto':
            top_df, bottom_df = filter_by_pareto(df)
        elif variant == 'fuzzy_pareto':
            top_df, bottom_df = filter_by_fuzzy_pareto(df)
        elif variant == 'constraints':
            top_df, bottom_df = filter_by_constraints(df, LOOK_AT_TOP_N)
        elif variant == 'objective':
            top_df, bottom_df = filter_by_objective(df, LOOK_AT_TOP_N)
        elif variant == 'utopia':
            top_df, bottom_df = filter_by_utopia(df, LOOK_AT_TOP_N)
        else:
            raise ValueError(f'Unknown variant: {variant}')

        # alternative_statistical_significance(full_df, metric_names)

        best_design_continuous, best_design_discrete = get_best_design(top_df, bottom_df)
        pprint_best_design(best_design_continuous, best_design_discrete, env_names[idx])

        if store:
            # Store the design distribution to csv
            best_design_continuous.to_csv(f'{path}/best_design_continuous_{variant}.csv')
            best_design_discrete.to_csv(f'{path}/best_design_discrete_{variant}.csv')

        top_designs.append(top_df)
        bottom_designs.append(bottom_df)

    # Trow all environments together (what is the best env design over all problems?)
    print(f'Get overall best design regarding {variant}')
    best_design_continuous, best_design_discrete = get_best_design(
        pd.concat(top_designs), pd.concat(bottom_designs))
    pprint_best_design(best_design_continuous, best_design_discrete, name='Overall best design')

    print('---------------------------------')

    if store:
        one_path_lower = os.path.split(path)[0]
        best_design_continuous.to_csv(f'{one_path_lower}/best_design_continuous_{variant}.csv')
        best_design_discrete.to_csv(f'{one_path_lower}/best_design_discrete_{variant}.csv')



def filter_by_pareto(df) -> tuple[pd.DataFrame, pd.DataFrame]:
    top_df = df[df.non_dominated]
    bottom_df = df[~df.non_dominated]
    return top_df, bottom_df

def filter_by_fuzzy_pareto(df):
    flag = df.fuzzy_dominated + df.non_dominated
    top_df = df[flag]
    bottom_df = df[~flag]
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


def get_best_design(df, bottom_df):
    print('Number of considered designs: ', len(df))
    best_design_continuous = []
    best_design_discrete = []
    design_decisions = [c for c in df.columns if "params_" in c]

    for param in design_decisions:
        # Compute mean and std for continuous values
        if isinstance(df[param].iloc[0], float):
            overutilization = df[param].mean() / bottom_df[param].mean()
            # Test for statistical significance (is the distribution different from the overall distribution?)
            t, p_value = ttest_ind(df[param], bottom_df[param], equal_var=False)  # assumption: unequal variance
            data = {'Parameter': param, 'Mean': df[param].mean(), 'Std': df[param].std(), 'Overutilization': overutilization, 'p-value': p_value}
            best_design_continuous.append(data)
        else:
            # If boolean, string or int: return most used entry instead
            most_used_category = df[param].mode()[0]
            share_of_most_used = df[param].value_counts().iloc[0]/len(df)
            try:
                share_overall = bottom_df[param].value_counts()[most_used_category]/len(bottom_df)
                overutilization = share_of_most_used / share_overall
            except KeyError:
                # most_used_category is not in bottom_df
                overutilization = np.inf
            # Test for statistical significance
            contingency_table = pd.DataFrame({
                'Top Group': df[param].value_counts(),
                'Rest Group': bottom_df[param].value_counts()
            }).fillna(0)
            chi2, p_value, _, _ = chi2_contingency(contingency_table)
            data = {'Parameter': param, 'Most Used': most_used_category, 'Share': share_of_most_used, 'Overutilization': overutilization, 'p-value': p_value}
            best_design_discrete.append(data)

    best_design_continuous = pd.DataFrame(best_design_continuous)
    best_design_discrete = pd.DataFrame(best_design_discrete)

    for df in (best_design_continuous, best_design_discrete):
        df['Significant'] = df['p-value'] < 0.05

    return best_design_continuous, best_design_discrete


def pprint_best_design(best_design_continuous, best_design_discrete, name='Best design'):
    print(name, ':')
    print('Continuous:')
    print(best_design_continuous)
    print('Discrete:')
    print(best_design_discrete)
    total = len(best_design_continuous) + len(best_design_discrete)
    significance = best_design_continuous['Significant'].sum() + best_design_discrete['Significant'].sum()
    mean_significance = significance / total
    print('Overall significance: ', mean_significance)
    print('---------------------------------')


def alternative_statistical_significance(df, metrics):
    """ The test above checks if the top x designs are significantly different
    from the rest, however, without actually looking at the performance metrics,
    except using them for the split.
    Here, we perform the group split based on the design decisions and then
    look if the performance metrics are significantly different. """
    design_decisions = [c for c in df.columns if "params_" in c]
    for metric in metrics:
        continuous_design_decisions = []
        discrete_design_decisions = []
        for param in design_decisions:
            if isinstance(df[param].iloc[0], float):
                # Continuous
                top_50_percentile = df[param].quantile(0.5)
                top_df = df[df[param] > top_50_percentile]
                rest_df = df[df[param] <= top_50_percentile]
                # Test for statistical significance
                t, p_value = ttest_ind(top_df[metric], rest_df[metric], equal_var=False)
                data = {'Parameter': param[7:], 'Mean': df[param].mean(), 'Std': df[param].std(), 'p-value': p_value, 'Significant': p_value < 0.05}
                continuous_design_decisions.append(data)
            else:
                # Discrete
                most_used_category = df[param].mode()[0]
                most_used_df = df[df[param] == most_used_category]
                rest_df = df[df[param] != most_used_category]
                # Test for statistical significance
                t, p_value = ttest_ind(most_used_df[metric], rest_df[metric], equal_var=False)
                data = {'Parameter': param[7:], 'Most Used': most_used_category, 'p-value': p_value, 'Significant': p_value < 0.05}
                discrete_design_decisions.append(data)

        print(metric)
        print('Continuous:')
        print(pd.DataFrame(continuous_design_decisions))
        print('Discrete:')
        print(pd.DataFrame(discrete_design_decisions))
        print('---------------------------------')


if __name__ == '__main__':
    env_names = ('voltage', 'qmarket', 'load')  # 'eco', 'load', 'renewable', 
    paths = [f'HPC/auto_env_design/data/20241121_multi_GA_full/{env_name}' for env_name in env_names]

    for variant in VARIANTS:
        main(paths, env_names, variant)
