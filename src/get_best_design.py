
import os

import optuna
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency, chi2

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


def get_best_design(df, bottom_df, continuous_to_n_bins: int=None):
    print('Number of considered designs: ', len(df))
    best_design_continuous = []
    best_design_discrete = []

    # Treat continuous values as discrete by putting them into buckets
    # Probably not necessary because ttest does not compare the averages (which was the motivation for this) but compares the distributions
    for param in [c for c in df.columns if "params_" in c]:
        if isinstance(df[param].iloc[0], float) and continuous_to_n_bins:
            full_df = pd.concat([df, bottom_df])
            bins = np.linspace(full_df[param].min(), full_df[param].max(), continuous_to_n_bins+1)
            # Add new columns with discretized values
            df[param + '_discrete'] = pd.cut(df[param], bins, labels=False, include_lowest=True)
            bottom_df[param + '_discrete'] = pd.cut(bottom_df[param], bins, labels=False, include_lowest=True)

    for param in [c for c in df.columns if "params_" in c]:
        if isinstance(df[param].iloc[0], float):
            # Compute mean and std for continuous values
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
        try:
            df['significant'] = df['p-value'] < 0.05
        except KeyError:
            # Dataframe is empty
            pass

    return best_design_continuous, best_design_discrete


def pprint_best_design(best_design_continuous, best_design_discrete, name='Best design'):
    print(name, ':')
    print('Continuous:')
    print(best_design_continuous)
    print('Discrete:')
    print(best_design_discrete)
    mean_significance = compute_significance_share(best_design_discrete, best_design_continuous)
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
                data = {'Parameter': param[7:], 'Mean': df[param].mean(), 'Std': df[param].std(), 'p-value': p_value, 'significant': p_value < 0.05}
                continuous_design_decisions.append(data)
            else:
                # Discrete
                most_used_category = df[param].mode()[0]
                most_used_df = df[df[param] == most_used_category]
                rest_df = df[df[param] != most_used_category]
                # Test for statistical significance
                t, p_value = ttest_ind(most_used_df[metric], rest_df[metric], equal_var=False)
                data = {'Parameter': param[7:], 'Most Used': most_used_category, 'p-value': p_value, 'significant': p_value < 0.05}
                discrete_design_decisions.append(data)

        print(metric)
        print('Continuous:')
        print(pd.DataFrame(continuous_design_decisions))
        print('Discrete:')
        print(pd.DataFrame(discrete_design_decisions))
        print('---------------------------------')


def compute_significance_share(df_discrete, df_continuous):
    """ Compute the share of significant design decisions """
    total = len(df_discrete) + len(df_continuous)
    return (df_discrete['significant'].sum() + df_continuous['significant'].sum()) / total


def compute_combined_p_values(top_dfs: list, bottom_dfs: list) -> dict:
    """ Combine p-values of multiple tests """
    all_p_values = {}
    best_designs = [get_best_design(top_df, bottom_df) for top_df, bottom_df in zip(top_dfs, bottom_dfs)]

    continuous_best_designs = [d[0] for d in best_designs]
    discrete_best_designs = [d[1] for d in best_designs]

    for idx, param in enumerate(continuous_best_designs[0]['Parameter']):
        p_values = [df.at[idx, 'p-value'] for df in continuous_best_designs]
        overutilizations = [df.at[idx, 'Overutilization'] for df in continuous_best_designs]
        means = [df.at[idx, 'Mean'] for df in continuous_best_designs]
        p_value = fishers_method(p_values)
        if p_value < 0.05:
            all_p_values[param] = {'p-value': p_value,
                                'mean': np.mean(means),
                                'overutilization': np.mean(overutilizations)}

    for idx, param in enumerate(discrete_best_designs[0]['Parameter']):
        p_values = [df.at[idx, 'p-value'] for df in discrete_best_designs]
        # Unclear how to do the same here?!
        p_value = fishers_method(p_values)
        if p_value < 0.05:
            most_used_per_env = [df.at[idx, 'Most Used'] for df in discrete_best_designs]
            single_most_used = max(set(most_used_per_env), key=most_used_per_env.count)
            all_p_values[param] = {'p-value': p_value,
                                   'most_used': single_most_used}

    import pdb; pdb.set_trace()

    return all_p_values


def fishers_method(p_values):
    """ Combine p-values using Fisher's method """
    chi2_values = [-2 * np.log(p) for p in p_values]
    chi2_value = sum(chi2_values)
    degrees_of_freedom = 2 * len(p_values)
    combined_p_value = 1 - chi2.cdf(chi2_value, degrees_of_freedom)
    return combined_p_value


# if __name__ == '__main__':
#     env_names = ('voltage', 'qmarket', 'load')  # 'eco', 'load', 'renewable', 
#     paths = [f'HPC/auto_env_design/data/20241121_multi_GA_full/{env_name}' for env_name in env_names]

#     for variant in VARIANTS:
#         main(paths, env_names, variant)
