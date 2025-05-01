""" The idea is to store the best environment designs to tables, including the 
utilized evaluation criterion, the overutilization, the p-value. """

import os

import numpy as np
import pandas as pd

import get_best_design
import util



def create_significance_table(
        optuna_paths: list[str],
        env_names: list[str],
        top_n: int = 20,
        exclude_constraint_and_objective: bool = True,
        ) -> None:
    assert isinstance(optuna_paths[0], str)

    metrics = ['values_Invalid share', 'values_Mean error']

    columns = ['Design Decision']
    # Add env prefixes to the column names
    if not exclude_constraint_and_objective:
        columns += [env[0] + 'Constraints'for env in env_names]
        columns += [env[0] + 'Objective' for env in env_names]
    columns += [env[0] + 'Utopia' for env in env_names]
    columns += [env[0] + 'Pareto' for env in env_names]
    columns += ['Mean']
    significance_df = pd.DataFrame(columns=columns)

    raw_dfs = [util.load_optuna_data(path, metrics, None) for path in optuna_paths]

    eval_dfs = []
    if not exclude_constraint_and_objective:
        eval_dfs.append([get_best_design.filter_by_constraints(df, top_n) for df in raw_dfs])
        eval_dfs.append([get_best_design.filter_by_objective(df, top_n) for df in raw_dfs])
    eval_dfs.append([get_best_design.filter_by_utopia(df, top_n) for df in raw_dfs])
    eval_dfs.append([get_best_design.filter_by_pareto(df) for df in raw_dfs])

    parameter_names = list(eval_dfs[0][0][0].columns)
    parameter_names = [column for column in parameter_names if 'params_' in column]

    if exclude_constraint_and_objective:
        criteria = ['Utopia', 'Pareto']
    else:
        criteria = ['Constraints', 'Objective', 'Utopia', 'Pareto']
    criterion_df_pairs = tuple(zip(criteria, eval_dfs))

    for parameter in parameter_names:
        data = {'Design Decision': parameter.replace('params_', '')}
        for criterion, dfs in criterion_df_pairs:
            for i, env in enumerate(env_names):
                prefix = env[0]  # First letter
                best_design_continuous, best_design_discrete = get_best_design.get_best_design(*dfs[i])
                df = best_design_continuous if parameter in best_design_continuous['Parameter'].values else best_design_discrete
                significant = df.loc[df['Parameter'] == parameter, 'significant'].values[0]
                data[prefix + criterion] = 1 if significant else 0

        data['Mean'] = np.mean([data[env[0] + criterion] for env in env_names for criterion in criteria])
        significance_df = significance_df.append(data, ignore_index=True)

    # Sort table by 'Mean' column
    significance_df = significance_df.sort_values(by='Mean', ascending=False)

    mean_significance = {column: significance_df[column].mean()
                         for column in significance_df.columns
                         if column != 'Design Decision'}
    mean_significance['Design Decision'] = 'Mean'
    significance_df = significance_df.append(mean_significance, ignore_index=True)

    base_path = os.path.commonpath(optuna_paths)
    path = os.path.join(base_path, 'significance_table.txt')
    table_to_latex(significance_df, path)
    replace_latex_headers(path, criteria=criteria)


def replace_latex_headers(path: str, criteria: list[str]):
    with open(path, 'r') as f:
        latex_string = f.read()

    for criterion in criteria:
        existing_string = f"v{criterion} & e{criterion} & r{criterion} & l{criterion} & q{criterion}"
        latex_string = latex_string.replace(
            existing_string,
            "\\multicolumn{5}{l}" + "{" + criterion + "}"
        )

    with open(path, 'w') as f:
        f.write(latex_string)


def create_design_tables(optuna_path: str, top_n: int = 20):

    assert isinstance(optuna_path, str)

    metrics = ['values_Invalid share', 'values_Mean error']
    optuna_df = util.load_optuna_data(optuna_path, metrics, None)
    for eval_crit in ['constraints', 'objective', 'pareto', 'utopia']:
        if eval_crit == 'constraints':
            top_df, bottom_df = get_best_design.filter_by_constraints(optuna_df, top_n)
        elif eval_crit == 'objective':
            top_df, bottom_df = get_best_design.filter_by_objective(optuna_df, top_n)
        elif eval_crit == 'pareto':
            top_df, bottom_df = get_best_design.filter_by_pareto(optuna_df)
        elif eval_crit == 'utopia':
            top_df, bottom_df = get_best_design.filter_by_utopia(optuna_df, top_n)

        best_design_continuous, best_design_discrete = get_best_design.get_best_design(top_df, bottom_df)

        # Rounding to 3 decimal places
        best_design_continuous = best_design_continuous.round(3)
        best_design_discrete = best_design_discrete.round(3)

        # Remove `params_` prefix from all entries in 'Parameter' column
        best_design_continuous['Parameter'] =best_design_continuous['Parameter'].apply(lambda x: x.replace('params_', ''))
        best_design_discrete['Parameter'] = best_design_discrete['Parameter'].apply(lambda x: x.replace('params_', ''))

        path = os.path.join(optuna_path, f'best_design_{eval_crit}continuous.txt')
        table_to_latex(best_design_continuous, path)
        path = os.path.join(optuna_path, f'best_design_{eval_crit}discrete.txt')
        table_to_latex(best_design_discrete, path)


def table_to_latex(table: pd.DataFrame, path: str):
    table.to_latex(path, index=False)


if __name__ == '__main__':
    envs = ['voltage', 'eco', 'renewable', 'load', 'qmarket']
    paths = [f'HPC/auto_env_design/data/20241128_multi_GA_reduced/{env}/' for env in envs]
    create_significance_table(paths, envs, 20)
    # for path in paths:
    #     create_design_tables(path)