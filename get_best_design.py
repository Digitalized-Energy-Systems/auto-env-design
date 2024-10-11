
import os

import optuna

import pareto_front


EPSILON = 0.01


def main(paths):
    for path in paths:
        if path[-1] == '/':
            path = path[:-1]
        study_name = os.path.join(path, path.split('/')[-1])
        print(study_name)
        storage = f'sqlite:///{os.path.join(path, "storage.db")}'
        print(storage)
        study = optuna.study.load_study(study_name=study_name, storage=storage)
        metric_names = ["Invalid share", "Mean error"]        
        study.set_metric_names(metric_names)
        metric_names = ["values_" + m for m in metric_names]

        ## Compute the pareto front
        df = study.trials_dataframe()
        df = df[df.state == 'COMPLETE']
        df = df.dropna(subset=metric_names)
        df.reset_index(drop=True, inplace=True)
        # Normalize the dataframe
        for metric_name in metric_names:
            df[metric_name] -= df[metric_name].min()
            df[metric_name] /= df[metric_name].max() - df[metric_name].min()
        # comparison = df[metric_names].to_numpy() <= df[metric_names].values[:, None]
        # TODO: Maybe also create some fuzzy pareto front where solutions are not dominated if some tolerance to dominating solution
        # dominated_mask = (comparison[:, :, 0] & comparison[:, :, 1]).any(axis=1) & (comparison.sum(axis=2) > 1).any(axis=1)
        df['not_dominated'] = pareto_front.compute_pareto_front(df[metric_names])
        df['fuzzy_dominated'] = pareto_front.compute_fuzzy_dominance(df[metric_names], df['not_dominated'], EPSILON)
        df = df[df.not_dominated]
        print('Non-dominated: ', df.index.values)

        design_decisions = [c for c in df.columns if "params_" in c]
        print(design_decisions)
        for param in design_decisions:
            # Compute mean and std for continuous values
            if isinstance(df[param].iloc[0], float):
                print(f'{param}: {df[param].mean()} +- {df[param].std()}')
            else:
                # If boolean or string: return most used entry instead
                print(f'{param}: {df[param].mode()[0]} ({df[param].value_counts().iloc[0]}/{len(df)})')

        print('---------------------------------')


if __name__ == '__main__':
    main(('data/20240906_eco_multi','data/20240906_load_multi','data/20240906_renewable_multi','data/20240906_voltage_multi'))
