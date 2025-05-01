
import os

import matplotlib.pyplot as plt
import optuna


def main(paths):
    for path in paths:
        if path[-1] == '/':
            path = path[:-1]
        study_name = os.path.join(path, path.split('/')[-1])
        print(study_name)
        storage = f'sqlite:///{os.path.join(path, "storage.db")}'
        print(storage)
        study = optuna.study.load_study(study_name=study_name, storage=storage)
        study.set_metric_names(["Invalid share", "Mean error"])

        metrics = ('invalid share', 'error')
        fig = optuna.visualization.matplotlib.plot_pareto_front(study, target_names=metrics)
        plt.title('Pareto front (all samples)')
        plt.savefig(os.path.join(path, 'pareto_front_all.pdf'), format='pdf')

        fig = optuna.visualization.matplotlib.plot_pareto_front(study, target_names=metrics, include_dominated_trials=False)
        plt.title('Pareto front (only non-dominated)')
        plt.savefig(os.path.join(path, 'pareto_front.pdf'), format='pdf')

        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'hp_importance.pdf'), format='pdf')

        # Not supported for multi-objective
        # optuna.visualization.matplotlib.plot_terminator_improvement(study)
        # plt.savefig(os.path.join(path, 'terminator_improvement.pdf'), format='pdf')


if __name__ == '__main__':
    main(('data/20240906_qmarket_multi','data/20240906_eco_multi','data/20240906_load_multi','data/20240906_renewable_multi','data/20240906_voltage_multi'))
