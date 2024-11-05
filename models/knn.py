from models import base_model_
import sklearn
import abc
import optuna

class KNeighborsClassifier(base_model_.BaseModel, abc.ABC):

    def __init__(self, optuna_trial: optuna.trial.Trial):
        super().__init__(optuna_trial=optuna_trial)

    def define_model(self) -> sklearn.neighbors.KNeighborsClassifier:

        self.sampling = self.suggest_hyperparam_to_optuna('sampling')
        self.sampling_strategy = self.suggest_hyperparam_to_optuna('sampling_strategy')
        if self.sampling == "over" and self.sampling_strategy == "minmajority":
            self.sampling_strategy = "minority"
        if self.sampling == "under" and self.sampling_strategy == "minmajority":
            self.sampling_strategy = "majority"
        self.standardize_X = True
        self.x_scaler = sklearn.preprocessing.StandardScaler()
        self.impute = self.suggest_hyperparam_to_optuna('impute')

        n_neighbors = self.suggest_hyperparam_to_optuna('n_neighbors')
        weights = self.suggest_hyperparam_to_optuna('weights')
        algorithm = self.suggest_hyperparam_to_optuna('algorithm')
        leaf_size = self.suggest_hyperparam_to_optuna('leaf_size')
        p = self.suggest_hyperparam_to_optuna('p')

        return sklearn.neighbors.KNeighborsClassifier(
            n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size, p=p)

    def define_hyperparams_to_tune(self) -> dict:
        """
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the format.
        """
        return {
            'n_neighbors': {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 100
            },
            'leaf_size': {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 100
            },
            'p': {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 5
            },
            'weights': {
                'datatype': 'categorical',
                'list_of_values': ["uniform", "distance"],
            },
            'algorithm': {
                'datatype': 'categorical',
                'list_of_values': ["auto", "ball_tree", "kd_tree", "brute"],
            },
            'penalty': {
                'datatype': 'categorical',
                'list_of_values': ["l2", "l1", "elasticnet", None],
            }
        }



