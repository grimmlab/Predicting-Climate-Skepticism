from models import base_model_
import sklearn
import abc
import optuna

class LogisticRegression(base_model_.BaseModel, abc.ABC):

    def __init__(self, optuna_trial: optuna.trial.Trial):
        super().__init__(optuna_trial=optuna_trial)

    def define_model(self) -> sklearn.linear_model.LogisticRegression:

        self.sampling = self.suggest_hyperparam_to_optuna('sampling')
        self.sampling_strategy = self.suggest_hyperparam_to_optuna('sampling_strategy')
        if self.sampling == "over" and self.sampling_strategy == "minmajority":
            self.sampling_strategy = "minority"
        if self.sampling == "under" and self.sampling_strategy == "minmajority":
            self.sampling_strategy = "majority"
        self.standardize_X = True
        self.x_scaler = sklearn.preprocessing.StandardScaler()
        self.impute = self.suggest_hyperparam_to_optuna('impute')

        fit_intercept = self.suggest_hyperparam_to_optuna('fit_intercept')
        tol = self.suggest_hyperparam_to_optuna('tol')
        solver = self.suggest_hyperparam_to_optuna('solver')
        max_iter = self.suggest_hyperparam_to_optuna('max_iter')
        warm_start = self.suggest_hyperparam_to_optuna('warm_start')
        penalty = self.suggest_hyperparam_to_optuna('penalty')
        C = self.suggest_hyperparam_to_optuna('C')
        l1_ratio = self.suggest_hyperparam_to_optuna('l1_ratio')

        return sklearn.linear_model.LogisticRegression(
            penalty=penalty, C=C, max_iter=max_iter, warm_start=warm_start, n_jobs=-1, l1_ratio=l1_ratio,
            fit_intercept=fit_intercept, tol=tol, solver=solver, random_state=42)

    def define_hyperparams_to_tune(self) -> dict:
        """
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the format.
        """
        return {
            'fit_intercept': {
                'datatype': 'categorical',
                'list_of_values': [False, True],
            },
            'l1_ratio': {
                'datatype': 'float',
                'upper_bound': 1.0,
                'lower_bound': 0.0,
                'step': 0.1
            },
            'C': {
                'datatype': 'categorical',
                'list_of_values': [0.01, 0.1, 1, 10, 100],
            },
            'penalty': {
                'datatype': 'categorical',
                'list_of_values': ["l1"] # , None, "l2", , "elasticnet"
            },
            'warm_start': {
                'datatype': 'categorical',
                'list_of_values': [False, True],
            },
            'max_iter': {
                'datatype': 'int',
                'lower_bound': 100,
                'upper_bound': 1000,
                'step': 100
            },
            'tol': {
                'datatype': 'float',
                'lower_bound': 0.00001,
                'upper_bound': 0.1,
                'log': True
            },
            'solver': {
                'datatype': 'categorical',
                'list_of_values': ['saga', 'liblinear'],
            }
        }



