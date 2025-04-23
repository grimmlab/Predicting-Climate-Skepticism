from models import base_model_
import sklearn
import abc
import optuna

class RidgeClassifier(base_model_.BaseModel, abc.ABC):

    def __init__(self, optuna_trial: optuna.trial.Trial):
        super().__init__(optuna_trial=optuna_trial)

    def define_model(self) -> sklearn.linear_model.RidgeClassifier:

        self.sampling = self.suggest_hyperparam_to_optuna('sampling')
        self.sampling_strategy = self.suggest_hyperparam_to_optuna('sampling_strategy')
        if self.sampling == "over" and self.sampling_strategy == "minmajority":
            self.sampling_strategy = "minority"
        if self.sampling == "under" and self.sampling_strategy == "minmajority":
            self.sampling_strategy = "majority"
        self.standardize_X = True
        self.x_scaler = sklearn.preprocessing.StandardScaler()
        self.impute = self.suggest_hyperparam_to_optuna('impute')

        alpha = self.suggest_hyperparam_to_optuna('alpha')
        fit_intercept = self.suggest_hyperparam_to_optuna('fit_intercept')
        copy_X = self.suggest_hyperparam_to_optuna('copy_X')
        tol = self.suggest_hyperparam_to_optuna('tol')
        solver = self.suggest_hyperparam_to_optuna('solver')

        return sklearn.linear_model.RidgeClassifier(
            alpha=alpha, fit_intercept=fit_intercept, copy_X=copy_X, tol=tol, solver=solver, random_state=42)

    def define_hyperparams_to_tune(self) -> dict:
        """
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the format.
        """
        return {
            'alpha': {
                'datatype': 'float',
                'lower_bound': 0.01,
                'upper_bound': 100,
                'log': True
            },
            'fit_intercept': {
                'datatype': 'categorical',
                'list_of_values': [False, True],
            },
            'copy_X': {
                'datatype': 'categorical',
                'list_of_values': [False, True],
            },
            'tol': {
                'datatype': 'float',
                'lower_bound': 0.00001,
                'upper_bound': 0.1,
                'log': True
            },
            'solver': {
                'datatype': 'categorical',
                'list_of_values': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'],
            }
        }



