from models import base_model_
import sklearn
import abc
import optuna

class SVClassifier(base_model_.BaseModel, abc.ABC):

    def __init__(self, optuna_trial: optuna.trial.Trial):
        super().__init__(optuna_trial=optuna_trial)

    def define_model(self) -> sklearn.svm.SVC:

        self.sampling = self.suggest_hyperparam_to_optuna('sampling')
        self.sampling_strategy = self.suggest_hyperparam_to_optuna('sampling_strategy')
        if self.sampling == "over" and self.sampling_strategy == "minmajority":
            self.sampling_strategy = "minority"
        if self.sampling == "under" and self.sampling_strategy == "minmajority":
            self.sampling_strategy = "majority"
        self.standardize_X = self.suggest_hyperparam_to_optuna('standardize_X')
        if self.standardize_X:
            self.x_scaler = sklearn.preprocessing.StandardScaler()
        self.impute = self.suggest_hyperparam_to_optuna('impute')

        C = self.suggest_hyperparam_to_optuna('C')
        kernel = self.suggest_hyperparam_to_optuna('kernel')
        degree = self.suggest_hyperparam_to_optuna('degree')
        shrinking = self.suggest_hyperparam_to_optuna('shrinking')
        probability = self.suggest_hyperparam_to_optuna('probability')
        tol = self.suggest_hyperparam_to_optuna('tol')

        return sklearn.svm.SVC(
            C=C, kernel=kernel, degree=degree, random_state=42, shrinking=shrinking, probability=probability, tol=tol)

    def define_hyperparams_to_tune(self) -> dict:
        """
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the format.
        """
        return {
            'C': {
                'datatype': 'categorical',
                'list_of_values': [0.01, 0.1, 1, 10, 100],
            },
            'shrinking': {
                'datatype': 'categorical',
                'list_of_values': [True, False],
            },
            'probability': {
                'datatype': 'categorical',
                'list_of_values': [True, False],
            },
            'tol': {
                'datatype': 'float',
                'lower_bound': 0.0001,
                'upper_bound': 0.1,
                'log': True
            },
            'degree': {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 10
            },
            'kernel': {
                'datatype': 'categorical',
                'list_of_values': ["linear", "poly", "rbf", "sigmoid"]
            }
        }



