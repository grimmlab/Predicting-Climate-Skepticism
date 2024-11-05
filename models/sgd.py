from models import base_model_
import sklearn
import abc
import optuna

class SGDClassifier(base_model_.BaseModel, abc.ABC):

    def __init__(self, optuna_trial: optuna.trial.Trial):
        super().__init__(optuna_trial=optuna_trial)

    def define_model(self) -> sklearn.linear_model.SGDClassifier:

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
        shuffle = self.suggest_hyperparam_to_optuna('shuffle')
        tol = self.suggest_hyperparam_to_optuna('tol')
        loss = self.suggest_hyperparam_to_optuna('loss')
        early_stopping = self.suggest_hyperparam_to_optuna('early_stopping')
        warm_start = self.suggest_hyperparam_to_optuna('warm_start')
        average = self.suggest_hyperparam_to_optuna('average')
        penalty = self.suggest_hyperparam_to_optuna('penalty')
        l1_ration = self.suggest_hyperparam_to_optuna('l1_ration')
        max_iter = self.suggest_hyperparam_to_optuna('max_iter')
        epsilon = self.suggest_hyperparam_to_optuna('epsilon')
        learning_rate = self.suggest_hyperparam_to_optuna('learning_rate')
        validation_fraction = self.suggest_hyperparam_to_optuna('validation_fraction')
        n_iter_no_change = self.suggest_hyperparam_to_optuna('n_iter_no_change')

        return sklearn.linear_model.SGDClassifier(
            loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ration, fit_intercept=fit_intercept, max_iter=max_iter,
            tol=tol, shuffle=shuffle, epsilon=epsilon, random_state=42, learning_rate=learning_rate,
            early_stopping=early_stopping, validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change,
            warm_start=warm_start, average=average)

    def define_hyperparams_to_tune(self) -> dict:
        """
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the format.
        """
        return {
            'alpha': {
                'datatype': 'float',
                'lower_bound': 0.00001,
                'upper_bound': 100,
                'log': True
            },
            'max_iter': {
                'datatype': 'int',
                'lower_bound': 10,
                'upper_bound': 100
            },
            'n_iter_no_change': {
                'datatype': 'int',
                'lower_bound': 1,
                'upper_bound': 10
            },
            'l1_ration': {
                'datatype': 'float',
                'lower_bound': 0.01,
                'upper_bound': 1,
            },
            'validation_fraction': {
                'datatype': 'float',
                'lower_bound': 0.1,
                'upper_bound': 1,
                'step': 0.1
            },
            'epsilon': {
                'datatype': 'float',
                'lower_bound': 0.1,
                'upper_bound': 1,
                'step': 0.1
            },
            'fit_intercept': {
                'datatype': 'categorical',
                'list_of_values': [False, True],
            },
            'learning_rate': {
                'datatype': 'categorical',
                'list_of_values': ["constant", "optimal", "invscaling", "adaptive"],
            },
            'penalty': {
                'datatype': 'categorical',
                'list_of_values': ["l2", "l1", "elasticnet", None],
            },
            'average': {
                'datatype': 'categorical',
                'list_of_values': [False, True],
            },
            'warm_start': {
                'datatype': 'categorical',
                'list_of_values': [False, True],
            },
            'early_stopping': {
                'datatype': 'categorical',
                'list_of_values': [False, True],
            },
            'shuffle': {
                'datatype': 'categorical',
                'list_of_values': [False, True],
            },
            'tol': {
                'datatype': 'float',
                'lower_bound': 0.00001,
                'upper_bound': 0.1,
                'log': True
            },
            'loss': {
                'datatype': 'categorical',
                'list_of_values': ["hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron", "squared_error",
                                   "huber", "epsilon_insensitive", "squared_epsilon_insensitive"]
            }
        }



