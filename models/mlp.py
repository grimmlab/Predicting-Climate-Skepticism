from models import base_model_
import sklearn.neural_network
import abc
import optuna

class MLPClassifier(base_model_.BaseModel, abc.ABC):

    def __init__(self, optuna_trial: optuna.trial.Trial):
        super().__init__(optuna_trial=optuna_trial)

    def define_model(self) -> sklearn.neural_network.MLPClassifier:

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

        hidden_layer_sizes = self.suggest_hyperparam_to_optuna('hidden_layer_sizes')
        activation = self.suggest_hyperparam_to_optuna('activation')
        solver = self.suggest_hyperparam_to_optuna('solver')
        alpha = self.suggest_hyperparam_to_optuna('alpha')
        learning_rate_init = self.suggest_hyperparam_to_optuna('learning_rate_init')
        shuffle = self.suggest_hyperparam_to_optuna('shuffle')
        early_stopping = self.suggest_hyperparam_to_optuna('early_stopping')
        n_iter_no_change = self.suggest_hyperparam_to_optuna('n_iter_no_change')
        max_iter = self.suggest_hyperparam_to_optuna('max_iter')

        return sklearn.neural_network.MLPClassifier(
            random_state=42, hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha,
            learning_rate_init=learning_rate_init, shuffle=shuffle, early_stopping=early_stopping,
            n_iter_no_change=n_iter_no_change, max_iter=max_iter)

    def define_hyperparams_to_tune(self) -> dict:
        """
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the format.
        """
        return {
            'early_stopping': {
                'datatype': 'categorical',
                'list_of_values': [True, False],
            },
            'solver': {
                'datatype': 'categorical',
                'list_of_values': ["sgd", "adam"],
            },
            'shuffle': {
                'datatype': 'categorical',
                'list_of_values': [True, False],
            },
            'activation': {
                'datatype': 'categorical',
                'list_of_values': ["identity", "logistic", "tanh", "relu"],
            },
            'alpha': {
                'datatype': 'float',
                'lower_bound': 0.00001,
                'upper_bound': 0.1,
                'log': True
            },
            'learning_rate_init': {
                'datatype': 'float',
                'lower_bound': 0.00001,
                'upper_bound': 0.1,
                'log': True
            },
            'hidden_layer_sizes': {
                'datatype': 'int',
                'lower_bound': 5,
                'upper_bound': 100
            },
            'n_iter_no_change': {
                'datatype': 'int',
                'lower_bound': 5,
                'upper_bound': 100
            },
            'max_iter': {
                'datatype': 'int',
                'lower_bound': 2000,
                'upper_bound': 10000,
                'step': 1000
            }
        }