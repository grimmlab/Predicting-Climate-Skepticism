import xgboost
from models import base_model_
import sklearn
import torch
import abc
import optuna

class XGBoostClassifier(base_model_.BaseModel, abc.ABC):

    def __init__(self, optuna_trial: optuna.trial.Trial, dependent_variable: str):
        super().__init__(optuna_trial=optuna_trial, dependent_variable=dependent_variable)

    def define_model(self) -> xgboost.XGBModel:

        self.sampling = self.suggest_hyperparam_to_optuna('sampling')
        self.sampling_strategy = self.suggest_hyperparam_to_optuna('sampling_strategy')
        if self.sampling == "over" and self.sampling_strategy == "minmajority":
            self.sampling_strategy = "minority"
        if self.sampling == "under" and self.sampling_strategy == "minmajority":
            self.sampling_strategy = "majority"
        self.standardize_X = self.suggest_hyperparam_to_optuna('standardize_X')
        if self.standardize_X:
            self.x_scaler = sklearn.preprocessing.StandardScaler()

        max_depth = self.suggest_hyperparam_to_optuna('max_depth')
        gamma = self.suggest_hyperparam_to_optuna('gamma')
        reg_lambda = self.suggest_hyperparam_to_optuna('reg_lambda')
        reg_alpha = self.suggest_hyperparam_to_optuna('reg_alpha')
        n_estimators = self.suggest_hyperparam_to_optuna('n_estimators')
        learning_rate = self.suggest_hyperparam_to_optuna('learning_rate')
        subsample = self.suggest_hyperparam_to_optuna('subsample')
        colsample_bytree = self.suggest_hyperparam_to_optuna('colsample_bytree')

        return xgboost.XGBClassifier(
            random_state=42, verbosity=1, tree_method="hist", max_depth=max_depth, n_estimators=n_estimators,
            learning_rate=learning_rate, subsample=subsample, colsample_bytree=colsample_bytree, reg_alpha=reg_alpha,
            enable_categorical=True, device="cuda" if torch.cuda.is_available() else "cpu", gamma=gamma,
            reg_lambda=reg_lambda)

    def define_hyperparams_to_tune(self) -> dict:
        """
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the format.
        """
        return {
            'max_depth': {
                'datatype': 'int',
                'lower_bound': 0,
                'upper_bound': 1000,
            },
            'n_estimators': {
                'datatype': 'int',
                'lower_bound': 50,
                'upper_bound': 2000,
                'step': 50
            },
            'gamma': {
                'datatype': 'float',
                'lower_bound': 0.001,
                'upper_bound': 1.0,
                'log': True
            },
            'reg_lambda': {
                'datatype': 'float',
                'lower_bound': 0.1,
                'upper_bound': 100,
                'log': True
            },
            'reg_alpha': {
                'datatype': 'float',
                'lower_bound': 0.1,
                'upper_bound': 100,
                'log': True
            },
            'learning_rate': {
                'datatype': 'float',
                'lower_bound': 0.025,
                'upper_bound': 0.3,
                'step': 0.025
            },
            'subsample': {
                'datatype': 'float',
                'lower_bound': 0.05,
                'upper_bound': 1.0,
                'step': 0.05
            },
            'colsample_bytree': {
                'datatype': 'float',
                'lower_bound': 0.005,
                'upper_bound': 1.0,
                'step': 0.005
            }
        }



