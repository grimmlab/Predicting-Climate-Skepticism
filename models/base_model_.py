import abc
import optuna
import joblib
import pandas as pd
import numpy as np
import pathlib

class BaseModel(abc.ABC):

    # Constructor super class #
    def __init__(self, optuna_trial: optuna.trial.Trial, dependent_variable: str):

        self.all_hyperparams = self.define_hyperparams_to_tune()
        self.optuna_trial = optuna_trial
        self.model = self.define_model()
        self.dependent_variable = dependent_variable

    # Methods required by each child class #
    @abc.abstractmethod
    def define_hyperparams_to_tune(self) -> dict:
        """
        Defining the hyperparameters to tune
        """

    @abc.abstractmethod
    def define_model(self):
        """
        Defining the model
        """

    def retrain(self, retrain: pd.DataFrame):
        x_train = retrain.drop(self.dependent_variable, axis=1)
        y_train = retrain[self.dependent_variable]
        self.model.fit(x_train, y_train)

    def predict(self, X_in: pd.DataFrame) -> np.array:
        X_in = X_in.drop(self.dependent_variable, axis=1) if self.dependent_variable in X_in.columns else X_in
        prediction = self.model.predict(X_in)
        return prediction

    def train_val_loop(self, train: pd.DataFrame, val: pd.DataFrame) -> np.array:
        self.retrain(train)
        return self.predict(X_in=val)

    def suggest_hyperparam_to_optuna(self, hyperparam_name: str = None):
        if hyperparam_name in self.all_hyperparams:
            spec = self.all_hyperparams[hyperparam_name]
        else:
            raise Exception(hyperparam_name + ' not found in all_hyperparams dictionary.')

        if hyperparam_name in self.optuna_trial.params:
            counter = 1
            while True:
                current_name = hyperparam_name + '_' + str(counter)
                if current_name not in self.optuna_trial.params:
                    optuna_param_name = current_name
                    break
                counter += 1
        else:
            optuna_param_name = hyperparam_name

        # Read dict with specification for the hyperparamater and suggest it to the trial
        if spec['datatype'] == 'categorical':
            if 'list_of_values' not in spec:
                raise Exception(
                    '"list of values" for ' + hyperparam_name + ' not in hyperparams_dict. '
                    'Check define_hyperparams_to_tune() of the model.'
                )
            suggested_value = \
                self.optuna_trial.suggest_categorical(name=optuna_param_name, choices=spec['list_of_values'])
        elif spec['datatype'] in ['float', 'int']:
            if 'step' in spec:
                step = spec['step']
            else:
                step = None if spec['datatype'] == 'float' else 1
            log = spec['log'] if 'log' in spec else False
            if 'lower_bound' not in spec or 'upper_bound' not in spec:
                raise Exception(
                    '"lower_bound" or "upper_bound" for ' + hyperparam_name + ' not in all_hyperparams. '
                    'Check define_hyperparams_to_tune() of the model.'
                )
            if spec['datatype'] == 'int':
                suggested_value = self.optuna_trial.suggest_int(
                    name=optuna_param_name, low=spec['lower_bound'], high=spec['upper_bound'], step=step, log=log
                )
            else:
                suggested_value = self.optuna_trial.suggest_float(
                    name=optuna_param_name, low=spec['lower_bound'], high=spec['upper_bound'], step=step, log=log
                )
        else:
            raise Exception(
                spec['datatype'] + ' is not a valid parameter. Check define_hyperparams_to_tune() of the model.'
            )
        return suggested_value

    def save_model(self, path: pathlib.Path, filename: str):
        joblib.dump(self, path.joinpath(filename), compress=3)