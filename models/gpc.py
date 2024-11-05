from models import base_model_
import sklearn
import abc
import optuna
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, RationalQuadratic, Matern, PairwiseKernel
import sklearn.gaussian_process

class GaussianProcessClassifier(base_model_.BaseModel, abc.ABC):

    def __init__(self, optuna_trial: optuna.trial.Trial):
        super().__init__(optuna_trial=optuna_trial)

    def define_model(self) -> sklearn.gaussian_process.GaussianProcessClassifier:

        self.sampling = self.suggest_hyperparam_to_optuna('sampling')
        self.sampling_strategy = self.suggest_hyperparam_to_optuna('sampling_strategy')
        if self.sampling == "over" and self.sampling_strategy == "minmajority":
            self.sampling_strategy = "minority"
        if self.sampling == "under" and self.sampling_strategy == "minmajority":
            self.sampling_strategy = "majority"
        self.standardize_X = True
        self.x_scaler = sklearn.preprocessing.StandardScaler()
        self.impute = self.suggest_hyperparam_to_optuna('impute')

        warm_start = self.suggest_hyperparam_to_optuna('warm_start')
        multi_class = self.suggest_hyperparam_to_optuna('multi_class')
        kernel = self.suggest_hyperparam_to_optuna('kernel')

        return sklearn.gaussian_process.GaussianProcessClassifier(
            kernel=kernel, max_iter_predict=100000, warm_start=warm_start, random_state=42, multi_class=multi_class)

    def define_hyperparams_to_tune(self) -> dict:
        """
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the format.
        """
        return {
            'warm_start': {
                'datatype': 'categorical',
                'list_of_values': [False, True],
            },
            'multi_class': {
                'datatype': 'categorical',
                'list_of_values': ["one_vs_rest", "one_vs_one"],
            },
            'kernel': {
                'datatype': 'categorical',
                'list_of_values': [WhiteKernel(), RBF(), RationalQuadratic(), Matern(), PairwiseKernel()],
            }
        }



