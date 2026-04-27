import pandas as pd
import numpy as np
import optuna
import pathlib
import shutil
import csv
import shap
import copy
import matplotlib.pyplot as plt
import utils
from models import base_model_
import warnings
warnings.filterwarnings('ignore')
import sklearn
import joblib
import traceback
import imblearn

optuna.logging.set_verbosity(optuna.logging.WARNING)

class Optimizer:

    def __init__(self, data: pd.DataFrame = None, save_dir: pathlib.Path = None, dependent_variable: str = None):
        self.data = data
        self.save_dir = save_dir
        self.dependent_variable = dependent_variable
        self.folds = 5

    def objective(self, trial: optuna.trial.Trial, train_val):

        model_name_task = "xgb_regressor" if self.dependent_variable == "climate_eb_problem" else "xgb_classifier"

        unfitted_model: base_model_.BaseModel = utils.get_mapping_name_to_class()[model_name_task](optuna_trial=trial, dependent_variable=self.dependent_variable)

        self.save_dir.joinpath('temp').mkdir(parents=True, exist_ok=True)
        unfitted_model.save_model(path=self.save_dir.joinpath('temp'), filename=f'unfitted_model_trial {trial.number}')

        objective_values = []

        train_indexes, val_indexes = utils.get_indexes(df=train_val, target_column=self.dependent_variable, n_splits=self.folds)

        for fold in range(self.folds):
            model = copy.deepcopy(unfitted_model)

            train, val = (
                train_val.iloc[train_indexes[fold]],
                train_val.iloc[val_indexes[fold]],
            )

            if hasattr(model, 'sampling') and hasattr(model, 'sampling_strategy'):
                if model.sampling == "over":
                    sampler = imblearn.over_sampling.RandomOverSampler(
                        sampling_strategy=model.sampling_strategy, random_state=42)
                else:
                    sampler = imblearn.under_sampling.RandomUnderSampler(
                        sampling_strategy=model.sampling_strategy, random_state=42)
                train_X_sampled, train_y_sampled = sampler.fit_resample(
                    train.drop(self.dependent_variable, axis=1), train[self.dependent_variable]
                )
                train = pd.concat([train_X_sampled, train_y_sampled], axis=1)
                train = train.sample(frac=1).reset_index(drop=True)

            try:
                y_pred = model.train_val_loop(train=train, val=val)
            except ValueError as exc:
                print(traceback.format_exc())
                print(exc)
                print(trial.params)
                print('Trial failed. Error in model creation.')
                self.clean_up_after_exception(trial_number=trial.number)
                raise optuna.exceptions.TrialPruned()

            if self.dependent_variable != "climate_eb_problem":
                objective_value = sklearn.metrics.matthews_corrcoef(val[self.dependent_variable], y_pred)
            else:
                objective_value = sklearn.metrics.r2_score(val[self.dependent_variable], y_pred)

            objective_values.append(objective_value)

        current_val_result = float(np.mean(objective_values))

        return current_val_result

    def clean_up_after_exception(self, trial_number: int):
        if self.save_dir.joinpath('temp', f'unfitted_model_trial{trial_number}').exists():
            self.save_dir.joinpath('temp', f'unfitted_model_trial {trial_number}').unlink()

    def shap(self, final_model, train_val, test):
        explainer = shap.Explainer(final_model.predict, train_val.drop(self.dependent_variable, axis=1))
        shap_values = explainer(test.drop(self.dependent_variable, axis=1))

        joblib.dump(explainer, self.save_dir.joinpath('explainer.sav'))

        joblib.dump(shap_values, self.save_dir.joinpath('shapvalues.sav'))
        """
        for feature in self.data.drop(self.dependent_variable, axis=1).columns:
            shap.partial_dependence_plot(
                feature,
                final_model.predict,
                test,
                ice=False,
                model_expected_value=True,
                feature_expected_value=True,
                show=False
            )
            f = plt.gcf()
            self.save_dir.joinpath('partial_dependence_plots').mkdir(parents=True, exist_ok=True)
            f.savefig(self.save_dir.joinpath(f"partial_dependence_plots/shap.partial_dependence_plot_{feature}.pdf"),
                      format='pdf', bbox_inches='tight')
        """
        return explainer(test)

    def run_optimization(self):
        train_val, test = sklearn.model_selection.train_test_split(
            self.data, test_size=0.2, random_state=42, stratify=self.data[self.dependent_variable])

        study = utils.create_new_study()
        study.optimize(lambda trial: self.objective(trial=trial, train_val=train_val), n_trials=30, show_progress_bar=True)
        print(f"Best score: {study.best_trial.value}")

        # Move validation results and models of best trial
        files_to_keep_path = self.save_dir.joinpath('temp', f'*trial {study.best_trial.number}*')
        files_to_keep = pathlib.Path(files_to_keep_path.parent).expanduser().glob(files_to_keep_path.name)
        for file in files_to_keep:
            shutil.copyfile(file, self.save_dir.joinpath(file.name))
        shutil.rmtree(self.save_dir.joinpath('temp'))

        final_model = joblib.load(self.save_dir.joinpath(f'unfitted_model_trial {study.best_trial.number}'))

        final_model.retrain(train_val)

        joblib.dump(final_model, self.save_dir.joinpath(f'best_model'))

        predictions = final_model.predict(test)

        np.savetxt(self.save_dir.joinpath('predictions.csv'), predictions, delimiter=",")
        test.to_csv(self.save_dir.joinpath('test.csv'))
        with open(self.save_dir.joinpath('best_params.csv'), 'w+') as f:
            w = csv.writer(f)
            w.writerows(study.best_params.items())

        shap_values = self.shap(final_model, train_val, test)

        with open(self.save_dir.joinpath('score.txt'), 'w') as f:
            if self.dependent_variable != "climate_eb_problem":
                f.write(str(sklearn.metrics.matthews_corrcoef(y_true=test[self.dependent_variable], y_pred=predictions)))
            else:
                f.write(str(sklearn.metrics.r2_score(y_true=test[self.dependent_variable], y_pred=predictions)))

        np.savetxt(self.save_dir.joinpath('shap_values.csv'), shap_values.values, delimiter=",")

        return predictions, shap_values