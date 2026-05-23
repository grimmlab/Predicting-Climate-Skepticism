import pandas as pd
import optuna
import csv
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
import sklearn
import imblearn
import pathlib
import shutil
import numpy as np
import utils

optuna.logging.set_verbosity(optuna.logging.WARNING)

class Optimizer:

    def __init__(self, data: pd.DataFrame = None, save_dir: pathlib.Path = None, dependent_variable: str = None):
        self.data = data
        self.save_dir = save_dir
        self.dependent_variable = dependent_variable
        self.folds = 5

    def objective(self, trial: optuna.trial.Trial, train_val):

        if self.dependent_variable != "climate_eb_problem":
            sampling = trial.suggest_categorical('sampling', [None, "over", "under"])
            if sampling != None:
                sampling_strategy = trial.suggest_categorical('sampling_strategy',
                                                              ["all", "minmajority", "not majority", "not minority"])
                if sampling == "over" and sampling_strategy == "minmajority":
                    sampling_strategy = "minority"
                if sampling == "under" and sampling_strategy == "minmajority":
                    sampling_strategy = "majority"

        if utils.check_params_for_duplicate(current_params=trial.params, study=self.study):
            print('Trial params are a duplicate.')
            utils.clean_up_after_exception(trial_number=trial.number, save_dir=self.save_dir)
            raise optuna.exceptions.TrialPruned()

        self.save_dir.joinpath('temp').mkdir(parents=True, exist_ok=True)

        objective_values = []

        train_indexes, val_indexes = utils.get_indexes(df=train_val, target_column=self.dependent_variable, n_splits=self.folds)

        for fold in range(self.folds):

            train, val = (train_val.iloc[train_indexes[fold]], train_val.iloc[val_indexes[fold]])

            train, val = utils.impute_data(train, val)

            if self.dependent_variable != "climate_eb_problem":
                if sampling != None:
                    if sampling == "over":
                        sampler = imblearn.over_sampling.RandomOverSampler(sampling_strategy=sampling_strategy,
                                                                           random_state=42)
                    else:
                        sampler = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=sampling_strategy,
                                                                             random_state=42)
                    train_X_sampled, train_y_sampled = sampler.fit_resample(
                        train.drop(self.dependent_variable, axis=1), train[self.dependent_variable]
                    )
                    train = pd.concat([train_X_sampled, train_y_sampled], axis=1)
                    train = train.sample(frac=1).reset_index(drop=True)

                model = sm.OLS(exog=train.drop(self.dependent_variable, axis=1),
                                   endog=train[self.dependent_variable])
            else:
                model = sm.OLS(exog=train.drop(self.dependent_variable, axis=1), endog=train[self.dependent_variable])
            try:
                model = model.fit()
            except np.linalg.LinAlgError:
                print('np.linalg.LinAlgError!')
                utils.clean_up_after_exception(trial_number=trial.number, save_dir=self.save_dir)
                raise optuna.exceptions.TrialPruned()

            y_pred = model.predict(exog=val.drop(self.dependent_variable, axis=1) if self.dependent_variable in val.columns else val)

            if self.dependent_variable != "climate_eb_problem":
                objective_value = sklearn.metrics.r2_score(val[self.dependent_variable], y_pred)
            else:
                objective_value = sklearn.metrics.r2_score(val[self.dependent_variable], y_pred)

            objective_values.append(objective_value)

        current_val_result = float(np.mean(objective_values))

        return current_val_result

    def run_optimization(self):
        train_val, test = sklearn.model_selection.train_test_split(
            self.data, test_size=0.2, random_state=42, stratify=self.data[self.dependent_variable])

        self.study = utils.create_new_study()
        self.study.optimize(lambda trial: self.objective(trial=trial, train_val=train_val), n_trials=30, show_progress_bar=True)
        print(f"Best score: {self.study.best_trial.value}")

        # Move validation results and models of best trial
        files_to_keep_path = self.save_dir.joinpath('temp', f'*trial {self.study.best_trial.number}*')
        files_to_keep = pathlib.Path(files_to_keep_path.parent).expanduser().glob(files_to_keep_path.name)
        for file in files_to_keep:
            shutil.copyfile(file, self.save_dir.joinpath(file.name))
        shutil.rmtree(self.save_dir.joinpath('temp'))

        train_val, test = utils.impute_data(train_val, test)

        if self.dependent_variable != "climate_eb_problem":
            sampling = self.study.best_params["sampling"]
            if sampling != None:
                sampling_strategy = self.study.best_params["sampling_strategy"]
                if sampling == "over" and sampling_strategy == "minmajority":
                    sampling_strategy = "minority"
                if sampling == "under" and sampling_strategy == "minmajority":
                    sampling_strategy = "majority"

                if sampling == "over":
                    sampler = imblearn.over_sampling.RandomOverSampler(sampling_strategy=sampling_strategy,
                                                                       random_state=42)
                else:
                    sampler = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=sampling_strategy,
                                                                         random_state=42)
                train_val_X_sampled, train_val_y_sampled = sampler.fit_resample(
                    train_val.drop(self.dependent_variable, axis=1), train_val[self.dependent_variable]
                )
                train_val = pd.concat([train_val_X_sampled, train_val_y_sampled], axis=1)
                train_val = train_val.sample(frac=1).reset_index(drop=True)

            final_model = sm.OLS(exog=train_val.drop(self.dependent_variable, axis=1), endog=train_val[self.dependent_variable])
        else:
            final_model = sm.OLS(exog=train_val.drop(self.dependent_variable, axis=1), endog=train_val[self.dependent_variable])

        final_model = final_model.fit()
        predictions = final_model.predict(
            exog=test.drop(self.dependent_variable, axis=1) if self.dependent_variable in test.columns else test)

        final_model.pvalues.to_csv(self.save_dir.joinpath('pvalues.csv'), header=False)
        np.savetxt(self.save_dir.joinpath('predictions.csv'), predictions, delimiter=",")
        test.to_csv(self.save_dir.joinpath('test.csv'))
        with open(self.save_dir.joinpath('best_params.csv'), 'w+') as f:
            w = csv.writer(f)
            w.writerows(self.study.best_params.items())

        shap_values = utils.compute_shap(final_model, train_val, test, dependent_variable=self.dependent_variable, save_dir=self.save_dir)

        with open(self.save_dir.joinpath('score.txt'), 'w') as f:
            if self.dependent_variable != "climate_eb_problem":
                f.write(str(sklearn.metrics.r2_score(y_true=test[self.dependent_variable], y_pred=predictions)))
            else:
                f.write(str(sklearn.metrics.r2_score(y_true=test[self.dependent_variable], y_pred=predictions)))

        pd.DataFrame(shap_values.values, columns=shap_values.feature_names).to_csv(self.save_dir.joinpath('shap_values.csv'), index=False)

        return predictions, shap_values