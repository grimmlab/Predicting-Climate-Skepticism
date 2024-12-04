import pandas as pd
import numpy as np
import optuna
import datetime
import pathlib
import shutil
import csv
import shap
import copy
import matplotlib.pyplot as plt
import imblearn
import utils
from models import base_model_
import warnings
warnings.filterwarnings('ignore')
import sklearn
import joblib
import traceback

class Optimizer:

    def __init__(self, to_be_dropped_columns: list = None, data: pd.DataFrame = None, model_name: str = None):
        self.to_be_dropped_columns = to_be_dropped_columns
        self.data = data
        self.model_name = model_name
        self.save_dir = pathlib.Path(
            "results/" + model_name + "/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "/")

    def objective(self, trial: optuna.trial.Trial, train_val):

        unfitted_model: base_model_.BaseModel = utils.get_mapping_name_to_class()[self.model_name](optuna_trial=trial)

        self.save_dir.joinpath('temp').mkdir(parents=True, exist_ok=True)
        unfitted_model.save_model(
            path=self.save_dir.joinpath('temp'), filename='unfitted_model_trial' + str(trial.number))

        print("Params for Trial " + str(trial.number))
        print(trial.params)

        objective_values = []

        train_indexes, val_indexes = utils.get_indexes(df=train_val, target_column="climatedeniers")

        for fold in range(5):
            model = copy.deepcopy(unfitted_model)

            train, val = (
                train_val.iloc[train_indexes[fold]],
                train_val.iloc[val_indexes[fold]],
            )

            if model.impute:
                train, val = utils.impute_data(train, val)
            else:
                train, val = train.dropna(), val.dropna()

            train = utils.encode_data(train, self.save_dir)
            val = utils.encode_data(val, self.save_dir)

            for column in train.columns:
                if train[column].dtype == 'O':
                    train[column] = train[column].astype('category')
            for column in val.columns:
                if val[column].dtype == 'O':
                    val[column] = val[column].astype('category')

            if model.sampling != None:
                if model.sampling == "over":
                    sampler = imblearn.over_sampling.RandomOverSampler(
                        sampling_strategy=model.sampling_strategy, random_state=42)
                else:
                    sampler = imblearn.under_sampling.RandomUnderSampler(
                        sampling_strategy=model.sampling_strategy, random_state=42)
                train_X_sampled, train_y_sampled = sampler.fit_resample(
                    train.drop("climatedeniers", axis=1), train["climatedeniers"]
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
                self.clean_up_after_exception(
                    trial_number=trial.number, trial_params=trial.params, reason='model creation: ' + str(exc))
                raise optuna.exceptions.TrialPruned()

            objective_value = sklearn.metrics.matthews_corrcoef(val["climatedeniers"], y_pred)
            # objective_value = sklearn.metrics.recall_score(val["climatedeniers"], y_pred, pos_label=0)

            objective_values.append(objective_value)

        current_val_result = float(np.mean(objective_values))

        return current_val_result

    def clean_up_after_exception(self, trial_number: int, trial_params: dict, reason: str):
        if self.save_dir.joinpath('temp', 'unfitted_model_trial' + str(trial_number)).exists():
            self.save_dir.joinpath('temp', 'unfitted_model_trial' + str(trial_number)).unlink()

    def get_feature_importance(self, model) -> pd.DataFrame:
        feat_import_df = pd.DataFrame()
        feature_importances = model.feature_importances_
        sorted_idx = feature_importances.argsort()[::-1]
        feat_import_df["feature"] = self.data.drop(["climatedeniers"], axis=1).columns[sorted_idx]
        feat_import_df["feature_importance"] = feature_importances[sorted_idx]

        return feat_import_df

    def shap(self, final_model, test):
        explainer = shap.Explainer(final_model.predict, test)
        shap_values = explainer(test)

        joblib.dump(explainer, self.save_dir.joinpath('explainer.sav'))

        joblib.dump(shap_values, self.save_dir.joinpath('shapvalues.sav'))

        for feature in self.data.drop("climatedeniers", axis=1).columns:
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
            f.savefig(self.save_dir.joinpath(
                "partial_dependence_plots/shap.partial_dependence_plot" + feature + ".pdf"), format='pdf',
                bbox_inches='tight')

    def run_optimization(self):
        train_val, test = sklearn.model_selection.train_test_split(self.data, test_size=0.2, random_state=42, stratify=self.data["climatedeniers"])

        study = utils.create_new_study()
        study.optimize(lambda trial: self.objective(trial=trial, train_val=train_val), n_trials=100)
        print("Best matthews correlation score: " + str(study.best_trial.value))
        print("Best hyperparameters: " + str(study.best_params))

        # Move validation results and models of best trial
        files_to_keep_path = self.save_dir.joinpath('temp', '*trial' + str(study.best_trial.number) + '*')
        files_to_keep = pathlib.Path(files_to_keep_path.parent).expanduser().glob(files_to_keep_path.name)
        for file in files_to_keep:
            shutil.copyfile(file, self.save_dir.joinpath(file.name))
        shutil.rmtree(self.save_dir.joinpath('temp'))

        final_model = joblib.load(self.save_dir.joinpath('unfitted_model_trial' + str(study.best_trial.number)))

        if final_model.impute:
            train_val, test = utils.impute_data(train_val, test)
        else:
            train_val, test = train_val.dropna(), test.dropna()

        train_val = utils.encode_data(train_val, self.save_dir)
        test = utils.encode_data(test, self.save_dir)

        for column in train_val.columns:
            if train_val[column].dtype == 'O':
                train_val[column] = train_val[column].astype('category')

        final_model.retrain(train_val)

        joblib.dump(final_model, self.save_dir.joinpath('best_model_' + self.model_name))

        for column in test.columns:
            if test[column].dtype == 'O':
                test[column] = test[column].astype('category')

        predictions = final_model.predict(test)

        np.savetxt(self.save_dir.joinpath('predictions.csv'), predictions, delimiter=",")
        np.savetxt(self.save_dir.joinpath('test.csv'), test, delimiter=",", fmt='%s')
        with open(self.save_dir.joinpath('best_params.csv'), 'w+') as f:
            w = csv.writer(f)
            w.writerows(study.best_params.items())
        print(sklearn.metrics.classification_report(y_true=test["climatedeniers"], y_pred=predictions))
        print(sklearn.metrics.matthews_corrcoef(y_true=test["climatedeniers"], y_pred=predictions))
        disp = sklearn.metrics.ConfusionMatrixDisplay.from_predictions(
            test["climatedeniers"], predictions, cmap=plt.cm.Blues)

        print(disp.confusion_matrix)

        plt.show()

        if hasattr(final_model, "feature_importances_"):
            feat_import_df = self.get_feature_importance(model=final_model.model)
            feat_import_df.to_csv(
                self.save_dir.joinpath(
                    "final_model_feature_importances_" + self.model_name + "_" + str(self.to_be_dropped_columns) + ".csv"),
                sep=",", decimal=".", float_format="%.10f", index=False)

        self.shap(final_model, test)

        with open(self.save_dir.joinpath('matthews_corrcoef' + str(self.to_be_dropped_columns) + '.txt'), 'w') as f:
            f.write(
                "matthews_corrcoef: %.2f" % sklearn.metrics.matthews_corrcoef(
                    y_true=test["climatedeniers"], y_pred=predictions))