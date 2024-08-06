import pandas as pd
import numpy as np
import optuna
import datetime
import xgboost
import csv
import shap
from sklearn.model_selection import train_test_split
import copy
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, matthews_corrcoef
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from utils import impute_data, encode_data, get_indexes, run_optuna_optimization
import argparse

class ClimateDeniersClassifier:

    def preprocess_data(self, impute):

        if impute:
            self.data = impute_data(self.data)
        else:
            self.data.dropna(inplace=True)

        for column in self.data.columns:
            if self.data[column].dtype == 'O':
                self.data[column] = self.data[column].astype('category')

        # data = encode_data(data)

    def retrain(self, retrain: pd.DataFrame, model):
        x_train = retrain.drop(target_column, axis=1)
        y_train = retrain[target_column]
        model.fit(x_train, y_train)

    def predict(self, X_in: pd.DataFrame, model) -> np.array:
        X_in = X_in.drop(target_column, axis=1)
        prediction = model.predict(X_in)
        return prediction

    def train_val_loop(self, train: pd.DataFrame, val: pd.DataFrame, model) -> np.array:
        self.retrain(train, model)
        return self.predict(X_in=val, model=model)

    def objective(self, trial: optuna.trial.Trial):

        max_depth = trial.suggest_int("max_depth", 0, 1000, step=1)
        n_estimators = trial.suggest_int("n_estimators", 50, 2000, step=50)
        learning_rate = trial.suggest_float("learning_rate", 0.025, 0.3, step=0.025)
        subsample = trial.suggest_float("subsample", 0.05, 1.0, step=0.05)
        colsample_bytree = trial.suggest_float(
            "colsample_bytree", 0.005, 1.0, step=0.005
        )
        sampling = trial.suggest_categorical(
            "sampling",
            [None, "over", "under"])
        sampling_strategy = trial.suggest_categorical(
            "sampling_strategy",
            ["auto", "all", "minmajority", "not majority", "not minority"])
        if sampling == "over" and sampling_strategy == "minmajority":
            sampling_strategy = "minority"
        if sampling == "under" and sampling_strategy == "minmajority":
            sampling_strategy = "majority"
        max_leaves = trial.suggest_int("max_leaves", 0, 1000, step=1)
        objective = trial.suggest_categorical(
            "objective",
            ["binary:logistic", "binary:logitraw"])
        impute = trial.suggest_categorical(
            "impute",
            [True, False])

        unfitted_model = xgboost.XGBClassifier(
            random_state=42,
            verbosity=1,
            tree_method="auto",
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            max_leaves=max_leaves,
            objective=objective,
            enable_categorical=True,
        )

        objective_values = []

        self.data = data.copy()

        self.preprocess_data(impute=impute)
        self.train_val, self.test = train_test_split(self.data, test_size=0.2, random_state=42)

        train_indexes, val_indexes = get_indexes(df=self.train_val, target_column=target_column)

        for fold in range(5):
            model = copy.deepcopy(unfitted_model)

            train, val = (
                self.train_val.iloc[train_indexes[fold]],
                self.train_val.iloc[val_indexes[fold]],
            )

            if sampling != None:
                if sampling == "over":
                    sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
                else:
                    sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
                train_X_sampled, train_y_sampled = sampler.fit_resample(
                    train.drop(target_column, axis=1), train[target_column]
                )
                train = pd.concat([train_X_sampled, train_y_sampled], axis=1)
                train = train.sample(frac=1).reset_index(drop=True)

            y_pred = self.train_val_loop(train=train, val=val, model=model)

            objective_value = matthews_corrcoef(val[target_column], y_pred)

            objective_values.append(objective_value)

        current_val_result = float(np.mean(objective_values))

        return current_val_result

    def create_new_study(self) -> optuna.study.Study:
        study_name = (
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                + "_"
                + "-MODEL"
                + "XGBoost"
                + "-TRIALS"
                + str(200)
        )
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.PercentilePruner(percentile=80, n_min_trials=20),
            load_if_exists=False,
        )

        return study

    def get_feature_importance(self, model) -> pd.DataFrame:
        feat_import_df = pd.DataFrame()
        feature_importances = model.feature_importances_
        sorted_idx = feature_importances.argsort()[::-1]
        feat_import_df["feature"] = climate_deniers_classifier.data.drop([target_column], axis=1).columns[sorted_idx]
        feat_import_df["feature_importance"] = feature_importances[sorted_idx]

        return feat_import_df

    def shap(self):
        explainer = shap.Explainer(self.best_model.predict, climate_deniers_classifier.test.drop(target_column, axis=1))
        shap_values = explainer(climate_deniers_classifier.test.drop(target_column, axis=1))

        filename_expl = 'explainer.sav'
        pickle.dump(explainer, open("explainer/" + filename_expl, 'wb'))

        filename = 'shapvalues.sav'
        pickle.dump(shap_values, open("shapvalues/" + filename, 'wb'))

        for feature in climate_deniers_classifier.data.drop(target_column, axis=1).columns:
            shap.partial_dependence_plot(
                feature,
                self.best_model.predict,
                climate_deniers_classifier.test.drop(target_column, axis=1),
                ice=False,
                model_expected_value=True,
                feature_expected_value=True,
                show=False
            )
            f = plt.gcf()
            f.savefig("partial_dependence_plots/shap.partial_dependence_plot" + feature + ".pdf", format='pdf',
                      bbox_inches='tight')

    def run_pipeline(self):
        best_params = run_optuna_optimization(objective=self.objective)

        self.best_model = xgboost.XGBClassifier(**best_params, enable_categorical=True)

        self.retrain(self.train_val, self.best_model)

        pickle.dump(self.best_model, open("models/model_xgb", 'wb'))

        predictions = self.predict(X_in=self.test, model=self.best_model)

        np.savetxt("predictions/predictions_xgb" + str(to_be_dropped_columns) + ".csv", predictions, delimiter=",")
        self.test.to_csv("testsets/test_xgb.csv", index=False,
                                               sep=',', decimal='.', float_format='%.10f')
        with open('best_params/best_params_xgb' + str(to_be_dropped_columns) + '.csv', 'w') as f:
            w = csv.writer(f)
            w.writerows(best_params.items())
        print(classification_report(y_true=self.test[target_column], y_pred=predictions))
        print(matthews_corrcoef(y_true=self.test[target_column], y_pred=predictions))
        with open('matthews_corrcoef/matthews_corrcoef_xgb' + str(to_be_dropped_columns) + '.txt', 'w') as f:
            f.write("matthews_corrcoef: %.2f" % matthews_corrcoef(y_true=self.test[target_column],
                                                                  y_pred=predictions))

        feat_import_df = self.get_feature_importance(model=self.best_model)
        feat_import_df.to_csv(
            "final_model_feature_importances/final_model_feature_importances_xgb" + str(to_be_dropped_columns) + ".csv",
            sep=",",
            decimal=".",
            float_format="%.10f",
            index=False,
        )

        # self.shap()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-tbdc', '--to_be_dropped_columns', nargs='*', help='Define columns to drop.', required=True)
    parser.add_argument("-t", "--target_column", type=str, default="climatedeniers_1", help="specify target.")
    parser.add_argument("-d", "--dataset", type=str, default="Datasets/dataset_preprocessed.csv", help="specify dataset")

    args = parser.parse_args()

    to_be_dropped_columns = args.to_be_dropped_columns
    target_column = args.target_column
    dataset = args.dataset

    data = pd.read_csv(
        dataset,
        dtype={
            'LocationLatitude': 'float64', 'LocationLongitude': 'float64', 'gender': 'category', 'age': 'int64',
            'income': 'category', 'region_1': 'category', 'region_2': 'category', 'region_3': 'category',
            'plz': 'category', 'education': 'category', 'marital_status': 'category', 'children': 'category',
            'job': 'category', 'job_field_1': 'category', 'job_field_2': 'category', 'religion': 'category',
            'religion_practice': 'category', 'migration_b_germany': 'category', 'migration_b_g_city': 'category',
            'migration_b_g_drill_1': 'category', 'migration_b_g_drill_2': 'category',
            'migration_b_g_drill_3': 'category', 'migration_s_germany': 'category',
            'migration_s_g_city': 'category', 'migration_s_g_drill_1': 'category',
            'migration_s_g_drill_2': 'category', 'migration_s_g_drill_3': 'category',
            'migration_region': 'category', 'subject_well_being_1': 'float64', 'innovation': 'category',
            'b_q1_own_opinion': 'category', 'b_q1_point_guess': 'category', 'b_q1_distribution_1': 'int64',
            'b_q1_distribution_2': 'int64', 'b_q1_distribution_3': 'int64', 'b_q1_distribution_4': 'int64',
            'b_q1_distribution_5': 'int64', 'b_q2_population': 'category', 'b_q2_point_guess': 'category',
            'b_q2_distribution_1': 'float64', 'b_q2_distribution_2': 'float64', 'b_q2_distribution_3': 'float64',
            'b_q2_distribution_4': 'float64', 'b_q2_distribution_5': 'float64', 'e_q1_own_opinion': 'category',
            'e_q1_point_guess': 'category', 'e_q1_distribution_1': 'float64', 'e_q1_distribution_2': 'float64',
            'e_q1_distribution_3': 'float64', 'e_q1_distribution_4': 'float64', 'e_q1_distribution_5': 'float64',
            'e_q2_population': 'category', 'e_q2_point_guess': 'category', 'e_q2_distribution_1': 'float64',
            'e_q2_distribution_2': 'float64', 'e_q2_distribution_3': 'float64', 'e_q2_distribution_4': 'float64',
            'e_q2_distribution_5': 'float64', 'GPS_71': 'category', 'GPS_72_A': 'category', 'GPS_72_B': 'category',
            'GPS_72_C': 'category', 'GPS_72_D': 'category', 'GPS_73_A': 'category', 'GPS_73_B': 'category',
            'GPS_73_C': 'category', 'GPS_73_D': 'category', 'GPS_73_E': 'category', 'GPS_74': 'category',
            'GPS_105': 'category', 'GPS_106': 'float64', 'GPS_107': 'category', 'climate_know': 'category',
            'climate_eb_problem_1': 'float64', 'climate_eb_responsib': 'category',
            'climate_eb_renewable': 'category', 'climate_eb_efficient': 'category', 'climate_eb_ets': 'category',
            'climate_eb_statement_1': 'category', 'climate_eb_statement_2': 'category',
            'climate_eb_statement_3': 'category', 'climate_eb_statement_4': 'category',
            'climate_eb_statement_5': 'category', 'climate_eb_statement_6': 'category',
            'climate_flood_affect': 'category', 'climate_risk': 'category', 'climate_statements_1': 'category',
            'climate_statements_2': 'category', 'climate_statements_3': 'category',
            'climate_statements_4': 'category', 'climate_statements_5': 'category',
            'climate_statements_6': 'category', 'climate_statements_7': 'category',
            'climate_statements_8': 'category', 'climate_statements_9': 'category',
            'climate_statements_10': 'category', 'climate_statements_11': 'category',
            'climate_statements_12': 'category', 'climate_statements_13': 'category',
            'climate_statements_14': 'category', 'climate_statements_15': 'category',
            'climate_policies_1': 'category', 'climate_policies_2': 'category', 'climate_policies_3': 'category',
            'climate_policies_4': 'category', 'climate_pol_engage_1': 'category', 'climate_pol_engage_2': 'category',
            'climate_pol_engage_3': 'category', 'climate_pol_engage_4': 'category',
            'climate_pol_engage_5': 'category', 'climate_pol_engage_6': 'category',
            'climate_pol_engage_7': 'category', 'climate_trust_1': 'category', 'climate_trust_2': 'category',
            'climate_trust_3': 'category', 'climate_trust_4': 'category', 'climate_trust_5': 'category',
            'climate_trust_6': 'category', 'climate_trust_7': 'category', 'bioecon_products': 'category',
            'moral_right_wrong_1': 'category', 'moral_right_wrong_2': 'category', 'moral_right_wrong_3': 'category',
            'moral_right_wrong_4': 'category', 'moral_right_wrong_5': 'category', 'moral_right_wrong_6': 'category',
            'moral_right_wrong_7': 'category', 'moral_right_wrong_8': 'category', 'moral_right_wrong_9': 'category',
            'moral_right_wrong_10': 'category', 'moral_right_wrong_11': 'category',
            'moral_right_wrong_12': 'category', 'moral_right_wrong_13': 'category',
            'moral_right_wrong_14': 'category', 'moral_right_wrong_15': 'category',
            'moral_right_wrong_16': 'category', '\xa0moral_statements_1': 'category',
            '\xa0moral_statements_2': 'category', '\xa0moral_statements_3': 'category',
            '\xa0moral_statements_4': 'category', '\xa0moral_statements_5': 'category',
            '\xa0moral_statements_6': 'category', '\xa0moral_statements_7': 'category',
            '\xa0moral_statements_8': 'category', '\xa0moral_statements_9': 'category',
            '\xa0moral_statements_10': 'category', '\xa0moral_statements_11': 'category',
            '\xa0moral_statements_12': 'category', '\xa0moral_statements_13': 'category',
            '\xa0moral_statements_14': 'category', '\xa0moral_statements_15': 'category',
            '\xa0moral_statements_16': 'category', 'politics_orientation_1': 'float64', 'politics_vote': 'category',
            'politics_climate': 'category', 'soc_inequ_climate': 'category', 'covid_coping': 'category',
            'covid_finance': 'category', 'income.1': 'category', 'region_prefix': 'category',
            'unemployment_rate': 'category', 'unemployment_count': 'int64', 'patent': 'int64',
            'treat_info': 'category', 'endline_info': 'category', 'adequacy': 'category', 'goal': 'category',
            'climatedeniers_1': 'int64'
        }
    )

    data = data[data.climatedeniers_1 != 2]

    data = data.drop(columns=to_be_dropped_columns)

    if target_column == "climatedeniers_2":
        data.drop(columns=["climate_eb_problem", "climate_state_convinced", "climatedeniers_1", "climate_risk",
                           "d_q1_own_opinion"], inplace=True)
    elif target_column == "climatedeniers_1":
        data = data.drop(columns=["climate_risk", "climate_eb_problem_1", "politics_vote", "climate_eb_responsib"])

    data = data[data.columns.drop(list(data.filter(regex='climate_pol_engage_')))]
    data = data[data.columns.drop(list(data.filter(regex='climate_policies_')))]
    data = data[data.columns.drop(list(data.filter(regex='climate_statements_')))]

    climate_deniers_classifier = ClimateDeniersClassifier()
    climate_deniers_classifier.run_pipeline()


