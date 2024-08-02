import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import optuna
import datetime
from sklearn.neural_network import MLPClassifier
import csv
import shap
from sklearn.model_selection import train_test_split
import copy
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report, matthews_corrcoef
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer
import traceback
import sklearn
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

class ClimateDeniersClassifier:

    def preprocess_data(self, impute):
        data = pd.read_csv("Climate_Deniers_15K.csv")

        data["climatedeniers"] = data["climatedeniers_2"] + data["climatedeniers_1"]
        data.loc[(data['climatedeniers_1'] == 0) & (data['climatedeniers_2'] == 1), ['climatedeniers']] = 1
        data.loc[(data['climatedeniers_1'] == 1) & (data['climatedeniers_2'] == 0), ['climatedeniers']] = 2
        data.loc[(data['climatedeniers_1'] == 1) & (data['climatedeniers_2'] == 1), ['climatedeniers']] = 3
        self.drop_columns(data, ["climate_eb_problem", "climate_state_convinced", 'climatedeniers_1', "climatedeniers_2", "climate_risk", "d_q1_own_opinion"])

        if impute:
            train_val, _ = train_test_split(data, test_size=0.2, random_state=42, shuffle=False)
            str_columns = ["gender", "income_section", "education", "children", "migration_b_germany", "innovation",
                           "moral_rel_love_country", "eastger", "academic"]
            imp_str = SimpleImputer(strategy="most_frequent")
            imp_str.fit(train_val[str_columns])
            imputed_data_str = pd.DataFrame(imp_str.transform(data[str_columns]))
            imputed_data_str.columns = data[str_columns].columns
            imputed_data_str.index = data.index
            num_columns = [x for x in data.columns.tolist() if x not in str_columns]
            imp_num = IterativeImputer(random_state=42)
            imp_num.fit(train_val[num_columns])
            imputed_data_num = pd.DataFrame(imp_num.transform(data[num_columns]))
            imputed_data_num.columns = data[num_columns].columns
            imputed_data_num.index = data.index
            data = pd.concat([imputed_data_num, imputed_data_str], axis=1)
        else:
            data.dropna(inplace=True)

        data = pd.get_dummies(
            data,
            columns=["gender", "eastger", "academic", "migration_b_germany"], drop_first=True, dtype=float
        )
        enc = OrdinalEncoder(
            categories=[['<= 200 EUR', '200 - 300 EUR', '300 - 400 EUR', '400 - 500 EUR', '500 - 625 EUR', '625 - 750 EUR',
                         '750 - 875 EUR', '875 - 1000 EUR', '1000 - 1125 EUR', '1125 - 1250 EUR', '1250 - 1375 EUR',
                         '1375 - 1500 EUR', '1500 - 1750 EUR', '1750 - 2000 EUR', '2000 - 2250 EUR', '2250 - 2500 EUR',
                         '2500 - 2750 EUR', '2750 - 3000 EUR', '3000 - 4000 EUR', '4000 - 5000 EUR', '5000 - 7500 EUR',
                         '>=7500 EUR']])
        data["income_section"] = enc.fit_transform(data[["income_section"]])
        enc = OrdinalEncoder(categories=[['No degree', 'Hauptschule', 'Realschule', 'Abitur', 'Lehre', 'Hochschule',
                                          'Doktor, Habilitation']])
        data["education"] = enc.fit_transform(data[["education"]])
        enc = OrdinalEncoder(categories=[['None', '1', '2', '3', '4', '5 or more']])
        data["children"] = enc.fit_transform(data[["children"]])
        enc = OrdinalEncoder(categories=[['Completely disagree', 'Rather disagree', 'Neither agree nor disagree',
                                          'Rather agree', 'Completely agree']])
        data["innovation"] = enc.fit_transform(data[["innovation"]])
        enc = OrdinalEncoder(categories=[['Not at all relevant', 'Not very relevant', 'Somewhat relevant',
                                          'Slightly relevant', 'Very relevant', 'Extremely relevant']])
        data["moral_rel_love_country"] = enc.fit_transform(data[["moral_rel_love_country"]])
        self.data = data.copy()

    def drop_columns(self, df: pd.DataFrame, columns: list):
        df.drop(columns=columns, inplace=True)

    def get_indexes(self, df: pd.DataFrame, n_splits: int=10):
        train_indexes = []
        test_indexes = []
        splitter = StratifiedShuffleSplit(n_splits=n_splits, random_state=42)
        X = df.drop(columns=["climatedeniers"])
        y = df["climatedeniers"]
        for train_index, test_index in splitter.split(X, y):
            train_indexes.append(train_index)
            test_indexes.append(test_index)

        return train_indexes, test_indexes

    def retrain(self, retrain: pd.DataFrame, model):
        x_train = retrain.drop("climatedeniers", axis=1)
        self.X_scaler = sklearn.preprocessing.StandardScaler()
        x_train_standard = pd.DataFrame(self.X_scaler.fit_transform(x_train))
        x_train_standard.columns = x_train.columns
        y_train = retrain["climatedeniers"]
        model.fit(x_train_standard, y_train)

    def predict(self, X_in: pd.DataFrame, model) -> np.array:
        X_in = X_in.drop("climatedeniers", axis=1)
        X_in_standard = pd.DataFrame(self.X_scaler.transform(X_in))
        X_in_standard.columns = X_in.columns
        prediction = model.predict(X_in_standard)
        return prediction

    def train_val_loop(self, train: pd.DataFrame, val: pd.DataFrame, model) -> np.array:
        self.retrain(train, model)
        return self.predict(X_in=val, model=model)

    def objective(self, trial: optuna.trial.Trial):

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
        impute = trial.suggest_categorical(
            "impute",
            [False, True]) # TODO: also try True
        early_stopping = trial.suggest_categorical(
            "early_stopping",
            [False, True])
        shuffle = trial.suggest_categorical(
            "shuffle",
            [False, True])
        solver = trial.suggest_categorical(
            "solver",
            ["sgd", "adam"])
        activation = trial.suggest_categorical(
            "activation",
            ["identity", "logistic", "tanh", "relu"])
        hidden_layer_sizes = trial.suggest_int(
            "hidden_layer_sizes",
            5, 100)
        alpha = trial.suggest_float(
            "alpha", 0.00001, 0.1, log=True
        )
        n_iter_no_change = trial.suggest_int(
            "n_iter_no_change",
            5, 100)
        learning_rate_init = trial.suggest_float(
            "learning_rate_init", 0.00001, 0.1, log=True
        )
        max_iter = trial.suggest_int(
            "max_iter",
            2000, 10000, step=1000)

        unfitted_model = MLPClassifier(
            random_state=42,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            shuffle=shuffle,
            early_stopping=early_stopping,
            n_iter_no_change=n_iter_no_change,
            max_iter=max_iter
        )

        objective_values = []

        self.preprocess_data(impute=impute)
        self.train_val, self.test = train_test_split(self.data, test_size=0.2, random_state=42, shuffle=False)

        train_indexes, val_indexes = self.get_indexes(df=self.train_val)

        for fold in range(5):
            model = copy.deepcopy(unfitted_model)

            train, val = (
                self.train_val.iloc[train_indexes[fold]],
                self.train_val.iloc[val_indexes[fold]],
            )

            if sampling != None:
                if sampling == "over":
                    sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
                else:
                    sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
                try:
                    train_X_sampled, train_y_sampled = sampler.fit_resample(
                        train.drop("climatedeniers", axis=1), train["climatedeniers"]
                    )
                    train = pd.concat([train_X_sampled, train_y_sampled], axis=1)
                    train = train.sample(frac=1).reset_index(drop=True)
                except (sklearn.utils._param_validation.InvalidParameterError) as exc:
                    print(traceback.format_exc())
                    print(exc)
                    print('Trial failed. Error in optim loop.')
                    raise optuna.exceptions.TrialPruned()

            y_pred = self.train_val_loop(train=train, val=val, model=model)

            objective_value = matthews_corrcoef(val["climatedeniers"], y_pred)

            objective_values.append(objective_value)

        current_val_result = float(np.mean(objective_values))

        return current_val_result

    def create_new_study(self) -> optuna.study.Study:
        study_name = (
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                + "_"
                + "-MODEL"
                + "MLP"
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

    def run_optuna_optimization(self) -> dict:
        # Create a new study
        study = self.create_new_study()
        # Start optimization run
        study.optimize(lambda trial: self.objective(trial=trial), n_trials=200)
        print(study.best_trial.value)
        print(study.best_params)
        print(optuna.importance.get_param_importances(study))

        return study.best_params

climate_deniers_classifier = ClimateDeniersClassifier()

best_params = climate_deniers_classifier.run_optuna_optimization()
del best_params["sampling"]
del best_params["sampling_stragety"]
del best_params["impute"]

best_model = MLPClassifier(**best_params)

climate_deniers_classifier.retrain(climate_deniers_classifier.train_val, best_model)

pickle.dump(best_model, open("models/model_mlp", 'wb'))

predictions = climate_deniers_classifier.predict(X_in=climate_deniers_classifier.test, model=best_model)

np.savetxt("predictions/predictions_mlp.csv", predictions, delimiter=",")
climate_deniers_classifier.test.to_csv("testsets/test_mlp.csv", index=False,
            sep=',', decimal='.', float_format='%.10f')
with open('best_params/best_params_mlp.csv', 'w') as f:
    w = csv.writer(f)
    w.writerows(best_params.items())
print(classification_report(y_true=climate_deniers_classifier.test["climatedeniers"], y_pred=predictions))
print(matthews_corrcoef(y_true=climate_deniers_classifier.test["climatedeniers"], y_pred=predictions))
with open('matthews_corrcoef/matthews_corrcoef_mlp.txt', 'w') as f:
    f.write("matthews_corrcoef: %.2f" % matthews_corrcoef(y_true=climate_deniers_classifier.test["climatedeniers"], y_pred=predictions))

test = climate_deniers_classifier.X_scaler.transform(climate_deniers_classifier.test.drop("climatedeniers", axis=1))

explainer = shap.Explainer(best_model.predict, test)
shap_values = explainer(test)

filename_expl = 'explainer.sav'
pickle.dump(explainer, open("explainer/" + filename_expl, 'wb'))

filename = 'shapvalues.sav'
pickle.dump(shap_values, open("shapvalues/" + filename, 'wb'))

for feature in climate_deniers_classifier.data.drop("climatedeniers", axis=1).columns:
    shap.partial_dependence_plot(
        feature,
        best_model.predict,
        test,
        ice=False,
        model_expected_value=True,
        feature_expected_value=True,
        show=False
    )
    f = plt.gcf()
    f.savefig("partial_dependence_plots/shap.partial_dependence_plot" + feature + ".pdf", format='pdf', bbox_inches='tight')
