from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import OrdinalEncoder
import sklearn
import datetime
import optuna
import pandas as pd

def impute_data(data):
    train_val, _ = train_test_split(data, test_size=0.2, random_state=42)

    imp_str = SimpleImputer(strategy="most_frequent")
    imp_str = imp_str.fit(train_val.select_dtypes(include=['O']))
    imputed_data_str = pd.DataFrame(imp_str.transform(data.select_dtypes(include=['O'])))
    imputed_data_str.columns = data.select_dtypes(include=['O']).columns
    imputed_data_str.index = data.index

    num_columns = [x for x in data.columns.tolist() if x not in data.select_dtypes(include=['O']).columns.tolist() ]
    imp_num = IterativeImputer(random_state=42, estimator=sklearn.ensemble.HistGradientBoostingRegressor())
    imp_num = imp_num.fit(train_val[num_columns])
    imputed_data_num = pd.DataFrame(imp_num.transform(data[num_columns]))
    imputed_data_num.columns = data[num_columns].columns
    imputed_data_num.index = data.index

    return pd.concat([imputed_data_num, imputed_data_str], axis=1)

def encode_data(data):

    data = pd.get_dummies(
        data,
        columns=["gender", "eastger", "academic", "migration_b_germany"], drop_first=True, dtype=float
    )

    oe_income = OrdinalEncoder(
        categories=[
            ['<= 200 EUR', '200 - 300 EUR', '300 - 400 EUR', '400 - 500 EUR', '500 - 625 EUR', '625 - 750 EUR',
             '750 - 875 EUR', '875 - 1000 EUR', '1000 - 1125 EUR', '1125 - 1250 EUR', '1250 - 1375 EUR',
             '1375 - 1500 EUR', '1500 - 1750 EUR', '1750 - 2000 EUR', '2000 - 2250 EUR', '2250 - 2500 EUR',
             '2500 - 2750 EUR', '2750 - 3000 EUR', '3000 - 4000 EUR', '4000 - 5000 EUR', '5000 - 7500 EUR',
             '>=7500 EUR']])
    data["income_section"] = oe_income.fit_transform(data[["income_section"]])
    oe_education = OrdinalEncoder(categories=[['No degree', 'Hauptschule', 'Realschule', 'Abitur', 'Lehre', 'Hochschule',
                                      'Doktor, Habilitation']])
    data["education"] = oe_education.fit_transform(data[["education"]])
    oe_children = OrdinalEncoder(categories=[['None', '1', '2', '3', '4', '5 or more']])
    data["children"] = oe_children.fit_transform(data[["children"]])
    oe_innovation = OrdinalEncoder(categories=[['Completely disagree', 'Rather disagree', 'Neither agree nor disagree',
                                      'Rather agree', 'Completely agree']])
    data["innovation"] = oe_innovation.fit_transform(data[["innovation"]])
    oe_moral = OrdinalEncoder(categories=[['Not at all relevant', 'Not very relevant', 'Somewhat relevant',
                                      'Slightly relevant', 'Very relevant', 'Extremely relevant']])
    data["moral_rel_love_country"] = oe_moral.fit_transform(data[["moral_rel_love_country"]])

    return data

def drop_columns(df: pd.DataFrame, columns: list):
    df.drop(columns=columns, inplace=True)

def get_indexes(df: pd.DataFrame, n_splits: int=10, target_column: str=None):
    train_indexes = []
    test_indexes = []
    splitter = StratifiedShuffleSplit(n_splits=n_splits, random_state=42)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    for train_index, test_index in splitter.split(X, y):
        train_indexes.append(train_index)
        test_indexes.append(test_index)

    return train_indexes, test_indexes


def create_new_study() -> optuna.study.Study:
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

def run_optuna_optimization(objective) -> dict:
    study = create_new_study()
    study.optimize(lambda trial: objective(trial=trial), n_trials=100)
    print(study.best_trial.value)
    print(study.best_params)
    print(optuna.importance.get_param_importances(study))

    return study.best_params