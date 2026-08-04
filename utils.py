import pathlib
import numpy as np
np.random.seed(0)
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import sklearn
import datetime
import optuna
import pandas as pd
import os
import inspect
import importlib
import joblib
import shap

def encode_data(data, save_dir, featuresets, dependent_variable):

    if dependent_variable == "climate_eb_problem":
        data["climate_eb_problem"] = (
            data["climate_eb_problem"].replace(["No serious problem at all", "A very serious problem"], [1, 10])).astype(float)
    if dependent_variable == "climate_state_manmade":
        encoder = OrdinalEncoder(categories=[
            ['Completely disagree', 'Rather disagree', 'Rather agree', 'Completely agree']])
        data["climate_state_manmade"] = encoder.fit_transform(data[["climate_state_manmade"]])
    if dependent_variable == "climate_state_convinced":
        encoder = OrdinalEncoder(categories=[
            ['Completely disagree', 'Rather disagree', 'Rather agree', 'Completely agree']])
        data["climate_state_convinced"] = encoder.fit_transform(data[["climate_state_convinced"]])

    if "DEMOGRAPHICS" in featuresets:

        ## Ordinal enconding
        encoder = OrdinalEncoder(categories=[
            ['<= 200 EUR', '200 - 300 EUR', '300 - 400 EUR', '400 - 500 EUR', '500 - 625 EUR', '625 - 750 EUR',
             '750 - 875 EUR', '875 - 1000 EUR', '1000 - 1125 EUR', '1125 - 1250 EUR', '1250 - 1375 EUR', '1375 - 1500 EUR',
             '1500 - 1750 EUR', '1750 - 2000 EUR', '2000 - 2250 EUR', '2250 - 2500 EUR', '2500 - 2750 EUR',
             '2750 - 3000 EUR', '3000 - 4000 EUR', '4000 - 5000 EUR', '5000 - 7500 EUR', '>=7500 EUR']])
        data["income_section"] = encoder.fit_transform(data[["income_section"]])
        encoder = OrdinalEncoder(categories=[
            ["Other", "No degree", "Hauptschule", "Realschule", "Abitur", "Lehre", "Hochschule",
             "Doktor, Habilitation"]], handle_unknown="use_encoded_value", unknown_value=np.nan)
        data["education"] = encoder.fit_transform(data[["education"]])
        encoder = OrdinalEncoder(categories=[[
            "Never", "Less frequently", "Several times a year", "One too three times a month", "Once a week",
            "More than once a week"]])
        data["religion_practice"] = encoder.fit_transform(data[["religion_practice"]])

        ## Replacement of certain values
        data["children"] = data["children"].replace(["None", "5 or more"], [0, 6]).astype(float)
        data['migration_region'] = (data['migration_region'].replace(
            ['Neud', 'Weiß', 'KEIN', 'weiß', 'Gieß', '40Ja', 'Y200', 'Deut', 'nein', 'Acht', 'Gar ', '197q', 'kein',
             'fünf', 'xxxx', '75 J', '10 j', "Germ", "oooo", '000/', "19i8", "Draw", '1ß65', "Oooo", '2oo6'],
            [0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000, 1970, 0000, 0000, 0000, 0000, 0000, 0000,
             0000, 0000, 1998, 0000, 1965, 0000, 2006])).astype(float)
        data["climate_flood_affect"] = data["climate_flood_affect"].replace([2,3], [1,1]).astype(float)

        ## one hot encoding
        data = (
            pd.get_dummies(data, columns=[
                "job", "marital_status", "gender", "job_field_maingroup", "religion", "migration_b_germany",
                "migration_b_country", "migration_s_germany", "migration_s_country", "state", "nuts2"], dtype="int"))

    if "PERSONAL_CONVICTION" in featuresets:
        ## Ordinal enconding
        encoder = OrdinalEncoder(categories=[
            ['No information provided', 'Completely disagree', 'Rather disagree', 'Moderately disagree',
             'Neither agree nor disagree',
             'Slightly agree', 'Rather agree', 'Completely agree']])
        data["innovation"] = encoder.fit_transform(data[["innovation"]])
        encoder = OrdinalEncoder(categories=[
            ["Very unimportant", "Unimportant", "Neither unimportant nor important", "Important", "Very important"]])
        data["politics_climate"] = encoder.fit_transform(data[["politics_climate"]])
        encoder = OrdinalEncoder(categories=[
            ["Strongly decrease", "Slightly decrease", "No change", "Slightly increase", "Strongly increase"]])
        data["soc_inequ_climate"] = encoder.fit_transform(data[["soc_inequ_climate"]])

        ## Replacement of certain values
        data["subject_well_being"] = (
            data["subject_well_being"].replace(["Completely satisfied", "Not satisfied at all"], [1, 10])).astype(float)
        data["politics_vote"] = data["politics_vote"].replace(["Bündnis 90/Die Grünen", "CDU/CSU"],
                                                              ["Bündnis 90|Die Grünen", "CDU|CSU"])

        ## one hot encoding
        data = pd.get_dummies(data, columns=["politics_vote"], dtype="int")

    if "RESPONSIBILITY" in featuresets:

        ## one hot encoding
        data = pd.get_dummies(
            data,
            columns=["climate_eb_resp_nat_gov", "climate_eb_resp_eu", "climate_eb_resp_reg_gov",
                     "climate_eb_resp_industry", "climate_eb_resp_self", "climate_eb_resp_activists",
                     "climate_eb_resp_other", "climate_eb_resp_nobody", "climate_eb_resp_dont_know"],
            drop_first=True, dtype="int")

    if "POLICY_ACTIONS" in featuresets:

        ## Ordinal encoding
        ordinal_cols_climate_eb = ["climate_eb_renewable", "climate_eb_efficient", "climate_eb_ets"]
        for col in ordinal_cols_climate_eb:
            encoder = OrdinalEncoder(categories=[
                ["Not at all important", "Not very important", "Fairly important", "Very important"]])
            data[col] = encoder.fit_transform(data[[col]])
        ordinal_cols_eb = \
            ["climate_eb_state_expertise", "climate_eb_state_energy_security", "climate_eb_state_innovation",
             "climate_eb_state_transition", "climate_eb_state_pos_outcome", "climate_eb_state_min_emissions"]
        for col in ordinal_cols_eb:
            encoder = OrdinalEncoder(categories=[
                ['Completely disagree', 'Rather disagree', 'Rather agree', 'Completely agree']])
            data[col] = encoder.fit_transform(data[[col]])
        ordinal_cols_policies = \
            ["climate_policies_fund_research", "climate_policies_stop_coal", "climate_policies_carbon_tax",
             "climate_policies_tax_rabates"]
        for col in ordinal_cols_policies:
            encoder = OrdinalEncoder(categories=[
                ['Completely disagree', 'Rather disagree', 'Rather agree', 'Completely agree']])
            data[col] = encoder.fit_transform(data[[col]])

    if "CLIMATE_OPINION" in featuresets:

        ## Ordinal encoding
        encoder = OrdinalEncoder(categories=[['Not at all', 'Very little', 'Little', 'Much', 'Very much']])
        data["climate_know"] = encoder.fit_transform(data[["climate_know"]])
        encoder = OrdinalEncoder(categories=[["Very low", "Rather low", "Rather high", "Very high"]])
        data["climate_risk"] = (encoder.fit_transform(data[["climate_risk"]]))
        ordinal_cols_climate_state = [
            "climate_state_worry", "climate_state_damage", "climate_state_adhere_goal", "climate_state_together",
            "climate_state_single_person", "climate_state_forecasts", "climate_state_disagree",
            "climate_state_media", "climate_state_children",
            "climate_state_extreme_weather"]
        for col in ordinal_cols_climate_state:
            encoder = OrdinalEncoder(categories=[
                ['Completely disagree', 'Rather disagree', 'Rather agree', 'Completely agree']])
            data[col] = encoder.fit_transform(data[[col]])

    if "PERSONAL_ACTIONS" in featuresets:

        ## Ordinal Encoding
        ordinal_cols_climate_actions = \
            ["climate_actions_public_display", "climate_actions_donate", "climate_actions_volunteer",
             "climate_actions_discuss", "climate_actions_protest", "climate_actions_contact_news",
             "climate_actions_social_media"]
        for col in ordinal_cols_climate_actions:
            encoder = OrdinalEncoder(categories=[
                ['Definitely would not', 'Probably would not', 'Probably would', 'Definitely would',
                 'Already doing this']])
            data[col] = encoder.fit_transform(data[[col]])

        ## Label encoding
        label_cols_PERSONAL_ACTIONS = \
            ["bioecon_prod_clothes", "bioecon_prod_cleaning", "bioecon_prod_cosmetics", "bioecon_prod_furniture",
             "bioecon_prod_dishes", "bioecon_prod_bags", "bioecon_prod_dispos_dish", "bioecon_prod_packaging",
             "bioecon_prod_wood","bioecon_prod_building", "bioecon_prod_trashbags", "bioecon_prod_none",
             "bioecon_prod_other"]
        for col in label_cols_PERSONAL_ACTIONS:
            encoder = LabelEncoder()
            data[col] = (encoder.fit_transform(data[[col]]))
            save_dir.joinpath('encoder_classes').mkdir(parents=True, exist_ok=True)
            with open(save_dir.joinpath('encoder_classes/encoder_classes' + col + '.txt'), 'w') as f:
                f.write(str(encoder.classes_))

    return data


def get_indexes(df: pd.DataFrame, n_splits: int=5, target_column: str=None):
    train_indexes = []
    test_indexes = []
    splitter = sklearn.model_selection.StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True) if target_column != "climate_eb_problem" else sklearn.model_selection.KFold(n_splits=n_splits, random_state=42, shuffle=True)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    for train_index, test_index in splitter.split(X, y):
        train_indexes.append(train_index)
        test_indexes.append(test_index)

    return train_indexes, test_indexes


def create_new_study() -> optuna.study.Study:
    study_name = (datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    study = optuna.create_study(
        study_name=study_name, direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.PercentilePruner(percentile=80, n_min_trials=20), load_if_exists=False)

    return study


def check_params_for_duplicate(current_params: dict, study) -> bool:
    past_params = [trial.params for trial in study.trials[:-1]]
    return current_params in past_params


def clean_up_after_exception(trial_number: int, save_dir):
    if save_dir.joinpath('temp', f'unfitted_model_trial{trial_number}').exists():
        save_dir.joinpath('temp', f'unfitted_model_trial {trial_number}').unlink()


def get_mapping_name_to_class() -> dict:
    files = os.listdir('models')
    modules_mapped = {}
    for file in files:
        if file not in ['__init__.py', '__pycache__']:
            if file[-3:] != '.py':
                continue

            file_name = file[:-3]
            module_name = 'models.' + file_name
            for name, cls in inspect.getmembers(importlib.import_module(module_name), inspect.isclass):
                if cls.__module__ == module_name:
                    modules_mapped[file_name] = cls
    return modules_mapped


def preprocess_data(save_dir: pathlib.Path = None, dependent_variable: str = None, featuresets: list = None, modus: str = None) -> pd.DataFrame:

    usecols = [dependent_variable]

    DEMOGRAPHICS = [
        "gender", "age","income_section","state","nuts2","education","marital_status","children","job",
        "job_field_maingroup","religion","religion_practice","migration_b_germany","migration_b_country",
        "migration_s_germany","migration_s_country","migration_region","unemployment_rate","patent",
        "climate_flood_affect"
    ]
    PERSONAL_CONVICTION = [
        "subject_well_being","innovation", "politics_orientation","politics_vote","politics_climate","covid_coping",
        "covid_finance","soc_inequ_climate","climate_trust_city","climate_trust_state_gov","climate_trust_nat_gov",
        "climate_trust_companies","climate_trust_scientist","climate_trust_un","climate_trust_eu"
    ]
    MORAL_FOUNDATIONS = [
        "moral_rel_importance_values", "moral_universal_values", "moral_communal_values", "moral_care_score",
        "moral_fairness_score", "moral_in_group_score", "moral_authority_score", "moral_purity_score"
    ]
    ECONOMIC_PREFERENCES = [
        "patience", "risk", "posrecip", "negrecip", "altruism", "trust"
    ]
    RESPONSIBILITY = [
        "climate_eb_resp_nat_gov","climate_eb_resp_eu","climate_eb_resp_reg_gov",
        "climate_eb_resp_industry","climate_eb_resp_self","climate_eb_resp_activists","climate_eb_resp_other",
        "climate_eb_resp_nobody","climate_eb_resp_dont_know"
    ]
    POLICY_ACTIONS = [
        "climate_eb_renewable", "climate_eb_efficient","climate_eb_ets", "climate_eb_state_expertise",
        "climate_eb_state_energy_security","climate_eb_state_innovation","climate_eb_state_transition",
        "climate_eb_state_pos_outcome", "climate_eb_state_min_emissions","climate_policies_fund_research",
        "climate_policies_stop_coal","climate_policies_carbon_tax","climate_policies_tax_rabates",
    ]
    CLIMATE_OPINION = [
        "climate_know","climate_risk","climate_state_worry","climate_state_damage","climate_state_adhere_goal",
        "climate_state_together","climate_state_EU","climate_state_Germany","climate_state_region",
        "climate_state_single_person","climate_state_forecasts","climate_state_disagree","climate_state_media",
        "climate_state_children","climate_state_extreme_weather"
    ]
    PERSONAL_ACTIONS = [
        "climate_actions_public_display", "climate_actions_donate","climate_actions_volunteer",
        "climate_actions_discuss", "climate_actions_protest","climate_actions_contact_news","climate_actions_social_media",
        "bioecon_prod_clothes","bioecon_prod_cleaning","bioecon_prod_cosmetics","bioecon_prod_furniture",
        "bioecon_prod_dishes","bioecon_prod_bags","bioecon_prod_dispos_dish","bioecon_prod_packaging",
        "bioecon_prod_wood","bioecon_prod_building","bioecon_prod_trashbags","bioecon_prod_none","bioecon_prod_other"
    ]

    experiments_dict = {
        "DEMOGRAPHICS": DEMOGRAPHICS,"PERSONAL_CONVICTION": PERSONAL_CONVICTION,"MORAL_FOUNDATIONS": MORAL_FOUNDATIONS,
        "ECONOMIC_PREFERENCES": ECONOMIC_PREFERENCES, "RESPONSIBILITY": RESPONSIBILITY,
        "POLICY_ACTIONS": POLICY_ACTIONS,"PERSONAL_ACTIONS": PERSONAL_ACTIONS,"CLIMATE_OPINION": CLIMATE_OPINION
    }

    for featureset in featuresets:
        usecols.extend(experiments_dict[featureset])

    full_data = pd.read_csv("datasets/ClimateDeniers.csv", usecols=usecols, keep_default_na=False, na_values=[""])
    if "DEMOGRAPHICS" in featuresets:
        full_data[["religion_practice"]] = full_data[["religion_practice"]].fillna("Never")
        full_data[["migration_region"]] = full_data[["migration_region"]].fillna(0000)
    if "PERSONAL_CONVICTION" in featuresets:
        full_data[["innovation"]] = full_data[["innovation"]].fillna("No information provided")

    full_data = encode_data(full_data, save_dir, featuresets, dependent_variable)

    if modus == "hypothesis":
        for feature in full_data.columns:
            full_data[feature] = np.random.permutation(full_data[feature].values)

    dataset_dir = pathlib.Path("datasets/preprocessed")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    full_data.to_csv(dataset_dir.joinpath(f"{modus}_{dependent_variable}_{''.join(featuresets)}.csv"), index=False)

    return full_data

def impute_data(train: pd.DataFrame = None, test: pd.DataFrame = None, dependent_variable: str = None):

    imputer = sklearn.impute.SimpleImputer(strategy="most_frequent").fit(train)
    train_imp = pd.DataFrame(imputer.transform(train))
    test_imp = pd.DataFrame(imputer.transform(test))
    train_imp.columns = train.columns
    train_imp.index = train.index
    test_imp.columns = test.columns
    test_imp.index = test.index

    return train_imp, test_imp

def compute_shap(final_model, train_val, test, dependent_variable, save_dir):
    explainer = shap.Explainer(final_model.predict, train_val.drop(dependent_variable, axis=1))
    shap_values = explainer(test.drop(dependent_variable, axis=1))

    joblib.dump(explainer, save_dir.joinpath('explainer.sav'))

    joblib.dump(shap_values, save_dir.joinpath('shapvalues.sav'))
    return shap_values

def standardize_data(train, test, dependent_variable):
    column_names = train.columns.tolist()
    column_names.remove(dependent_variable)
    num_columns = ["age","income_section","education","children","religion_practice","migration_region",
                   "climate_flood_affect","unemployment_rate","patent"]
    cat_columns = list(set(column_names) - set(num_columns))
    column_names = num_columns + cat_columns + [dependent_variable]
    if len(num_columns) > 1:
        scaler = sklearn.preprocessing.StandardScaler()
        scaler = scaler.fit(X=train[num_columns])
        train = pd.DataFrame(
            np.concatenate(
                (scaler.transform(train[num_columns]),
                 train[cat_columns].to_numpy(),
                 train[dependent_variable].to_frame().to_numpy()), axis=1),
            columns=column_names)
        test = pd.DataFrame(
            np.concatenate(
                (scaler.transform(test[num_columns]),
                 test[cat_columns].to_numpy(),
                 test[dependent_variable].to_frame().to_numpy()), axis=1),
            columns=column_names)
    return train, test

