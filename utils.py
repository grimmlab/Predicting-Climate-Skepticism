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
from sklearn.experimental import enable_iterative_imputer

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

    if "PERSONAL_DATA" in featuresets:

        ## Ordinal enconding
        encoder = OrdinalEncoder(categories=[["low", "middle", "high"]])
        data["income_group"] = encoder.fit_transform(data[["income_group"]])
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
        encoder = OrdinalEncoder(categories=[
            ['No information provided', 'Completely disagree', 'Rather disagree', 'Moderately disagree', 'Neither agree nor disagree',
             'Slightly agree', 'Rather agree', 'Completely agree']])
        data["innovation"] = encoder.fit_transform(data[["innovation"]])

        ## Replacement of certain values
        data["subject_well_being"] = (
            data["subject_well_being"].replace(["Completely satisfied", "Not satisfied at all"], [1, 10]).astype(float))
        data["children"] = data["children"].replace(["None", "5 or more"], [0, 6]).astype(float)
        data['migration_region'] = (data['migration_region'].replace(
            ['Neud', 'Weiß', 'KEIN', 'weiß', 'Gieß', '40Ja', 'Y200', 'Deut', 'nein', 'Acht', 'Gar ', '197q', 'kein',
             'fünf', 'xxxx', '75 J', '10 j', "Germ", "oooo", '000/', "19i8", "Draw", '1ß65', "Oooo", '2oo6'],
            [0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000, 1970, 0000, 0000, 0000, 0000, 0000, 0000,
             0000, 0000, 1998, 0000, 1965, 0000, 2006])).astype(int)

        ## Label encoding
        label_cols_personal = [
            "state", "nuts2", "district", "religion", "religion_christian", "religion_islam", "migration_b_germany",
            "migration_b_g_city", "migration_b_g_state", "migration_b_g_nuts2", "migration_b_g_district",
            "migration_b_country", "migration_s_germany", "migration_s_g_city", "migration_s_g_state",
            "migration_s_g_nuts2", "migration_s_g_district", "migration_s_country", "job_field_maingroup",
            "job_field_group"]
        for col in label_cols_personal:
            encoder = LabelEncoder()
            data[col] = (encoder.fit_transform(data[[col]]))
            save_dir.joinpath('encoder_classes').mkdir(parents=True, exist_ok=True)
            with open(save_dir.joinpath(f'encoder_classes/encoder_classes_{col}.txt'), 'w') as f:
                f.write(str(encoder.classes_))

        ## one hot encoding
        data = pd.get_dummies(data, columns=["job", "marital_status"], dtype=int)
        data = pd.get_dummies(data, columns=["gender"], dtype=int, drop_first=True)

    if "CLIMATE_EUROBAROMETER" in featuresets:

        ## Ordinal encoding
        encoder = OrdinalEncoder(categories=[['Not at all', 'Very little', 'Little', 'Much', 'Very much']])
        data["climate_know"] = encoder.fit_transform(data[["climate_know"]])
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

        ## one hot encoding
        data = pd.get_dummies(
            data,
            columns=["climate_eb_resp_nat_gov", "climate_eb_resp_eu", "climate_eb_resp_reg_gov",
                     "climate_eb_resp_industry", "climate_eb_resp_self", "climate_eb_resp_activists",
                     "climate_eb_resp_other", "climate_eb_resp_nobody", "climate_eb_resp_dont_know"],
            dtype=int, drop_first=True)

        if dependent_variable != "climate_eb_problem":
            data["climate_eb_problem"] = (
                data["climate_eb_problem"].replace(["No serious problem at all", "A very serious problem"],
                                                   [1, 10])).astype(float)

    if "CLIMATE_KNOWLEDGE" in featuresets:

        ## Ordinal encoding
        encoder = OrdinalEncoder(categories=[["Very low", "Rather low", "Rather high", "Very high"]])
        data["climate_risk"] = (encoder.fit_transform(data[["climate_risk"]]))
        ordinal_cols_climate_state = [
            "climate_state_worry", "climate_state_damage", "climate_state_adhere_goal", "climate_state_together",
            "climate_state_single_person", "climate_state_forecasts", "climate_state_disagree",
            "climate_state_media", "climate_state_children",
            "climate_state_extreme_weather"]
        if dependent_variable != "climate_state_manmade":
            ordinal_cols_climate_state.extend(["climate_state_manmade"])
        if dependent_variable != "climate_state_convinced":
            ordinal_cols_climate_state.extend(["climate_state_convinced"])
        for col in ordinal_cols_climate_state:
            encoder = OrdinalEncoder(categories=[
                ['Completely disagree', 'Rather disagree', 'Rather agree', 'Completely agree']])
            data[col] = encoder.fit_transform(data[[col]])

    if "CLIMATE_POLICIES_ACTIONS" in featuresets:

        ## Ordinal encoding
        ordinal_cols_policies = \
            ["climate_policies_fund_research", "climate_policies_stop_coal", "climate_policies_carbon_tax",
             "climate_policies_tax_rabates"]
        for col in ordinal_cols_policies:
            encoder = OrdinalEncoder(categories=[
                ['Completely disagree', 'Rather disagree', 'Rather agree', 'Completely agree']])
            data[col] = encoder.fit_transform(data[[col]])
        ordinal_cols_climate_actions = \
            ["climate_actions_public_display", "climate_actions_donate", "climate_actions_volunteer",
             "climate_actions_discuss", "climate_actions_protest", "climate_actions_contact_news",
             "climate_actions_social_media"]
        for col in ordinal_cols_climate_actions:
            encoder = OrdinalEncoder(categories=[
                ['Definitely would not', 'Probably would not', 'Probably would', 'Definitely would', 'Already doing this']])
            data[col] = encoder.fit_transform(data[[col]])

    if "BIOECONOMY" in featuresets:

        ## Label encoding
        label_cols_bioeconomy = \
            ["bioecon_prod_clothes", "bioecon_prod_cleaning", "bioecon_prod_cosmetics", "bioecon_prod_furniture",
             "bioecon_prod_dishes", "bioecon_prod_bags", "bioecon_prod_dispos_dish", "bioecon_prod_packaging",
             "bioecon_prod_wood","bioecon_prod_building", "bioecon_prod_trashbags", "bioecon_prod_none",
             "bioecon_prod_other"]
        for col in label_cols_bioeconomy:
            encoder = LabelEncoder()
            data[col] = (encoder.fit_transform(data[[col]]))
            save_dir.joinpath('encoder_classes').mkdir(parents=True, exist_ok=True)
            with open(save_dir.joinpath('encoder_classes/encoder_classes' + col + '.txt'), 'w') as f:
                f.write(str(encoder.classes_))

    if "MORAL_VALUES" in featuresets:

        ## Ordinal encoding
        ordinal_cols_moral_rel = \
            ["moral_rel_suffering", "moral_rel_treat_diff", "moral_rel_love_country", "moral_rel_lack_respect",
             "moral_rel_violate_purity", "moral_rel_math", "moral_rel_care", "moral_rel_unfair", "moral_rel_betray",
             "moral_rel_traditions", "moral_rel_disgusting", "moral_rel_cruelty", "moral_rel_deny_rights",
             "moral_rel_lack_loyalty", "moral_rel_disorder", "moral_rel_god_approve"]
        for col in ordinal_cols_moral_rel:
            encoder = OrdinalEncoder(categories=[
                ['Not at all relevant', 'Not very relevant', 'Somewhat relevant', 'Slightly relevant', 'Very relevant',
                 'Extremely relevant']])
            data[col] = encoder.fit_transform(data[[col]])
        ordinal_cols_moral_state = \
            ["moral_state_compassion", "moral_state_laws_fair", "moral_state_proud", "moral_state_child_respect",
             "moral_state_disgusting", "moral_state_good_than_bad", "moral_state_hurt_animals", "moral_state_justice",
             "moral_state_loyal_family", "moral_state_diff_roles", "moral_state_unnatural", "moral_state_never_kill",
             "moral_state_inherit", "moral_state_team_player", "moral_state_obey", "moral_state_chastity"]
        for col in ordinal_cols_moral_state:
            encoder = OrdinalEncoder(categories=[
                ['Completely disagree', 'Slightly disagree', 'Moderately disagree', 'Slightly agree',
                 'Moderately agree', 'Completely agree']])
            data[col] = encoder.fit_transform(data[[col]])

    if "ADDITIONAL_VARIABLES" in featuresets:

        ## Ordinal encoding
        encoder = OrdinalEncoder(categories=[
            ["Very unimportant", "Unimportant", "Neither unimportant nor important", "Important", "Very important"]])
        data["politics_climate"] = encoder.fit_transform(data[["politics_climate"]])
        encoder = OrdinalEncoder(categories=[
            ["Strongly decrease", "Slightly decrease", "No change", "Slightly increase", "Strongly increase"]])
        data["soc_inequ_climate"] = encoder.fit_transform(data[["soc_inequ_climate"]])

        ## Replacement of certain values
        data["politics_vote"] = data["politics_vote"].replace(["Bündnis 90/Die Grünen", "CDU/CSU"], ["Bündnis 90|Die Grünen", "CDU|CSU"])

        ## Label encoding
        label_cols_personal = ["unemployment_rate"]
        for col in label_cols_personal:
            encoder = LabelEncoder()
            data[col] = (encoder.fit_transform(data[[col]]))
            save_dir.joinpath('encoder_classes').mkdir(parents=True, exist_ok=True)
            with open(save_dir.joinpath(f'encoder_classes/encoder_classes_{col}.txt'), 'w') as f:
                f.write(str(encoder.classes_))

        ## one hot encoding
        data = pd.get_dummies(data, columns=["politics_vote"], dtype=int)

    return data


def get_indexes(df: pd.DataFrame, n_splits: int=5, target_column: str=None):
    train_indexes = []
    test_indexes = []
    splitter = sklearn.model_selection.StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
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

    PERSONAL = [
        "gender", "age","income_group","income_section","state","nuts2","district","plz","education","marital_status",
        "children","job","job_field_maingroup","job_field_group","religion","religion_practice","religion_christian",
        "religion_islam","migration_b_germany","migration_b_g_city","migration_b_g_state","migration_b_g_nuts2",
        "migration_b_g_district","migration_b_country","migration_s_germany","migration_s_g_city","migration_s_g_state",
        "migration_s_g_nuts2","migration_s_g_district","migration_s_country","migration_region","subject_well_being",
        "innovation"
    ]
    ECONOMIC_PREFERENCES = [
        "patience", "risk", "posrecip", "negrecip", "altruism", "trust"
    ]
    CLIMATE_EUROBAROMETER = [
        "climate_know","climate_eb_resp_nat_gov","climate_eb_resp_eu","climate_eb_resp_reg_gov",
        "climate_eb_resp_industry","climate_eb_resp_self","climate_eb_resp_activists","climate_eb_resp_other",
        "climate_eb_resp_nobody","climate_eb_resp_dont_know","climate_eb_renewable","climate_eb_efficient",
        "climate_eb_ets","climate_eb_state_expertise","climate_eb_state_energy_security","climate_eb_state_innovation",
        "climate_eb_state_transition","climate_eb_state_pos_outcome","climate_eb_state_min_emissions"
    ]
    if dependent_variable != "climate_eb_problem":
        CLIMATE_EUROBAROMETER.extend(["climate_eb_problem"])
    CLIMATE_KNOWLEDGE = [
        "climate_flood_affect","climate_risk","climate_state_worry","climate_state_damage","climate_state_adhere_goal",
        "climate_state_together","climate_state_EU","climate_state_Germany","climate_state_region",
        "climate_state_single_person","climate_state_forecasts","climate_state_disagree","climate_state_media",
        "climate_state_children","climate_state_extreme_weather"
    ]
    if dependent_variable != "climate_state_manmade":
        CLIMATE_KNOWLEDGE.extend(["climate_state_manmade"])
    if dependent_variable != "climate_state_convinced":
        CLIMATE_KNOWLEDGE.extend(["climate_state_convinced"])
    CLIMATE_POLICIES_ACTIONS = [
        "climate_policies_fund_research","climate_policies_stop_coal","climate_policies_carbon_tax",
        "climate_policies_tax_rabates","climate_actions_public_display","climate_actions_donate",
        "climate_actions_volunteer","climate_actions_discuss","climate_actions_protest",
        "climate_actions_contact_news","climate_actions_social_media"
    ]
    CLIMATE_TRUST = [
       "climate_trust_city",
        "climate_trust_state_gov","climate_trust_nat_gov","climate_trust_companies","climate_trust_scientist",
        "climate_trust_un","climate_trust_eu"
    ]
    BIOECONOMY = [
        "bioecon_prod_clothes","bioecon_prod_cleaning","bioecon_prod_cosmetics","bioecon_prod_furniture",
        "bioecon_prod_dishes","bioecon_prod_bags","bioecon_prod_dispos_dish","bioecon_prod_packaging",
        "bioecon_prod_wood","bioecon_prod_building","bioecon_prod_trashbags","bioecon_prod_none","bioecon_prod_other"
    ]
    MORAL_VALUES = [
        "moral_rel_suffering","moral_rel_treat_diff","moral_rel_love_country","moral_rel_lack_respect",
        "moral_rel_violate_purity","moral_rel_math","moral_rel_care","moral_rel_unfair","moral_rel_betray",
        "moral_rel_traditions","moral_rel_disgusting","moral_rel_cruelty","moral_rel_deny_rights",
        "moral_rel_lack_loyalty","moral_rel_disorder","moral_rel_god_approve","moral_state_compassion",
        "moral_state_laws_fair","moral_state_proud","moral_state_child_respect","moral_state_disgusting",
        "moral_state_good_than_bad","moral_state_hurt_animals","moral_state_justice","moral_state_loyal_family",
        "moral_state_diff_roles","moral_state_unnatural","moral_state_never_kill","moral_state_inherit",
        "moral_state_team_player","moral_state_obey","moral_state_chastity"
    ]
    ADDITIONAL_VARIABLES = [
        "politics_orientation","politics_vote","politics_climate","soc_inequ_climate","covid_coping","covid_finance",
        "unemployment_rate","patent"
    ]

    experiments_dict = {
        "PERSONAL_DATA": PERSONAL, "ECONOMIC_PREFERENCES": ECONOMIC_PREFERENCES,
        "CLIMATE_EUROBAROMETER": CLIMATE_EUROBAROMETER, "CLIMATE_KNOWLEDGE": CLIMATE_KNOWLEDGE,
        "CLIMATE_POLICIES_ACTIONS": CLIMATE_POLICIES_ACTIONS, "CLIMATE_TRUST": CLIMATE_TRUST,
        "BIOECONOMY": BIOECONOMY, "MORAL_VALUES": MORAL_VALUES, "ADDITIONAL_VARIABLES": ADDITIONAL_VARIABLES
    }

    for featureset in featuresets:
        usecols.extend(experiments_dict[featureset])

    full_data = pd.read_csv("datasets/ClimateDeniers.csv", usecols=usecols)
    if "PERSONAL_DATA" in featuresets:
        full_data[["religion_practice"]] = full_data[["religion_practice"]].fillna("Never")
        full_data[["innovation"]] = full_data[["innovation"]].fillna("No information provided")
        full_data[["migration_region"]] = full_data[["migration_region"]].fillna(0000)

    full_data = encode_data(full_data, save_dir, featuresets, dependent_variable)

    if modus == "hypothesis":
        full_data[dependent_variable] = np.random.permutation(full_data[dependent_variable].values)

    dataset_dir = pathlib.Path("datasets/preprocessed")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    full_data.to_csv(dataset_dir.joinpath(f"{modus}_{dependent_variable}_{''.join(featuresets)}.csv"), index=False)

    return full_data


def impute_data(train: pd.DataFrame = None, test: pd.DataFrame = None):
    train_str_imp = pd.DataFrame(train.select_dtypes(include=['O']))
    test_str_imp = pd.DataFrame(test.select_dtypes(include=['O']))
    if train.select_dtypes(include=['O']).size > 0:
        imp_str = sklearn.impute.SimpleImputer(strategy="most_frequent").fit(train.select_dtypes(include=['O']))
        train_str_imp = pd.DataFrame(imp_str.transform(train_str_imp))
        test_str_imp = pd.DataFrame(imp_str.transform(test_str_imp))
        train_str_imp.columns = train.select_dtypes(include=['O']).columns
        train_str_imp.index = train.index
        test_str_imp.columns = test.select_dtypes(include=['O']).columns
        test_str_imp.index = test.index

    num_columns = [x for x in train.columns.tolist() if x not in train.select_dtypes(include=['O']).columns.tolist()]
    imp_num = sklearn.impute.IterativeImputer(
        random_state=42, estimator=sklearn.ensemble.HistGradientBoostingRegressor()).fit(train[num_columns])
    train_num_imp = pd.DataFrame(imp_num.transform(train[num_columns]))
    test_num_imp = pd.DataFrame(imp_num.transform(test[num_columns]))
    train_num_imp.columns = train[num_columns].columns
    train_num_imp.index = train.index
    test_num_imp.columns = test[num_columns].columns
    test_num_imp.index = test.index

    return pd.concat([pd.concat([train_num_imp, train_str_imp], axis=1), pd.concat([test_num_imp, test_str_imp], axis=1)])
