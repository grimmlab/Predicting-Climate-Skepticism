from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import sklearn
import datetime
import optuna
import pandas as pd
import os
import inspect
import importlib
from sklearn.experimental import enable_iterative_imputer

def encode_data(data, save_dir):

    data["subject_well_being"] = data["subject_well_being"].replace("Completely satisfied", 10)
    data["subject_well_being"] = data["subject_well_being"].replace("Not satisfied at all", 1).astype(float)
    data["climate_eb_resp_all"] = data["climate_eb_resp_all"].replace(",", "", regex=True).astype(int)
    data["bioecon_prod_all"] = data["bioecon_prod_all"].replace(",", "", regex=True).astype(int)
    data["unemployment_rate"] = data["unemployment_rate"].replace(",", "", regex=True).astype(int)
    data['migration_region'] = (data['migration_region'].replace(
        ['Neud', 'Weiß', 'KEIN', 'weiß', 'Gieß', '40Ja', 'Y200', 'Deut', 'nein', 'Acht', 'Gar ', '197q', 'kein', 'fünf',
         'xxxx', '75 J', '10 j', "Germ", "oooo", '000/', "19i8", "Draw", '1ß65', "Oooo", '2oo6'],
        [0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000,
        0000, 0000, 0000, 0000, 1965, 0000, 2006]).astype("int"))

    label_cols = \
        ["gender", "plz", "marital_status", "job", "migration_b_g_city", "migration_s_g_city", "politics_vote",
         "region_prefix", "adequacy", "goal", "state", "nuts2", "district", "job_field_group", "migration_b_g_state",
         "migration_b_g_nuts2", "migration_b_g_district", "migration_s_g_state", "migration_s_g_nuts2",
         "migration_s_g_district", "b_q1_attention", "b_q2_attention", "climate_eb_resp_nat_gov", "climate_eb_resp_eu",
         "climate_eb_resp_reg_gov", "climate_eb_resp_industry", "climate_eb_resp_self", "climate_eb_resp_dont_know",
         "bioecon_prod_cleaning", "bioecon_prod_cosmetics", "bioecon_prod_furniture", "bioecon_prod_dishes",
         "bioecon_prod_bags", "bioecon_prod_dispos_dish", "bioecon_prod_packaging", "bioecon_prod_wood",
         "bioecon_prod_trashbags", "bioecon_prod_clothes", "migration_b_germany", "migration_s_germany",
         "climate_eb_resp_activists", "climate_eb_resp_other", "climate_eb_resp_nobody", "attention1_d", "attention2_d",
         "bioecon_prod_building", "bioecon_prod_none", "bioecon_prod_other"] # "religion", "job_field_maingroup",
    for col in label_cols:
        encoder = LabelEncoder()
        data[col] = (encoder.fit_transform(data[[col]]))
        save_dir.joinpath('encoder_classes').mkdir(parents=True, exist_ok=True)
        with open(save_dir.joinpath('encoder_classes/encoder_classes' + col + '.txt'), 'w') as f:
            f.write(str(encoder.classes_))

    cols_climate_actions = \
        ["climate_actions_public_display", "climate_actions_donate", "climate_actions_volunteer",
        "climate_actions_discuss", "climate_actions_protest", "climate_actions_contact_news",
         "climate_actions_social_media"]
    for col in cols_climate_actions:
        oe_climate_actions = OrdinalEncoder(categories=[
            ['Definitely would not', 'Probably would not', 'Probably would', 'Definitely would', 'Already doing this']])
        data[col] = oe_climate_actions.fit_transform(data[[col]])
    cols_moral_rel = \
        ["moral_rel_suffering", "moral_rel_treat_diff", "moral_rel_love_country", "moral_rel_lack_respect",
        "moral_rel_violate_purity", "moral_rel_math", "moral_rel_care", "moral_rel_unfair", "moral_rel_betray",
         "moral_rel_traditions", "moral_rel_disgusting", "moral_rel_cruelty", "moral_rel_deny_rights",
         "moral_rel_lack_loyalty", "moral_rel_disorder", "moral_rel_god_approve"]
    for col in cols_moral_rel:
        oe_moral_rel = OrdinalEncoder(categories=[
            ['Not at all relevant', 'Not very relevant', 'Somewhat relevant', 'Slightly relevant', 'Very relevant',
             'Extremely relevant']])
        data[col] = oe_moral_rel.fit_transform(data[[col]])
    cols_moral_state = \
        ["moral_state_compassion", "moral_state_laws_fair", "moral_state_proud", "moral_state_child_respect",
        "moral_state_disgusting", "moral_state_good_than_bad", "moral_state_hurt_animals", "moral_state_justice",
         "moral_state_loyal_family", "moral_state_diff_roles", "moral_state_unnatural", "moral_state_never_kill",
         "moral_state_inherit", "moral_state_team_player", "moral_state_obey", "moral_state_chastity"]
    for col in cols_moral_state:
        oe_moral_state = OrdinalEncoder(categories=[
            ['Completely disagree', 'Slightly disagree', 'Moderately disagree', 'Slightly agree', 'Moderately agree',
             'Completely agree']])
        data[col] = oe_moral_state.fit_transform(data[[col]])
    oe_religion_practice = (OrdinalEncoder(categories=[
        ['Keine Angabe', 'Never', 'Less frequently', 'Several times a year', 'One too three times a month',
         'Once a week', 'More than once a week']]))
    data["religion_practice"] = oe_religion_practice.fit_transform(data[["religion_practice"]])
    cols_innovation = \
        ["innovation", "climate_policies_fund_research", "climate_policies_carbon_tax", "climate_policies_tax_rabates",
         "climate_state_worry", "climate_state_damage", "climate_state_adhere_goal", "climate_state_single_person",
         "climate_state_manmade", "climate_state_forecasts", "climate_state_disagree", "climate_state_convinced",
         "climate_state_media", "climate_state_children", "climate_state_extreme_weather", "climate_eb_state_expertise",
         "climate_eb_state_energy_security", "climate_eb_state_innovation", "climate_eb_state_transition",
         "climate_eb_state_pos_outcome", "climate_eb_state_min_emissions", "climate_state_together",
         "climate_policies_stop_coal"]
    for col in cols_innovation:
        oe_innovation = OrdinalEncoder(categories=[
            ['Keine Angabe', 'Completely disagree', 'Rather disagree', 'Moderately disagree',
             'Neither agree nor disagree', 'Slightly agree', 'Rather agree', 'Completely agree']])
        data[col] = oe_innovation.fit_transform(data[[col]])
    cols_eb_q12 = ["b_q1_own_opinion", "b_q1_point_guess", "b_q2_population", "b_q2_point_guess"]
    for col in cols_eb_q12:
        oe_eb_q12 = OrdinalEncoder(categories=[
            ['Keine Angabe', 'Completely oppose', 'Rather oppose', 'Neither oppose nor suport', 'Rather suport',
             'Completely suport']])
        data[col] = oe_eb_q12.fit_transform(data[[col]])
    oe_climate_know = OrdinalEncoder(categories=[['Not at all', 'Very little', 'Little', 'Much', 'Very much']])
    data["climate_know"] = oe_climate_know.fit_transform(data[["climate_know"]])
    cols_climate_eb = ["climate_eb_renewable", "climate_eb_efficient", "climate_eb_ets"]
    for col in cols_climate_eb:
        oe_climate_eb = OrdinalEncoder(categories=[
            ["Not at all important", "Not very important", "Fairly important", "Very important"]])
        data[col] = oe_climate_eb.fit_transform(data[[col]])
    oe_climate_risk = OrdinalEncoder(categories=[["Very low", "Rather low", "Rather high", "Very high"]])
    data["climate_risk"] = (oe_climate_risk.fit_transform(data[["climate_risk"]]))
    oe_politics_climate = OrdinalEncoder(categories=[
        ["Very unimportant", "Unimportant", "Neither unimportant nor important", "Important", "Very important"]])
    data["politics_climate"] = oe_politics_climate.fit_transform(data[["politics_climate"]])
    oe_soc_inequ_climate = OrdinalEncoder(categories=[
        ["Strongly decrease", "Slightly decrease", "No change", "Slightly increase", "Strongly increase"]])
    data["soc_inequ_climate"] = oe_soc_inequ_climate.fit_transform(data[["soc_inequ_climate"]])
    oe_income_group = OrdinalEncoder(categories=[["low", "middle", "high"]])
    data["income_group"] = oe_income_group.fit_transform(data[["income_group"]])
    oe_income_section = OrdinalEncoder(categories=[
        ['<= 200 EUR', '200 - 300 EUR', '300 - 400 EUR', '400 - 500 EUR', '500 - 625 EUR', '625 - 750 EUR',
         '750 - 875 EUR', '875 - 1000 EUR', '1000 - 1125 EUR', '1125 - 1250 EUR', '1250 - 1375 EUR', '1375 - 1500 EUR',
         '1500 - 1750 EUR', '1750 - 2000 EUR', '2000 - 2250 EUR', '2250 - 2500 EUR', '2500 - 2750 EUR',
         '2750 - 3000 EUR', '3000 - 4000 EUR', '4000 - 5000 EUR', '5000 - 7500 EUR', '>=7500 EUR']])
    data["income_section"] = oe_income_section.fit_transform(data[["income_section"]])
    oe_education = OrdinalEncoder(categories=[
        ["Other", "No degree", "Hauptschule", "Realschule", "Abitur", "Lehre", "Hochschule", "Doktor, Habilitation"]])
    data["education"] = oe_education.fit_transform(data[["education"]])

    return data


def get_indexes(df: pd.DataFrame, n_splits: int=10, target_column: str=None):
    train_indexes = []
    test_indexes = []
    splitter = sklearn.model_selection.StratifiedShuffleSplit(n_splits=n_splits, random_state=42)
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


def preprocess_data(to_be_dropped_columns: list = None):

    full_data = pd.read_csv("datasets/Climate_Deniers_24_10_15.csv", low_memory=False)

    full_data["climate_eb_problem"].replace("A very serious problem", int(10), inplace=True)
    full_data["climate_eb_problem"].replace("No serious problem at all", int(1), inplace=True)
    full_data["climate_eb_problem"] = full_data["climate_eb_problem"].astype("float")
    full_data["climatedeniers"] = 2
    full_data["climatedeniers"][full_data["climate_eb_problem"] <= 5] = 1
    full_data["climatedeniers"][full_data["climate_eb_problem"] > 5] = 0

    full_data = full_data.drop(columns=to_be_dropped_columns, axis=1)

    full_data.dropna(axis=1, thresh=11000, inplace=True)

    full_data = full_data[full_data.climatedeniers != 2]

    return full_data


def impute_data(data: pd.DataFrame = None):
    train_val, _ = sklearn.model_selection.train_test_split(data, test_size=0.2, random_state=42)

    imp_str = sklearn.impute.SimpleImputer(strategy="most_frequent")
    imp_str = imp_str.fit(train_val.select_dtypes(include=['O']))
    imputed_data_str = pd.DataFrame(imp_str.transform(data.select_dtypes(include=['O'])))
    imputed_data_str.columns = data.select_dtypes(include=['O']).columns
    imputed_data_str.index = data.index

    num_columns = [x for x in data.columns.tolist() if x not in data.select_dtypes(include=['O']).columns.tolist()]
    imp_num = sklearn.impute.IterativeImputer(random_state=42, estimator=sklearn.ensemble.HistGradientBoostingRegressor())
    imp_num = imp_num.fit(train_val[num_columns])
    imputed_data_num = pd.DataFrame(imp_num.transform(data[num_columns]))
    imputed_data_num.columns = data[num_columns].columns
    imputed_data_num.index = data.index

    return pd.concat([imputed_data_num, imputed_data_str], axis=1)