from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
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

    num_columns = [x for x in data.columns.tolist() if x not in data.select_dtypes(include=['O']).columns.tolist()]
    imp_num = IterativeImputer(random_state=42, estimator=sklearn.ensemble.HistGradientBoostingRegressor())
    imp_num = imp_num.fit(train_val[num_columns])
    imputed_data_num = pd.DataFrame(imp_num.transform(data[num_columns]))
    imputed_data_num.columns = data[num_columns].columns
    imputed_data_num.index = data.index

    return pd.concat([imputed_data_num, imputed_data_str], axis=1)

def encode_data(data):

    label_cols = ["gender", "region_1", "region_2", "region_3", "plz", "marital_status", "job", "religion",
                 "migration_b_germany", "migration_b_g_city", "migration_b_g_drill_1", "migration_b_g_drill_2",
                 "migration_b_g_drill_3", "migration_s_germany", "migration_s_g_city", "migration_s_g_drill_1",
                 "migration_s_g_drill_2", "migration_s_g_drill_3", "GPS_74", "GPS_107", "climate_eb_responsib",
                 "bioecon_products", "politics_vote", "region_prefix", "job_field_1", "job_field_2"]
    for col in label_cols:
        data[col] = (LabelEncoder().fit_transform(data[[col]]))

    data['GPS_71'] = data['GPS_71'].replace(["Weiß nicht"], [-1]).astype("int")
    data['GPS_72_A'] = data['GPS_72_A'].replace(
        ["Weiß nicht", "10 Sehr bereit, dies zu tun", "0 Überhaupt nicht bereit, dies zu tun"], [-1, 10, 0]).astype("int")
    data['GPS_72_B'] = data['GPS_72_B'].replace(
        ["Weiß nicht", "10 Sehr bereit, dies zu tun", "0 Überhaupt nicht bereit, dies zu tun"], [-1, 10, 0]).astype("int")
    data['GPS_72_C'] = data['GPS_72_C'].replace(
        ["Weiß nicht", "10 Sehr bereit, dies zu tun", "0 Überhaupt nicht bereit, dies zu tun"], [-1, 10, 0]).astype("int")
    data['GPS_72_D'] = data['GPS_72_D'].replace(
        ["Weiß nicht", "10 Sehr bereit, dies zu tun", "0 Überhaupt nicht bereit, dies zu tun"], [-1, 10, 0]).astype("int")
    data['GPS_73_A'] = data['GPS_73_A'].replace(
        ["Weiß nicht", "10 Beschreibt mich perfekt", "0 Beschreibt mich überhaupt nicht"], [-1, 10, 0]).astype("int")
    data['GPS_73_B'] = data['GPS_73_B'].replace(
        ["Weiß nicht", "10 Beschreibt mich perfekt", "0 Beschreibt mich überhaupt nicht"], [-1, 10, 0]).astype("int")
    data['GPS_73_C'] = data['GPS_73_C'].replace(
        ["Weiß nicht", "10 Beschreibt mich perfekt", "0 Beschreibt mich überhaupt nicht"], [-1, 10, 0]).astype("int")
    data['GPS_73_D'] = data['GPS_73_D'].replace(
        ["Weiß nicht", "10 Beschreibt mich perfekt", "0 Beschreibt mich überhaupt nicht"], [-1, 10, 0]).astype("int")
    data['GPS_73_E'] = data['GPS_73_E'].replace(
        ["Weiß nicht", "10 Beschreibt mich perfekt", "0 Beschreibt mich überhaupt nicht"], [-1, 10, 0]).astype("int")
    data['migration_region'] = data['migration_region'].replace(
        ['Neud', 'Weiß', 'KEIN', 'weiß', 'Gieß', '40Ja', 'Y200', 'Deut', 'nein', 'Acht', 'Gar ', '197q', 'kein', 'fünf', 'xxxx', '75 J', '10 j',
         "Germ", "oooo", '000/', "19i8", "Draw", '1ß65', "Oooo", '2oo6'],
        [0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000, 0000,
         0000, 1965, 0000, 2006]).astype("int")

    oe_income = OrdinalEncoder(categories=[['Unter 200 €', '200 bis unter 300 €', '300 bis unter 400 €',
                                            '400 bis unter 500 €', '500 bis unter 625 €', '625 bis unter 750 €',
                                            '750 bis unter 875 €', '875 bis unter 1000 €', '1000 bis unter 1125 €',
                                            '1125 bis unter 1250 €', '1250 bis unter 1375 €', '1375 bis unter 1500 €',
                                            '1500 bis unter 1750 €', '1750 bis unter 2000 €', '2000 bis unter 2250 €',
                                            '2250 bis unter 2500 €', '2500 bis unter 2750 €', '2750 bis unter 3000 €',
                                            '3000 bis unter 4000 €', '4000 bis unter 5000 €', '5000 bis unter 7500 €',
                                            '7500 € und mehr']])
    data["income"] = oe_income.fit_transform(data[["income"]])
    oe_education = OrdinalEncoder(categories=[['Keine Angabe', 'Anderer Abschluss', '(Noch) kein Abschluss',
                                               'Hauptschulabschluss (Volksschulabschluss) oder gleichwertiger Abschluss',
                                               'Realschulabschluss (Mittlere Reife) oder gleichwertiger Abschluss',
                                               'Allgemeine oder fachgebundene Hochschulreife/Abitur (Gymnasium bzw. FOS)',
                                               'Berufsausbildung, Lehre oder Ausbildung an einer Fachhochschule',
                                               '(Fach-)Hochschulabschluss (Bachelor, Master, Magister, Diplom, Staatsexamen)',
                                               'Doktorgrad oder Habilitation']])
    data["education"] = oe_education.fit_transform(data[["education"]])
    oe_children = OrdinalEncoder(categories=[['Keine Angabe', 'Ich habe keine Kinder', '1', '2', '3', '4', '5 oder mehr']])
    data["children"] = oe_children.fit_transform(data[["children"]])
    oe_religion_practice = OrdinalEncoder(categories=[['Keine Angabe', 'Nie', 'Seltener', 'Mehrmals pro Jahr',
                                                       'Ein- bis dreimal im Monat', 'Einmal in der Woche',
                                                       'Mehr als einmal in der Woche']])
    data["religion_practice"] = oe_religion_practice.fit_transform(data[["religion_practice"]])

    cols_innovation = ["innovation", "climate_eb_statement_1", "climate_eb_statement_2", "climate_eb_statement_3",
                       "climate_eb_statement_4", "climate_eb_statement_5",	"climate_eb_statement_6"]
    for col in cols_innovation:
        oe_innovation = OrdinalEncoder(categories=[['Keine Angabe', 'Stimme überhaupt nicht zu', 'Stimme eher nicht zu',
                                                    'Weder noch', 'Stimme eher zu', 'Stimme voll und ganz zu']])
        data[col] = oe_innovation.fit_transform(data[[col]])

    cols_eb_q12 = ["b_q1_own_opinion", "b_q1_point_guess", "b_q2_population", "b_q2_point_guess", "e_q1_own_opinion",
                   "e_q1_point_guess", "e_q2_population", "e_q2_point_guess"]
    for col in cols_eb_q12:
        oe_eb_q12 = OrdinalEncoder(categories=[['Keine Angabe', 'Voll und ganz ablehnen', 'Eher ablehnen',
                                                'Weder ablehnen noch unterstützen', 'Eher unterstützen',
                                                'Voll und ganz unterstützen']])
        data[col] = oe_eb_q12.fit_transform(data[[col]])

    oe_GPS_105 = OrdinalEncoder(categories=[['Weiß nicht', 'Nein, ich würde kein Geschenk geben',
                                             'Das Geschenk im Wert von 5 Euro', 'Das Geschenk im Wert von 10 Euro',
                                             'Das Geschenk im Wert von 15 Euro', 'Das Geschenk im Wert von 20 Euro',
                                             'Das Geschenk im Wert von 25 Euro', 'Das Geschenk im Wert von 30 Euro']])
    data["GPS_105"] = oe_GPS_105.fit_transform(data[["GPS_105"]])
    oe_climate_know = OrdinalEncoder(categories=[['Gar nicht', 'Sehr wenig', 'Wenig', 'Viel', 'Sehr viel']])
    data["climate_know"] = oe_climate_know.fit_transform(data[["climate_know"]])

    cols_climate_eb = ["climate_eb_renewable", "climate_eb_efficient", "climate_eb_ets"]
    for col in cols_climate_eb:
        oe_climate_eb = OrdinalEncoder(categories=[["Überhaupt nicht wichtig", "Nicht so wichtig", "Ziemlich wichtig",
                                                    "Sehr wichtig"]])
        data[col] = oe_climate_eb.fit_transform(data[[col]])

    oe_climate_flood = OrdinalEncoder(categories=[["Gar nicht", "Sehr wenig", "Wenig", "Etwas", "Sehr stark"]])
    data["climate_flood_affect"] = (oe_climate_flood.fit_transform(data[["climate_flood_affect"]]))
    oe_climate_flood = OrdinalEncoder(categories=[["Sehr gering", "Eher gering", "Eher groß", "Sehr groß"]])
    data["climate_risk"] = (oe_climate_flood.fit_transform(data[["climate_risk"]]))

    cols_climate_trust = ["climate_trust_1", "climate_trust_2", "climate_trust_3", "climate_trust_4", "climate_trust_5",
                          "climate_trust_6", "climate_trust_7"]
    for col in cols_climate_trust:
        oe_climate_trust = OrdinalEncoder(categories=[["Vertraue überhaupt nicht", "Vertraue eher nicht", "Vertraue eher",
                                                       "Vertraue voll und ganz"]])
        data[col] = oe_climate_trust.fit_transform(data[[col]])

    cols_moral = ["moral_right_wrong_1", "moral_right_wrong_2", "moral_right_wrong_3", "moral_right_wrong_4",
    "moral_right_wrong_5", "moral_right_wrong_6", "moral_right_wrong_7", "moral_right_wrong_8", "moral_right_wrong_9",
    "moral_right_wrong_10", "moral_right_wrong_11", "moral_right_wrong_12", "moral_right_wrong_13",
    "moral_right_wrong_14",	"moral_right_wrong_15", "moral_right_wrong_16"]
    for col in cols_moral:
        oe_moral = OrdinalEncoder(categories=[['Sehr relevant', 'Extrem relevant', 'Überhaupt nicht relevant',
                                               'Wenig relevant', 'Nicht sehr relevant', 'Einigermaßen relevant']])
        data[col] = oe_moral.fit_transform(data[[col]])

    cols_moral_statements = [" moral_statements_1", " moral_statements_2", " moral_statements_3",
                             " moral_statements_4", " moral_statements_5", " moral_statements_6",
                             " moral_statements_7", " moral_statements_8", " moral_statements_9",
                             " moral_statements_10", " moral_statements_11", " moral_statements_12",
                             " moral_statements_13", " moral_statements_14", " moral_statements_15",
                             " moral_statements_16"]
    for col in cols_moral_statements:
        oe_moral_statements = OrdinalEncoder(categories=[["Lehne voll und ganz ab", "Lehne etwas ab", "Lehne ein wenig ab",
                                                          "Stimme ein wenig zu","Stimme etwas zu",
                                                          "Stimme voll und ganz zu"]])
        data[col] = oe_moral_statements.fit_transform(data[[col]])

    oe_politics_climate = OrdinalEncoder(categories=[["Sehr unwichtig", "Unwichtig", "Weder unwichtig noch wichtig",
                                                      "Wichtig", "Sehr wichtig"]])
    data["politics_climate"] = oe_politics_climate.fit_transform(data[["politics_climate"]])
    oe_soc_inequ_climate = OrdinalEncoder(categories=[["Wird viel kleiner", "Wird etwas kleiner", "Unverändert",
                                                       "Wird etwas größer", "Wird viel größer"]])
    data["soc_inequ_climate"] = oe_soc_inequ_climate.fit_transform(data[["soc_inequ_climate"]])
    oe_covid_coping = OrdinalEncoder(categories=[["Sehr schlecht", "Schlecht", "Neutral", "Unverändert", "Gut",
                                                  "Sehr gut"]])
    data["covid_coping"] = oe_covid_coping.fit_transform(data[["covid_coping"]])
    oe_covid_finance = OrdinalEncoder(categories=[["Nein", "Sehr wenig", "Wenig", "Viel", "Sehr viel"]])
    data["covid_finance"] = oe_covid_finance.fit_transform(data[["covid_finance"]])
    oe_income1 = OrdinalEncoder(categories=[["low", "middle", "high"]])
    data["income.1"] = oe_income1.fit_transform(data[["income.1"]])

    return data


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
    study.optimize(lambda trial: objective(trial=trial), n_trials=200)
    print(study.best_trial.value)
    print(study.best_params)
    print(optuna.importance.get_param_importances(study))

    return study.best_params