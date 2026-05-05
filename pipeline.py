import optimizer
import utils
import pathlib
import scipy.stats as stats
import os
import shutil
import numpy as np

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def run():
    DEPENDENT_VARIABLES = ["climate_eb_problem", "climate_state_manmade", "climate_state_convinced"]
    EXPERIMENTS = ["DEMOGRAPHICS", "PERSONAL_CONVICTION", "MORAL_FOUNDATIONS", "ECONOMIC_PREFERENCES",
        "RESPONSIBILITY", "POLICY_ACTIONS", "CLIMATE_OPINION", "PERSONAL_ACTIONS"]

    if os.path.isdir("results"):
        shutil.rmtree("results")

    for dependent_variable in DEPENDENT_VARIABLES:
        for experiment in EXPERIMENTS:
            featuresets = EXPERIMENTS[:EXPERIMENTS.index(experiment) + 1]

            predictions = []
            shap_values = []

            for modus in ["normal", "hypothesis"]:
                print(f'{bcolors.HEADER}{dependent_variable, featuresets, modus}{bcolors.ENDC}')

                save_dir = pathlib.Path(f"results/{dependent_variable}/{'#'.join(featuresets)}/{modus}/")
                save_dir.mkdir(parents=True, exist_ok=True)

                data = utils.preprocess_data(save_dir=save_dir, dependent_variable=dependent_variable, featuresets=featuresets, modus=modus)

                optimizer_run = optimizer.Optimizer(data=data, save_dir=save_dir, dependent_variable=dependent_variable)
                preds, shaps = optimizer_run.run_optimization()
                predictions.append(preds)
                shap_values.append(shaps)

            ttest_predictions = stats.ttest_ind(predictions[0], predictions[1], equal_var=False)
            print(f"T-Test Predictions: {ttest_predictions.pvalue}")
            np.savetxt(save_dir.parent.joinpath('ttest_predictions.csv'), ttest_predictions.pvalue.reshape(-1, 1), delimiter=",")

            ttest_shap_values = stats.ttest_ind(shap_values[0].values, shap_values[1].values, equal_var=False)
            print(f"T-Test SHAP Values: {ttest_shap_values.pvalue}")
            np.savetxt(save_dir.parent.joinpath('ttest_shap_values.csv'), ttest_shap_values.pvalue, delimiter=",")

            if len(featuresets) > 1:

                predictions = []
                shap_values = []

                for modus in ["normal", "hypothesis"]:

                    print(f'{bcolors.HEADER}{dependent_variable, featuresets, modus}{bcolors.ENDC}')

                    save_dir = pathlib.Path(f"results/{dependent_variable}/{experiment}/{modus}/")
                    save_dir.mkdir(parents=True, exist_ok=True)

                    featuresets = [experiment]

                    data = utils.preprocess_data(save_dir=save_dir, dependent_variable=dependent_variable, featuresets=featuresets, modus=modus)

                    optimizer_run = optimizer.Optimizer(data=data,save_dir=save_dir,dependent_variable=dependent_variable)
                    preds, shaps = optimizer_run.run_optimization()
                    predictions.append(preds)
                    shap_values.append(shaps)

                ttest_predictions = stats.ttest_ind(predictions[0], predictions[1], equal_var=False)
                print(f"T-Test Predictions: {ttest_predictions.pvalue}")
                np.savetxt(save_dir.parent.joinpath('ttest_predictions.csv'), ttest_predictions.pvalue.reshape(-1, 1), delimiter=",")

                ttest_shap_values = stats.ttest_ind(shap_values[0].values, shap_values[1].values, equal_var=False)
                print(f"T-Test SHAP Values: {ttest_shap_values.pvalue}")
                np.savetxt(save_dir.parent.joinpath('ttest_shap_values.csv'), ttest_shap_values.pvalue, delimiter=",")

        featuresets = ["RESPONSIBILITY", "POLICY_ACTIONS", "CLIMATE_OPINION", "PERSONAL_ACTIONS"]

        predictions = []
        shap_values = []

        for modus in ["normal", "hypothesis"]:
            print(f'{bcolors.HEADER}{dependent_variable, featuresets, modus}{bcolors.ENDC}')

            save_dir = pathlib.Path(f"results/{dependent_variable}/{'#'.join(featuresets)}/{modus}/")
            save_dir.mkdir(parents=True, exist_ok=True)

            data = utils.preprocess_data(save_dir=save_dir, dependent_variable=dependent_variable,
                                         featuresets=featuresets, modus=modus)

            optimizer_run = optimizer.Optimizer(data=data, save_dir=save_dir, dependent_variable=dependent_variable)
            preds, shaps = optimizer_run.run_optimization()
            predictions.append(preds)
            shap_values.append(shaps)

        ttest_predictions = stats.ttest_ind(predictions[0], predictions[1], equal_var=False)
        print(f"T-Test Predictions: {ttest_predictions.pvalue}")
        np.savetxt(save_dir.parent.joinpath('ttest_predictions.csv'), ttest_predictions.pvalue.reshape(-1, 1), delimiter=",")

        ttest_shap_values = stats.ttest_ind(shap_values[0].values, shap_values[1].values, equal_var=False)
        print(f"T-Test SHAP Values: {ttest_shap_values.pvalue}")
        np.savetxt(save_dir.parent.joinpath('ttest_shap_values.csv'), ttest_shap_values.pvalue, delimiter=",")

if __name__ == "__main__":

    run()
