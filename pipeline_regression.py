import warnings
warnings.filterwarnings('ignore')
import pathlib
import scipy.stats as stats
import os
import shutil
import numpy as np
import utils
import optimizer_regression
import pandas as pd

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
    DEPENDENT_VARIABLES = ["climate_state_manmade", "climate_eb_problem", "climate_state_convinced"]
    EXPERIMENTS = ["DEMOGRAPHICS"]

    if os.path.isdir("results_regression"):
        shutil.rmtree("results_regression")

    for dependent_variable in DEPENDENT_VARIABLES:
        for experiment in EXPERIMENTS:
            featuresets = EXPERIMENTS[:EXPERIMENTS.index(experiment) + 1]

            predictions = []
            shap_values = []

            for modus in ["normal", "hypothesis"]:
                print(f'{bcolors.HEADER}{dependent_variable, featuresets, modus}{bcolors.ENDC}')

                save_dir = pathlib.Path(f"results_regression/{dependent_variable}/{'#'.join(featuresets)}/{modus}/")
                save_dir.mkdir(parents=True, exist_ok=True)

                data = utils.preprocess_data(save_dir=save_dir, dependent_variable=dependent_variable, featuresets=featuresets, modus=modus)

                optimizer_run = optimizer_regression.Optimizer(data=data, save_dir=save_dir, dependent_variable=dependent_variable)
                preds, shaps = optimizer_run.run_optimization()
                predictions.append(preds)
                shap_values.append(shaps)

            ttest_predictions = stats.ttest_ind(predictions[0], predictions[1], equal_var=False)
            print(f"T-Test Predictions: {ttest_predictions.pvalue}")
            np.savetxt(save_dir.parent.joinpath('ttest_predictions.csv'), ttest_predictions.pvalue.reshape(-1, 1), delimiter=",")

            ttest_shap_values = pd.DataFrame(
                stats.ttest_ind(pd.DataFrame(shap_values[0].values, columns=shap_values[0].feature_names),
                                pd.DataFrame(shap_values[1].values, columns=shap_values[1].feature_names),
                                equal_var=False).pvalue.reshape(1, -1), columns=shap_values[0].feature_names)
            print(f"T-Test SHAP Values: {ttest_shap_values}")
            ttest_shap_values.to_csv(save_dir.parent.joinpath('ttest_shap_values.csv'), index=False)

if __name__ == "__main__":

    run()