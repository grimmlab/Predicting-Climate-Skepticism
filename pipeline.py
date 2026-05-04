import optimizer
import utils
import pathlib
import scipy.stats as stats
import os
import shutil

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
                predictions.append(optimizer_run.run_optimization()[0])
                shap_values.append(optimizer_run.run_optimization()[1])

            ttest_predictions = stats.ttest_ind(predictions[1], predictions[0])
            print(f"T-Test Predictions: {ttest_predictions}")
            with open(save_dir.parent.joinpath('ttest_predictions.txt'), 'w') as f:
                f.write(str(ttest_predictions))

            ttest_shap_values = stats.ttest_ind(shap_values[1].values, shap_values[0].values)
            print(ttest_shap_values)
            with open(save_dir.parent.joinpath('ttest_shap_values.txt'), 'w') as f:
                f.write(str(ttest_shap_values))

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
                    predictions.append(optimizer_run.run_optimization()[0])
                    shap_values.append(optimizer_run.run_optimization()[1])

                ttest_predictions = stats.ttest_ind(predictions[1], predictions[0])
                print(f"T-Test Predictions: {ttest_predictions}")
                with open(save_dir.parent.joinpath('ttest_predictions.txt'), 'w') as f:
                    f.write(str(ttest_predictions))

                ttest_shap_values = stats.ttest_ind(shap_values[1].values, shap_values[0].values)
                print(ttest_shap_values)
                with open(save_dir.parent.joinpath('ttest_shap_values.txt'), 'w') as f:
                    f.write(str(ttest_shap_values))

        featuresets = ["RESPONSIBILITY", "POLICY_ACTIONS", "CLIMATE_OPINION", "PERSONAL_ACTIONS"]

        for modus in ["normal", "hypothesis"]:
            print(f'{bcolors.HEADER}{dependent_variable, featuresets, modus}{bcolors.ENDC}')

            save_dir = pathlib.Path(f"results/{dependent_variable}/{'#'.join(featuresets)}/{modus}/")
            save_dir.mkdir(parents=True, exist_ok=True)

            data = utils.preprocess_data(save_dir=save_dir, dependent_variable=dependent_variable,
                                         featuresets=featuresets, modus=modus)

            optimizer_run = optimizer.Optimizer(data=data, save_dir=save_dir, dependent_variable=dependent_variable)
            predictions.append(optimizer_run.run_optimization()[0])
            shap_values.append(optimizer_run.run_optimization()[1])

        ttest_predictions = stats.ttest_ind(predictions[1], predictions[0])
        print(f"T-Test Predictions: {ttest_predictions}")
        with open(save_dir.parent.joinpath('ttest_predictions.txt'), 'w') as f:
            f.write(str(ttest_predictions))

        ttest_shap_values = stats.ttest_ind(shap_values[1].values, shap_values[0].values)
        print(ttest_shap_values)
        with open(save_dir.parent.joinpath('ttest_shap_values.txt'), 'w') as f:
            f.write(str(ttest_shap_values))

if __name__ == "__main__":

    run()
