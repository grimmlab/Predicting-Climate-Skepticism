import optimizer_ML
import utils
import pathlib
import os
import shutil
import argparse

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

def run(DEPENDENT_VARIABLES: list):

    EXPERIMENTS = ["DEMOGRAPHICS", "PERSONAL_CONVICTION", "MORAL_FOUNDATIONS", "ECONOMIC_PREFERENCES", "RESPONSIBILITY",
                   "POLICY_ACTIONS", "CLIMATE_OPINION", "PERSONAL_ACTIONS",
                   "DEMOGRAPHICS#PERSONAL_CONVICTION", "DEMOGRAPHICS#PERSONAL_CONVICTION#MORAL_FOUNDATIONS",
                   "DEMOGRAPHICS#PERSONAL_CONVICTION#MORAL_FOUNDATIONS#ECONOMIC_PREFERENCES",
                   "DEMOGRAPHICS#PERSONAL_CONVICTION#MORAL_FOUNDATIONS#ECONOMIC_PREFERENCES#RESPONSIBILITY",
                   "DEMOGRAPHICS#PERSONAL_CONVICTION#MORAL_FOUNDATIONS#ECONOMIC_PREFERENCES#RESPONSIBILITY#POLICY_ACTIONS",
                   "DEMOGRAPHICS#PERSONAL_CONVICTION#MORAL_FOUNDATIONS#ECONOMIC_PREFERENCES#RESPONSIBILITY#POLICY_ACTIONS#CLIMATE_OPINION",
                   "DEMOGRAPHICS#PERSONAL_CONVICTION#MORAL_FOUNDATIONS#ECONOMIC_PREFERENCES#RESPONSIBILITY#POLICY_ACTIONS#CLIMATE_OPINION#PERSONAL_ACTIONS"]

    for dependent_variable in DEPENDENT_VARIABLES:

        if os.path.isdir(f"results_ML/{dependent_variable}"):
            shutil.rmtree(f"results_ML/{dependent_variable}")

        for experiment in EXPERIMENTS:
            featuresets = experiment.split("#")

            save_dir = pathlib.Path(f"results/{dependent_variable}/{experiment}/")
            save_dir.mkdir(parents=True, exist_ok=True)

            data = utils.preprocess_data(save_dir=save_dir, dependent_variable=dependent_variable, featuresets=featuresets)

            for seed in range(30):

                print(f'{bcolors.HEADER}{dependent_variable, featuresets, seed}{bcolors.ENDC}')

                save_dir_seed = save_dir.joinpath(str(seed))

                optimizer_run = optimizer_ML.Optimizer(
                    data=data, save_dir=save_dir_seed, dependent_variable=dependent_variable, seed=seed)
                optimizer_run.run_optimization()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-DV','--DEPENDENT_VARIABLES', nargs='+', required=True)

    args = vars(parser.parse_args())

    run(**args)
