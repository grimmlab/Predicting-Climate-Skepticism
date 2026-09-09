import optimizer
import utils
import pathlib
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

            save_dir = pathlib.Path(f"results/{dependent_variable}/{'#'.join(featuresets)}/")
            save_dir.mkdir(parents=True, exist_ok=True)

            data = utils.preprocess_data(save_dir=save_dir, dependent_variable=dependent_variable, featuresets=featuresets)

            for seed in range(30):

                print(f'{bcolors.HEADER}{dependent_variable, featuresets, seed}{bcolors.ENDC}')

                save_dir_seed = save_dir.joinpath(str(seed))

                optimizer_run = optimizer.Optimizer(
                    data=data, save_dir=save_dir_seed, dependent_variable=dependent_variable, seed=seed)
                optimizer_run.run_optimization()

            if len(featuresets) > 1:

                save_dir = pathlib.Path(f"results/{dependent_variable}/{experiment}/")
                save_dir.mkdir(parents=True, exist_ok=True)

                featuresets = [experiment]

                data = utils.preprocess_data(
                    save_dir=save_dir, dependent_variable=dependent_variable, featuresets=featuresets)

                for seed in range(30):

                    print(f'{bcolors.HEADER}{dependent_variable, featuresets}{bcolors.ENDC}')

                    save_dir_seed = save_dir.joinpath(str(seed))

                    optimizer_run = optimizer.Optimizer(
                        data=data,save_dir=save_dir_seed,dependent_variable=dependent_variable, seed=seed)
                    optimizer_run.run_optimization()

if __name__ == "__main__":

    run()
