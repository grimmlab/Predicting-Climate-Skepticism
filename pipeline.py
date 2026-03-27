import optimizer
import utils
import pathlib
import datetime
import scipy.stats as stats

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
    EXPERIMENTS = ["PERSONAL_DATA", "ADDITIONAL_VARIABLES", "MORAL_VALUES", "ECONOMIC_PREFERENCES", "CLIMATE_EUROBAROMETER",
        "CLIMATE_KNOWLEDGE", "CLIMATE_POLICIES_ACTIONS", "CLIMATE_TRUST", "BIOECONOMY"]

    for dependent_variable in DEPENDENT_VARIABLES:
        for experiment in EXPERIMENTS:
            featuresets = EXPERIMENTS[:EXPERIMENTS.index(experiment) + 1]

            predictions = []

            for modus in ["normal", "hypothesis"]:
                print(f'{bcolors.HEADER}{dependent_variable, featuresets, modus}{bcolors.ENDC}')

                save_dir = pathlib.Path(f"results/{dependent_variable}/{'#'.join(featuresets)}/{modus}/")
                save_dir.mkdir(parents=True, exist_ok=True)

                data = utils.preprocess_data(save_dir=save_dir, dependent_variable=dependent_variable, featuresets=featuresets, modus=modus)

                optimizer_run = optimizer.Optimizer(experiment=experiment, data=data, save_dir=save_dir, dependent_variable=dependent_variable)
                predictions.append(optimizer_run.run_optimization())

            with open(save_dir.parent.joinpath('ttest.txt'), 'w') as f:
                f.write(str(stats.ttest_ind(predictions[1], predictions[0])))

            if len(featuresets) > 1:

                predictions = []

                for modus in ["normal", "hypothesis"]:

                    print(f'{bcolors.HEADER}{dependent_variable, featuresets, modus}{bcolors.ENDC}')

                    save_dir = pathlib.Path(f"results/{dependent_variable}/{experiment}/{modus}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/")
                    save_dir.mkdir(parents=True, exist_ok=True)

                    featuresets = [experiment]

                    data = utils.preprocess_data(save_dir=save_dir, dependent_variable=dependent_variable, featuresets=featuresets, modus=modus)

                    optimizer_run = optimizer.Optimizer(experiment=experiment, data=data, save_dir=save_dir, dependent_variable=dependent_variable)
                    predictions.append(optimizer_run.run_optimization())

                print(stats.ttest_ind(predictions[1], predictions[0]))
                with open(save_dir.joinpath('ttest.txt'), 'w') as f:
                    f.write(str(stats.ttest_ind(predictions[1], predictions[0])))

if __name__ == "__main__":

    run()
