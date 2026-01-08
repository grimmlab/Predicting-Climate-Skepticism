import optimizer
import utils
import pathlib
import datetime

def run(model_name: str = None):
    DEPENDENT_VARIABLES = ["climate_state_manmade", "climate_eb_problem", "climate_state_convinced"]
    EXPERIMENTS = [
        "PERSONAL_DATA", "ADDITIONAL_VARIABLES",
        "MORAL_VALUES", "ECONOMIC_PREFERENCES", "CLIMATE_EUROBAROMETER",
        "CLIMATE_KNOWLEDGE", "CLIMATE_POLICIES_ACTIONS", "CLIMATE_TRUST", "BIOECONOMY"]

    for dependent_variable in DEPENDENT_VARIABLES:
        for experiment in EXPERIMENTS:
            featuresets = EXPERIMENTS[:EXPERIMENTS.index(experiment) + 1]
            print(dependent_variable, featuresets)
            save_dir = pathlib.Path(
                f"results/{dependent_variable}/{'#'.join(featuresets)}/{model_name}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/")
            save_dir.mkdir(parents=True, exist_ok=True)

            data = utils.preprocess_data(
                save_dir=save_dir, dependent_variable=dependent_variable, featuresets=featuresets)

            optimizer_run = optimizer.Optimizer(
                experiment=experiment, data=data, model_name=model_name, save_dir=save_dir,
                dependent_variable=dependent_variable)
            optimizer_run.run_optimization()

            if len(featuresets) > 1:
                print(dependent_variable, experiment)
                save_dir = pathlib.Path(
                    f"results/{dependent_variable}/{experiment}/{model_name}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/")
                save_dir.mkdir(parents=True, exist_ok=True)

                featuresets = [experiment]

                data = utils.preprocess_data(
                    save_dir=save_dir, dependent_variable=dependent_variable, featuresets=featuresets)

                optimizer_run = optimizer.Optimizer(
                    experiment=experiment, data=data, model_name=model_name, save_dir=save_dir,
                    dependent_variable=dependent_variable)
                optimizer_run.run_optimization()
