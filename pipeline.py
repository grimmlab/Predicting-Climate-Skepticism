import optimizer
import utils
import pathlib
import datetime

def run(model_name: str = None):
    DEPENDENT_VARIABLES = ["climate_state_manmade", "climate_eb_problem", "climate_state_convinced"]
    EXPERIMENTS = [
        "PERSONAL_DATA", "CLIMATE_EUROBAROMETER", "CLIMATE_KNOWLEDGE", "CLIMATE_POLICIES_ACTIONS", "CLIMATE_TRUST",
        "BIOECONOMY", "MORAL_VALUES", "ECONOMIC_PREFERENCES", "ADDITIONAL_VARIABLES"]

    for dependent_variable in DEPENDENT_VARIABLES:
        for experiment in EXPERIMENTS:
            print(dependent_variable, experiment)
            save_dir = pathlib.Path(f"results/{dependent_variable}/{experiment}/{model_name}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/")
            save_dir.mkdir(parents=True, exist_ok=True)

            featuresets = EXPERIMENTS[:EXPERIMENTS.index(experiment)+1]

            data = utils.preprocess_data(
                save_dir=save_dir, dependent_variable=dependent_variable, featuresets=featuresets)

            optimizer_run = optimizer.Optimizer(
                experiment=experiment, data=data, model_name=model_name, save_dir=save_dir,
                dependent_variable=dependent_variable)
            optimizer_run.run_optimization()