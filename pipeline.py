import optimizer
import utils
import pathlib
import datetime

def run(model_name: str = None):
    DEPENDENT_VARIABLES = ["climatedeniers", "climate_eb_problem", "climate_belief_score"]
    EXPERIMENTS = [
        "PERSONAL_DATA", "ECONOMIC_PREFERENCES", "CLIMATE_EUROBAROMETER", "CLIMATE_KNOWLEDGE",
        "CLIMATE_POLICIES_ACTIONS", "CLIMATE_TRUST", "BIOECONOMY", "MORAL_VALUES"]

    for dependent_variable in DEPENDENT_VARIABLES:
        for experiment in EXPERIMENTS:
            save_dir = pathlib.Path(f"results/{dependent_variable}/{experiment}/{model_name}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/")
            save_dir.mkdir(parents=True, exist_ok=True)

            featuresets = EXPERIMENTS[:EXPERIMENTS.index(experiment)+1]

            data = utils.preprocess_data(
                save_dir=save_dir, dependent_variable=dependent_variable, featuresets=featuresets)

            optimizer_run = optimizer.Optimizer(
                experiment=experiment, data=data, model_name=model_name, save_dir=save_dir,
                dependent_variable=dependent_variable)
            optimizer_run.run_optimization()