import optimizer
import utils
import pathlib
import datetime

def run(experiment: str = None, climate_belief_score: bool = None, model_name: str = None):

    save_dir = pathlib.Path(
        "results/" + model_name + "/" + experiment + "/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "/")

    data = utils.preprocess_data(experiment=experiment, climate_belief_score=climate_belief_score, save_dir=save_dir)

    optimizer_run = optimizer.Optimizer(experiment=experiment, data=data, model_name=model_name, save_dir=save_dir)
    optimizer_run.run_optimization()