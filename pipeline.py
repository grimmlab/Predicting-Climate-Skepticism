import optimizer
import utils
import pathlib
import datetime

def run(experiment: str = None, model_name: str = None):

    save_dir = pathlib.Path(
        "results/" + model_name + "/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "/")

    data = utils.preprocess_data(experiment=experiment, save_dir=save_dir)

    optimizer_run = optimizer.Optimizer(experiment=experiment, data=data, model_name=model_name, save_dir=save_dir)
    optimizer_run.run_optimization()