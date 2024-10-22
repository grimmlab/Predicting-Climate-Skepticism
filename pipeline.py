import optimizer
import utils

def run(to_be_dropped_columns: list = None, model_name: str = None):

    data = utils.preprocess_data(to_be_dropped_columns=to_be_dropped_columns)

    optimizer_run = optimizer.Optimizer(to_be_dropped_columns=to_be_dropped_columns, data=data, model_name=model_name)
    optimizer_run.run_optimization()