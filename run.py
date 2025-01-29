import argparse
import pipeline

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--experiment', help='Define experiment.', required=True)
    parser.add_argument('-cbs', '--climate_belief_score', action=argparse.BooleanOptionalAction,  help='Define climate_belief_score.', required=True)
    parser.add_argument('-mn', '--model_name', help='Define model name.', required=True)

    args = vars(parser.parse_args())

    pipeline.run(**args)