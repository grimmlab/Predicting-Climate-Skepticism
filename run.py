import argparse
import pipeline

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-tbdc', '--to_be_dropped_columns', nargs='*', help='Define columns to drop.', required=True)
    parser.add_argument('-mn', '--model_name', help='Define model name.', required=True)

    args = vars(parser.parse_args())

    pipeline.run(**args)