from model.sem_eval_model import SemEvalModel

import argparse

parser = argparse.ArgumentParser(description="Model Parameters")
parser.add_argument("--config_file", help="Path to the config file.")
args = parser.parse_args()

with open(args.config_file, "r") as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)  # noqa: 701

if __name__ == "__main__":
    for epochs in range(10, 11):
        fine_tuner = SemEvalModel(config)
        fine_tuner.train(epochs)
