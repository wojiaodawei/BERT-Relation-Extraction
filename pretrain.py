import sys
import warnings

from model.mtb_model import MTBModel

import argparse

parser = argparse.ArgumentParser(description="Model Parameters")
parser.add_argument("--config_file", help="Path to the config file.")
args = parser.parse_args()

with open(args.config_file, "r") as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)  # noqa: 701

if not sys.warnoptions:
    warnings.simplefilter("ignore")

if __name__ == "__main__":
    pretrainer = MTBModel(config)
    output = pretrainer.train(
        save_best_model_only=config.get("save_best_model_only")
    )
