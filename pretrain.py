import sys
import warnings

from ml_utils.config_parser import ConfigParser
from ml_utils.console_args import args

from model.mtb_model import MTBModel

config = ConfigParser(
    args.config_file, console_args=dict(args._get_kwargs())
).parse()


if not sys.warnoptions:
    warnings.simplefilter("ignore")

if __name__ == "__main__":
    pretrainer = MTBModel(config)
    output = pretrainer.train(
        save_best_model_only=config.get("save_best_model_only")
    )
