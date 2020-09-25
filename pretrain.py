import logging

from ml_utils.config_parser import ConfigParser
from ml_utils.console_args import args

from constants import LOG_DATETIME_FORMAT, LOG_FORMAT, LOG_LEVEL
from model.mtb_model import MTBModel

logging.basicConfig(
    format=LOG_FORMAT, datefmt=LOG_DATETIME_FORMAT, level=LOG_LEVEL
)
logger = logging.getLogger("__file__")

config = ConfigParser(
    args.config_file, console_args=dict(args._get_kwargs())
).parse()

if __name__ == "__main__":
    pretrainer = MTBModel(config)
    output = pretrainer.train(
        save_best_model_only=config.get("save_best_model_only")
    )
