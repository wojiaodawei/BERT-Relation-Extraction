from ml_utils.config_parser import ConfigParser
from ml_utils.console_args import args

from model.sem_eval_model import SemEvalModel

config = ConfigParser(
    args.config_file, console_args=dict(args._get_kwargs())
).parse()


if __name__ == "__main__":
    for epochs in range(10, 11):
        fine_tuner = SemEvalModel(config)
        fine_tuner.train(epochs)
