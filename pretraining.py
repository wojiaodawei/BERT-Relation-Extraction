import logging
from argparse import ArgumentParser

from constants import LOG_DATETIME_FORMAT, LOG_FORMAT, LOG_LEVEL
from src.trainer import train_and_fit

logging.basicConfig(
    format=LOG_FORMAT, datefmt=LOG_DATETIME_FORMAT, level=LOG_LEVEL
)
logger = logging.getLogger("__file__")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--pretrain_data",
        type=str,
        default="./data/cnn.txt",
        help="pre-training data.txt file path",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=0,
        help="""1: Freeze most layers until classifier layers
        \n0: Don\'t freeze
        \n(Probably best not to freeze if GPU memory is sufficient)""",
    )
    parser.add_argument(
        "--gradient_acc_steps",
        type=int,
        default=2,
        help="No. of steps of gradient accumulation",
    )
    parser.add_argument(
        "--max_norm", type=float, default=1.0, help="Clipped gradient norm"
    )
    parser.add_argument(
        "--fp16",
        type=int,
        default=0,
        help="1: use mixed precision ; 0: use floating point 32",
    )  # mixed precision doesn't seem to train well
    parser.add_argument(
        "--num_epochs", type=int, default=18, help="No of epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="learning rate"
    )

    args = parser.parse_args()

    output = train_and_fit(args)
