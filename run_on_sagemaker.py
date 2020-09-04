import logging

import sagemaker
from ml_utils.config_parser import ConfigParser
from ml_utils.console_args import args
from ml_utils.path_operations import strip_to_tmp
from sagemaker.pytorch import PyTorch

from constants import LOG_DATETIME_FORMAT, LOG_FORMAT, LOG_LEVEL

logging.basicConfig(
    format=LOG_FORMAT, datefmt=LOG_DATETIME_FORMAT, level=LOG_LEVEL
)
logger = logging.getLogger("__file__")

sagemaker_session = sagemaker.Session()

config = ConfigParser(
    args.config_file, console_args=dict(args._get_kwargs())
).parse()

METRIC_DEFINITION = config.get("sagemaker_metric_definitions", [])


if __name__ == "__main__":
    hyperparameters = {"config_file": args.config_file}

    logger.info("Strip dir")
    source_dir = strip_to_tmp(
        exclude=[
            "./requirements.txt",
            "./data/cnn_dm.txt",
            "./data/cnn.txt",
            "./data/cnn_small.txt",
        ],
        file_ext=[".py", ".yaml", ".yml", ".sh", ".txt", ".sql"],
    )

    estimator = PyTorch(
        entry_point="pretrain.sh",
        hyperparameters=hyperparameters,
        source_dir=source_dir,
        instance_type=config.get("aws_train_instance_type", "local"),
        instance_count=1,
        dependencies=["ssh"],
        role="AmazonSageMaker-ExecutionRole-20180822T000617",
        framework_version="1.6",
        py_version="py3",
        base_job_name=config.get("experiment_name", None),
        metric_definitions=METRIC_DEFINITION,
        enable_sagemaker_metrics=True,
        debugger_hook_config=False,
    )

    if config.get("aws_train_instance_type", "local") == "local":
        inputs = {"training": f"file://data/pretraining_small"}
    else:
        inputs = {"training": config.get("aws_data_input")}

    logger.info("Start training")
    estimator.fit(inputs)
