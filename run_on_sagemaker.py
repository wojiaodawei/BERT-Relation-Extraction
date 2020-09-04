import sagemaker
from ml_utils.config_parser import ConfigParser
from ml_utils.console_args import args
from ml_utils.path_operations import strip_to_tmp
from sagemaker.pytorch import PyTorch

from logger import logger

sagemaker_session = sagemaker.Session()

config = ConfigParser(
    args.config_file, console_args=dict(args._get_kwargs())
).parse()


if __name__ == "__main__":
    hyperparameters = {"config_file": args.config_file}

    metric_definitions = [
        {
            "Name": "Train Loss",
            "Regex": "Train Loss: (.*?)!",
        },
        {
            "Name": "Train LM Accuracy",
            "Regex": "Train LM Accuracy: (.*?)!",
        },
        {
            "Name": "Validation LM Accuracy",
            "Regex": "Validation LM Accuracy: (.*?)!",
        },
        {
            "Name": "Validation MTB Binary Cross Entropy",
            "Regex": "Validation MTB Binary Cross Entropy: (.*?)!",
        },
    ]
    logger.info("Strip dir")
    source_dir = strip_to_tmp(
        exclude=[
            "./requirements.txt",
            "./data/cnn_dm.txt",
            "./data/cnn.txt",
            "./data/cnn-small.txt",
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
        metric_definitions=metric_definitions,
        enable_sagemaker_metrics=True,
    )

    if config.get("aws_train_instance_type", "local") == "local":
        inputs = {"training": f"file://data/pretraining_small"}
    else:
        inputs = {"training": config.get("aws_data_input")}

    logger.info("Start training")
    estimator.fit(inputs)
