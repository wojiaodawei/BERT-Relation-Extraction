import logging
import sys

LOG_FORMAT = "%(asctime)s [%(levelname)s]: %(message)s"
LOG_DATETIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"
LOG_LEVEL = logging.INFO

logging.basicConfig(
    format=LOG_FORMAT, datefmt=LOG_DATETIME_FORMAT, level=LOG_LEVEL
)

logger = logging.getLogger()

# Needed to be able to see Outputs on AWS Sagemaker
# See: https://github.com/aws/sagemaker-training-toolkit/issues/37
logger.removeHandler(logger.handlers[0])
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(LOG_LEVEL)
formatter = logging.Formatter(LOG_FORMAT)
handler.setFormatter(formatter)
logger.addHandler(handler)
