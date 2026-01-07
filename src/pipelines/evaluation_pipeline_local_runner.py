"""
Run evaluation pipelines locally for testing and development.
"""

import os
import sys
import logging
import subprocess

from dotenv import load_dotenv

from src.modules.log_config import setup_logging
from src.modules.utils import read_yaml

logger = logging.getLogger(__name__)

load_dotenv()

PROJECT_ROOT = os.getenv("REPOSITORY_ROOT")
sys.path.insert(0, PROJECT_ROOT)


def run_evaluation_pipeline(config_path: str) -> None:
    """
    Run the evaluation pipeline locally.
    Pipeline consists of 1 step:
    1. Evaluate multiple models and save the backtest scores to file
    Args:
        config_path: Path to the evaluation pipeline YAML configuration file
    """
    setup_logging()
    logger.info("Starting local evaluation pipeline run...")
    config = read_yaml(config_path)
    logger.info(f"Loaded pipeline config: {config}")

    # Step 1: Evaluate models
    PATHS_TO_MODELS = config["inputs"]["evaluate_models__paths_to_models"]["default"]
    TARGET_TRAINING_DATA_PATH = config["inputs"][
        "evaluate_models__target_training_data_path"
    ]["default"]
    TARGET_TEST_DATA_PATH = config["inputs"]["evaluate_models__target_test_data_path"][
        "default"
    ]
    PAST_COVARIATES_PATH = config["inputs"]["evaluate_models__past_covariates_path"][
        "default"
    ]
    FUTURE_COVARIATES_PATH = config["inputs"][
        "evaluate_models__future_covariates_path"
    ]["default"]
    FUTURE_COVARIATES_COLUMNS = config["inputs"][
        "evaluate_models__future_covariates_columns"
    ]["default"]
    PAST_COVARIATES_COLUMNS = config["inputs"][
        "evaluate_models__past_covariates_columns"
    ]["default"]
    TARGET_COLUMN_NAME = config["inputs"]["evaluate_models__target_column_name"][
        "default"
    ]
    TIME_COLUMN_NAME = config["inputs"]["evaluate_models__time_column_name"]["default"]
    SCORES_OUTPUT = config["inputs"]["evaluate_models__scores_output"]["default"]

    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.components.evaluation.evaluate_models",
            "--paths_to_models",
        ]
        + PATHS_TO_MODELS
        + [
            "--target_training_data_path",
            TARGET_TRAINING_DATA_PATH,
            "--target_test_data_path",
            TARGET_TEST_DATA_PATH,
            "--past_covariates_path",
            PAST_COVARIATES_PATH,
            "--future_covariates_path",
            FUTURE_COVARIATES_PATH,
            "--future_covariates_columns",
        ]
        + FUTURE_COVARIATES_COLUMNS
        + [
            "--past_covariates_columns",
        ]
        + PAST_COVARIATES_COLUMNS
        + [
            "--target_column_name",
            TARGET_COLUMN_NAME,
            "--time_column_name",
            TIME_COLUMN_NAME,
            "--scores_output",
            SCORES_OUTPUT,
        ],
        check=True,
    )


if __name__ == "__main__":
    run_evaluation_pipeline("app_config/dev/evaluation_pipeline.yaml")
