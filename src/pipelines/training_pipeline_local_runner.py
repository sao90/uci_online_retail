"""
Run pipelines locally for testing and development.
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


def run_training_pipeline(config_path: str) -> None:
    """
    Run the training pipeline locally.
    Pipeline consists of 2 steps:
    1. Train model and save trained model to file
    2. Backtest model in training period save backtest scores to file
    Args:
        config_path: Path to the training pipeline YAML configuration file
    """
    setup_logging()
    logger.info("Starting local training pipeline run...")
    config = read_yaml(config_path)
    logger.info(f"Loaded pipeline config: {config}")
    
    # Step 1: Train model
    MODEL_CONFIG = config["inputs"]["train_model__model_config"]["default"]
    TARGET_TRAINING_DATA_PATH = config["inputs"][
        "train_model__target_training_data_path"
    ]["default"]
    PAST_COVARIATES_PATH = config["inputs"]["train_model__past_covariates_path"][
        "default"
    ]
    FUTURE_COVARIATES_PATH = config["inputs"]["train_model__future_covariates_path"][
        "default"
    ]
    FUTURE_COVARIATES_COLUMNS = config["inputs"][
        "train_model__future_covariates_columns"
    ]["default"]
    PAST_COVARIATES_COLUMNS = config["inputs"]["train_model__past_covariates_columns"][
        "default"
    ]
    TARGET_COLUMN_NAME = config["inputs"]["train_model__target_column_name"]["default"]
    TIME_COLUMN_NAME = config["inputs"]["train_model__time_column_name"]["default"]
    MODEL_OUTPUT = config["inputs"]["train_model__model_output"]["default"]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.components.training.train_model",
            "--model_config",
            MODEL_CONFIG,
            "--target_training_data_path",
            TARGET_TRAINING_DATA_PATH,
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
            "--model_output",
            MODEL_OUTPUT,
        ],
        check=True,
    )

    # Step 2: Backtest model
    
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.components.training.backtest_model",
            "--model_path",
            MODEL_OUTPUT,
            "--target_training_data_path",
            TARGET_TRAINING_DATA_PATH,
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
            "--backtest_start",
            str(config["inputs"]["backtest_model__backtest_start"]["default"]),
            "--scores_output_path",
            config["inputs"]["backtest_model__scores_output_path"]["default"],
            
        ],
        check=True,
    )

if __name__ == "__main__":
    run_training_pipeline("pipelines/training_pipeline.yaml")
