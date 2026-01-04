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


def run_preprocessing_pipeline(config_path: str) -> None:
    """
    Run the preprocessing pipeline locally.
    Pipeline consists of 4 steps:

    1. Ingest data from SQLite to Parquet (local file)
    2. Clean data and save cleaned data to Parquet (local file)
    3. Split data into train/test and save to Parquet (local file)
    4. Generate features and save to Parquet (local file)

    Args:
        config_path: Path to the preprocessing pipeline YAML configuration file
    """
    setup_logging()
    logger.info("Starting local preprocessing pipeline run...")

    config = read_yaml(config_path)
    logger.info(f"Loaded pipeline config: {config}")

    # Step 1: Ingest data
    DB_PATH = config["inputs"]["ingest_data__db_path"]["default"]
    DB_INPUT_TABLE_NAME = config["inputs"]["ingest_data__table_name"]["default"]
    RAW_DATA_OUTPUT_PATH = config["inputs"]["ingest_data__output_data"]["default"]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.components.preprocessing.ingest_data",
            "--db_path",
            DB_PATH,
            "--table_name",
            DB_INPUT_TABLE_NAME,
            "--output_data",
            RAW_DATA_OUTPUT_PATH,
        ],
        check=True,
    )

    # Step 2: Clean data
    COUNTRIES = config["inputs"]["clean_data__countries"]["default"]
    CLEANED_DATA_OUTPUT_PATH = config["inputs"]["clean_data__output_data"]["default"]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.components.preprocessing.clean_data",
            "--input_data",
            RAW_DATA_OUTPUT_PATH,
            "--countries",
            COUNTRIES,
            "--output_data",
            CLEANED_DATA_OUTPUT_PATH,
        ],
        check=True,
    )

    # Step 3: Split data
    TARGET_COLUMN = config["inputs"]["split_data__target_column"]["default"]
    DATE_COLUMN = config["inputs"]["split_data__date_column"]["default"]
    DAYS_IN_TEST_SPLIT = config["inputs"]["split_data__days_in_test_split"]["default"]
    SPLIT_OUTPUT_TRAIN_TARGETS = config["inputs"]["split_data__output_train_targets"][
        "default"
    ]
    SPLIT_OUTPUT_TEST_TARGETS = config["inputs"]["split_data__output_test_targets"][
        "default"
    ]
    SPLIT_OUTPUT_FEATURES_RAW = config["inputs"]["split_data__output_features"][
        "default"
    ]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.components.preprocessing.split_data",
            "--input_data",
            CLEANED_DATA_OUTPUT_PATH,
            "--target_column",
            TARGET_COLUMN,
            "--date_column",
            DATE_COLUMN,
            "--days_in_test_split",
            str(DAYS_IN_TEST_SPLIT),
            "--output_train_targets",
            SPLIT_OUTPUT_TRAIN_TARGETS,
            "--output_test_targets",
            SPLIT_OUTPUT_TEST_TARGETS,
            "--output_features",
            SPLIT_OUTPUT_FEATURES_RAW,
        ],
        check=True,
    )

    # Step 4: Feature engineering
    CUSTOMER_ID_COLUMN = config["inputs"]["feature_engineering__customer_id_column"][
        "default"
    ]
    TRANSACTION_ID_COLUMN = config["inputs"][
        "feature_engineering__transaction_id_column"
    ]["default"]
    ARTICLE_ID_COLUMN = config["inputs"]["feature_engineering__article_id_column"][
        "default"
    ]
    REVENUE_COLUMN = config["inputs"]["feature_engineering__revenue_column"]["default"]
    FEATURE_ENGINEERING_OUTPUT_TRAIN_TARGETS = config["inputs"][
        "feature_engineering__output_train_targets"
    ]["default"]
    FEATURE_ENGINEERING_OUTPUT_TEST_TARGETS = config["inputs"][
        "feature_engineering__output_test_targets"
    ]["default"]
    FEATURE_ENGINEERING_OUTPUT_PAST_COVARIATES = config["inputs"][
        "feature_engineering__output_past_covariates"
    ]["default"]
    FEATURE_ENGINEERING_OUTPUT_FUTURE_COVARIATES = config["inputs"][
        "feature_engineering__output_future_covariates"
    ]["default"]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.components.preprocessing.feature_engineering",
            "--target_train_file",
            SPLIT_OUTPUT_TRAIN_TARGETS,
            "--target_test_file",
            SPLIT_OUTPUT_TEST_TARGETS,
            "--features_raw_file",
            SPLIT_OUTPUT_FEATURES_RAW,
            "--target_column",
            TARGET_COLUMN,
            "--date_column",
            DATE_COLUMN,
            "--transaction_id_column",
            TRANSACTION_ID_COLUMN,
            "--customer_id_column",
            CUSTOMER_ID_COLUMN,
            "--article_id_column",
            ARTICLE_ID_COLUMN,
            "--revenue_column",
            REVENUE_COLUMN,
            "--output_train_targets",
            FEATURE_ENGINEERING_OUTPUT_TRAIN_TARGETS,
            "--output_test_targets",
            FEATURE_ENGINEERING_OUTPUT_TEST_TARGETS,
            "--output_past_covariates",
            FEATURE_ENGINEERING_OUTPUT_PAST_COVARIATES,
            "--output_future_covariates",
            FEATURE_ENGINEERING_OUTPUT_FUTURE_COVARIATES,
        ],
        check=True,
    )


def run_training_pipeline(config_path: str) -> None:
    """
    Run the training pipeline locally.
    Pipeline consists of 1 step:
    1. Train model and save trained model to file
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


if __name__ == "__main__":
    run_preprocessing_pipeline("pipelines/preprocessing_pipeline.yaml")
    run_training_pipeline("pipelines/training_pipeline.yaml")
