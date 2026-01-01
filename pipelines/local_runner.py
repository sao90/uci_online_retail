"""
Run pipelines locally for testing and development.
"""

import os
import sys
import logging
import subprocess

from dotenv import load_dotenv

from src.log_config import setup_logging

logger = logging.getLogger(__name__)

load_dotenv()

PROJECT_ROOT = os.getenv("REPOSITORY_ROOT")
sys.path.insert(0, PROJECT_ROOT)


def run_preprocessing_pipeline():
    """
    Pipeline consists of 4 steps:

    1. Ingest data from SQLite to Parquet (local file)
    2. Clean data and save cleaned data to Parquet (local file)
    3. Split data into train/test and save to Parquet (local file)
    4. Generate features and save to Parquet (local file)
    """
    setup_logging()
    logger.info("Starting local preprocessing pipeline run...")

    # Step 1: Ingest data")
    DB_PATH = os.getenv("DB_PATH")
    DB_INPUT_TABLE_NAME = os.getenv("DB_INPUT_TABLE_NAME")
    RAW_DATA_OUTPUT_PATH = os.getenv("RAW_DATA_OUTPUT_PATH")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "components.preprocessing.ingest_data",
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
    COUNTRIES = os.getenv("COUNTRIES")
    CLEANED_DATA_OUTPUT_PATH = os.getenv("CLEANED_DATA_OUTPUT_PATH")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "components.preprocessing.clean_data",
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
    TARGET_COLUMN = os.getenv("TARGET_COLUMN")
    DATE_COLUMN = os.getenv("DATE_COLUMN")
    DAYS_IN_TEST_SPLIT = os.getenv("DAYS_IN_TEST_SPLIT")
    SPLIT_OUTPUT_TRAIN_TARGETS = os.getenv("SPLIT_OUTPUT_TRAIN_TARGETS")
    SPLIT_OUTPUT_TEST_TARGETS = os.getenv("SPLIT_OUTPUT_TEST_TARGETS")
    SPLIT_OUTPUT_FEATURES_RAW = os.getenv("SPLIT_OUTPUT_FEATURES_RAW")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "components.preprocessing.split_data",
            "--input_data",
            CLEANED_DATA_OUTPUT_PATH,
            "--target_column",
            TARGET_COLUMN,
            "--date_column",
            DATE_COLUMN,
            "--days_in_test_split",
            DAYS_IN_TEST_SPLIT,
            "--output_train_targets",
            SPLIT_OUTPUT_TRAIN_TARGETS,
            "--output_test_targets",
            SPLIT_OUTPUT_TEST_TARGETS,
            "--output_features",
            SPLIT_OUTPUT_FEATURES_RAW,
        ],
        check=True,
    )

    # Step 4: Generate features
    CUSTOMER_ID_COLUMN = os.getenv("CUSTOMER_ID_COLUMN")
    TRANSACTION_ID_COLUMN = os.getenv("TRANSACTION_ID_COLUMN")
    ARTICLE_ID_COLUMN = os.getenv("ARTICLE_ID_COLUMN")
    REVENUE_COLUMN = os.getenv("REVENUE_COLUMN")
    FEATURE_ENGINEERING_OUTPUT_TRAIN_TARGETS = os.getenv(
        "FEATURE_ENGINEERING_OUTPUT_TRAIN_TARGETS"
    )
    FEATURE_ENGINEERING_OUTPUT_TEST_TARGETS = os.getenv(
        "FEATURE_ENGINEERING_OUTPUT_TEST_TARGETS"
    )
    FEATURE_ENGINEERING_OUTPUT_PAST_COVARIATES = os.getenv(
        "FEATURE_ENGINEERING_OUTPUT_PAST_COVARIATES"
    )
    FEATURE_ENGINEERING_OUTPUT_FUTURE_COVARIATES = os.getenv(
        "FEATURE_ENGINEERING_OUTPUT_FUTURE_COVARIATES"
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "components.preprocessing.feature_engineering",
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


if __name__ == "__main__":
    run_preprocessing_pipeline()
