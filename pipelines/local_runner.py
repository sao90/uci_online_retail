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
    OUTPUT_TRAIN_TARGETS = os.getenv("OUTPUT_TRAIN_TARGETS")
    OUTPUT_TEST_TARGETS = os.getenv("OUTPUT_TEST_TARGETS")
    OUTPUT_FEATURES_RAW = os.getenv("OUTPUT_FEATURES_RAW")
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
            OUTPUT_TRAIN_TARGETS,
            "--output_test_targets",
            OUTPUT_TEST_TARGETS,
            "--output_features",
            OUTPUT_FEATURES_RAW,
        ],
        check=True,
    )

    # Step 4: Generate features


if __name__ == "__main__":
    run_preprocessing_pipeline()
