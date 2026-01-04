import sys
import argparse
import logging
from pathlib import Path

import pandas as pd

from src.modules.data_processing.data_splitter import DataSplitter
from src.modules.log_config import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Data splitting component")

    parser.add_argument(
        "--input_data",
        type=str,
        required=True,
        help="Path to the input Parquet file",
    )
    parser.add_argument(
        "--date_column",
        type=str,
        required=True,
        help="Name of the date column for chronological splitting",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        required=True,
        help="Name of the target column to separate from features",
    )
    parser.add_argument(
        "--days_in_test_split",
        type=int,
        required=True,
        help="Number of calendar days to include in test set",
    )
    parser.add_argument(
        "--output_train_targets",
        type=str,
        required=True,
        help="Path to save the training targets Parquet file",
    )
    parser.add_argument(
        "--output_test_targets",
        type=str,
        required=True,
        help="Path to save the test targets Parquet file",
    )
    parser.add_argument(
        "--output_features",
        type=str,
        required=True,
        help="Path to save the features Parquet file",
    )
    return parser.parse_args()


def main():
    """Data splitting component entry point."""
    setup_logging()
    args = parse_args()
    logger.info("Starting data splitting component...")
    try:
        logger.info(f"Reading input data from {args.input_data}...")
        df = pd.read_parquet(Path(args.input_data))

        data_splitter = DataSplitter()
        train_targets, test_targets, features = data_splitter.run(
            df=df,
            date_column=args.date_column,
            target_column=args.target_column,
            days_in_test_split=args.days_in_test_split,
        )

        # Save train targets
        train_path = Path(args.output_train_targets)
        train_path.parent.mkdir(parents=True, exist_ok=True)
        train_targets.to_parquet(train_path, index=False)
        logger.info(
            f"Train targets saved to {train_path} (shape: {train_targets.shape})"
        )

        # Save test targets
        test_path = Path(args.output_test_targets)
        test_path.parent.mkdir(parents=True, exist_ok=True)
        test_targets.to_parquet(test_path, index=False)
        logger.info(f"Test targets saved to {test_path} (shape: {test_targets.shape})")

        # Save features
        features_path = Path(args.output_features)
        features_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(features_path, index=False)
        logger.info(f"Features saved to {features_path} (shape: {features.shape})")

        logger.info("Data splitting component completed successfully.")

    except Exception:
        logger.error("Error during data splitting component", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
