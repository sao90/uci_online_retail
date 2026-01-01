import sys
import argparse
import logging
from pathlib import Path

import pandas as pd

from src.data_processing.feature_engineer import FeatureEngineer
from src.log_config import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Data splitting component")

    parser.add_argument(
        "--target_train_file",
        type=str,
        required=True,
        help="Path to the target train-split Parquet file",
    )
    parser.add_argument(
        "--target_test_file",
        type=str,
        required=True,
        help="Path to the target test-split Parquet file",
    )
    parser.add_argument(
        "--features_raw_file",
        type=str,
        required=True,
        help="Path to the raw features Parquet file",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        required=True,
        help="Name of the target column to separate from features",
    )
    parser.add_argument(
        "--date_column",
        type=str,
        required=True,
        help="Name of the date column for chronological splitting",
    )
    parser.add_argument(
        "--transaction_id_column",
        type=str,
        required=True,
        help="Name of the transaction ID column",
    )
    parser.add_argument(
        "--customer_id_column",
        type=str,
        required=True,
        help="Name of the customer ID column",
    )
    parser.add_argument(
        "--article_id_column",
        type=str,
        required=True,
        help="Name of the article ID column",
    )
    parser.add_argument(
        "--revenue_column",
        type=str,
        required=True,
        help="Name of the revenue column",
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
        "--output_past_covariates",
        type=str,
        required=True,
        help="Path to save the past covariates Parquet file",
    )
    parser.add_argument(
        "--output_future_covariates",
        type=str,
        required=True,
        help="Path to save the future covariates Parquet file",
    )
    return parser.parse_args()


def main():
    """Feature Engineering component entry point."""
    setup_logging()
    args = parse_args()
    logger.info("Starting feature engineering component...")
    try:
        logger.info(
            f"Reading input data from {args.target_train_file}, {args.target_test_file}, and {args.features_raw_file}"
        )
        target_train = pd.read_parquet(Path(args.target_train_file))
        target_test = pd.read_parquet(Path(args.target_test_file))
        features_raw = pd.read_parquet(Path(args.features_raw_file))

        feature_engineer = FeatureEngineer(
            target_col_name=args.target_column,
            date_col_name=args.date_column,
            transaction_id_col_name=args.transaction_id_column,
            customer_id_col_name=args.customer_id_column,
            article_id_col_name=args.article_id_column,
            revenue_col_name=args.revenue_column,
        )
        target_train, target_test, past_covariates, future_covariates = (
            feature_engineer.run(
                target_train=target_train,
                target_test=target_test,
                features_raw=features_raw,
            )
        )

        # Save train split
        target_train_path = Path(args.output_train_targets)
        target_train_path.parent.mkdir(parents=True, exist_ok=True)
        target_train.to_parquet(target_train_path, index=False)
        logger.info(
            f"Train targets saved to {target_train_path} (shape: {target_train.shape})"
        )

        # Save test split
        target_test_path = Path(args.output_test_targets)
        target_test_path.parent.mkdir(parents=True, exist_ok=True)
        target_test.to_parquet(target_test_path, index=False)
        logger.info(
            f"Test targets saved to {target_test_path} (shape: {target_test.shape})"
        )

        # Save past_covariates
        past_covariates_path = Path(args.output_past_covariates)
        past_covariates_path.parent.mkdir(parents=True, exist_ok=True)
        past_covariates.to_parquet(past_covariates_path, index=False)
        logger.info(
            f"Past covariates saved to {past_covariates_path} (shape: {past_covariates.shape})"
        )

        # Save future_covariates
        future_covariates_path = Path(args.output_future_covariates)
        future_covariates_path.parent.mkdir(parents=True, exist_ok=True)
        future_covariates.to_parquet(future_covariates_path, index=False)
        logger.info(
            f"Future covariates saved to {future_covariates_path} (shape: {future_covariates.shape})"
        )

        logger.info("Feature engineering component completed successfully.")

    except Exception:
        logger.error("Error during feature engineering component", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
