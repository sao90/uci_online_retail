import sys
import argparse
import logging
from pathlib import Path
import pickle
import json

import pandas as pd
from darts import TimeSeries

from src.modules.model_handling.model_handler import ModelHandler
from src.modules.log_config import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Model backtest component")
    parser.add_argument(
        "--model_input",
        type=str,
        required=True,
        help="Path to the trained model pickle file",
    )
    parser.add_argument(
        "--target_training_data_path",
        type=str,
        required=True,
        help="Path to the target training data Parquet file",
    )
    parser.add_argument(
        "--past_covariates_path",
        type=str,
        required=True,
        help="Path to the past covariates training data Parquet file",
    )
    parser.add_argument(
        "--future_covariates_path",
        type=str,
        required=True,
        help="Path to the future covariates training data Parquet file",
    )
    parser.add_argument(
        "--future_covariates_columns",
        type=str,
        nargs="+",
        required=True,
        help="List of future covariate columns to use",
    )
    parser.add_argument(
        "--past_covariates_columns",
        type=str,
        nargs="+",
        required=True,
        help="List of past covariate columns to use",
    )
    parser.add_argument(
        "--target_column_name",
        type=str,
        required=True,
        help="Name of the target column",
    )
    parser.add_argument(
        "--time_column_name",
        type=str,
        required=True,
        help="Name of the time column",
    )
    parser.add_argument(
        "--backtest_start",
        type=float,
        default=0.7,
        help="Fraction of series to start backtest (0.0-1.0)",
    )
    parser.add_argument(
        "--scores_output",
        type=str,
        required=True,
        help="Path to save the backtest scores JSON file",
    )
    return parser.parse_args()


def main():
    """Model backtest component entry point."""
    setup_logging()
    args = parse_args()
    logger.info("Starting model backtest component...")

    future_covariates_columns = args.future_covariates_columns
    past_covariates_columns = args.past_covariates_columns
    target_column = [args.target_column_name]
    time_column = args.time_column_name
    target_train_df_path = args.target_training_data_path
    past_covariates_df_path = args.past_covariates_path
    future_covariates_df_path = args.future_covariates_path
    model_input_path = Path(args.model_input)
    scores_output_path = Path(args.scores_output)
    backtest_start = args.backtest_start

    # Load trained model
    try:
        logger.info(f"Loading model from {model_input_path}")
        with open(model_input_path, "rb") as f:
            trained_model = pickle.load(f)
    except Exception:
        logger.error("Error loading the model", exc_info=True)
        sys.exit(1)

    # Load data
    try:
        target_train_df = pd.read_parquet(target_train_df_path)
        past_covariates_df = pd.read_parquet(past_covariates_df_path)
        future_covariates_df = pd.read_parquet(future_covariates_df_path)
    except Exception:
        logger.error("Error reading training data files", exc_info=True)
        sys.exit(1)

    # Convert to Darts TimeSeries
    try:
        target_train = TimeSeries.from_dataframe(
            target_train_df,
            time_col=time_column,
            value_cols=target_column,
            fill_missing_dates=True,
            fillna_value=0,
            freq="D",
        )
        past_covariates = TimeSeries.from_dataframe(
            past_covariates_df,
            time_col=time_column,
            value_cols=past_covariates_columns,
            fill_missing_dates=True,
            fillna_value=0,
            freq="D",
        )
        future_covariates = TimeSeries.from_dataframe(
            future_covariates_df,
            time_col=time_column,
            value_cols=future_covariates_columns,
            fill_missing_dates=True,
            fillna_value=0,
            freq="D",
        )
    except Exception:
        logger.error("Error converting data to TimeSeries", exc_info=True)
        sys.exit(1)

    # Backtest model on training data
    logger.info("Running backtest on training data...")
    model_handler = ModelHandler()
    backtest_scores = model_handler.backtest_model(
        model=trained_model,
        target_series=target_train,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        start=backtest_start,
        metrics=["rmse", "wmape"],
    )

    logger.info(f"Backtest scores: {backtest_scores}")

    # Save backtest scores
    try:
        scores_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scores_output_path, "w") as f:
            json.dump(backtest_scores, f, indent=2)
        logger.info(f"Backtest scores saved to {scores_output_path}")
    except Exception:
        logger.error("Error saving backtest scores", exc_info=True)
        sys.exit(1)

    logger.info("Model backtest component completed successfully.")


if __name__ == "__main__":
    main()
