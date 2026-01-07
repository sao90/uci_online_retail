import sys
import argparse
import logging
from pathlib import Path
import pickle

import pandas as pd
from darts import TimeSeries

from src.modules.model_handling.model_handler import ModelHandler
from src.modules.log_config import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Model training component")
    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        help="Name of model in MODEL CATALOGUE",
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
        "--model_output",
        type=str,
        required=True,
        help="Path to save the trained model",
    )
    return parser.parse_args()


def main():
    """Model training component entry point."""
    setup_logging()
    args = parse_args()
    logger.info("Starting model training component...")
    model_key = args.model_config
    logger.info(f"Using model config: {args.model_config}")
    future_covariates_columns = args.future_covariates_columns
    past_covariates_columns = args.past_covariates_columns
    target_column = [args.target_column_name]
    time_column = args.time_column_name
    target_train_df_path = args.target_training_data_path
    past_covariates_df_path = args.past_covariates_path
    future_covariates_df_path = args.future_covariates_path
    model_output_path = Path(args.model_output)

    # Load data
    try:
        target_train_df = pd.read_parquet(target_train_df_path)
        past_covariates_df = pd.read_parquet(past_covariates_df_path)
        future_covariates_df = pd.read_parquet(future_covariates_df_path)
    except Exception:
        logger.error("Error reading training data files", exc_info=True)
        sys.exit(1)
    # Convert to Darts TimeSeries
    time_series_kwargs = {
        "time_col": time_column,
        "fill_missing_dates": True,
        "fillna_value": 0,
        "freq": "D",
    }
    try:
        target_train = TimeSeries.from_dataframe(
            target_train_df,
            value_cols=target_column,
            **time_series_kwargs,
        )
        past_covariates = TimeSeries.from_dataframe(
            past_covariates_df,
            **time_series_kwargs,
            value_cols=past_covariates_columns,
        )
        future_covariates = TimeSeries.from_dataframe(
            future_covariates_df,
            value_cols=future_covariates_columns,
            **time_series_kwargs,
        )
    except Exception:
        logger.error("Error converting data to TimeSeries", exc_info=True)
        sys.exit(1)

    # Train model
    model_handler = ModelHandler()
    trained_model = model_handler.train_model(
        model_key=model_key,
        target_series=target_train,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
    )
    # Save model
    try:
        model_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_output_path, "wb") as f:
            pickle.dump(trained_model, f)
        logger.info(f"Model saved to {model_output_path}")
    except Exception:
        logger.error("Error saving the model", exc_info=True)
        sys.exit(1)

    logger.info("Model training component completed successfully.")


if __name__ == "__main__":
    main()
