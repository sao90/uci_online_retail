import sys
import argparse
import logging
from pathlib import Path
import pickle
import json

import pandas as pd
from darts import TimeSeries, concatenate

from src.modules.model_handling.model_handler import ModelHandler
from src.modules.log_config import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Model backtest component")
    parser.add_argument(
        "--paths_to_models",
        nargs="+",
        type=str,
        required=True,
        help="List of paths to the trained model pickle files",
    )
    parser.add_argument(
        "--target_training_data_path",
        type=str,
        required=True,
        help="Path to the target training split data Parquet file",
    )
    parser.add_argument(
        "--target_test_data_path",
        type=str,
        required=True,
        help="Path to the target test split data Parquet file",
    )
    parser.add_argument(
        "--past_covariates_path",
        type=str,
        required=True,
        help="Path to the past covariates data Parquet file",
    )
    parser.add_argument(
        "--future_covariates_path",
        type=str,
        required=True,
        help="Path to the future covariates data Parquet file",
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
    target_test_df_path = args.target_test_data_path
    past_covariates_df_path = args.past_covariates_path
    future_covariates_df_path = args.future_covariates_path
    paths_to_models = [Path(p) for p in args.paths_to_models]
    scores_output_path = Path(args.scores_output)

    # Load data
    try:
        target_train_df = pd.read_parquet(target_train_df_path)
        target_test_df = pd.read_parquet(target_test_df_path)
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
        target_test = TimeSeries.from_dataframe(
            target_test_df,
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
        # TODO: better handling of split date. Both here and in split component
        split_date = target_train.end_time()
        target_full = concatenate([target_train, target_test], axis=0)
    except Exception:
        logger.error("Error converting data to TimeSeries", exc_info=True)
        sys.exit(1)

    model_handler = ModelHandler()

    # Iterate over models to evaluate
    # TODO: parallelize this loop to speed up evaluation
    evaluation_results = {}
    for model_input_path in paths_to_models:
        model_name = model_input_path.stem
        logger.info(f"Backtesting model from {model_input_path}")
        # Load trained model
        try:
            logger.info(f"Loading model from {model_input_path}")
            with open(model_input_path, "rb") as f:
                trained_model = pickle.load(f)
        except Exception:
            logger.error("Error loading the model", exc_info=True)
            sys.exit(1)

        # Backtest model on full data
        logger.info("Running backtest on full data...")
        try:
            backtest_scores = model_handler.backtest_model(
                model=trained_model,
                target_series=target_full,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
                start=split_date,
                metrics=[
                    "rmse",
                    "wmape",
                ],  # TODO: parameterize metrics in case new ones are added
            )
            logger.info(f"Backtest scores: {backtest_scores}")
        except Exception:
            logger.error("Error during backtesting", exc_info=True)
            sys.exit(1)

        # Assign scores to evaluation results
        evaluation_results[model_name] = backtest_scores

    # Save evaluation results
    try:
        scores_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scores_output_path, "w") as f:
            json.dump(evaluation_results, f, indent=2)
        logger.info(f"Evaluation results saved to {scores_output_path}")
    except Exception:
        logger.error("Error saving evaluation results", exc_info=True)
        sys.exit(1)

    logger.info("Model backtest component completed successfully.")


if __name__ == "__main__":
    main()
