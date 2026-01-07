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
    parser = argparse.ArgumentParser(description="Evaluate models component")
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
    """Model evaluation component entry point."""
    setup_logging()
    args = parse_args()
    logger.info("Starting model evaluation component...")

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
        logger.error("Error reading data files", exc_info=True)
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
        target_test = TimeSeries.from_dataframe(
            target_test_df,
            value_cols=target_column,
            **time_series_kwargs,
        )
        past_covariates = TimeSeries.from_dataframe(
            past_covariates_df,
            value_cols=past_covariates_columns,
            **time_series_kwargs,
        )
        future_covariates = TimeSeries.from_dataframe(
            future_covariates_df,
            value_cols=future_covariates_columns,
            **time_series_kwargs,
        )
        # TODO: better handling of split date. Both here and in split component
        split_date = target_train.end_time()
        target_full = concatenate([target_train, target_test], axis=0)
    except Exception:
        logger.error("Error converting data to TimeSeries", exc_info=True)
        sys.exit(1)

    evaluation_dict = {}
    model_handler = ModelHandler()

    # Iterate over models to evaluate
    # TODO: parallelize this loop to speed up evaluation
    logger.info("Loading models for evaluation...")
    for model_path in paths_to_models:
        model_name = model_path.stem
        try:
            with open(model_path, "rb") as pickled_model:
                trained_model = pickle.load(pickled_model)
            logger.info(f"Model '{model_name}' loaded successfully.")
        except FileNotFoundError:
            logger.error(f"Model file not found: {model_path}", exc_info=True)
            logger.error("Skipping to next model...")
            continue
        except Exception:
            logger.error(f"Error loading model from {model_path}", exc_info=True)
            logger.error("Skipping to next model...")
            continue

        # Backtest model on full data
        logger.info(f"Running backtest on full data for model: {model_name}...")
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
            logger.info(f"Backtest scores for {model_name}: {backtest_scores}")
        except Exception:
            logger.error(f"Error during backtesting for {model_name}", exc_info=True)
            logger.error("Skipping to next model...")
            continue

        evaluation_dict[model_name] = backtest_scores

    # Save evaluation results
    if not evaluation_dict:
        logger.error("No evaluation results to save. Exiting with failure.")
        sys.exit(1)
    try:
        scores_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scores_output_path, "w") as f:
            json.dump(evaluation_dict, f, indent=2)
        logger.info(f"Evaluation results saved to {scores_output_path}")
    except Exception:
        logger.error("Error saving evaluation results", exc_info=True)
        sys.exit(1)

    logger.info("Evaluation component completed successfully.")


if __name__ == "__main__":
    main()
