from typing import Dict, List, Union
import logging

import pandas as pd
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.metrics import rmse, wmape

from src.modules.model_handling.model_catalogue import MODEL_CATALOGUE

logger = logging.getLogger(__name__)


class ModelHandler:
    """
    Handles training and backtesting of time series forecasting models.

    This class provides methods for training individual models and evaluating
    them using backtesting on specified datasets.
    """

    def __init__(self):
        self.available_metrics = {
            "rmse": rmse,
            "wmape": wmape,
        }

    def train_model(
        self,
        model_key: str,
        target_series: TimeSeries,
        past_covariates: TimeSeries,
        future_covariates: TimeSeries,
    ) -> ForecastingModel:
        """
        Train a model specified by model_key using the provided time series data.

        Args:
            model_key (str): Key to identify the model in the MODEL_CATALOGUE.
            target_series (TimeSeries): The target time series for training.
            past_covariates (Optional[TimeSeries]): Past covariate time series.
            future_covariates (Optional[TimeSeries]): Future covariate time series.
        Returns:
            ForecastingModel: The trained forecasting model.
        Raises:
            ValueError: If the model_key is not found in the MODEL_CATALOGUE.
        """
        if model_key not in MODEL_CATALOGUE:
            logger.error(
                f"Model key '{model_key}' not found in MODEL_CATALOGUE.", exc_info=True
            )
            raise ValueError

        model_class = MODEL_CATALOGUE[model_key]
        model = model_class()
        try:
            model.fit(
                series=target_series,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
            )
        except Exception:
            logger.error(f"Error training model '{model_key}'", exc_info=True)
            raise
        logger.info(f"Model '{model_key}' trained successfully.")
        return model

    def backtest_model(
        self,
        model: ForecastingModel,
        target_series: TimeSeries,
        past_covariates: TimeSeries,
        future_covariates: TimeSeries,
        start: Union[float, int, pd.Timestamp],
        metrics: List[str],
    ) -> Dict[str, float]:
        """
        Backtest the provided model on the target series using specified metrics.
        NOTE: This method is used in both training and evaluation pipelines.
        What varies is the target series passed.
        In training pipeline, it is the training target series,
        while in evaluation pipeline, it is the full target series (train + test).

        Data leakage is not and issue, as it is handled by the Darts backtest method, which
        mimics the inference scenario through its parameters.

        Args:
            model: The trained forecasting model to backtest.
            target_series: The target time series for backtesting.
            past_covariates: Past covariate time series.
            future_covariates: Future covariate time series.
            start: Fraction (0.0-1.0), absolute index (int), or timestamp to start backtest.
            metrics: List of metric names to evaluate.
        Returns:
            A dictionary with metric names as keys and their computed values.
        """
        metric_functions = []
        for metric_name in metrics:
            if metric_name in self.available_metrics:
                metric_functions.append(self.available_metrics[metric_name])
            else:
                logger.error(
                    f"Metric '{metric_name}' is not available. \n"
                    f"Available metrics: {list(self.available_metrics.keys())}"
                )
                raise ValueError

        # Backtest default parameters to match assumed(!) inference requirements:
        # - forecast_horizon=7: 7-day ahead forecasts
        # - stride=1: Generate new forecast daily
        # - retrain=True: Refit model daily with latest data
        # - last_points_only=True: Return scalar metrics (not forecast series)
        try:
            backtest_scores = model.backtest(
                series=target_series,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
                forecast_horizon=7,
                stride=1,
                retrain=True,
                last_points_only=True,
                metric=metric_functions,
                start=start,
            )
        except Exception:
            logger.error("Error during backtesting", exc_info=True)
            raise

        # If only one metric, return single value; else return dict
        if len(metric_functions) == 1:
            return {metrics[0]: backtest_scores}
        else:
            return {metrics[i]: score for i, score in enumerate(backtest_scores)}
