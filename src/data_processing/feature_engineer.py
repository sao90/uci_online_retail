import logging
from typing import Tuple

import holidays
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature Engineering module for UCI Online Retail dataset.
    ML logic: Working with Darts library going forward, we need to create past (observed) covariates
    and future (known) covariates

    features generated:
    Target:
        - daily aggregated sales (sum of Quantity)
    Past covariates:
        - average basket size (items) per day
        - average unit price per day (weighted by quantity sold)
        - number of transactions per day
        - number of unique customers per day
        - number of unique articles sold per day
    Future covariates:
        - holiday indicator (is_holiday)


    TODO before production grade:
        - parameterize column names in class instance, in case external data schema changes.
        - get access to future planned marketing activities from business (crucial for future covariates)
        - validation of input data schema
        - validation of output data schema
        - logging
        - unit tests
        - Ideally, create use-case specific feature store and
            potentially pipeline it outside ML pipeline.
            ML pipeline would then read ready features based on pipeline configuration.
        - solve zero-activity (missing) days when they fall in the train/test split boundary. Not handled yet
    """

    def __init__(
        self,
        target_col_name: str,
        date_col_name: str,
        transaction_id_col_name: str,
        customer_id_col_name: str,
        article_id_col_name: str,
        revenue_col_name: str,
    ):
        self.target_col_name = target_col_name
        self.date_col_name = date_col_name
        self.transaction_id_col_name = transaction_id_col_name
        self.customer_id_col_name = customer_id_col_name
        self.article_id_col_name = article_id_col_name
        self.revenue_col_name = revenue_col_name

    def run(
        self,
        target_train: pd.DataFrame,
        target_test: pd.DataFrame,
        features_raw: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Execute full data engineering procedure.

        Args:
            target_train: DataFrame with training target data.
            target_test: DataFrame with testing target data.
            features_raw: DataFrame with raw features data.
        Returns:
            Tuple of (engineered_train_targets, engineered_test_targets, past_covariates, future_covariates)
        """

        # Aggregate target data to daily level for train and test set
        agg_train, agg_test = self.aggregate_targets(
            df_train=target_train,
            df_test=target_test,
        )
        # Create past observed covariates
        past_covariates = self.compute_past_covariates(
            features_raw=features_raw,
        )
        # Create future (known in advance) covariates
        future_covariates = self.compute_future_covariates(
            df=features_raw,
        )
        return agg_train, agg_test, past_covariates, future_covariates

    def aggregate_targets(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aggregate dataframe with target variable to daily level.
        Args:
            df: Input DataFrame with target variable.
            date_col: Name of the date column.
        Returns:
            Tuple of train and test DataFrames aggregated to daily level.
        """
        agg_train = df_train.copy()
        agg_test = df_test.copy()
        agg_train = (
            agg_train.groupby(self.date_col_name)
            .agg({self.target_col_name: "sum"})
            .reset_index()
        )
        agg_test = (
            agg_test.groupby(self.date_col_name)
            .agg({self.target_col_name: "sum"})
            .reset_index()
        )
        return agg_train, agg_test

    def compute_past_covariates(
        self,
        features_raw: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute past observed covariates from raw features DataFrame.
        Args:
            features_raw: Input DataFrame with raw features.
        Returns:
            dataframe with past observed covariates.
        """
        avg_basket_size = self._compute_avg_basket_size(features_raw)
        avg_unit_price = self._compute_avg_unit_price(features_raw)
        business_indicators = self._compute_business_indicators(features_raw)
        past_covariates = business_indicators.merge(
            avg_basket_size, on=self.date_col_name, how="left"
        ).merge(avg_unit_price, on=self.date_col_name, how="left")
        return past_covariates

    def compute_future_covariates(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create a DataFrame with future known covariates.
        Only holiday indicator for now.
        Args:
            df: Input DataFrame containing the date index column
        Returns:
            DataFrame with date column and 'is_holiday' column.
        """
        # Create full date range based on min/max
        min_date = df[self.date_col_name].min()
        max_date = df[self.date_col_name].max()
        full_date_range = pd.date_range(start=min_date, end=max_date, freq="D")

        uk_holidays = holidays.UK(years=(2010, 2011))

        future_covariates = pd.DataFrame(
            {
                self.date_col_name: full_date_range,
                "is_holiday": [
                    1 if d.date() in uk_holidays else 0 for d in full_date_range
                ],
            }
        )
        return future_covariates

    def _compute_avg_basket_size(
        self,
        features_raw: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute average basket size (items per transaction) per day.
        Args:
            features_raw: Input DataFrame with transaction-level data.
        Returns:
            DataFrame with InvoiceDate and avg_basket_size columns.
        """
        # Basket size per transaction (preserve InvoiceDate)
        basket_sizes = (
            features_raw.groupby([self.date_col_name, self.transaction_id_col_name])[
                self.target_col_name
            ]
            .sum()
            .reset_index()
        )
        # Average basket sizes per day
        avg_basket_size_daily = (
            basket_sizes.groupby(self.date_col_name)[self.target_col_name]
            .mean()
            .reset_index()
            .rename(columns={self.target_col_name: "avg_basket_size"})
        )
        return avg_basket_size_daily

    def _compute_avg_unit_price(
        self,
        features_raw: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute weighted average unit price per day.
        Weighted by quantity sold.

        Args:
            features_raw: Input DataFrame with Revenue and Quantity columns.
        Returns:
            DataFrame with InvoiceDate and avg_unit_price columns.
        """
        # Compute total daily revenue and quantity
        daily_agg = (
            features_raw.groupby(self.date_col_name)
            .agg({self.revenue_col_name: "sum", self.target_col_name: "sum"})
            .reset_index()
        )
        # Compute average unit price per day
        daily_agg["avg_unit_price"] = (
            daily_agg[self.revenue_col_name] / daily_agg[self.target_col_name]
        )
        return daily_agg[[self.date_col_name, "avg_unit_price"]]

    def _compute_business_indicators(
        self,
        features_raw: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute daily business indicators:
        - Number of transactions per day
        - Number of unique customers per day
        - Number of unique articles sold per day

        Args:
            df: Input DataFrame with transaction data.
        Returns:
            DataFrame with time index and business indicator columns.
        """
        business_indicators = (
            features_raw.groupby(self.date_col_name)
            .agg(
                {
                    self.transaction_id_col_name: "nunique",
                    self.customer_id_col_name: "nunique",
                    self.article_id_col_name: "nunique",
                }
            )
            .reset_index()
            .rename(
                columns={
                    self.transaction_id_col_name: "num_transactions",
                    self.customer_id_col_name: "num_unique_customers",
                    self.article_id_col_name: "num_unique_articles",
                }
            )
        )

        return business_indicators
