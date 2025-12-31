import logging
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class DataSplitter:
    """
    Stateless data splitting.
    ML logic: Working with Darts library, we do not need to split features. Only targets.
    Features will be generated into past- and future covariates later.

    Performs data splitting operations:
    - convert date column to datetime
    - split in features and target
    - train and test split (validation splits are rolling windows in train set)


    Example usage:
    ```python
        TODO: example usage
    ```
    """

    def __init__(self):
        """Initialize DataSplitter. Stateless"""
        pass

    def run(
        self,
        df: pd.DataFrame,
        date_column: str,
        target_column: str,
        days_in_test_split: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Execute full data splitting procedure.

        Args:
            df: Input DataFrame with cleaned transaction data.
            date_column: name of date column
            target_column: name of target column
            days_in_test_split: number of calendar days to include in test set
        Returns:
            Tuple of (train_targets, test_targets, features)
        """
        logger.info(f"Starting data splitting component. Input shape: {df.shape}")

        df = self.convert_date_column_to_datetime(df, date_column)
        df_targets, df_features_raw = self.split_targets_and_features(
            df, target_column, date_column
        )
        train_targets, test_targets = self.split_train_test(
            target_df=df_targets,
            date_column=date_column,
            days_in_test_split=days_in_test_split,
        )
        logger.info(
            f"Completed data splitting. "
            f"Train length: {len(train_targets[date_column].unique())}, "
            f"Test targets length: {len(test_targets[date_column].unique())}, "
            f"Features length: {len(df_features_raw[date_column].unique())}"
        )
        return train_targets, test_targets, df_features_raw

    def convert_date_column_to_datetime(
        self,
        df: pd.DataFrame,
        date_column: str,
    ) -> pd.DataFrame:
        """
        convert date column to datetime format

        Args:
            df: Input DataFrame.
            date_column: name of date column
        Returns:
            DataFrame with date column converted to datetime
        """
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found in DataFrame")
        df = df.sort_values(by=date_column).copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df[date_column] = df[date_column].dt.normalize()  # set time to 00:00:00
        return df

    def split_targets_and_features(
        self, df: pd.DataFrame, target_column: str, date_column: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataframe into targets and features

        Args:
            df: Input DataFrame.
            target_column: name of target column
            date_column: name of date column
        Returns:
            Tuple of (targets DataFrame, features DataFrame)
        Raises:
            KeyError: if target_column or date_column not in df
        """
        if target_column not in df.columns or date_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' or date column '{date_column}' "
                "not found in DataFrame"
            )
        df_targets = df[[date_column, target_column]].copy()
        df_features = df.drop(columns=[target_column]).copy()
        logger.info(
            f"Split data into features (shape: {df_features.shape}) and "
            f"targets (shape: {df_targets.shape})"
        )
        return df_targets, df_features

    def split_train_test(
        self, target_df: pd.DataFrame, date_column: str, days_in_test_split: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split target dataframe into train and test
        Business logic:
            - Zero-activity days are not yet imputed.
            - Split is done by calendar date, and not by number of rows.
            - Test split uses last N calendar days from max date.
        Args:
            target_df: targets DataFrame.
            date_column: name of date column
            days_in_test_split: number of calendar days to include in test set
        Returns:
            Tuple of (train_targets, test_targets)
        """
        if days_in_test_split <= 0:
            raise ValueError("days_in_test_split must be positive integer")

        max_date = target_df[date_column].max()
        split_date = max_date - pd.Timedelta(days=days_in_test_split)
        train_targets = target_df[target_df[date_column] <= split_date].copy()
        test_targets = target_df[target_df[date_column] > split_date].copy()
        logger.info(f"Splitting Test set after date: {split_date.date()}. ")

        return train_targets, test_targets
