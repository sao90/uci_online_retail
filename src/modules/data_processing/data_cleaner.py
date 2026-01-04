import logging
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Stateless data cleaning for retail transaction data.

    Performs data quality operations:
    - Remove non-positive values in Quantity or UnitPrice
    - Remove non-product StockCodes (alphabetic prefixes)
    - Create Revenue column (Quantity * UnitPrice)
    - Filter to specified countries

    TODO before production grade:
    - cancelled transactions can have a counterpart in positive sales, not just remove cancellations.
        potentially complex matching logic
        - which columns to match on?
        - partial cancellations?
        - reasons for cancellations? are all cancellations equal?
    - validation of input data schema
    - validation of output data schema


    """

    def __init__(self):
        """Initialize DataCleaner. Stateless - no configuration needed."""
        pass

    def run(
        self, df: pd.DataFrame, countries: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Execute full data cleaning procedure.

        Args:
            df: Input DataFrame with raw transaction data.
            countries: Optional list of country names to keep. If None, keeps all.
        Returns:
            Cleaned DataFrame.
        """
        self._validate_input_dataframe_schema(df)
        logger.info(f"Starting data cleaning pipeline. Input shape: {df.shape}")
        initial_rows = len(df)

        df_cleaned = self.remove_non_positive_values(df)
        df_cleaned = self.remove_articles_with_alphabetic_prefix(df_cleaned)
        df_cleaned = self.create_revenue_column(df_cleaned)

        if countries is not None:
            df_cleaned = self.keep_countries(df_cleaned, countries)

        self._validate_output_dataframe_schema(df_cleaned)

        final_rows = len(df_cleaned)
        rows_removed = initial_rows - final_rows
        removal_pct = (rows_removed / initial_rows * 100) if initial_rows > 0 else 0

        logger.info(
            f"Data cleaning complete. Removed {rows_removed:,} rows ({removal_pct:.1f}%). "
            f"Final shape: {df_cleaned.shape}"
        )

        return df_cleaned

    def remove_non_positive_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with non-positive Quantity or UnitPrice values.
        Business rule: Transactions must have positive quantities and prices.

        Args:
            df: Input DataFrame.
        Returns:
            DataFrame with non-positive values removed.
        """
        initial_rows = len(df)
        df_filtered = df[
            (df["Quantity"] > 0)
            & (df["UnitPrice"] > 0)
            & df["Quantity"].notna()
            & df["UnitPrice"].notna()
        ].copy()
        removed = initial_rows - len(df_filtered)
        logger.info(f"Removed {removed:,} rows with non-positive Quantity or UnitPrice")
        return df_filtered

    def remove_articles_with_alphabetic_prefix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows where StockCode starts with alphabetic characters.
        Business rule: Valid product StockCodes are numeric. Alphabetic prefixes
        indicate non-product items (shipping fees, adjustments, samples, etc.).

        Args:
            df: Input DataFrame.
        Returns:
            DataFrame with non-product StockCodes removed.
        """
        initial_rows = len(df)
        df_filtered = df[~df["StockCode"].astype(str).str.match(r"^[A-Za-z]")].copy()
        removed = initial_rows - len(df_filtered)
        logger.info(f"Removed {removed:,} rows with alphabetic StockCode prefix")
        return df_filtered

    def create_revenue_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create Revenue column by multiplying Quantity and UnitPrice.

        Args:
            df: Input DataFrame with Quantity and UnitPrice columns.
        Returns:
            DataFrame with Revenue column added.
        """
        df = df.copy()
        df["Revenue"] = df["Quantity"] * df["UnitPrice"]
        logger.info("Created Revenue column")
        return df

    def keep_countries(self, df: pd.DataFrame, countries: List[str]) -> pd.DataFrame:
        """
        Filter DataFrame to keep only specified countries.

        Args:
            df: Input DataFrame.
            countries: List of country names to keep (exact match).
        Returns:
            DataFrame filtered to specified countries.
        """
        initial_rows = len(df)
        if not countries:
            logger.info("No countries specified for filtering. Returning all countries")
            return df.copy()
        df_filtered = df[df["Country"].isin(countries)].copy()
        removed = initial_rows - len(df_filtered)
        logger.info(f"Removed {removed:,} rows not in specified countries: {countries}")
        return df_filtered

    def _validate_input_dataframe_schema(self, df: pd.DataFrame) -> None:
        """TODO: schema validation logic."""
        pass

    def _validate_output_dataframe_schema(self, df: pd.DataFrame) -> None:
        """TODO: schema validation logic."""
        pass
