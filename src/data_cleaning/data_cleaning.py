import sqlite3
import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Class for cleaning data in an SQLite database.
    Provides methods to clean data by removing unwanted records based on
    specific criteria.

    To be used as a context manager to ensure connection is opened and closed properly.
    Example usage:
    ```
        with DataCleaner(db_path="path/to/db.sqlite") as cleaner:
            cleaned_data_df = cleaner.run(
                source_table="my_table",
                target_table="cleaned_table",
                countries=["United Kingdom"]
            )
    ```
    """

    def __init__(self, db_path: str):
        self.db_path = db_path

    def __enter__(self):
        """
        Establish DB connection and initialize cursor within context.

        Returns:
            self: DataCleaner instance with active database connection.
        """
        self.db_connection = sqlite3.connect(self.db_path)
        self.cursor = self.db_connection.cursor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Secure handling of data manipulations within context.
        Closes database connection when exiting context manager.
        Commit changes to DB if no exception, else rollback.

        Args: (captured automatically within context block)
            exc_type: Exception type caught, or None if no exception.
            exc_val: Exception instance caught, or None if no exception.
            exc_tb: Exception traceback object caught, or None if no exception.
        """
        if exc_type is None:
            self.db_connection.commit()
            logger.info("Database changes committed successfully")
        else:
            self.db_connection.rollback()
            logger.error(
                f"Error during data cleaning: {exc_type.__name__}: {exc_val}",
                exc_info=(exc_type, exc_val, exc_tb),
            )
        self.db_connection.close()

    def run(
        self, source_table: str, target_table: str, countries: List[str]
    ) -> pd.DataFrame:
        """
        Run the full data cleaning process:
        1. Copy source_table to target_table
        2. Remove cancelled transactions
        3. Remove negative values
        4. Remove articles with alphanumeric StockCode prefix
        5. Keep only specified countries

        Args:
            source_table: Name of the source table.
            target_table: Name of the target table.
            countries: List of country names to keep.

        Returns:
            DataFrame containing the cleaned data.
        """
        self.copy_table(source_table, target_table)
        self.remove_cancelled_transactions(target_table)
        self.remove_negative_values(target_table)
        self.remove_articles_with_alphanummeric_prefix(target_table)
        self.keep_countries(target_table, countries)
        query = f""" SELECT * FROM {target_table} """
        df = self.query_to_df(query=query)
        return df

    def copy_table(self, source_table: str, target_table: str) -> str:
        """
        Copy data from source_table to target_table.
        Args:
            source_table: Name of the source table.
            target_table: Name of the target table.
        Returns:
            Name of the newly created table.
        """
        self.cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {target_table} AS
            SELECT * FROM {source_table}
            """
        )
        logger.info(f"Copied table '{source_table}' to '{target_table}'")
        return target_table

    def remove_cancelled_transactions(
        self,
        table_name: str,
    ) -> None:
        """
        Remove transactions starting with 'C' in InvoiceNo.
        """
        self.cursor.execute(
            f"""
            DELETE FROM {table_name}
            WHERE InvoiceNo LIKE 'C%'
            """
        )
        deleted_count = self.cursor.rowcount
        logger.info(
            f"Removed {deleted_count} cancelled transactions from '{table_name}'"
        )

    def remove_negative_values(
        self,
        table_name: str,
    ) -> None:
        """
        Remove rows with zero and negative values in 'Quantity' and 'UnitPrice' columns.
        Args:
            table_name: Name of the table to clean.
        """
        self.cursor.execute(
            f"""
            DELETE FROM {table_name}
            WHERE Quantity <= 0 OR UnitPrice <= 0
            """
        )
        deleted_count = self.cursor.rowcount
        logger.info(
            f"Removed {deleted_count} rows with zero or negative values in 'Quantity' or 'UnitPrice' from '{table_name}'"
        )

    def remove_articles_with_alphanummeric_prefix(
        self,
        table_name: str,
    ) -> None:
        """
        Remove rows where 'StockCode' starts with letters.
        Investigation shows these are mainly non-product codes, with a few grey areas.
        Args:
            table_name: Name of the table to clean.
        """
        self.cursor.execute(
            f"""
            DELETE FROM {table_name}
            WHERE StockCode GLOB '[A-Za-z]*'
            """
        )
        deleted_count = self.cursor.rowcount
        logger.info(
            f"Removed {deleted_count} rows with alphanumeric 'StockCode' from '{table_name}'"
        )

    def keep_countries(self, table_name: str, countries: List[str]) -> None:
        """
        Keep only rows where 'Country' is in the specified list of countries.
        Args:
            table_name: Name of the table to clean.
            countries: List of country names to keep.
        """
        countries_str = ", ".join(f"'{country}'" for country in countries)
        self.cursor.execute(
            f"""
            DELETE FROM {table_name}
            WHERE Country NOT IN ({countries_str})
            """
        )
        deleted_count = self.cursor.rowcount
        logger.info(
            f"Removed {deleted_count} rows not in specified countries from '{table_name}'"
        )

    def query_to_df(self, query: str) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame.
        Args:
            query: SQL query to execute.
        Returns:
            DataFrame containing the table data.
        """
        df = pd.read_sql_query(query, self.db_connection)
        logger.info(f"Retrieved {len(df):,} rows from query")
        return df
