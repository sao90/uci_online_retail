import sqlite3
import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DataCleaner:
    def __init__(self, db_path):
        self.db_path = db_path

    def __enter__(self):
        """
        Open database connection when entering context manager.

        Called automatically by 'with' statement. Establishes connection
        and cursor for use in context block.

        Returns:
            self: DataCleaner instance with active database connection.
        """
        self.db_connection = sqlite3.connect(self.db_path)
        self.cursor = self.db_connection.cursor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Close database connection when exiting context manager.
        Commit changes if no exception, else rollback.
        Captures exceptions automatically from context block.

        Args:
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
