import sqlite3
import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Class for loading data in a SQLite database.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path

    def load_table_to_df(
        self,
        table_name: str,
    ) -> pd.DataFrame:
        """
        Load data for specified countries into a pandas DataFrame.
        Args:
            table_name: Name of the table to load.
            countries: List of countries to filter the data.
        Returns:
            DataFrame containing the table data.
        """
        with sqlite3.connect(self.db_path) as conn:
            query = f"""
                SELECT * FROM {table_name}
            """
            df = pd.read_sql_query(query, conn)
        logger.info(f"Loaded {len(df):,} rows from table '{table_name}'")
        return df
