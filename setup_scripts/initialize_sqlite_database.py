"""
Initialize SQLite database with UCI Online Retail dataset.

This script adds the UCI online retail dataset to a local SQLite database.
The local database serves as a mock for a system database that would exist in a real
project.
It contains a single table with all the data from the dataset, without any transformations.
"""

import os
import sqlite3
import logging

import pandas as pd
from dotenv import load_dotenv

from src.log_config import setup_logging

logger = logging.getLogger(__name__)


def main():
    setup_logging()
    load_dotenv()
    logging.info("Starting SQLite database initialization...")

    excel_path = os.getenv("INPUT_DATA_FILE")
    db_path = os.getenv("DB_FILE")
    table_name = os.getenv("DB_TABLE_NAME")

    logger.info(f"Reading {excel_path}...")
    df = pd.read_excel(excel_path)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    logger.info(f"Creating database at {db_path}...")
    conn = sqlite3.connect(database=db_path)
    df.to_sql(name=table_name, con=conn, if_exists="replace", index=False)
    conn.close()

    logger.info("Done!")


if __name__ == "__main__":
    main()
