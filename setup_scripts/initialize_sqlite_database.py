"""Initialize SQLite database with UCI Online Retail dataset."""

# python-native imports
import os
import sqlite3
import logging

# third-party imports
import pandas as pd
from dotenv import load_dotenv

# project imports
from src.log_config import setup_logging

logger = logging.getLogger(__name__)


def main():
    logging.info("Starting SQLite database initialization...")
    setup_logging()
    load_dotenv()

    excel_path = os.getenv("INPUT_DATA_FILE")
    db_path = os.getenv("DB_FILE")

    logger.info(f"Reading {excel_path}...")
    df = pd.read_excel(excel_path)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    logger.info(f"Creating database at {db_path}...")
    conn = sqlite3.connect(database=db_path)
    df.to_sql(name="transactions", con=conn, if_exists="replace", index=False)
    conn.close()

    logger.info("Done!")


if __name__ == "__main__":
    main()
