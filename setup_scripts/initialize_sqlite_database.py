"""
Initialize SQLite database with UCI Online Retail dataset.

This script adds the UCI online retail dataset to a local SQLite database.
The local database serves as a mock for a system database that would exist in
a real project.

It performs some cleaning, as I would expect a pipelines database to have
rudimentary cleaning done (e.g. removing original transactions that were
later cancelled).
Such cleaning should not be part of an ML pipeline.

Final database will have 3 tables:
- raw transactions table: all data as-is from the Excel file
- cancelled transactions table: only transactions that were later cancelled
- processed transactions table: all transactions except those cancelled
    (original debited transaction + later credited transaction)
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
    logger.info("Starting SQLite database initialization...")

    excel_path = os.getenv("INPUT_DATA_FILE")
    db_path = os.getenv("DB_FILE")
    table_name_raw = os.getenv("DB_TABLE_NAME_RAW")
    table_name_processed = os.getenv("DB_TABLE_NAME_PROCESSED")
    cancelled_table_name = "cancelled_transactions"  # TODO: don't hardcode name
    logger.info(f"Reading {excel_path}...")
    df = pd.read_excel(excel_path)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    logger.info(f"Creating database at {db_path}...")

    # Context manager to commit to DB only on successful completion.
    with sqlite3.connect(database=db_path) as conn:
        try:
            # Write raw table
            logger.info(f"Creating raw transactions table: '{table_name_raw}'...")
            df.to_sql(name=table_name_raw, con=conn, if_exists="replace", index=False)
            logger.info(f"Table '{table_name_raw}' created with {len(df):,} rows.")
            # Write cancelled transactions table
            create_cancelled_transactions(
                df=df, cancelled_table_name=cancelled_table_name, conn=conn
            )
            # Write processed table where cancelled transactions are removed
            remove_original_transactions_cancelled_later(
                raw_table_name=table_name_raw,
                processed_table_name=table_name_processed,
                cancelled_table_name=cancelled_table_name,
                conn=conn,
            )
        except Exception as e:
            logger.error(f"Error during database initialization: {e}")
            raise
        conn.commit()

    logger.info("Done!")


def create_cancelled_transactions(
    df: pd.DataFrame, cancelled_table_name: str, conn: sqlite3.Connection
) -> None:
    """Create SQLite table with only cancelled transactions.
    logic:
        - Identify cancelled transactions by "InvoiceNo" starting with "C"/"c"
    Args:
        df: DataFrame containing the raw transactions.
        cancelled_table_name: Name of the SQLite table to store cancelled transactions.
        conn: SQLite connection object.
    """
    # Cancelled transactions have "InvoiceNo" starting with "C"
    # Filter out NaN values first to avoid masking errors
    cancelled_df = df[
        df["InvoiceNo"].notna() & df["InvoiceNo"].str.startswith(("C", "c"))
    ]
    logger.info(
        f"Creating table '{cancelled_table_name}' with cancelled transactions..."
    )
    cancelled_df.to_sql(
        name=cancelled_table_name, con=conn, if_exists="replace", index=False
    )
    logger.info(
        f"Table '{cancelled_table_name}' created with {len(cancelled_df):,} rows."
    )
    return


def remove_original_transactions_cancelled_later(
    raw_table_name: str,
    processed_table_name: str,
    cancelled_table_name: str,
    conn: sqlite3.Connection,
) -> None:
    """Create SQLite table where cancelled transactions are filtered out.

    Operations:
    - Remove transactions with InvoiceNo starting with 'C' (cancelled transactions)
    - Remove original transactions that were later cancelled.
        Original transactions are identified by matching StockCode, CustomerID, ABS(Quantity),
        Description, Country, and ABS(UnitPrice) with cancelled_transactions table.
        Original transaction's InvoiceDate must be before cancelled transaction's InvoiceDate,
        and within 50 days. (since we cannot match on transaction IDs)
    Args:
        raw_table_name: Name of the SQLite table with raw transactions.
        processed_table_name: Name of the SQLite table to store processed transactions.
        cancelled_table_name: Name of the SQLite table with cancelled transactions.
        conn: SQLite connection object.
    """
    logger.info(
        f"Creating table '{processed_table_name}' without transactions that were later cancelled..."
    )
    query = f"""
    DROP TABLE IF EXISTS {processed_table_name};

    CREATE TABLE {processed_table_name} AS
    SELECT *
    FROM {raw_table_name} o
    WHERE (o.InvoiceNo NOT LIKE 'C%' AND o.InvoiceNo NOT LIKE 'c%')
        AND NOT EXISTS (
            SELECT 1
            FROM {cancelled_table_name} c
            WHERE o.StockCode = c.StockCode
                AND o.CustomerID = c.CustomerID
                AND ABS(o.Quantity) = ABS(c.Quantity)
                AND o.Description = c.Description
                AND o.Country = c.Country
                AND ABS(o.UnitPrice) = ABS(c.UnitPrice)
                AND o.InvoiceDate <= c.InvoiceDate
                AND (julianday(c.InvoiceDate) - julianday(o.InvoiceDate)) < 50 -- TODO: parameterize this threshold
        );
    """
    conn.executescript(query)
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {processed_table_name}")
    count = cursor.fetchone()[0]
    logger.info(f"Table '{processed_table_name}' created with {count:,} rows.")
    return


if __name__ == "__main__":
    main()
