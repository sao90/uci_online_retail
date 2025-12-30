import sys
import argparse
import logging
from pathlib import Path

from src.data_processing.data_loader import DataLoader
from src.log_config import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Data ingestion component")
    parser.add_argument(
        "--db_path",
        type=str,
        required=True,
        help="Path to the SQLite database file",
    )
    parser.add_argument(
        "--output_data",
        type=str,
        required=True,
        help="Path to save the output Parquet file",
    )
    parser.add_argument(
        "--table_name",
        type=str,
        default="transactions",
        help="Name of the table to load from the database",
    )
    return parser.parse_args()


def main():
    """Data ingenstion component entry point."""
    setup_logging()
    args = parse_args()
    logger.info("Starting data ingestion component...")
    try:
        logger.info(
            f"Loading data from table {args.table_name} in database {args.db_path}..."
        )
        data_loader = DataLoader(db_path=args.db_path)
        df = data_loader.load_table_to_df(
            table_name=args.table_name,
        )

        output_path = Path(args.output_data)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)

        logger.info(f"Ingested data saved to {output_path}")
        logger.info(f"Dataframe shape: {df.shape}")
        logger.info("Data ingestion component completed successfully.")

    except Exception:
        logger.error("Error during data ingestion component", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
