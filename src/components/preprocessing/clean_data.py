import sys
import argparse
import logging
from pathlib import Path

import pandas as pd

from src.modules.data_processing.data_cleaner import DataCleaner
from src.modules.log_config import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Data cleaning component")

    parser.add_argument(
        "--input_data",
        type=str,
        required=True,
        help="Path to the input Parquet file",
    )
    parser.add_argument(
        "--countries",
        type=str,
        nargs="+",
        default=None,
        help="Comma-separated list of countries to keep in the data",
    )
    parser.add_argument(
        "--output_data",
        type=str,
        required=True,
        help="Path to save the output Parquet file",
    )
    return parser.parse_args()


def main():
    """Data cleaning component entry point."""
    setup_logging()
    args = parse_args()
    logger.info("Starting data cleaning component...")
    try:
        logger.info(f"Reading input data from {args.input_data}...")
        df = pd.read_parquet(Path(args.input_data))

        data_cleaner = DataCleaner()
        df = data_cleaner.run(
            df=df,
            countries=args.countries,
        )

        output_path = Path(args.output_data)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)

        logger.info(f"Cleaned data saved to {output_path}")
        logger.info(f"Cleaned data shape: {df.shape}")
        logger.info("Data cleaning component completed successfully.")

    except Exception:
        logger.error("Error during data cleaning component", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
