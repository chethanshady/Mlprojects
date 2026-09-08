"""Dataset download module for Credit Card Fraud Detection.

This module automates the retrieval of the Kaggle Credit Card Fraud dataset
via the official Kaggle API with fallback instructions for manual acquisition.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Union

from src.utils.logger import get_logger

logger = get_logger(__name__)

KAGGLE_DATASET_IDENTIFIER = "mlg-ulb/creditcardfraud"
DEFAULT_RAW_DIR = Path("data/raw")
EXPECTED_CSV_FILENAME = "creditcard.csv"


def print_manual_download_instructions(
    output_dir: Union[str, Path],
    dataset_name: str = KAGGLE_DATASET_IDENTIFIER,
) -> None:
    """Print structured step-by-step instructions for manual dataset download.

    Args:
        output_dir: Target directory where dataset should reside.
        dataset_name: Kaggle dataset slug/identifier.
    """
    target_path = Path(output_dir) / EXPECTED_CSV_FILENAME
    kaggle_url = f"https://www.kaggle.com/datasets/{dataset_name}"
    kaggle_json_path = (
        Path.home() / ".kaggle" / "kaggle.json"
        if os.name != "nt"
        else Path(os.environ.get("USERPROFILE", str(Path.home()))) / ".kaggle" / "kaggle.json"
    )

    instructions = f"""
{"=" * 80}
[MANUAL DOWNLOAD INSTRUCTIONS]
Could not automatically download dataset via Kaggle API.

Option 1: Manual Web Download
--------------------------------------------------------------------------------
1. Navigate to: {kaggle_url}
2. Sign in to Kaggle and click 'Download' (archive.zip ~66 MB).
3. Extract '{EXPECTED_CSV_FILENAME}' into your project directory:
   Target file path: {target_path.resolve()}

Option 2: Configure Kaggle API Authentication
--------------------------------------------------------------------------------
1. Log in to Kaggle -> Go to Account Settings -> 'Create New Token'.
2. Move downloaded 'kaggle.json' to:
   {kaggle_json_path}
3. Run: pip install kaggle
4. Re-run this script: python src/data/download.py --output-dir {output_dir}
{"=" * 80}
"""
    logger.warning(instructions)


def download_dataset(
    output_dir: Union[str, Path] = DEFAULT_RAW_DIR,
    dataset_name: str = KAGGLE_DATASET_IDENTIFIER,
    force: bool = False,
) -> Path:
    """Download and extract the Kaggle Credit Card Fraud dataset.

    Attempts automated download via Kaggle API. If Kaggle is unavailable
    or unauthenticated, guides the user on manual download procedures.

    Args:
        output_dir: Directory where the dataset CSV should be saved.
        dataset_name: Kaggle dataset identifier (owner/dataset-name).
        force: If True, re-downloads even if the CSV file already exists.

    Returns:
        Path: Path to the downloaded CSV file or target directory.

    Raises:
        RuntimeError: If download fails and file is not present.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    target_csv = out_path / EXPECTED_CSV_FILENAME

    if target_csv.exists() and not force:
        file_size_mb = target_csv.stat().st_size / (1024 * 1024)
        logger.info(
            f"Dataset already exists at {target_csv.resolve()} ({file_size_mb:.2f} MB). Skipping download."
        )
        return target_csv

    logger.info(
        f"Attempting to download dataset '{dataset_name}' to '{out_path.resolve()}'..."
    )

    try:
        # Attempt to import and authenticate Kaggle API
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        logger.info("Kaggle API successfully authenticated.")

        # Download and extract dataset files
        logger.info(f"Downloading files for dataset '{dataset_name}'...")
        api.dataset_download_files(
            dataset=dataset_name,
            path=str(out_path),
            unzip=True,
            quiet=False,
        )

        if target_csv.exists():
            file_size_mb = target_csv.stat().st_size / (1024 * 1024)
            logger.info(
                f"Dataset successfully downloaded and unpacked to {target_csv.resolve()} ({file_size_mb:.2f} MB)."
            )
            return target_csv
        else:
            logger.warning(
                f"Download completed, but expected file '{EXPECTED_CSV_FILENAME}' not found in {out_path}."
            )
            return out_path

    except ImportError:
        logger.error(
            "The 'kaggle' Python package is not installed. Run: pip install kaggle"
        )
        print_manual_download_instructions(output_dir=out_path, dataset_name=dataset_name)
    except Exception as exc:
        logger.error(f"Failed to download dataset via Kaggle API: {exc}")
        print_manual_download_instructions(output_dir=out_path, dataset_name=dataset_name)

    if target_csv.exists():
        return target_csv

    logger.warning(
        f"Dataset file '{target_csv.resolve()}' is not currently present on disk."
    )
    return target_csv


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download Credit Card Fraud Detection dataset from Kaggle."
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=str(DEFAULT_RAW_DIR),
        help=f"Target directory for downloaded data (default: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default=KAGGLE_DATASET_IDENTIFIER,
        help=f"Kaggle dataset slug (default: {KAGGLE_DATASET_IDENTIFIER})",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-download even if target file already exists.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info("Executing dataset download script...")
    download_dataset(
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        force=args.force,
    )
