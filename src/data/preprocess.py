"""Data preprocessing pipeline for Credit Card Fraud Detection.

Handles data loading, missing value verification, outlier-robust feature scaling,
feature engineering integration, stratified dataset splitting, and parquet export.
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from src.features.engineering import build_features
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load raw or processed dataset from CSV or Parquet format.

    Args:
        filepath: Path to the target data file (.csv or .parquet).

    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If file format is not supported.
    """
    path = Path(filepath)
    if not path.is_file():
        logger.error(f"Data file not found at: {path.resolve()}")
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    logger.info(f"Loading data from: {path.resolve()}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in [".parquet", ".pq"]:
        df = pd.read_parquet(path)
    else:
        raise ValueError(
            f"Unsupported file format '{path.suffix}'. Supported formats: .csv, .parquet"
        )

    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    logger.info(
        f"Data loaded successfully. Shape: {df.shape[0]:,} rows x {df.shape[1]} columns. "
        f"Memory usage: {memory_mb:.2f} MB."
    )
    return df


def preprocess(
    df: pd.DataFrame,
    run_feature_engineering: bool = True,
    amount_col: str = "Amount",
    time_col: str = "Time",
    target_col: str = "Class",
) -> pd.DataFrame:
    """Preprocess transaction dataset.

    Pipeline operations:
    1. Inspect and log missing value statistics.
    2. Run feature engineering (if enabled and original columns exist).
    3. Scale 'Amount' using RobustScaler (outlier-robust scaling).
    4. Scale 'Time' by normalizing elapsed seconds to continuous hours.
    5. Drop original 'Amount' and 'Time' columns, retaining scaled variants.

    Args:
        df: Input raw transaction DataFrame.
        run_feature_engineering: Whether to generate engineered features (default: True).
        amount_col: Name of the transaction amount column (default: 'Amount').
        time_col: Name of the timestamp column in seconds (default: 'Time').
        target_col: Name of the fraud label column (default: 'Class').

    Returns:
        pd.DataFrame: Fully preprocessed and scaled DataFrame.
    """
    logger.info("Initiating dataset preprocessing pipeline...")
    processed = df.copy()

    # 1. Missing Value Check & Verification
    missing_series = processed.isnull().sum()
    total_missing = missing_series.sum()
    if total_missing > 0:
        missing_report = missing_series[missing_series > 0].to_dict()
        logger.warning(
            f"Dataset contains {total_missing:,} missing values across columns: {missing_report}"
        )
        # Fill numeric NaNs with median as standard safeguard
        processed = processed.fillna(processed.median(numeric_only=True))
        logger.info("Missing values filled with column medians.")
    else:
        logger.info("Missing value check passed: No missing values detected in dataset.")

    # 2. Feature Engineering (before dropping original Amount/Time)
    if run_feature_engineering:
        logger.info("Applying feature engineering transforms...")
        processed = build_features(processed)

    # 3. Scale 'Amount' with RobustScaler
    if amount_col in processed.columns:
        logger.info(f"Scaling '{amount_col}' feature using RobustScaler...")
        robust_scaler = RobustScaler()
        processed["scaled_amount"] = robust_scaler.fit_transform(
            processed[[amount_col]]
        ).flatten()
        logger.info(
            f"'{amount_col}' scaled. RobustScaler Median: {robust_scaler.center_[0]:.4f}, "
            f"IQR Scale: {robust_scaler.scale_[0]:.4f}"
        )
    else:
        logger.warning(
            f"Column '{amount_col}' not found for RobustScaler; skipping amount scaling."
        )

    # 4. Scale 'Time' by normalizing seconds to hours
    if time_col in processed.columns:
        logger.info(f"Normalizing '{time_col}' from seconds to hours...")
        processed["scaled_time"] = (processed[time_col] / 3600.0).astype(float)
        logger.info(
            f"'{time_col}' normalized to hours. Range: [{processed['scaled_time'].min():.2f}h, {processed['scaled_time'].max():.2f}h]"
        )
    else:
        logger.warning(
            f"Column '{time_col}' not found; skipping time normalization."
        )

    # 5. Drop original Amount and Time columns
    cols_to_drop = [col for col in [amount_col, time_col] if col in processed.columns]
    if cols_to_drop:
        logger.info(f"Dropping original unscaled columns: {cols_to_drop}")
        processed = processed.drop(columns=cols_to_drop)

    # Convert categorical columns to string for parquet portability
    for cat_col in processed.select_dtypes(include=["category"]).columns:
        processed[cat_col] = processed[cat_col].astype(str)

    logger.info(
        f"Preprocessing completed. Final DataFrame shape: {processed.shape[0]:,} rows x {processed.shape[1]} columns."
    )
    return processed


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    target_col: str = "Class",
) -> Dict[str, pd.DataFrame]:
    """Perform a stratified two-stage train/val/test split.

    Preserves the rare fraud class ratio across all partitions.

    Args:
        df: Preprocessed DataFrame containing features and target.
        test_size: Fraction of the dataset to allocate to test set (default: 0.2).
        val_size: Fraction of the entire dataset to allocate to validation set (default: 0.1).
        random_state: Seed for reproducible random splits (default: 42).
        target_col: Target column name for stratification (default: 'Class').

    Returns:
        Dict[str, pd.DataFrame]: Dictionary with keys 'train', 'val', 'test'.

    Raises:
        ValueError: If split proportions are invalid or target_col is missing.
    """
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in DataFrame. Available: {list(df.columns)}"
        )

    if not (0.0 < test_size < 1.0) or not (0.0 <= val_size < 1.0) or (test_size + val_size >= 1.0):
        raise ValueError(
            f"Invalid split fractions: test_size={test_size}, val_size={val_size}. "
            "test_size + val_size must be strictly < 1.0."
        )

    logger.info(
        f"Splitting dataset with test_size={test_size:.1%}, val_size={val_size:.1%}, "
        f"train_size={1.0 - test_size - val_size:.1%} (stratified on '{target_col}')."
    )

    # Stage 1: Separate Test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col],
    )

    # Stage 2: Separate Train and Validation sets
    if val_size > 0.0:
        # Proportion of validation set relative to train_val_df
        val_relative_size = val_size / (1.0 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_relative_size,
            random_state=random_state,
            stratify=train_val_df[target_col],
        )
    else:
        train_df = train_val_df
        val_df = pd.DataFrame(columns=df.columns)

    splits = {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }

    # Log class distribution for each split
    for split_name, split_df in splits.items():
        if len(split_df) == 0:
            continue
        total_rows = len(split_df)
        fraud_count = int(split_df[target_col].sum())
        non_fraud_count = total_rows - fraud_count
        fraud_ratio = fraud_count / total_rows if total_rows > 0 else 0.0

        logger.info(
            f"[{split_name.upper():5s}] Total: {total_rows:7,d} | "
            f"Non-Fraud: {non_fraud_count:7,d} ({1.0 - fraud_ratio:6.2%}) | "
            f"Fraud: {fraud_count:5,d} ({fraud_ratio:6.4%})"
        )

    return splits


def save_processed(
    splits: Dict[str, pd.DataFrame],
    output_dir: Union[str, Path] = "data/processed",
) -> Dict[str, Path]:
    """Save split datasets as Parquet files.

    Args:
        splits: Dictionary containing 'train', 'val', 'test' DataFrames.
        output_dir: Target directory path for processed parquet files.

    Returns:
        Dict[str, Path]: Dictionary mapping split names to saved file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: Dict[str, Path] = {}
    logger.info(f"Saving processed splits to: {out_dir.resolve()}")

    for split_name, split_df in splits.items():
        if len(split_df) == 0:
            continue

        file_path = out_dir / f"{split_name}.parquet"
        split_df.to_parquet(file_path, index=False, engine="pyarrow")
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        saved_paths[split_name] = file_path

        logger.info(
            f"Saved '{split_name}' -> {file_path.name} "
            f"({len(split_df):,d} rows, {file_size_mb:.2f} MB)"
        )

    return saved_paths


def run_pipeline(
    raw_data_path: Union[str, Path] = "data/raw/creditcard.csv",
    output_dir: Union[str, Path] = "data/processed",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Dict[str, Path]:
    """Execute the end-to-end data loading, preprocessing, splitting, and saving pipeline.

    Args:
        raw_data_path: Path to the raw input dataset.
        output_dir: Target directory for processed parquet files.
        test_size: Test split ratio (default: 0.2).
        val_size: Validation split ratio (default: 0.1).
        random_state: Random seed for splitting (default: 42).

    Returns:
        Dict[str, Path]: Map of split names to saved parquet file paths.
    """
    logger.info("=" * 80)
    logger.info("Starting Full Data Preprocessing Pipeline")
    logger.info("=" * 80)

    # 1. Load Data
    raw_df = load_data(raw_data_path)

    # 2. Preprocess & Feature Engineering
    processed_df = preprocess(raw_df, run_feature_engineering=True)

    # 3. Stratified Split
    splits = split_data(
        processed_df,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    # 4. Save Processed Parquet Files
    saved_files = save_processed(splits, output_dir=output_dir)

    logger.info("=" * 80)
    logger.info("Data Preprocessing Pipeline Finished Successfully!")
    logger.info("=" * 80)
    return saved_files


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="Preprocess credit card fraud dataset and export train/val/test splits."
    )
    parser.add_argument(
        "--input-file",
        "-i",
        type=str,
        default="data/raw/creditcard.csv",
        help="Path to raw dataset CSV/Parquet (default: data/raw/creditcard.csv)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="data/processed",
        help="Directory to save processed splits (default: data/processed)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set fraction (default: 0.2)",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Validation set fraction (default: 0.1)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for data splitting (default: 42)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raw_path = Path(args.input_file)

    if not raw_path.exists():
        logger.warning(
            f"Input file not found at '{raw_path.resolve()}'. "
            "Please run 'python src/data/download.py' or provide a valid --input-file."
        )
    else:
        run_pipeline(
            raw_data_path=raw_path,
            output_dir=args.output_dir,
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.random_state,
        )
