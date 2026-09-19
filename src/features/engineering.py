"""Feature engineering module for Credit Card Fraud Detection.

This module provides functions to transform raw temporal, financial, and PCA
features into informative indicators for fraud classification.
"""

from typing import Optional
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_time_features(
    df: pd.DataFrame,
    time_col: str = "Time",
    inplace: bool = False,
) -> pd.DataFrame:
    """Create temporal features from the transaction timestamp column.

    Extracts:
    - 'hour_of_day': Floating-point / integer hour of the day (0-23) based on seconds elapsed.
    - 'is_night': Binary flag (1 if transaction occurred between 23:00 and 06:00, 0 otherwise).

    Args:
        df: Input DataFrame containing the transaction time column.
        time_col: Name of the timestamp column in seconds (default: 'Time').
        inplace: Whether to modify the DataFrame in place (default: False).

    Returns:
        pd.DataFrame: DataFrame containing new time-based features.

    Raises:
        KeyError: If time_col is not present in df.
    """
    if time_col not in df.columns:
        if "scaled_time" in df.columns:
            logger.warning(
                f"Column '{time_col}' not found. Using 'scaled_time' as hour reference."
            )
            data = df if inplace else df.copy()
            hours_elapsed = data["scaled_time"]
            data["hour_of_day"] = (hours_elapsed % 24).astype(float)
            data["is_night"] = (
                (data["hour_of_day"] >= 23.0) | (data["hour_of_day"] < 6.0)
            ).astype(int)
            return data
        raise KeyError(
            f"Missing required time column '{time_col}'. Available columns: {list(df.columns)}"
        )

    data = df if inplace else df.copy()
    logger.info(f"Generating time features from '{time_col}' column.")

    # Time column in Credit Card Fraud dataset is seconds elapsed from the first transaction
    # Normalize seconds to hours: 1 hour = 3600 seconds
    hours_elapsed = data[time_col] / 3600.0
    data["hour_of_day"] = (hours_elapsed % 24).astype(float)

    # Transactions between 11 PM (23:00) and 6 AM (06:00)
    data["is_night"] = (
        (data["hour_of_day"] >= 23.0) | (data["hour_of_day"] < 6.0)
    ).astype(int)

    night_count = data["is_night"].sum()
    night_ratio = night_count / len(data) if len(data) > 0 else 0.0
    logger.info(
        f"Time features created. Night transactions: {night_count:,} ({night_ratio:.2%})"
    )

    return data


def create_amount_features(
    df: pd.DataFrame,
    amount_col: str = "Amount",
    inplace: bool = False,
) -> pd.DataFrame:
    """Create financial amount transformation and binning features.

    Extracts:
    - 'amount_log': Natural log transform log1p(x) of the transaction amount.
    - 'amount_bin': Categorical binning:
        - 'small': < $10
        - 'medium': $10 - < $100
        - 'large': $100 - < $1,000
        - 'very_large': >= $1,000

    Args:
        df: Input DataFrame containing the transaction amount column.
        amount_col: Name of the transaction amount column (default: 'Amount').
        inplace: Whether to modify the DataFrame in place (default: False).

    Returns:
        pd.DataFrame: DataFrame containing new amount-based features.

    Raises:
        KeyError: If amount_col is not present in df.
    """
    if amount_col not in df.columns:
        raise KeyError(
            f"Missing required amount column '{amount_col}'. Available columns: {list(df.columns)}"
        )

    data = df if inplace else df.copy()
    logger.info(f"Generating amount features from '{amount_col}' column.")

    # Log1p transformation (handles 0 amounts smoothly)
    # Clip negative values at 0 in case of noisy data
    non_neg_amount = np.maximum(data[amount_col], 0.0)
    data["amount_log"] = np.log1p(non_neg_amount)

    # Binning categories
    bins = [-np.inf, 10.0, 100.0, 1000.0, np.inf]
    labels = ["small", "medium", "large", "very_large"]

    data["amount_bin"] = pd.cut(
        data[amount_col],
        bins=bins,
        labels=labels,
        right=False,  # [a, b) intervals: small < 10, medium < 100, large < 1000, very_large >= 1000
    )

    bin_counts = data["amount_bin"].value_counts().to_dict()
    logger.info(f"Amount features created. Distribution across bins: {bin_counts}")

    return data


def create_interaction_features(
    df: pd.DataFrame,
    v1_col: str = "V1",
    v2_col: str = "V2",
    v3_col: str = "V3",
    inplace: bool = False,
) -> pd.DataFrame:
    """Create interaction features between top PCA components.

    Extracts:
    - 'V1_V2_product': Product of PCA component 1 and component 2.
    - 'V1_V3_product': Product of PCA component 1 and component 3.

    Args:
        df: Input DataFrame containing PCA features.
        v1_col: Name of the V1 column (default: 'V1').
        v2_col: Name of the V2 column (default: 'V2').
        v3_col: Name of the V3 column (default: 'V3').
        inplace: Whether to modify the DataFrame in place (default: False).

    Returns:
        pd.DataFrame: DataFrame containing new interaction features.

    Raises:
        KeyError: If any of V1, V2, or V3 columns are missing.
    """
    required_cols = [v1_col, v2_col, v3_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required PCA columns for interaction features: {missing}. Available columns: {list(df.columns)}"
        )

    data = df if inplace else df.copy()
    logger.info(f"Generating PCA interaction features for ({v1_col}, {v2_col}, {v3_col}).")

    data["V1_V2_product"] = data[v1_col] * data[v2_col]
    data["V1_V3_product"] = data[v1_col] * data[v3_col]

    logger.info(
        f"Interaction features 'V1_V2_product' and 'V1_V3_product' successfully created."
    )

    return data


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run full feature engineering pipeline on the dataset.

    Executes:
    1. Time features: hour_of_day, is_night
    2. Amount features: amount_log, amount_bin
    3. Interaction features: V1_V2_product, V1_V3_product

    Args:
        df: Input raw or semi-processed DataFrame.

    Returns:
        pd.DataFrame: DataFrame enriched with engineered features.
    """
    logger.info(
        f"Starting feature engineering pipeline on DataFrame with shape {df.shape}."
    )
    result_df = df.copy()

    # 1. Time features
    if "Time" in result_df.columns or "scaled_time" in result_df.columns:
        result_df = create_time_features(result_df, inplace=True)
    else:
        logger.warning("No time column found; skipping time feature generation.")

    # 2. Amount features
    if "Amount" in result_df.columns:
        result_df = create_amount_features(result_df, inplace=True)
    else:
        logger.warning("No 'Amount' column found; skipping amount feature generation.")

    # 3. PCA Interaction features
    if all(col in result_df.columns for col in ["V1", "V2", "V3"]):
        result_df = create_interaction_features(result_df, inplace=True)
    else:
        logger.warning("V1, V2, or V3 columns missing; skipping PCA interaction features.")

    logger.info(
        f"Feature engineering pipeline completed. New shape: {result_df.shape}. "
        f"Columns added: {len(result_df.columns) - len(df.columns)}"
    )
    return result_df


def feature_engineering_pipeline(
    df: pd.DataFrame, config: Optional[dict] = None
) -> pd.DataFrame:
    """Wrapper pipeline function for feature engineering compatibility.

    Args:
        df: Input DataFrame.
        config: Optional configuration dictionary.

    Returns:
        pd.DataFrame: Engineered features DataFrame.
    """
    return build_features(df)


if __name__ == "__main__":
    # Self-test demonstration with synthetic sample
    logger.info("Running feature engineering self-test...")
    sample_data = {
        "Time": [0.0, 3600.0, 7200.0, 82800.0, 86400.0],  # 0h, 1h, 2h, 23h, 24h
        "V1": [-1.35, 1.19, -2.31, 0.45, -0.89],
        "V2": [-0.07, 0.26, 0.46, -0.12, 0.35],
        "V3": [2.53, 0.16, 1.88, 1.20, -0.45],
        "Amount": [149.62, 2.69, 378.66, 1200.00, 0.00],
        "Class": [0, 0, 0, 1, 0],
    }
    sample_df = pd.DataFrame(sample_data)
    engineered_df = build_features(sample_df)
    print("\n--- Engineered DataFrame Sample ---")
    print(engineered_df.head())

#making chnadu understand git hub!