"""
Unit tests for data preprocessing and splitting pipelines.

Tests data scaling, time normalization, stratified partitioning,
missing value handling, and data leakage prevention.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import preprocess, split_data


@pytest.fixture
def sample_creditcard_data() -> pd.DataFrame:
    """
    Generate synthetic credit card transaction DataFrame.
    Mimics Kaggle Credit Card Fraud dataset schema:
    - Time: float seconds elapsed
    - V1 to V28: float PCA transformed features
    - Amount: transaction amount
    - Class: binary target (0 = legitimate, 1 = fraud)
    """
    np.random.seed(42)
    n_samples = 200
    n_fraud = 20  # 10% fraud rate for robust test statistics

    # Generate PCA features V1-V28
    v_features = {f"V{i}": np.random.randn(n_samples) for i in range(1, 29)}

    # Generate Time (0 to 172,800 seconds = 48 hours)
    time = np.sort(np.random.uniform(0, 172800, n_samples))

    # Generate Amount (log-normal distribution)
    amount = np.random.exponential(scale=100.0, size=n_samples) + 1.0

    # Generate Class labels (stratified distribution)
    labels = np.array([0] * (n_samples - n_fraud) + [1] * n_fraud)
    np.random.shuffle(labels)

    data = {
        "Time": time,
        **v_features,
        "Amount": amount,
        "Class": labels,
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_data_with_missing(sample_creditcard_data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate synthetic dataset injected with intentional missing (NaN) values.
    """
    df_missing = sample_creditcard_data.copy()
    # Inject missing values across financial, temporal, and PCA features
    df_missing.loc[5:10, "Amount"] = np.nan
    df_missing.loc[20:25, "Time"] = np.nan
    df_missing.loc[30:35, "V1"] = np.nan
    df_missing.loc[40:45, "V10"] = np.nan
    return df_missing


def test_preprocess_scales_amount(sample_creditcard_data: pd.DataFrame):
    """
    Verify that the 'Amount' feature is scaled using RobustScaler:
    - Original 'Amount' is transformed into 'scaled_amount'
    - 'scaled_amount' column exists in processed output
    - Original unscaled 'Amount' is dropped
    - Scaled values contain no NaNs and exhibit scaled properties
    """
    processed = preprocess(sample_creditcard_data, run_feature_engineering=True)

    assert "scaled_amount" in processed.columns, "Output must contain 'scaled_amount'"
    assert "Amount" not in processed.columns, "Original 'Amount' should be dropped"
    assert processed["scaled_amount"].isnull().sum() == 0, "No NaNs allowed in 'scaled_amount'"
    assert isinstance(processed["scaled_amount"].iloc[0], (float, np.floating)), "Scaled amount must be float"


def test_preprocess_scales_time(sample_creditcard_data: pd.DataFrame):
    """
    Verify that 'Time' feature is normalized from seconds into continuous hours:
    - Original 'Time' is transformed into 'scaled_time'
    - 'scaled_time' = Time / 3600.0
    - Original unscaled 'Time' is dropped
    - 'scaled_time' is within valid expected range [0, 48]
    """
    original_time = sample_creditcard_data["Time"].copy()
    processed = preprocess(sample_creditcard_data, run_feature_engineering=True)

    assert "scaled_time" in processed.columns, "Output must contain 'scaled_time'"
    assert "Time" not in processed.columns, "Original 'Time' should be dropped"
    assert processed["scaled_time"].isnull().sum() == 0, "No NaNs allowed in 'scaled_time'"

    # Verify scaling arithmetic (seconds / 3600 = hours)
    expected_scaled = original_time / 3600.0
    np.testing.assert_allclose(
        processed["scaled_time"].values,
        expected_scaled.values,
        rtol=1e-5,
        err_msg="'scaled_time' values do not match expected seconds-to-hours normalization",
    )


def test_split_preserves_ratio(sample_creditcard_data: pd.DataFrame):
    """
    Verify that stratified train/val/test split maintains the fraud-to-legitimate ratio.
    """
    processed = preprocess(sample_creditcard_data, run_feature_engineering=True)
    original_fraud_ratio = processed["Class"].mean()

    splits = split_data(
        processed,
        test_size=0.2,
        val_size=0.1,
        random_state=42,
        target_col="Class",
    )

    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    train_ratio = train_df["Class"].mean()
    val_ratio = val_df["Class"].mean()
    test_ratio = test_df["Class"].mean()

    # Verify ratios are close across all partitions (tolerance within 5% due to integer rounding)
    assert abs(train_ratio - original_fraud_ratio) < 0.05, f"Train fraud ratio ({train_ratio:.3f}) deviated from original ({original_fraud_ratio:.3f})"
    assert abs(val_ratio - original_fraud_ratio) < 0.05, f"Val fraud ratio ({val_ratio:.3f}) deviated from original ({original_fraud_ratio:.3f})"
    assert abs(test_ratio - original_fraud_ratio) < 0.05, f"Test fraud ratio ({test_ratio:.3f}) deviated from original ({original_fraud_ratio:.3f})"


def test_split_no_data_leakage(sample_creditcard_data: pd.DataFrame):
    """
    Verify that train, validation, and test partitions have zero data leakage or overlapping samples:
    - Sum of split lengths equals total dataset length
    - Unique index identification confirms zero overlap between any two splits
    """
    processed = preprocess(sample_creditcard_data, run_feature_engineering=True)
    # Add a unique identifier column to strictly trace samples
    processed["_uid"] = [f"uid_{i}" for i in range(len(processed))]

    splits = split_data(
        processed,
        test_size=0.2,
        val_size=0.1,
        random_state=42,
        target_col="Class",
    )

    train_ids = set(splits["train"]["_uid"])
    val_ids = set(splits["val"]["_uid"])
    test_ids = set(splits["test"]["_uid"])

    # Verify total partition conservation
    assert len(train_ids) + len(val_ids) + len(test_ids) == len(processed)

    # Verify pairwise mutually exclusive sets (zero leakage)
    assert len(train_ids.intersection(val_ids)) == 0, "Data leakage detected between Train and Val sets"
    assert len(train_ids.intersection(test_ids)) == 0, "Data leakage detected between Train and Test sets"
    assert len(val_ids.intersection(test_ids)) == 0, "Data leakage detected between Val and Test sets"


def test_handles_missing_values(sample_data_with_missing: pd.DataFrame):
    """
    Verify that missing values (NaNs) are handled gracefully by imputation during preprocessing.
    """
    assert sample_data_with_missing.isnull().sum().sum() > 0, "Input test data must contain missing values"

    processed = preprocess(sample_data_with_missing, run_feature_engineering=True)

    # All numeric columns must have zero NaNs after preprocessing
    total_remaining_missing = processed.isnull().sum().sum()
    assert total_remaining_missing == 0, f"Processed DataFrame still contains {total_remaining_missing} missing values"
    assert len(processed) == len(sample_data_with_missing), "Row count must remain invariant after missing value handling"
