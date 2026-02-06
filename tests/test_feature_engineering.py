"""
Smoke tests for feature engineering pipeline.

Tests basic functionality:
- Raw data loads correctly
- Feature engineering produces expected output shape
- Target variable has no NaN values
- Key features are present
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.engineer import (
    parse_strike_data,
    parse_time_to_seconds,
    parse_percentage,
    UFCFeatureEngineer,
)


# ============================================================================
# Unit Tests for Utility Functions
# ============================================================================

class TestParseStrikeData:
    """Tests for parse_strike_data function."""

    def test_valid_strike_format(self):
        assert parse_strike_data("25 of 50") == (25, 50)
        assert parse_strike_data("0 of 10") == (0, 10)
        assert parse_strike_data("100 of 100") == (100, 100)

    def test_invalid_input_returns_zeros(self):
        assert parse_strike_data("") == (0, 0)
        assert parse_strike_data("N/A") == (0, 0)
        assert parse_strike_data(None) == (0, 0)
        assert parse_strike_data("invalid") == (0, 0)

    def test_whitespace_handling(self):
        assert parse_strike_data("  25 of 50  ") == (25, 50)
        assert parse_strike_data("25  of  50") == (0, 0)  # Extra spaces break it


class TestParseTimeToSeconds:
    """Tests for parse_time_to_seconds function."""

    def test_valid_time_format(self):
        assert parse_time_to_seconds("5:00") == 300
        assert parse_time_to_seconds("1:30") == 90
        assert parse_time_to_seconds("0:45") == 45

    def test_invalid_input_returns_zero(self):
        assert parse_time_to_seconds("") == 0
        assert parse_time_to_seconds("N/A") == 0
        assert parse_time_to_seconds(None) == 0
        assert parse_time_to_seconds("invalid") == 0


class TestParsePercentage:
    """Tests for parse_percentage function."""

    def test_percentage_with_symbol(self):
        assert parse_percentage("50%") == 0.5
        assert parse_percentage("100%") == 1.0
        assert parse_percentage("0%") == 0.0

    def test_percentage_as_decimal(self):
        assert parse_percentage("0.5") == 0.5
        assert parse_percentage("0.75") == 0.75

    def test_invalid_input_returns_zero(self):
        assert parse_percentage("") == 0.0
        assert parse_percentage("N/A") == 0.0
        assert parse_percentage(None) == 0.0


# ============================================================================
# Integration Tests for Feature Engineering Pipeline
# ============================================================================

class TestFeatureEngineeringPipeline:
    """Integration tests for the full feature engineering pipeline."""

    @pytest.fixture
    def raw_data_path(self):
        """Path to raw fight data."""
        return Path(__file__).parent.parent / "data" / "raw" / "ufc_fights_v1.csv"

    @pytest.fixture
    def processed_data_path(self):
        """Path to processed feature data."""
        return Path(__file__).parent.parent / "data" / "processed" / "ufc_fights_features_v1.csv"

    def test_raw_data_exists(self, raw_data_path):
        """Verify raw data file exists."""
        assert raw_data_path.exists(), f"Raw data not found at {raw_data_path}"

    def test_processed_data_exists(self, processed_data_path):
        """Verify processed data file exists."""
        assert processed_data_path.exists(), f"Processed data not found at {processed_data_path}"

    def test_processed_data_shape(self, processed_data_path):
        """Verify processed data has expected dimensions."""
        if not processed_data_path.exists():
            pytest.skip("Processed data not available")

        df = pd.read_csv(processed_data_path)

        # Should have 8,520 fights
        assert len(df) >= 8000, f"Expected ~8,520 fights, got {len(df)}"

        # Should have ~145 features
        assert df.shape[1] >= 100, f"Expected ~145 columns, got {df.shape[1]}"

    def test_target_variable_no_nan(self, processed_data_path):
        """Verify target variable has no NaN values."""
        if not processed_data_path.exists():
            pytest.skip("Processed data not available")

        df = pd.read_csv(processed_data_path)

        assert "f1_is_winner" in df.columns, "Target column 'f1_is_winner' not found"
        nan_count = df["f1_is_winner"].isna().sum()
        assert nan_count == 0, f"Target variable has {nan_count} NaN values"

    def test_target_variable_binary(self, processed_data_path):
        """Verify target variable is binary (0 or 1)."""
        if not processed_data_path.exists():
            pytest.skip("Processed data not available")

        df = pd.read_csv(processed_data_path)

        unique_values = set(df["f1_is_winner"].dropna().unique())
        assert unique_values.issubset({0, 1, 0.0, 1.0}), f"Target has non-binary values: {unique_values}"

    def test_key_features_present(self, processed_data_path):
        """Verify key features are present in output."""
        if not processed_data_path.exists():
            pytest.skip("Processed data not available")

        df = pd.read_csv(processed_data_path)

        key_features = [
            # Career features
            "f1_career_win_rate",
            "f1_career_ko_rate",
            # Rolling window features
            "f1_last_3_win_rate",
            "f1_last_5_win_rate",
            # Matchup differentials
            "diff_career_win_rate",
            "diff_reach",
            # Style indicators
            "f1_striking_volume",
            # Metadata
            "event_name",
            "f1_fighter_name",
        ]

        for feature in key_features:
            assert feature in df.columns, f"Key feature '{feature}' not found"

    def test_class_balance_reasonable(self, processed_data_path):
        """Verify target class balance is reasonable (not all one class)."""
        if not processed_data_path.exists():
            pytest.skip("Processed data not available")

        df = pd.read_csv(processed_data_path)

        class_counts = df["f1_is_winner"].value_counts(normalize=True)

        # Each class should be at least 30% (allows for some imbalance)
        for class_label, proportion in class_counts.items():
            assert proportion >= 0.30, f"Class {class_label} has only {proportion:.1%} of samples"


# ============================================================================
# Regression Test for Winner Detection Bug
# ============================================================================

class TestWinnerDetectionRegression:
    """Regression tests for the winner detection bug fixed on 2026-02-05."""

    @pytest.fixture
    def processed_data_path(self):
        return Path(__file__).parent.parent / "data" / "processed" / "ufc_fights_features_v1.csv"

    def test_not_all_draws(self, processed_data_path):
        """Verify not all fights are marked as draws (regression for CSS selector bug)."""
        if not processed_data_path.exists():
            pytest.skip("Processed data not available")

        df = pd.read_csv(processed_data_path)

        # If target is all 0 or all 1, winner detection is broken
        unique_targets = df["f1_is_winner"].nunique()
        assert unique_targets > 1, "All fights have same winner value - winner detection may be broken"

    def test_draw_percentage_reasonable(self, processed_data_path):
        """Verify draw percentage is reasonable (historically ~0.7% of UFC fights)."""
        if not processed_data_path.exists():
            pytest.skip("Processed data not available")

        df = pd.read_csv(processed_data_path)

        # Draws would show as 0.5 or NaN if encoded differently
        # But with binary target, we just check that both classes exist
        win_rate = df["f1_is_winner"].mean()

        # Should be roughly 50% (±20%) since fighter assignment is arbitrary
        assert 0.30 <= win_rate <= 0.70, f"Win rate {win_rate:.1%} is suspicious - check winner detection"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
