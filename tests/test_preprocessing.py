"""Unit tests for preprocessing functions."""

import pandas as pd
import numpy as np
import tempfile
import os

from src.preprocessing import preprocess_titanic


class TestPreprocessTitanic:
    """Test cases for preprocess_titanic function."""

    def test_preprocess_with_valid_data(self):
        """Test preprocessing with valid Titanic-like data."""
        # Create temporary train and test files
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = os.path.join(tmpdir, "train.csv")
            test_path = os.path.join(tmpdir, "test.csv")
            output_path = os.path.join(tmpdir, "preprocessed.csv")

            # Create sample train data
            train_data = pd.DataFrame(
                {
                    "PassengerId": [1, 2, 3],
                    "Survived": [1, 0, 1],
                    "Pclass": [1, 2, 3],
                    "Name": ["Mr. John Doe", "Miss Jane Smith", "Mrs. Mary Johnson"],
                    "Sex": ["male", "female", "female"],
                    "Age": [30, 25, 35],
                    "SibSp": [0, 1, 0],
                    "Parch": [0, 0, 2],
                    "Ticket": ["A123", "B456", "C789"],
                    "Fare": [50.0, 30.0, 40.0],
                    "Cabin": ["C1", None, "C2"],
                    "Embarked": ["S", "C", "Q"],
                }
            )
            train_data.to_csv(train_path, index=False)

            # Create sample test data
            test_data = pd.DataFrame(
                {
                    "PassengerId": [4, 5],
                    "Pclass": [2, 1],
                    "Name": ["Mr. Bob Wilson", "Miss Alice Brown"],
                    "Sex": ["male", "female"],
                    "Age": [28, 22],
                    "SibSp": [1, 0],
                    "Parch": [1, 0],
                    "Ticket": ["D123", "E456"],
                    "Fare": [35.0, 60.0],
                    "Cabin": [None, "D1"],
                    "Embarked": ["S", "C"],
                }
            )
            test_data.to_csv(test_path, index=False)

            # Run preprocessing
            result = preprocess_titanic(
                raw_train_path=train_path, raw_test_path=test_path, output_path=output_path
            )

            # Assertions
            assert result is not None
            assert isinstance(result, pd.DataFrame)
            assert os.path.exists(output_path)

            # Check that PassengerId and Name are dropped
            assert "PassengerId" not in result.columns
            assert "Name" not in result.columns
            assert "Ticket" not in result.columns
            assert "Cabin" not in result.columns

            # Check that Sex is encoded
            assert "Sex" in result.columns
            assert result["Sex"].dtype in [int, np.int64]

            # Check that Title feature exists
            assert "Title" in result.columns

    def test_preprocess_with_missing_files(self):
        """Test preprocessing with missing input files."""
        result = preprocess_titanic(
            raw_train_path="nonexistent_train.csv", raw_test_path="nonexistent_test.csv"
        )
        assert result is None

    def test_preprocess_handles_missing_values(self):
        """Test that preprocessing handles missing values correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = os.path.join(tmpdir, "train.csv")
            test_path = os.path.join(tmpdir, "test.csv")
            output_path = os.path.join(tmpdir, "preprocessed.csv")

            # Create data with missing values
            train_data = pd.DataFrame(
                {
                    "PassengerId": [1, 2, 3],
                    "Survived": [1, 0, 1],
                    "Pclass": [1, 2, 3],
                    "Name": ["Mr. John", "Miss Jane", "Mrs. Mary"],
                    "Sex": ["male", "female", "female"],
                    "Age": [30, None, 35],  # Missing value
                    "SibSp": [0, 1, 0],
                    "Parch": [0, 0, 2],
                    "Ticket": ["A123", "B456", "C789"],
                    "Fare": [50.0, None, 40.0],  # Missing value
                    "Cabin": ["C1", None, "C2"],
                    "Embarked": ["S", None, "Q"],  # Missing value
                }
            )
            train_data.to_csv(train_path, index=False)

            test_data = pd.DataFrame(
                {
                    "PassengerId": [4, 5],
                    "Pclass": [2, 1],
                    "Name": ["Mr. Bob", "Miss Alice"],
                    "Sex": ["male", "female"],
                    "Age": [28, 22],
                    "SibSp": [1, 0],
                    "Parch": [1, 0],
                    "Ticket": ["D123", "E456"],
                    "Fare": [35.0, 60.0],
                    "Cabin": [None, "D1"],
                    "Embarked": ["S", "C"],
                }
            )
            test_data.to_csv(test_path, index=False)

            result = preprocess_titanic(
                raw_train_path=train_path, raw_test_path=test_path, output_path=output_path
            )

            assert result is not None
            # Check that missing values are filled
            assert result["Age"].isna().sum() == 0
            assert result["Fare"].isna().sum() == 0
            assert result["Embarked"].isna().sum() == 0
