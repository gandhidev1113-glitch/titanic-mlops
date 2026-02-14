"""Unit tests for training functions."""

import pytest
import pandas as pd
import tempfile
import os
from sklearn.ensemble import RandomForestClassifier

from src.train import (
    load_preprocessed_data,
    get_original_train_size,
    split_merged_df,
    build_features,
    train_baseline_model,
    save_model,
    evaluate_model,
    save_kaggle_predictions,
)


class TestLoadPreprocessedData:
    """Test cases for load_preprocessed_data function."""

    def test_load_existing_file(self):
        """Test loading an existing preprocessed file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
            df.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            result = load_preprocessed_data(temp_path)
            assert result is not None
            assert isinstance(result, pd.DataFrame)
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_file(self):
        """Test loading a non-existent file returns None."""
        result = load_preprocessed_data("nonexistent_file.csv")
        assert result is None


class TestGetOriginalTrainSize:
    """Test cases for get_original_train_size function."""

    def test_get_size_from_file(self):
        """Test getting size from existing file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df = pd.DataFrame({"col1": [1, 2, 3, 4, 5]})
            df.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            size = get_original_train_size(temp_path)
            assert size == 5
        finally:
            os.unlink(temp_path)

    def test_get_default_size(self):
        """Test getting default size when file doesn't exist."""
        size = get_original_train_size("nonexistent.csv")
        assert size == 891  # Default value


class TestSplitMergedDf:
    """Test cases for split_merged_df function."""

    def test_split_dataframe(self, monkeypatch):
        """Test splitting a merged dataframe."""
        df = pd.DataFrame(
            {"Survived": [1, 0, 1, 0, 1], "Pclass": [1, 2, 3, 1, 2], "Age": [30, 25, 35, 28, 22]}
        )

        # Mock get_original_train_size to return 3
        monkeypatch.setattr("src.train.get_original_train_size", lambda: 3)

        train_df, test_df = split_merged_df(df)
        assert len(train_df) == 3
        assert len(test_df) == 2


class TestBuildFeatures:
    """Test cases for build_features function."""

    def test_build_features_basic(self):
        """Test building features from train and test dataframes."""
        train_df = pd.DataFrame(
            {"Survived": [1, 0, 1], "Pclass": [1, 2, 3], "Sex": [1, 0, 1], "Age": [30, 25, 35]}
        )

        test_df = pd.DataFrame({"Pclass": [2, 1], "Sex": [0, 1], "Age": [28, 22]})

        X, y, X_kaggle = build_features(train_df, test_df)

        assert "Survived" not in X.columns
        assert len(y) == 3
        assert len(X_kaggle) == 2
        assert X.shape[1] == X_kaggle.shape[1]  # Same number of features

    def test_build_features_missing_survived(self):
        """Test that missing Survived column raises error."""
        train_df = pd.DataFrame({"Pclass": [1, 2, 3], "Age": [30, 25, 35]})

        test_df = pd.DataFrame({"Pclass": [2, 1], "Age": [28, 22]})

        with pytest.raises(ValueError, match="Survived"):
            build_features(train_df, test_df)


class TestTrainBaselineModel:
    """Test cases for train_baseline_model function."""

    def test_train_model(self):
        """Test training a baseline model."""
        X_train = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [2, 3, 4, 5, 6]})
        y_train = pd.Series([0, 1, 0, 1, 0])

        model = train_baseline_model(X_train, y_train)

        assert isinstance(model, RandomForestClassifier)
        assert hasattr(model, "predict")

    def test_model_predictions(self):
        """Test that trained model can make predictions."""
        X_train = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [2, 3, 4, 5, 6]})
        y_train = pd.Series([0, 1, 0, 1, 0])

        model = train_baseline_model(X_train, y_train)
        predictions = model.predict(X_train)

        assert len(predictions) == len(y_train)
        assert all(pred in [0, 1] for pred in predictions)


class TestSaveModel:
    """Test cases for save_model function."""

    def test_save_model(self):
        """Test saving a model to file."""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        y = pd.Series([0, 1, 0])
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_model.pkl")
            save_model(model, output_path)
            assert os.path.exists(output_path)


class TestEvaluateModel:
    """Test cases for evaluate_model function."""

    def test_evaluate_model(self):
        """Test model evaluation returns metrics."""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_train = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [2, 3, 4, 5, 6]})
        y_train = pd.Series([0, 1, 0, 1, 0])
        X_val = pd.DataFrame({"feature1": [6, 7], "feature2": [7, 8]})
        y_val = pd.Series([1, 0])

        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_train, y_train, X_val, y_val)

        assert isinstance(metrics, dict)
        assert "train_accuracy" in metrics
        assert "val_accuracy" in metrics
        assert "val_precision" in metrics
        assert "val_recall" in metrics
        assert "val_f1" in metrics
        assert all(0 <= v <= 1 for v in metrics.values())


class TestTrainBaselineModelWithParams:
    """Test cases for train_baseline_model with custom parameters."""

    def test_train_with_custom_params(self):
        """Test training with custom hyperparameters."""
        X_train = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [2, 3, 4, 5, 6]})
        y_train = pd.Series([0, 1, 0, 1, 0])

        model = train_baseline_model(
            X_train,
            y_train,
            n_estimators=50,
            max_depth=5,
            random_state=123,
            class_weight="balanced",
        )

        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 50
        assert model.max_depth == 5
        assert model.random_state == 123


class TestSaveKagglePredictions:
    """Test cases for save_kaggle_predictions function."""

    def test_save_kaggle_predictions(self):
        """Test saving Kaggle predictions file."""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        y_train = pd.Series([0, 1, 0])
        model.fit(X_train, y_train)

        X_kaggle = pd.DataFrame({"feature1": [7, 8], "feature2": [9, 10]})

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "predictions.csv")
            save_kaggle_predictions(model, X_kaggle, output_path=output_path)

            assert os.path.exists(output_path)
            result_df = pd.read_csv(output_path)
            assert "PassengerId" in result_df.columns
            assert "Survived" in result_df.columns
            assert len(result_df) == 2

    def test_save_kaggle_predictions_with_raw_test(self):
        """Test saving predictions with raw test file."""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        y_train = pd.Series([0, 1, 0])
        model.fit(X_train, y_train)

        X_kaggle = pd.DataFrame({"feature1": [7, 8], "feature2": [9, 10]})

        # Create a temporary raw test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            raw_test = pd.DataFrame({"PassengerId": [100, 101], "Pclass": [1, 2]})
            raw_test.to_csv(f.name, index=False)
            raw_test_path = f.name

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = os.path.join(tmpdir, "predictions.csv")
                save_kaggle_predictions(
                    model, X_kaggle, raw_test_path=raw_test_path, output_path=output_path
                )

                assert os.path.exists(output_path)
                result_df = pd.read_csv(output_path)
                assert list(result_df["PassengerId"]) == [100, 101]
        finally:
            os.unlink(raw_test_path)
