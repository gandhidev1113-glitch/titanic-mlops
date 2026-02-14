"""Unit tests for utility functions."""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import os

from src.utils import load_data_safely, ensure_dir_exists, get_project_root


class TestLoadDataSafely:
    """Test cases for load_data_safely function."""

    def test_load_valid_csv(self):
        """Test loading a valid CSV file."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("col1,col2\n1,2\n3,4\n")
            temp_path = f.name

        try:
            df = load_data_safely(temp_path)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert list(df.columns) == ["col1", "col2"]
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_file(self):
        """Test loading a non-existent file raises an error."""
        with pytest.raises(FileNotFoundError):
            load_data_safely("nonexistent_file.csv")


class TestEnsureDirExists:
    """Test cases for ensure_dir_exists function."""

    def test_create_new_directory(self):
        """Test creating a new directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, "test_dir")
            result = ensure_dir_exists(new_dir)
            assert os.path.exists(new_dir)
            assert isinstance(result, Path)

    def test_existing_directory(self):
        """Test with an existing directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = ensure_dir_exists(tmpdir)
            assert os.path.exists(tmpdir)
            assert isinstance(result, Path)


class TestGetProjectRoot:
    """Test cases for get_project_root function."""

    def test_returns_path_object(self):
        """Test that get_project_root returns a Path object."""
        root = get_project_root()
        assert isinstance(root, Path)
        assert root.exists()
