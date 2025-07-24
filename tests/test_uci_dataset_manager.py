"""
Tests for UCI Dataset Manager v2 with ModelData and DataSplit support

This module contains comprehensive tests for the updated UCIDatasetManager class.

Author: Dmatryus Detry
License: Apache 2.0
"""

import contextlib
import json
import shutil
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from dmdslab.datasets import DataSplit, ModelData
from dmdslab.datasets.uci_dataset_manager import (
    DatasetInfo,
    Domain,
    TaskType,
    UCIDatasetManager,
    print_dataset_summary,
)


def create_mock_dataset(
    n_samples,
    n_features,
    n_classes=2,
    targets_shape=None,
    use_pandas=False,
    no_target=False,
):
    """Helper function to create properly configured mock dataset."""
    mock_dataset = Mock()
    mock_dataset.data = Mock()

    if use_pandas:
        mock_dataset.data.features = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"col_{i}" for i in range(n_features)],
        )
        if not no_target:
            mock_dataset.data.targets = pd.Series(
                np.random.randint(0, n_classes, n_samples)
            )
        else:
            mock_dataset.data.targets = None
    else:
        mock_dataset.data.features = np.random.randn(n_samples, n_features)
        if not no_target:
            if targets_shape is not None:
                mock_dataset.data.targets = np.random.randint(
                    0, n_classes, targets_shape
                )
            else:
                mock_dataset.data.targets = np.random.randint(0, n_classes, n_samples)
        else:
            mock_dataset.data.targets = None

    mock_dataset.data.feature_names = None
    return mock_dataset


class TestDatasetInfo:
    """Test enhanced DatasetInfo functionality."""

    def test_dataset_info_with_feature_names(self):
        """Test DatasetInfo with feature names."""
        feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        dataset = DatasetInfo(
            id=53,
            name="Iris",
            url="https://test.com",
            n_instances=150,
            n_features=4,
            task_type=TaskType.MULTICLASS_CLASSIFICATION,
            domain=Domain.BIOLOGY,
            feature_names=feature_names,
            target_name="species",
        )

        assert dataset.feature_names == feature_names
        assert dataset.target_name == "species"

    def test_to_data_info_conversion(self):
        """Test conversion to DataInfo object."""
        dataset = DatasetInfo(
            id=73,
            name="Mushroom",
            url="https://archive.ics.uci.edu/dataset/73/mushroom",
            n_instances=8124,
            n_features=22,
            task_type=TaskType.BINARY_CLASSIFICATION,
            domain=Domain.BIOLOGY,
            class_balance={"edible": 0.52, "poisonous": 0.48},
            description="Classification of mushrooms",
            year=1987,
            is_imbalanced=False,
        )

        data_info = dataset.to_data_info()

        assert data_info.name == "Mushroom"
        assert data_info.source == "https://archive.ics.uci.edu/dataset/73/mushroom"
        assert data_info.description == "Classification of mushrooms"
        assert data_info.metadata["uci_id"] == 73
        assert data_info.metadata["task_type"] == "binary_classification"
        assert data_info.metadata["domain"] == "biology"
        assert data_info.metadata["class_balance"] == {
            "edible": 0.52,
            "poisonous": 0.48,
        }
        assert data_info.metadata["year"] == 1987

    def test_to_dict_from_dict_with_feature_names(self):
        """Test serialization with feature names."""
        feature_names = ["feature1", "feature2", "feature3"]
        dataset = DatasetInfo(
            id=1,
            name="Test",
            url="https://test.com",
            n_instances=100,
            n_features=3,
            task_type=TaskType.REGRESSION,
            domain=Domain.PHYSICS,
            feature_names=feature_names,
            target_name="output",
        )

        # Convert to dict and back
        data_dict = dataset.to_dict()
        assert data_dict["feature_names"] == json.dumps(feature_names)
        assert data_dict["target_name"] == "output"

        # Recreate from dict
        dataset_restored = DatasetInfo.from_dict(data_dict)
        assert dataset_restored.feature_names == feature_names
        assert dataset_restored.target_name == "output"


class TestUCIDatasetManager:
    """Test UCIDatasetManager with ModelData and DataSplit functionality."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database for testing."""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test_uci_datasets.db"
        yield db_path

        # Cleanup
        import gc
        import time

        gc.collect()
        time.sleep(0.1)

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                shutil.rmtree(temp_dir)
                break
            except PermissionError:
                if attempt < max_attempts - 1:
                    time.sleep(0.5)
                else:
                    with contextlib.suppress(Exception):
                        db_path.unlink(missing_ok=True)

    @pytest.fixture
    def manager(self, temp_db_path):
        """Create a UCIDatasetManager instance with temporary database."""
        mgr = UCIDatasetManager(db_path=temp_db_path)
        yield mgr
        mgr.close()

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample DatasetInfo for testing."""
        return DatasetInfo(
            id=73,
            name="Mushroom",
            url="https://archive.ics.uci.edu/dataset/73/mushroom",
            n_instances=8124,
            n_features=22,
            task_type=TaskType.BINARY_CLASSIFICATION,
            domain=Domain.BIOLOGY,
            class_balance={"edible": 0.52, "poisonous": 0.48},
            description="Classification of mushrooms",
            is_imbalanced=False,
            feature_names=[f"feature_{i}" for i in range(22)],
            target_name="edibility",
        )

    @patch("dmdslab.datasets.uci_dataset_manager.fetch_ucirepo")
    def test_load_dataset_returns_model_data(self, mock_fetch, manager, sample_dataset):
        """Test that load_dataset returns ModelData object."""
        # Add dataset to database
        manager.add_dataset(sample_dataset)

        # Mock the fetch_ucirepo response
        mock_fetch.return_value = create_mock_dataset(100, 22)

        # Load dataset
        model_data = manager.load_dataset(73)

        # Verify it's a ModelData object
        assert isinstance(model_data, ModelData)
        assert model_data.n_samples == 100
        assert model_data.n_features == 22
        assert model_data.feature_names == [f"feature_{i}" for i in range(22)]

        # Check DataInfo
        assert model_data.info is not None
        assert model_data.info.name == "Mushroom"
        assert model_data.info.metadata["uci_id"] == 73
        assert model_data.info.metadata["task_type"] == "binary_classification"

    @patch("dmdslab.datasets.uci_dataset_manager.fetch_ucirepo")
    def test_load_dataset_with_pandas_data(self, mock_fetch, manager, sample_dataset):
        """Test loading dataset that returns pandas DataFrames."""
        # Add dataset to database
        manager.add_dataset(sample_dataset)

        # Mock the fetch_ucirepo response with pandas data
        mock_fetch.return_value = create_mock_dataset(50, 22, use_pandas=True)

        # Load dataset
        model_data = manager.load_dataset(73)

        # Verify conversion to numpy
        assert isinstance(model_data.features, np.ndarray)
        assert isinstance(model_data.target, np.ndarray)
        assert model_data.n_samples == 50

    @patch("dmdslab.datasets.uci_dataset_manager.fetch_ucirepo")
    def test_load_dataset_split(self, mock_fetch, manager, sample_dataset):
        """Test loading dataset with automatic train/test split."""
        # Add dataset to database
        manager.add_dataset(sample_dataset)

        # Mock the fetch_ucirepo response
        mock_fetch.return_value = create_mock_dataset(1000, 22)

        # Load dataset with split
        split = manager.load_dataset_split(73, test_size=0.2, random_state=42)

        # Verify it's a DataSplit object
        assert isinstance(split, DataSplit)
        assert split.train.n_samples == 800
        assert split.test.n_samples == 200
        assert split.split_info["dataset_name"] == "Mushroom"
        assert split.split_info["dataset_id"] == 73

    @patch("dmdslab.datasets.uci_dataset_manager.fetch_ucirepo")
    def test_load_dataset_split_with_validation(
        self, mock_fetch, manager, sample_dataset
    ):
        """Test loading dataset with train/validation/test split."""
        # Add dataset to database
        manager.add_dataset(sample_dataset)

        # Mock the fetch_ucirepo response
        mock_fetch.return_value = create_mock_dataset(1000, 22)

        # Load dataset with full split
        split = manager.load_dataset_split(
            73, test_size=0.2, validation_size=0.2, random_state=42
        )

        # Verify splits
        assert split.train.n_samples == 600
        assert split.validation.n_samples == 200
        assert split.test.n_samples == 200
        assert split.has_validation
        assert split.has_test

    @patch("dmdslab.datasets.uci_dataset_manager.fetch_ucirepo")
    def test_load_dataset_split_auto_stratify(self, mock_fetch, manager):
        """Test that stratification is auto-enabled for classification tasks."""
        # Add classification dataset
        dataset = DatasetInfo(
            id=1,
            name="Binary Classification",
            url="https://test.com",
            n_instances=1000,
            n_features=10,
            task_type=TaskType.BINARY_CLASSIFICATION,
            domain=Domain.FINANCE,
        )
        manager.add_dataset(dataset)

        # Mock the fetch_ucirepo response with imbalanced data
        mock_dataset = Mock()
        mock_dataset.data = Mock()
        mock_dataset.data.features = np.random.randn(1000, 10)
        mock_dataset.data.targets = np.array([0] * 700 + [1] * 300)
        mock_dataset.data.feature_names = None
        mock_fetch.return_value = mock_dataset

        # Load with split (stratify should be auto-enabled)
        split = manager.load_dataset_split(1, test_size=0.2, random_state=42)

        # Check that class distribution is preserved
        train_ratio = np.mean(split.train.target == 1)
        test_ratio = np.mean(split.test.target == 1)
        assert abs(train_ratio - 0.3) < 0.05  # Should be close to 30%
        assert abs(test_ratio - 0.3) < 0.05

    @patch("dmdslab.datasets.uci_dataset_manager.fetch_ucirepo")
    def test_load_dataset_kfold(self, mock_fetch, manager, sample_dataset):
        """Test loading dataset with k-fold cross-validation splits."""
        # Add dataset to database
        manager.add_dataset(sample_dataset)

        # Mock the fetch_ucirepo response
        mock_fetch.return_value = create_mock_dataset(150, 22)

        # Load dataset with k-fold splits
        splits = manager.load_dataset_kfold(73, n_splits=5, random_state=42)

        # Verify splits
        assert len(splits) == 5
        for i, split in enumerate(splits):
            assert isinstance(split, DataSplit)
            assert split.train.n_samples == 120
            assert split.validation.n_samples == 30
            assert split.split_info["fold"] == i
            assert split.split_info["dataset_name"] == "Mushroom"
            assert split.split_info["dataset_id"] == 73

    @patch("dmdslab.datasets.uci_dataset_manager.fetch_ucirepo")
    def test_load_dataset_with_missing_target(self, mock_fetch, manager):
        """Test loading dataset without target variable (unsupervised)."""
        # Add clustering dataset
        dataset = DatasetInfo(
            id=999,
            name="Clustering Dataset",
            url="https://test.com",
            n_instances=200,
            n_features=5,
            task_type=TaskType.CLUSTERING,
            domain=Domain.ARTIFICIAL,
        )
        manager.add_dataset(dataset)

        # Mock response without targets
        mock_fetch.return_value = create_mock_dataset(200, 5, no_target=True)

        # Load dataset
        model_data = manager.load_dataset(999)

        # Verify no target
        assert model_data.target is None
        assert not model_data.has_target
        assert model_data.n_samples == 200
        assert model_data.n_features == 5

    @patch("dmdslab.datasets.uci_dataset_manager.fetch_ucirepo")
    def test_load_dataset_handles_2d_target(self, mock_fetch, manager, sample_dataset):
        """Test that 2D targets are converted to 1D."""
        # Add dataset to database
        manager.add_dataset(sample_dataset)

        # Mock response with 2D target
        mock_fetch.return_value = create_mock_dataset(100, 22, targets_shape=(100, 1))

        # Load dataset
        model_data = manager.load_dataset(73)

        # Verify target is 1D
        assert model_data.target.ndim == 1
        assert len(model_data.target) == 100

    def test_database_schema_includes_new_fields(self, manager):
        """Test that database schema includes feature_names and target_name."""
        with sqlite3.connect(manager._get_db_path()) as conn:
            cursor = conn.execute("PRAGMA table_info(datasets)")
            columns = {row[1] for row in cursor.fetchall()}

        assert "feature_names" in columns
        assert "target_name" in columns

    def test_full_integration_workflow(self, manager):
        """Test complete workflows with ModelData and DataSplit."""
        # Add a dataset
        dataset = DatasetInfo(
            id=100,
            name="Integration Test Dataset",
            url="https://test.com",
            n_instances=5000,
            n_features=20,
            task_type=TaskType.MULTICLASS_CLASSIFICATION,
            domain=Domain.ARTIFICIAL,
            feature_names=[f"x{i}" for i in range(20)],
            target_name="class",
            description="Dataset for integration testing",
        )
        manager.add_dataset(dataset)

        # Mock the fetch
        with patch("dmdslab.datasets.uci_dataset_manager.fetch_ucirepo") as mock_fetch:
            mock_fetch.return_value = create_mock_dataset(5000, 20, n_classes=3)

            # 1. Load as ModelData
            model_data = manager.load_dataset(100)
            assert isinstance(model_data, ModelData)
            assert model_data.shape == (5000, 20)
            assert model_data.info.name == "Integration Test Dataset"

            # 2. Convert to pandas and back
            features_df, target_series = model_data.to_pandas()
            assert isinstance(features_df, pd.DataFrame)
            assert features_df.columns.tolist() == [f"x{i}" for i in range(20)]

            # 3. Create train/test split
            split = manager.load_dataset_split(100, test_size=0.3, random_state=42)
            assert split.train.n_samples == 3500
            assert split.test.n_samples == 1500

            # 4. Create k-fold splits
            kfold_splits = manager.load_dataset_kfold(100, n_splits=3, random_state=42)
            assert len(kfold_splits) == 3

            # 5. Work with a fold
            first_fold = kfold_splits[0]
            # Sample from training data
            train_sample = first_fold.train.sample(n=100, random_state=42)
            assert train_sample.n_samples == 100


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.fixture
    def manager(self):
        return UCIDatasetManager()

    def test_load_nonexistent_dataset(self, manager):
        """Test loading dataset that doesn't exist in database."""
        with pytest.raises(ValueError, match="Dataset with ID 99999 not found"):
            manager.load_dataset(99999)

    def test_load_split_nonexistent_dataset(self, manager):
        """Test creating split for nonexistent dataset."""
        with pytest.raises(ValueError, match="Dataset with ID 99999 not found"):
            manager.load_dataset_split(99999)

    def test_load_kfold_nonexistent_dataset(self, manager):
        """Test creating k-fold for nonexistent dataset."""
        with pytest.raises(ValueError, match="Dataset with ID 99999 not found"):
            manager.load_dataset_kfold(99999)

    @patch("dmdslab.datasets.uci_dataset_manager.fetch_ucirepo")
    def test_network_error_handling(self, mock_fetch, manager):
        """Test handling of network errors during fetch."""
        # Add dataset
        dataset = DatasetInfo(
            id=1,
            name="Test",
            url="https://test.com",
            n_instances=100,
            n_features=5,
            task_type=TaskType.REGRESSION,
            domain=Domain.PHYSICS,
        )
        manager.add_dataset(dataset)

        # Mock network error
        mock_fetch.side_effect = Exception("Network connection failed")

        with pytest.raises(Exception, match="Network connection failed"):
            manager.load_dataset(1)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_print_dataset_summary(self, capsys):
        """Test dataset summary printing."""
        datasets = [
            DatasetInfo(
                id=1,
                name="Test Dataset 1",
                url="http://test1.com",
                n_instances=1000,
                n_features=10,
                task_type=TaskType.BINARY_CLASSIFICATION,
                domain=Domain.FINANCE,
                is_imbalanced=True,
            ),
            DatasetInfo(
                id=2,
                name="Test Dataset 2",
                url="http://test2.com",
                n_instances=50000,
                n_features=100,
                task_type=TaskType.REGRESSION,
                domain=Domain.PHYSICS,
                is_imbalanced=False,
            ),
        ]

        print_dataset_summary(datasets)

        captured = capsys.readouterr()
        output = captured.out

        assert "Found 2 datasets:" in output
        assert "Test Dataset 1" in output
        assert "Test Dataset 2" in output
        assert "1,000" in output  # Formatted number
        assert "50,000" in output  # Formatted number
        assert "finance" in output
        assert "physics" in output
        assert "Yes" in output  # Imbalanced
        assert "No" in output  # Not imbalanced


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
