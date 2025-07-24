"""
Tests for UCI Dataset Manager

This module contains comprehensive tests for the UCIDatasetManager class.

Author: Dmatryus Detry
License: Apache 2.0
"""

import pytest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sqlite3
import json
import numpy as np

from dmdslab.datasets.uci_dataset_manager import (
    DatasetInfo,
    TaskType,
    Domain,
    UCIDatasetManager,
    print_dataset_summary,
)


class TestDatasetInfo:
    """Test DatasetInfo dataclass functionality."""

    def test_dataset_info_creation(self):
        """Test creating a DatasetInfo instance."""
        dataset = DatasetInfo(
            id=1,
            name="Test Dataset",
            url="https://test.com",
            n_instances=1000,
            n_features=10,
            task_type=TaskType.BINARY_CLASSIFICATION,
            domain=Domain.FINANCE,
            class_balance={"positive": 0.7, "negative": 0.3},
            description="Test description",
            year=2023,
            has_missing_values=True,
            is_imbalanced=True,
            imbalance_ratio=2.33,
        )

        assert dataset.id == 1
        assert dataset.name == "Test Dataset"
        assert dataset.n_instances == 1000
        assert dataset.n_features == 10
        assert dataset.task_type == TaskType.BINARY_CLASSIFICATION
        assert dataset.domain == Domain.FINANCE
        assert dataset.class_balance == {"positive": 0.7, "negative": 0.3}
        assert dataset.is_imbalanced is True
        assert dataset.imbalance_ratio == 2.33

    def test_dataset_info_to_dict(self):
        """Test converting DatasetInfo to dictionary."""
        dataset = DatasetInfo(
            id=1,
            name="Test Dataset",
            url="https://test.com",
            n_instances=1000,
            n_features=10,
            task_type=TaskType.BINARY_CLASSIFICATION,
            domain=Domain.FINANCE,
            class_balance={"positive": 0.7, "negative": 0.3},
        )

        data_dict = dataset.to_dict()

        assert data_dict["id"] == 1
        assert data_dict["name"] == "Test Dataset"
        assert data_dict["task_type"] == "binary_classification"
        assert data_dict["domain"] == "finance"
        assert data_dict["class_balance"] == '{"positive": 0.7, "negative": 0.3}'

    def test_dataset_info_from_dict(self):
        """Test creating DatasetInfo from dictionary."""
        data_dict = {
            "id": 1,
            "name": "Test Dataset",
            "url": "https://test.com",
            "n_instances": 1000,
            "n_features": 10,
            "task_type": "binary_classification",
            "domain": "finance",
            "class_balance": '{"positive": 0.7, "negative": 0.3}',
            "description": "Test description",
            "year": 2023,
            "has_missing_values": True,
            "is_imbalanced": True,
            "imbalance_ratio": 2.33,
        }

        dataset = DatasetInfo.from_dict(data_dict)

        assert dataset.id == 1
        assert dataset.name == "Test Dataset"
        assert dataset.task_type == TaskType.BINARY_CLASSIFICATION
        assert dataset.domain == Domain.FINANCE
        assert dataset.class_balance == {"positive": 0.7, "negative": 0.3}
        assert dataset.is_imbalanced is True

    def test_dataset_info_minimal(self):
        """Test creating DatasetInfo with minimal required fields."""
        dataset = DatasetInfo(
            id=1,
            name="Minimal Dataset",
            url="https://test.com",
            n_instances=100,
            n_features=5,
            task_type=TaskType.REGRESSION,
            domain=Domain.PHYSICS,
        )

        assert dataset.class_balance is None
        assert dataset.description == ""
        assert dataset.year is None
        assert dataset.has_missing_values is False
        assert dataset.is_imbalanced is False
        assert dataset.imbalance_ratio is None


class TestUCIDatasetManager:
    """Test UCIDatasetManager functionality."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir) / "test_uci_datasets.db"
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def manager(self, temp_db_path):
        """Create a UCIDatasetManager instance with temporary database."""
        return UCIDatasetManager(db_path=temp_db_path)

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
        )

    def test_init_creates_database(self, temp_db_path):
        """Test that initialization creates database file and schema."""
        manager = UCIDatasetManager(db_path=temp_db_path)

        # Check database file exists
        assert temp_db_path.exists()

        # Check table exists
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='datasets'"
            )
            assert cursor.fetchone() is not None

    def test_add_dataset(self, manager, sample_dataset):
        """Test adding a dataset to the database."""
        manager.add_dataset(sample_dataset)

        # Verify dataset was added
        dataset_info = manager.get_dataset_info(73)
        assert dataset_info is not None
        assert dataset_info.name == "Mushroom"
        assert dataset_info.n_instances == 8124

    def test_add_dataset_update(self, manager, sample_dataset):
        """Test updating an existing dataset."""
        # Add dataset
        manager.add_dataset(sample_dataset)

        # Update dataset
        sample_dataset.n_instances = 9000
        sample_dataset.description = "Updated description"
        manager.add_dataset(sample_dataset)

        # Verify update
        dataset_info = manager.get_dataset_info(73)
        assert dataset_info.n_instances == 9000
        assert dataset_info.description == "Updated description"

    def test_get_dataset_info_not_found(self, manager):
        """Test getting non-existent dataset."""
        dataset_info = manager.get_dataset_info(99999)
        assert dataset_info is None

    def test_filter_datasets_by_task_type(self, manager):
        """Test filtering datasets by task type."""
        # Add multiple datasets
        datasets = [
            DatasetInfo(
                id=1,
                name="Binary1",
                url="http://test1.com",
                n_instances=1000,
                n_features=10,
                task_type=TaskType.BINARY_CLASSIFICATION,
                domain=Domain.FINANCE,
            ),
            DatasetInfo(
                id=2,
                name="Regression1",
                url="http://test2.com",
                n_instances=2000,
                n_features=20,
                task_type=TaskType.REGRESSION,
                domain=Domain.PHYSICS,
            ),
            DatasetInfo(
                id=3,
                name="Binary2",
                url="http://test3.com",
                n_instances=3000,
                n_features=30,
                task_type=TaskType.BINARY_CLASSIFICATION,
                domain=Domain.MEDICINE,
            ),
        ]

        for dataset in datasets:
            manager.add_dataset(dataset)

        # Filter by task type
        binary_datasets = manager.filter_datasets(
            task_type=TaskType.BINARY_CLASSIFICATION
        )
        assert len(binary_datasets) == 2
        assert all(
            d.task_type == TaskType.BINARY_CLASSIFICATION for d in binary_datasets
        )

    def test_filter_datasets_by_size(self, manager):
        """Test filtering datasets by instance count."""
        # Add datasets with different sizes
        for i, size in enumerate([500, 1500, 2500, 3500]):
            manager.add_dataset(
                DatasetInfo(
                    id=i,
                    name=f"Dataset{i}",
                    url=f"http://test{i}.com",
                    n_instances=size,
                    n_features=10,
                    task_type=TaskType.BINARY_CLASSIFICATION,
                    domain=Domain.FINANCE,
                )
            )

        # Filter by size range
        filtered = manager.filter_datasets(min_instances=1000, max_instances=3000)
        assert len(filtered) == 2
        assert all(1000 <= d.n_instances <= 3000 for d in filtered)

    def test_filter_datasets_by_imbalance(self, manager):
        """Test filtering datasets by imbalance."""
        # Add balanced and imbalanced datasets
        datasets = [
            DatasetInfo(
                id=1,
                name="Balanced",
                url="http://test1.com",
                n_instances=1000,
                n_features=10,
                task_type=TaskType.BINARY_CLASSIFICATION,
                domain=Domain.FINANCE,
                is_imbalanced=False,
            ),
            DatasetInfo(
                id=2,
                name="Imbalanced1",
                url="http://test2.com",
                n_instances=2000,
                n_features=20,
                task_type=TaskType.BINARY_CLASSIFICATION,
                domain=Domain.MEDICINE,
                is_imbalanced=True,
                imbalance_ratio=5.0,
            ),
            DatasetInfo(
                id=3,
                name="Imbalanced2",
                url="http://test3.com",
                n_instances=3000,
                n_features=30,
                task_type=TaskType.BINARY_CLASSIFICATION,
                domain=Domain.PHYSICS,
                is_imbalanced=True,
                imbalance_ratio=10.0,
            ),
        ]

        for dataset in datasets:
            manager.add_dataset(dataset)

        # Filter imbalanced datasets
        imbalanced = manager.filter_datasets(is_imbalanced=True)
        assert len(imbalanced) == 2
        assert all(d.is_imbalanced for d in imbalanced)

        # Filter by imbalance ratio
        highly_imbalanced = manager.filter_datasets(
            is_imbalanced=True, max_imbalance_ratio=7.0
        )
        assert len(highly_imbalanced) == 1
        assert highly_imbalanced[0].name == "Imbalanced1"

    def test_filter_datasets_multiple_criteria(self, manager):
        """Test filtering with multiple criteria."""
        # Add diverse datasets
        manager.add_dataset(
            DatasetInfo(
                id=1,
                name="Match",
                url="http://test1.com",
                n_instances=5000,
                n_features=50,
                task_type=TaskType.BINARY_CLASSIFICATION,
                domain=Domain.FINANCE,
                is_imbalanced=True,
                imbalance_ratio=3.0,
            )
        )
        manager.add_dataset(
            DatasetInfo(
                id=2,
                name="NoMatch1",
                url="http://test2.com",
                n_instances=500,  # Too small
                n_features=50,
                task_type=TaskType.BINARY_CLASSIFICATION,
                domain=Domain.FINANCE,
                is_imbalanced=True,
                imbalance_ratio=3.0,
            )
        )
        manager.add_dataset(
            DatasetInfo(
                id=3,
                name="NoMatch2",
                url="http://test3.com",
                n_instances=5000,
                n_features=50,
                task_type=TaskType.REGRESSION,  # Wrong type
                domain=Domain.FINANCE,
            )
        )

        # Apply multiple filters
        filtered = manager.filter_datasets(
            task_type=TaskType.BINARY_CLASSIFICATION,
            domain=Domain.FINANCE,
            min_instances=1000,
            is_imbalanced=True,
        )

        assert len(filtered) == 1
        assert filtered[0].name == "Match"

    @patch("uci_dataset_manager.fetch_ucirepo")
    def test_load_dataset_success(self, mock_fetch, manager, sample_dataset):
        """Test successful dataset loading."""
        # Add dataset to database
        manager.add_dataset(sample_dataset)

        # Mock the fetch_ucirepo response
        mock_dataset = Mock()
        mock_dataset.data.features = [[1, 2, 3], [4, 5, 6]]
        mock_dataset.data.targets = [0, 1]
        mock_fetch.return_value = mock_dataset

        # Load dataset
        X, y = manager.load_dataset(73)

        assert X == [[1, 2, 3], [4, 5, 6]]
        assert y == [0, 1]
        mock_fetch.assert_called_once_with(id=73)

    @patch("uci_dataset_manager.fetch_ucirepo")
    def test_load_dataset_with_metadata(self, mock_fetch, manager, sample_dataset):
        """Test loading dataset with metadata."""
        # Add dataset to database
        manager.add_dataset(sample_dataset)

        # Mock the fetch_ucirepo response
        mock_dataset = Mock()
        mock_dataset.data.features = [[1, 2, 3]]
        mock_dataset.data.targets = [0]
        mock_fetch.return_value = mock_dataset

        # Load dataset with metadata
        X, y, info = manager.load_dataset(73, return_metadata=True)

        assert X == [[1, 2, 3]]
        assert y == [0]
        assert info.name == "Mushroom"
        assert info.n_instances == 8124

    def test_load_dataset_not_found(self, manager):
        """Test loading non-existent dataset."""
        with pytest.raises(ValueError, match="Dataset with ID 99999 not found"):
            manager.load_dataset(99999)

    @patch("uci_dataset_manager.fetch_ucirepo")
    def test_load_dataset_fetch_error(self, mock_fetch, manager, sample_dataset):
        """Test handling fetch errors."""
        # Add dataset to database
        manager.add_dataset(sample_dataset)

        # Mock fetch error
        mock_fetch.side_effect = Exception("Network error")

        # Should raise the exception
        with pytest.raises(Exception, match="Network error"):
            manager.load_dataset(73)

    @patch("uci_dataset_manager.fetch_ucirepo")
    def test_load_dataset_caching(self, mock_fetch, manager, sample_dataset):
        """Test that dataset loading is cached."""
        # Add dataset to database
        manager.add_dataset(sample_dataset)

        # Mock the fetch_ucirepo response
        mock_dataset = Mock()
        mock_dataset.data.features = [[1, 2, 3]]
        mock_dataset.data.targets = [0]
        mock_fetch.return_value = mock_dataset

        # Load dataset twice
        X1, y1 = manager.load_dataset(73)
        X2, y2 = manager.load_dataset(73)

        # Should only fetch once due to caching
        mock_fetch.assert_called_once()
        assert X1 == X2
        assert y1 == y2

    def test_get_statistics_empty(self, manager):
        """Test getting statistics from empty database."""
        stats = manager.get_statistics()

        assert stats["total_datasets"] == 0
        assert stats["by_task_type"] == {}
        assert stats["by_domain"] == {}
        assert stats["imbalanced_datasets"] == 0
        assert stats["avg_instances"] == 0
        assert stats["avg_features"] == 0

    def test_get_statistics_with_data(self, manager):
        """Test getting statistics with datasets."""
        # Add various datasets
        datasets = [
            DatasetInfo(
                id=1,
                name="Dataset1",
                url="http://test1.com",
                n_instances=1000,
                n_features=10,
                task_type=TaskType.BINARY_CLASSIFICATION,
                domain=Domain.FINANCE,
                is_imbalanced=True,
            ),
            DatasetInfo(
                id=2,
                name="Dataset2",
                url="http://test2.com",
                n_instances=2000,
                n_features=20,
                task_type=TaskType.BINARY_CLASSIFICATION,
                domain=Domain.MEDICINE,
                is_imbalanced=False,
            ),
            DatasetInfo(
                id=3,
                name="Dataset3",
                url="http://test3.com",
                n_instances=3000,
                n_features=30,
                task_type=TaskType.REGRESSION,
                domain=Domain.FINANCE,
                is_imbalanced=False,
            ),
        ]

        for dataset in datasets:
            manager.add_dataset(dataset)

        stats = manager.get_statistics()

        assert stats["total_datasets"] == 3
        assert stats["by_task_type"]["binary_classification"] == 2
        assert stats["by_task_type"]["regression"] == 1
        assert stats["by_domain"]["finance"] == 2
        assert stats["by_domain"]["medicine"] == 1
        assert stats["imbalanced_datasets"] == 1
        assert stats["avg_instances"] == 2000
        assert stats["avg_features"] == 20

    def test_delete_dataset(self, manager, sample_dataset):
        """Test deleting a dataset from the database."""
        # Add dataset
        manager.add_dataset(sample_dataset)

        # Verify it was added
        assert manager.get_dataset_info(73) is not None

        # Delete dataset
        deleted = manager.delete_dataset(73)
        assert deleted is True

        # Verify it was deleted
        assert manager.get_dataset_info(73) is None

    def test_delete_dataset_not_found(self, manager):
        """Test deleting non-existent dataset."""
        deleted = manager.delete_dataset(99999)
        assert deleted is False

    def test_delete_all_datasets(self, manager):
        """Test deleting all datasets."""
        # Add multiple datasets
        for i in range(1, 4):
            manager.add_dataset(
                DatasetInfo(
                    id=i,
                    name=f"Dataset{i}",
                    url=f"http://test{i}.com",
                    n_instances=1000 * i,
                    n_features=10 * i,
                    task_type=TaskType.BINARY_CLASSIFICATION,
                    domain=Domain.FINANCE,
                )
            )

        # Verify they were added
        stats = manager.get_statistics()
        assert stats["total_datasets"] == 3

        # Delete all
        count = manager.delete_all_datasets()
        assert count == 3

        # Verify all deleted
        stats = manager.get_statistics()
        assert stats["total_datasets"] == 0


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


class TestIntegration:
    """Integration tests for the complete workflow."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir) / "test_integration.db"
        shutil.rmtree(temp_dir)

    def test_complete_workflow(self, temp_db_path):
        """Test a complete workflow of initializing, filtering, and loading."""
        # Create manager
        manager = UCIDatasetManager(db_path=temp_db_path)

        # Initialize with some datasets
        test_datasets = [
            DatasetInfo(
                id=1,
                name="Small Balanced",
                url="http://test1.com",
                n_instances=1000,
                n_features=10,
                task_type=TaskType.BINARY_CLASSIFICATION,
                domain=Domain.FINANCE,
                is_imbalanced=False,
                class_balance={"pos": 0.5, "neg": 0.5},
            ),
            DatasetInfo(
                id=2,
                name="Large Imbalanced",
                url="http://test2.com",
                n_instances=50000,
                n_features=100,
                task_type=TaskType.BINARY_CLASSIFICATION,
                domain=Domain.MEDICINE,
                is_imbalanced=True,
                imbalance_ratio=10.0,
                class_balance={"pos": 0.91, "neg": 0.09},
            ),
            DatasetInfo(
                id=3,
                name="Medium Regression",
                url="http://test3.com",
                n_instances=5000,
                n_features=50,
                task_type=TaskType.REGRESSION,
                domain=Domain.PHYSICS,
            ),
        ]

        for dataset in test_datasets:
            manager.add_dataset(dataset)

        # Test various filters

        # 1. Find all binary classification datasets
        binary_datasets = manager.filter_datasets(
            task_type=TaskType.BINARY_CLASSIFICATION
        )
        assert len(binary_datasets) == 2

        # 2. Find imbalanced datasets
        imbalanced = manager.filter_datasets(is_imbalanced=True)
        assert len(imbalanced) == 1
        assert imbalanced[0].name == "Large Imbalanced"

        # 3. Find datasets in specific size range
        medium_datasets = manager.filter_datasets(
            min_instances=1000, max_instances=10000
        )
        assert len(medium_datasets) == 2  # Small and Medium

        # 4. Complex query
        specific_datasets = manager.filter_datasets(
            task_type=TaskType.BINARY_CLASSIFICATION,
            min_instances=10000,
            is_imbalanced=True,
        )
        assert len(specific_datasets) == 1
        assert specific_datasets[0].name == "Large Imbalanced"

        # 5. Get statistics
        stats = manager.get_statistics()
        assert stats["total_datasets"] == 3
        assert stats["imbalanced_datasets"] == 1
        assert stats["by_task_type"]["binary_classification"] == 2
        assert stats["by_task_type"]["regression"] == 1

        # Clean up
        manager.delete_all_datasets()

    @pytest.mark.slow
    @pytest.mark.skipif(
        not os.environ.get("RUN_SLOW_TESTS", False),
        reason="Skipping slow test. Set RUN_SLOW_TESTS=1 to run.",
    )
    def test_real_dataset_loading(self, temp_db_path):
        """Test loading a real dataset from UCI repository."""
        # Create manager
        manager = UCIDatasetManager(db_path=temp_db_path)

        # Add Iris dataset info (small and reliable for testing)
        iris_info = DatasetInfo(
            id=53,  # Iris dataset ID
            name="Iris",
            url="https://archive.ics.uci.edu/dataset/53/iris",
            n_instances=150,
            n_features=4,
            task_type=TaskType.MULTICLASS_CLASSIFICATION,
            domain=Domain.BIOLOGY,
            description="Classic iris flower classification dataset",
        )
        manager.add_dataset(iris_info)

        try:
            # Load the dataset
            X, y, info = manager.load_dataset(53, return_metadata=True)

            # Verify basic properties
            assert X is not None
            assert y is not None
            assert info.name == "Iris"
            assert X.shape[0] == 150  # Should have 150 samples
            assert X.shape[1] == 4  # Should have 4 features

            # Check that data is numeric
            assert hasattr(X, "dtype") or hasattr(X, "dtypes")

            print(f"Successfully loaded {info.name} dataset")
            print(f"Shape: {X.shape}")
            print(
                f"Target classes: {np.unique(y) if hasattr(y, '__len__') else 'Unknown'}"
            )

        except ImportError:
            pytest.skip("ucimlrepo not installed")
        except Exception as e:
            # If network error or dataset not available, skip
            # sourcery skip: no-conditionals-in-tests
            if "Network" in str(e) or "HTTP" in str(e):
                pytest.skip(f"Network error or dataset unavailable: {e}")
            else:
                raise
        finally:
            # Clean up
            manager.delete_dataset(53)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
