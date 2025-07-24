"""
UCI Dataset Manager for DmDSLab

This module provides a convenient interface for working with UCI Machine Learning Repository datasets.
It includes a local database of dataset metadata and allows filtering and loading datasets based on various criteria.

Author: Dmatryus Detry
License: Apache 2.0
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from functools import lru_cache

try:
    from ucimlrepo import fetch_ucirepo
except ImportError as e:
    raise ImportError(
        "ucimlrepo package is required. Install it with: pip install ucimlrepo"
    ) from e


# Configure logging
logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of machine learning tasks."""

    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"


class Domain(Enum):
    """Dataset domains/application areas."""

    BIOLOGY = "biology"
    FINANCE = "finance"
    MEDICINE = "medicine"
    PHYSICS = "physics"
    SOCIAL = "social"
    CYBERSECURITY = "cybersecurity"
    TELECOMMUNICATIONS = "telecommunications"
    ASTRONOMY = "astronomy"
    MANUFACTURING = "manufacturing"
    ENERGY = "energy"
    COMPUTER_VISION = "computer_vision"
    NEUROSCIENCE = "neuroscience"
    ECOMMERCE = "ecommerce"
    SMART_BUILDINGS = "smart_buildings"
    CHEMISTRY = "chemistry"
    ROBOTICS = "robotics"
    MATERIALS = "materials"
    DOCUMENT_ANALYSIS = "document_analysis"
    ARTIFICIAL = "artificial"


@dataclass
class DatasetInfo:
    """
    Container for dataset metadata.

    Attributes:
        id: UCI repository ID
        name: Dataset name
        url: URL to dataset page
        n_instances: Number of instances/examples
        n_features: Number of features
        task_type: Type of ML task
        domain: Application domain
        class_balance: Dictionary with class distribution (for classification tasks)
        description: Brief description of the dataset
        year: Year of dataset creation/publication
        has_missing_values: Whether dataset contains missing values
        is_imbalanced: Whether dataset is imbalanced (for classification)
        imbalance_ratio: Ratio of majority to minority class
    """

    id: int
    name: str
    url: str
    n_instances: int
    n_features: int
    task_type: TaskType
    domain: Domain
    class_balance: Optional[Dict[str, float]] = None
    description: str = ""
    year: Optional[int] = None
    has_missing_values: bool = False
    is_imbalanced: bool = False
    imbalance_ratio: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data["task_type"] = self.task_type.value
        data["domain"] = self.domain.value
        if self.class_balance:
            data["class_balance"] = json.dumps(self.class_balance)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetInfo":
        """Create instance from dictionary."""
        data = data.copy()
        data["task_type"] = TaskType(data["task_type"])
        data["domain"] = Domain(data["domain"])
        if data.get("class_balance") and isinstance(data["class_balance"], str):
            data["class_balance"] = json.loads(data["class_balance"])
        return cls(**data)


class UCIDatasetManager:
    """
    Manager for UCI Machine Learning Repository datasets.

    This class provides functionality to:
    - Store and retrieve dataset metadata in a local SQLite database
    - Filter datasets by various criteria
    - Load datasets using ucimlrepo
    - Cache loaded datasets for efficiency
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        """
        Initialize the dataset manager.

        Args:
            db_path: Path to SQLite database file. If None, uses default location.
        """
        if db_path is None:
            db_path = Path(__file__).parent / "db" / "uci_datasets.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create database connection
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    url TEXT NOT NULL,
                    n_instances INTEGER NOT NULL,
                    n_features INTEGER NOT NULL,
                    task_type TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    class_balance TEXT,
                    description TEXT,
                    year INTEGER,
                    has_missing_values BOOLEAN,
                    is_imbalanced BOOLEAN,
                    imbalance_ratio REAL
                )
            """
            )
            conn.commit()

    def add_dataset(self, dataset_info: DatasetInfo) -> None:
        """
        Add a dataset to the database.

        Args:
            dataset_info: Dataset metadata
        """
        with sqlite3.connect(self.db_path) as conn:
            data = dataset_info.to_dict()
            columns = ", ".join(data.keys())
            placeholders = ", ".join(["?" for _ in data])
            query = (
                f"INSERT OR REPLACE INTO datasets ({columns}) VALUES ({placeholders})"
            )
            conn.execute(query, list(data.values()))
            conn.commit()
        logger.info(f"Added dataset: {dataset_info.name} (ID: {dataset_info.id})")

    def get_dataset_info(self, dataset_id: int) -> Optional[DatasetInfo]:
        """
        Get dataset information by ID.

        Args:
            dataset_id: UCI repository dataset ID

        Returns:
            DatasetInfo object or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
            row = cursor.fetchone()

        return DatasetInfo.from_dict(dict(row)) if row else None

    def filter_datasets(
        self,
        task_type: Optional[TaskType] = None,
        domain: Optional[Domain] = None,
        min_instances: Optional[int] = None,
        max_instances: Optional[int] = None,
        min_features: Optional[int] = None,
        max_features: Optional[int] = None,
        is_imbalanced: Optional[bool] = None,
        max_imbalance_ratio: Optional[float] = None,
        has_missing_values: Optional[bool] = None,
    ) -> List[DatasetInfo]:
        """
        Filter datasets based on criteria.

        Args:
            task_type: Type of ML task
            domain: Application domain
            min_instances: Minimum number of instances
            max_instances: Maximum number of instances
            min_features: Minimum number of features
            max_features: Maximum number of features
            is_imbalanced: Whether to filter for imbalanced datasets
            max_imbalance_ratio: Maximum imbalance ratio (for imbalanced datasets)
            has_missing_values: Whether to filter for datasets with missing values

        Returns:
            List of DatasetInfo objects matching the criteria
        """
        conditions = []
        params: list[Any] = []

        if task_type:
            conditions.append("task_type = ?")
            params.append(task_type.value)

        if domain:
            conditions.append("domain = ?")
            params.append(domain.value)

        if min_instances:
            conditions.append("n_instances >= ?")
            params.append(min_instances)

        if max_instances:
            conditions.append("n_instances <= ?")
            params.append(max_instances)

        if min_features:
            conditions.append("n_features >= ?")
            params.append(min_features)

        if max_features:
            conditions.append("n_features <= ?")
            params.append(max_features)

        if is_imbalanced is not None:
            conditions.append("is_imbalanced = ?")
            params.append(int(is_imbalanced))

        if max_imbalance_ratio:
            conditions.append("imbalance_ratio <= ?")
            params.append(max_imbalance_ratio)

        if has_missing_values is not None:
            conditions.append("has_missing_values = ?")
            params.append(int(has_missing_values))

        query = "SELECT * FROM datasets"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        return [DatasetInfo.from_dict(dict(row)) for row in rows]

    @lru_cache(maxsize=32)
    def load_dataset(
        self, dataset_id: int, return_metadata: bool = False
    ) -> Union[Tuple[Any, Any], Tuple[Any, Any, DatasetInfo]]:
        """
        Load a dataset from UCI repository.

        Args:
            dataset_id: UCI repository dataset ID
            return_metadata: Whether to return dataset metadata along with data

        Returns:
            If return_metadata is False: (X, y) tuple
            If return_metadata is True: (X, y, dataset_info) tuple

        Raises:
            ValueError: If dataset not found in database
            Exception: If dataset loading fails
        """
        dataset_info = self.get_dataset_info(dataset_id)
        if not dataset_info:
            raise ValueError(f"Dataset with ID {dataset_id} not found in database")

        logger.info(f"Loading dataset: {dataset_info.name} (ID: {dataset_id})")

        try:
            dataset = fetch_ucirepo(id=dataset_id)
            X = dataset.data.features
            y = dataset.data.targets

            # Handle multi-column targets
            if y is not None and len(y.shape) > 1 and y.shape[1] == 1:
                y = y.iloc[:, 0] if hasattr(y, "iloc") else y[:, 0]

            return (X, y, dataset_info) if return_metadata else (X, y)
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_id}: {str(e)}")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about datasets in the database.

        Returns:
            Dictionary with statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            # Total datasets
            cursor = conn.execute("SELECT COUNT(*) FROM datasets")
            stats = {"total_datasets": cursor.fetchone()[0]}
            # By task type
            cursor = conn.execute(
                "SELECT task_type, COUNT(*) FROM datasets GROUP BY task_type"
            )
            stats["by_task_type"] = dict(cursor.fetchall())

            # By domain
            cursor = conn.execute(
                "SELECT domain, COUNT(*) FROM datasets GROUP BY domain"
            )
            stats["by_domain"] = dict(cursor.fetchall())

            # Imbalanced datasets
            cursor = conn.execute(
                "SELECT COUNT(*) FROM datasets WHERE is_imbalanced = 1"
            )
            stats["imbalanced_datasets"] = cursor.fetchone()[0]

            # Average dataset size
            cursor = conn.execute(
                "SELECT AVG(n_instances), AVG(n_features) FROM datasets"
            )
            avg_instances, avg_features = cursor.fetchone()
            stats["avg_instances"] = round(avg_instances) if avg_instances else 0
            stats["avg_features"] = round(avg_features) if avg_features else 0

        return stats

    def delete_dataset(self, dataset_id: int) -> bool:
        """
        Delete a dataset from the database.

        Args:
            dataset_id: UCI repository dataset ID

        Returns:
            True if dataset was deleted, False if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
            deleted = cursor.rowcount > 0
            conn.commit()

        if deleted:
            logger.info(f"Deleted dataset with ID: {dataset_id}")
        else:
            logger.warning(f"Dataset with ID {dataset_id} not found")

        return deleted

    def delete_all_datasets(self) -> int:
        """
        Delete all datasets from the database.

        Returns:
            Number of datasets deleted
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM datasets")
            count = cursor.rowcount
            conn.commit()

        logger.info(f"Deleted all {count} datasets from database")
        return count


# Example usage and utility functions
def print_dataset_summary(datasets: List[DatasetInfo]) -> None:
    """Print a formatted summary of datasets."""
    print(f"\nFound {len(datasets)} datasets:")
    print("-" * 80)
    print(
        f"{'Name':<30} {'Instances':>10} {'Features':>10} {'Domain':<20} {'Imbalanced':<10}"
    )
    print("-" * 80)

    for ds in datasets:
        imbalanced = "Yes" if ds.is_imbalanced else "No"
        print(
            f"{ds.name:<30} {ds.n_instances:>10,} {ds.n_features:>10} "
            f"{ds.domain.value:<20} {imbalanced:<10}"
        )


if __name__ == "__main__":
    # Example: Use the manager with existing database
    manager = UCIDatasetManager()

    # Get statistics
    stats = manager.get_statistics()
    print("Database Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Example: Find imbalanced binary classification datasets
    print("\n" + "=" * 80)
    print("Imbalanced Binary Classification Datasets (1K-50K instances):")
    datasets = manager.filter_datasets(
        task_type=TaskType.BINARY_CLASSIFICATION,
        is_imbalanced=True,
        min_instances=1000,
        max_instances=50000,
    )
    print_dataset_summary(datasets)

    # Example: Load a specific dataset (if database is populated)
    if stats["total_datasets"] > 0:
        print("\n" + "=" * 80)
        print("Loading first available dataset...")
        first_dataset = datasets[0] if datasets else None
        if first_dataset:
            X, y, info = manager.load_dataset(first_dataset.id, return_metadata=True)
            print(f"Dataset: {info.name}")
            print(f"Shape: {X.shape}")
            print(f"Class distribution: {info.class_balance}")
    else:
        print("\nNo datasets found. Run initialize_uci_database.py first.")
