"""
UCI Dataset Manager for DmDSLab (v2)

This module provides a convenient interface for working with UCI Machine Learning Repository datasets.
It uses ModelData and DataSplit containers for better integration with the DmDSLab ecosystem.

Author: Dmatryus Detry
License: Apache 2.0
"""

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from ucimlrepo import fetch_ucirepo
except ImportError:
    raise ImportError(
        "ucimlrepo package is required. Install it with: pip install ucimlrepo"
    ) from None

# Import our data structures
from .ml_data_container import (
    DataInfo,
    DataSplit,
    ModelData,
    create_data_split,
    create_kfold_data,
)

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
        feature_names: List of feature names (if available)
        target_name: Name of the target variable (if available)
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
    feature_names: Optional[List[str]] = None
    target_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data["task_type"] = self.task_type.value
        data["domain"] = self.domain.value
        if self.class_balance:
            data["class_balance"] = json.dumps(self.class_balance)
        if self.feature_names:
            data["feature_names"] = json.dumps(self.feature_names)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetInfo":
        """Create instance from dictionary."""
        data = data.copy()
        data["task_type"] = TaskType(data["task_type"])
        data["domain"] = Domain(data["domain"])
        if data.get("class_balance") and isinstance(data["class_balance"], str):
            data["class_balance"] = json.loads(data["class_balance"])
        if data.get("feature_names") and isinstance(data["feature_names"], str):
            data["feature_names"] = json.loads(data["feature_names"])
        return cls(**data)

    def to_data_info(self) -> DataInfo:
        """Convert to DataInfo for use with ModelData."""
        metadata = {
            "uci_id": self.id,
            "task_type": self.task_type.value,
            "domain": self.domain.value,
            "n_instances": self.n_instances,
            "n_features": self.n_features,
            "has_missing_values": self.has_missing_values,
        }

        if self.class_balance:
            metadata["class_balance"] = self.class_balance
        if self.is_imbalanced:
            metadata["is_imbalanced"] = self.is_imbalanced
            metadata["imbalance_ratio"] = self.imbalance_ratio
        if self.year:
            metadata["year"] = self.year
        if self.target_name:
            metadata["target_name"] = self.target_name

        return DataInfo(
            name=self.name,
            description=self.description or f"UCI ML Repository: {self.name}",
            source=self.url,
            version="1.0.0",
            metadata=metadata,
        )


class UCIDatasetManager:
    """
    Manager for UCI Machine Learning Repository datasets.

    This class provides functionality to:
    - Store and retrieve dataset metadata in a local SQLite database
    - Filter datasets by various criteria
    - Load datasets as ModelData objects
    - Create DataSplit objects with train/validation/test splits
    - Cache loaded datasets for efficiency
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        """
        Initialize the dataset manager.

        Args:
            db_path: Path to SQLite database file. If None, uses default location.
                    Can be ":memory:" for in-memory database (useful for testing).
        """
        if db_path is None:
            self.db_path = Path(__file__).parent / "db" / "uci_datasets.db"

        else:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create database connection
        self._init_db()

    def _get_db_path(self) -> str:
        """Get database path as string, handling special cases like :memory:."""
        return str(self.db_path) if self.db_path != ":memory:" else ":memory:"

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self._get_db_path()) as conn:
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
                    imbalance_ratio REAL,
                    feature_names TEXT,
                    target_name TEXT
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
        with sqlite3.connect(self._get_db_path()) as conn:
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
        with sqlite3.connect(self._get_db_path()) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
            row = cursor.fetchone()

        if row:
            return DatasetInfo.from_dict(dict(row))
        return None

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
        params = []

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

        with sqlite3.connect(self._get_db_path()) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        return [DatasetInfo.from_dict(dict(row)) for row in rows]

    def load_dataset(self, dataset_id: int) -> ModelData:
        """
        Load a dataset from UCI repository as ModelData.

        Args:
            dataset_id: UCI repository dataset ID

        Returns:
            ModelData object containing the dataset

        Raises:
            ValueError: If dataset not found in database
            Exception: If dataset loading fails
        """
        dataset_info = self.get_dataset_info(dataset_id)
        if not dataset_info:
            raise ValueError(f"Dataset with ID {dataset_id} not found in database")

        logger.info(f"Loading dataset: {dataset_info.name} (ID: {dataset_id})")

        try:
            # Fetch dataset from UCI repository
            dataset = fetch_ucirepo(id=dataset_id)
            X = dataset.data.features
            y = dataset.data.targets

            # Convert target to 1D array if necessary
            if y is not None and len(y.shape) > 1 and y.shape[1] == 1:
                y = y.ravel()

            # Try to get feature names from the fetched dataset
            feature_names = dataset_info.feature_names
            if feature_names is None:
                if (
                    hasattr(dataset.data, "feature_names")
                    and dataset.data.feature_names is not None
                ):
                    try:
                        feature_names = list(dataset.data.feature_names)
                    except (TypeError, AttributeError):
                        feature_names = None
                elif hasattr(X, "columns"):
                    try:
                        feature_names = list(X.columns)
                    except (TypeError, AttributeError):
                        feature_names = None

            # Create ModelData with metadata
            return ModelData(
                features=X,
                target=y,
                feature_names=feature_names,
                info=dataset_info.to_data_info(),
            )

        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_id}: {str(e)}")
            raise

    def load_dataset_split(
        self,
        dataset_id: int,
        test_size: Optional[float] = 0.2,
        validation_size: Optional[float] = None,
        random_state: Optional[int] = None,
        stratify: bool = None,
    ) -> DataSplit:
        """
        Load a dataset and create train/validation/test splits.

        Args:
            dataset_id: UCI repository dataset ID
            test_size: Proportion of data for test set (default: 0.2)
            validation_size: Proportion of data for validation set (default: None)
            random_state: Random seed for reproducibility
            stratify: Whether to stratify splits. If None, auto-detect based on task type

        Returns:
            DataSplit object with train/validation/test sets

        Raises:
            ValueError: If dataset not found or splitting parameters invalid
        """
        # Load the dataset
        model_data = self.load_dataset(dataset_id)

        # Auto-detect stratification for classification tasks
        if stratify is None and model_data.info:
            task_type = model_data.info.metadata.get("task_type")
            stratify = task_type in [
                "binary_classification",
                "multiclass_classification",
            ]

        # Create the split
        split = create_data_split(
            features=model_data.features,
            y=model_data.target,
            test_size=test_size,
            validation_size=validation_size,
            random_state=random_state,
            stratify=stratify,
        )

        # Add dataset info to split metadata
        split.split_info["dataset_name"] = model_data.info.name
        split.split_info["dataset_id"] = dataset_id

        return split

    def load_dataset_kfold(
        self,
        dataset_id: int,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = None,
    ) -> List[DataSplit]:
        """
        Load a dataset and create k-fold cross-validation splits.

        Args:
            dataset_id: UCI repository dataset ID
            n_splits: Number of folds (default: 5)
            shuffle: Whether to shuffle data before splitting
            random_state: Random seed for reproducibility

        Returns:
            List of DataSplit objects, one for each fold

        Raises:
            ValueError: If dataset not found
        """
        # Load the dataset
        model_data = self.load_dataset(dataset_id)

        # Create k-fold splits
        splits = create_kfold_data(
            features=model_data.features,
            y=model_data.target,
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )

        # Add dataset info to each split
        for split in splits:
            split.split_info["dataset_name"] = model_data.info.name
            split.split_info["dataset_id"] = dataset_id

        return splits

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about datasets in the database.

        Returns:
            Dictionary with statistics
        """
        with sqlite3.connect(self._get_db_path()) as conn:
            stats = {}

            # Total datasets
            cursor = conn.execute("SELECT COUNT(*) FROM datasets")
            stats["total_datasets"] = cursor.fetchone()[0]

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
        with sqlite3.connect(self._get_db_path()) as conn:
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
        with sqlite3.connect(self._get_db_path()) as conn:
            cursor = conn.execute("DELETE FROM datasets")
            count = cursor.rowcount
            conn.commit()

        logger.info(f"Deleted all {count} datasets from database")
        return count

    def close(self):
        """
        Close any open database connections and clear cache.

        This method is useful for cleanup, especially on Windows where
        file handles may remain open.
        """

        # Force garbage collection to close any lingering connections
        import gc

        gc.collect()


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

    # Example: Load a specific dataset as ModelData
    if stats["total_datasets"] > 0 and datasets:
        print("\n" + "=" * 80)
        print("Loading first available dataset as ModelData...")
        first_dataset = datasets[0]

        # Load as ModelData
        model_data = manager.load_dataset(first_dataset.id)
        print(f"\nDataset: {model_data.info.name}")
        print(f"Shape: {model_data.shape}")
        print(f"Features: {model_data.n_features}")
        print(f"Samples: {model_data.n_samples}")

        # Create train/test split
        print("\nCreating train/test split...")
        split = manager.load_dataset_split(
            first_dataset.id, test_size=0.2, random_state=42
        )
        print(f"Train size: {split.train.n_samples}")
        print(f"Test size: {split.test.n_samples}")
        print(f"Split ratios: {split.get_split_ratios()}")

        # Create k-fold splits
        print("\nCreating 5-fold cross-validation splits...")
        kfold_splits = manager.load_dataset_kfold(
            first_dataset.id, n_splits=5, random_state=42
        )
        print(f"Number of folds: {len(kfold_splits)}")
        print(
            f"First fold - Train: {kfold_splits[0].train.n_samples}, "
            f"Val: {kfold_splits[0].validation.n_samples}"
        )
    else:
        print("\nNo datasets found. Run initialize_uci_database.py first.")
