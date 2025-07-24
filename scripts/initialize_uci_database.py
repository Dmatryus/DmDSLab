"""
Initialize UCI Database Script

This script initializes the UCI dataset database with predefined datasets
suitable for PU learning research. Run this script once to create and populate
the database that will be stored in the repository.

Usage:
    python initialize_uci_database.py

The database will be created at: data/uci_datasets.db

Author: DmDSLab Team
License: MIT
"""

import sys

from dmdslab.datasets.uci_dataset_manager import (
    DatasetInfo,
    Domain,
    TaskType,
    UCIDatasetManager,
)


def initialize_database():
    """Initialize the UCI dataset database with predefined datasets."""

    print("UCI Dataset Database Initialization")
    print("=" * 50)

    # Create manager
    manager = UCIDatasetManager()

    # Check if database already has data
    stats = manager.get_statistics()
    if stats["total_datasets"] > 0:
        response = input(
            f"\nDatabase already contains {stats['total_datasets']} datasets. "
            "Do you want to clear and reinitialize? (y/N): "
        )
        if response.lower() != "y":
            print("Initialization cancelled.")
            return
        else:
            count = manager.delete_all_datasets()
            print(f"Cleared {count} existing datasets.")

    print("\nInitializing database with predefined datasets...")

    # Define all datasets for PU learning research
    datasets = [
        # Balanced datasets
        DatasetInfo(
            id=73,
            name="Mushroom",
            url="https://archive.ics.uci.edu/dataset/73/mushroom",
            n_instances=8124,
            n_features=22,
            task_type=TaskType.BINARY_CLASSIFICATION,
            domain=Domain.BIOLOGY,
            class_balance={"edible": 0.52, "poisonous": 0.48},
            description="Classification of mushrooms as edible or poisonous",
            is_imbalanced=False,
        ),
        DatasetInfo(
            id=2,
            name="Adult",
            url="https://archive.ics.uci.edu/dataset/2/adult",
            n_instances=48842,
            n_features=14,
            task_type=TaskType.BINARY_CLASSIFICATION,
            domain=Domain.SOCIAL,
            class_balance={"<=50K": 0.76, ">50K": 0.24},
            description="Predict whether income exceeds $50K/yr",
            is_imbalanced=True,
            imbalance_ratio=3.17,
        ),
        DatasetInfo(
            id=94,
            name="Spambase",
            url="https://archive.ics.uci.edu/dataset/94/spambase",
            n_instances=4601,
            n_features=57,
            task_type=TaskType.BINARY_CLASSIFICATION,
            domain=Domain.CYBERSECURITY,
            class_balance={"non-spam": 0.60, "spam": 0.40},
            description="Email spam classification",
            is_imbalanced=False,
        ),
        # Moderately imbalanced datasets
        DatasetInfo(
            id=522,
            name="South German Credit",
            url="https://archive.ics.uci.edu/dataset/522/south+german+credit",
            n_instances=1000,
            n_features=21,
            task_type=TaskType.BINARY_CLASSIFICATION,
            domain=Domain.FINANCE,
            class_balance={"good": 0.70, "bad": 0.30},
            description="Credit risk assessment",
            is_imbalanced=True,
            imbalance_ratio=2.33,
        ),
        DatasetInfo(
            id=144,
            name="German Credit (Statlog)",
            url="https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data",
            n_instances=1000,
            n_features=20,
            task_type=TaskType.BINARY_CLASSIFICATION,
            domain=Domain.FINANCE,
            class_balance={"good": 0.70, "bad": 0.30},
            description="Binary credit risk classification",
            is_imbalanced=True,
            imbalance_ratio=2.33,
        ),
        DatasetInfo(
            id=563,
            name="Iranian Churn",
            url="https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset",
            n_instances=3150,
            n_features=13,
            task_type=TaskType.BINARY_CLASSIFICATION,
            domain=Domain.TELECOMMUNICATIONS,
            class_balance={"no_churn": 0.775, "churn": 0.225},
            description="Customer churn prediction",
            is_imbalanced=True,
            imbalance_ratio=3.44,
        ),
        DatasetInfo(
            id=350,
            name="Default of Credit Card Clients",
            url="https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
            n_instances=30000,
            n_features=23,
            task_type=TaskType.BINARY_CLASSIFICATION,
            domain=Domain.FINANCE,
            class_balance={"no_default": 0.7788, "default": 0.2212},
            description="Credit card default prediction",
            is_imbalanced=True,
            imbalance_ratio=3.52,
        ),
        DatasetInfo(
            id=159,
            name="MAGIC Gamma Telescope",
            url="https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope",
            n_instances=19020,
            n_features=10,
            task_type=TaskType.BINARY_CLASSIFICATION,
            domain=Domain.PHYSICS,
            class_balance={"background": 0.65, "signal": 0.35},
            description="Distinguish gamma rays from hadronic showers",
            is_imbalanced=True,
            imbalance_ratio=1.86,
        ),
        # Highly imbalanced datasets
        DatasetInfo(
            id=222,
            name="Bank Marketing",
            url="https://archive.ics.uci.edu/dataset/222/bank+marketing",
            n_instances=45211,
            n_features=16,
            task_type=TaskType.BINARY_CLASSIFICATION,
            domain=Domain.FINANCE,
            class_balance={"no": 0.887, "yes": 0.113},
            description="Bank term deposit subscription prediction",
            is_imbalanced=True,
            imbalance_ratio=7.85,
        ),
        DatasetInfo(
            id=179,
            name="SECOM",
            url="https://archive.ics.uci.edu/dataset/179/secom",
            n_instances=1567,
            n_features=591,
            task_type=TaskType.BINARY_CLASSIFICATION,
            domain=Domain.MANUFACTURING,
            class_balance={"pass": 0.934, "fail": 0.066},
            description="Semiconductor manufacturing quality control",
            is_imbalanced=True,
            imbalance_ratio=14.15,
            has_missing_values=True,
        ),
        DatasetInfo(
            id=572,
            name="Taiwanese Bankruptcy Prediction",
            url="https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction",
            n_instances=6819,
            n_features=95,
            task_type=TaskType.BINARY_CLASSIFICATION,
            domain=Domain.FINANCE,
            class_balance={"no_bankrupt": 0.968, "bankrupt": 0.032},
            description="Company bankruptcy prediction",
            is_imbalanced=True,
            imbalance_ratio=30.25,
        ),
        DatasetInfo(
            id=365,
            name="Polish Companies Bankruptcy",
            url="https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data",
            n_instances=10503,
            n_features=65,
            task_type=TaskType.BINARY_CLASSIFICATION,
            domain=Domain.FINANCE,
            class_balance={"no_bankrupt": 0.96, "bankrupt": 0.04},
            description="Bankruptcy prediction 1-5 years ahead",
            is_imbalanced=True,
            imbalance_ratio=24.0,
        ),
        DatasetInfo(
            id=372,
            name="HTRU2",
            url="https://archive.ics.uci.edu/dataset/372/htru2",
            n_instances=17898,
            n_features=8,
            task_type=TaskType.BINARY_CLASSIFICATION,
            domain=Domain.ASTRONOMY,
            class_balance={"no_pulsar": 0.91, "pulsar": 0.09},
            description="Pulsar star classification",
            is_imbalanced=True,
            imbalance_ratio=10.11,
        ),
        DatasetInfo(
            id=296,
            name="Diabetes 130-US Hospitals",
            url="https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008",
            n_instances=101766,
            n_features=47,
            task_type=TaskType.BINARY_CLASSIFICATION,
            domain=Domain.MEDICINE,
            class_balance={"no_readmission": 0.89, "readmission": 0.11},
            description="Early hospital readmission prediction",
            is_imbalanced=True,
            imbalance_ratio=8.09,
            has_missing_values=True,
        ),
        # Additional diverse datasets
        DatasetInfo(
            id=171,
            name="Madelon",
            url="https://archive.ics.uci.edu/dataset/171/madelon",
            n_instances=2600,
            n_features=500,
            task_type=TaskType.BINARY_CLASSIFICATION,
            domain=Domain.ARTIFICIAL,
            class_balance={"class_0": 0.50, "class_1": 0.50},
            description="Artificial dataset for feature selection",
            is_imbalanced=False,
        ),
        DatasetInfo(
            id=327,
            name="Phishing Websites",
            url="https://archive.ics.uci.edu/dataset/327/phishing+websites",
            n_instances=11055,
            n_features=30,
            task_type=TaskType.BINARY_CLASSIFICATION,
            domain=Domain.CYBERSECURITY,
            class_balance={"legitimate": 0.45, "phishing": 0.55},
            description="Phishing website detection",
            is_imbalanced=False,
        ),
        DatasetInfo(
            id=229,
            name="Skin Segmentation",
            url="https://archive.ics.uci.edu/dataset/229/skin+segmentation",
            n_instances=245057,
            n_features=3,
            task_type=TaskType.BINARY_CLASSIFICATION,
            domain=Domain.COMPUTER_VISION,
            class_balance={"non_skin": 0.79, "skin": 0.21},
            description="Skin/non-skin pixel classification",
            is_imbalanced=True,
            imbalance_ratio=3.76,
        ),
        DatasetInfo(
            id=264,
            name="EEG Eye State",
            url="https://archive.ics.uci.edu/dataset/264/eeg+eye+state",
            n_instances=14980,
            n_features=14,
            task_type=TaskType.BINARY_CLASSIFICATION,
            domain=Domain.NEUROSCIENCE,
            class_balance={"closed": 0.55, "open": 0.45},
            description="Eye state detection from EEG signals",
            is_imbalanced=False,
        ),
        DatasetInfo(
            id=468,
            name="Online Shoppers Purchasing Intention",
            url="https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset",
            n_instances=12330,
            n_features=17,
            task_type=TaskType.BINARY_CLASSIFICATION,
            domain=Domain.ECOMMERCE,
            class_balance={"no_revenue": 0.845, "revenue": 0.155},
            description="Purchase intention prediction",
            is_imbalanced=True,
            imbalance_ratio=5.45,
        ),
        DatasetInfo(
            id=357,
            name="Occupancy Detection",
            url="https://archive.ics.uci.edu/dataset/357/occupancy+detection",
            n_instances=20560,
            n_features=5,
            task_type=TaskType.BINARY_CLASSIFICATION,
            domain=Domain.SMART_BUILDINGS,
            class_balance={"not_occupied": 0.79, "occupied": 0.21},
            description="Room occupancy detection from sensor data",
            is_imbalanced=True,
            imbalance_ratio=3.76,
        ),
        DatasetInfo(
            id=199,
            name="MiniBooNE Particle Identification",
            url="https://archive.ics.uci.edu/dataset/199/miniboone+particle+identification",
            n_instances=130065,
            n_features=50,
            task_type=TaskType.BINARY_CLASSIFICATION,
            domain=Domain.PHYSICS,
            class_balance={"background": 0.72, "signal": 0.28},
            description="Neutrino particle identification",
            is_imbalanced=True,
            imbalance_ratio=2.57,
        ),
        DatasetInfo(
            id=224,
            name="Gas Sensor Array Drift",
            url="https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset",
            n_instances=13910,
            n_features=128,
            task_type=TaskType.MULTICLASS_CLASSIFICATION,
            domain=Domain.CHEMISTRY,
            description="Chemical gas classification with sensor drift",
            has_missing_values=False,
        ),
        DatasetInfo(
            id=601,
            name="AI4I 2020 Predictive Maintenance",
            url="https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset",
            n_instances=10000,
            n_features=14,
            task_type=TaskType.BINARY_CLASSIFICATION,
            domain=Domain.MANUFACTURING,
            class_balance={"no_failure": 0.966, "failure": 0.034},
            description="Industrial equipment failure prediction",
            is_imbalanced=True,
            imbalance_ratio=28.41,
        ),
        DatasetInfo(
            id=471,
            name="Electrical Grid Stability",
            url="https://archive.ics.uci.edu/dataset/471/electrical+grid+stability+simulated+data",
            n_instances=10000,
            n_features=12,
            task_type=TaskType.BINARY_CLASSIFICATION,
            domain=Domain.ENERGY,
            class_balance={"stable": 0.63, "unstable": 0.37},
            description="Power grid stability prediction",
            is_imbalanced=True,
            imbalance_ratio=1.70,
        ),
        DatasetInfo(
            id=459,
            name="Avila",
            url="https://archive.ics.uci.edu/dataset/459/avila",
            n_instances=20867,
            n_features=10,
            task_type=TaskType.MULTICLASS_CLASSIFICATION,
            domain=Domain.DOCUMENT_ANALYSIS,
            description="Medieval manuscript writer identification",
        ),
        DatasetInfo(
            id=507,
            name="Swarm Behaviour",
            url="https://archive.ics.uci.edu/dataset/507/swarm+behaviour",
            n_instances=24017,
            n_features=2400,
            task_type=TaskType.MULTICLASS_CLASSIFICATION,
            domain=Domain.ROBOTICS,
            description="Robot swarm behavior classification",
        ),
        DatasetInfo(
            id=464,
            name="Superconductivty",
            url="https://archive.ics.uci.edu/dataset/464/superconductivty+data",
            n_instances=21263,
            n_features=81,
            task_type=TaskType.REGRESSION,
            domain=Domain.MATERIALS,
            description="Superconducting material property prediction",
        ),
        DatasetInfo(
            id=102,
            name="Thyroid Disease",
            url="https://archive.ics.uci.edu/dataset/102/thyroid+disease",
            n_instances=9172,
            n_features=30,
            task_type=TaskType.MULTICLASS_CLASSIFICATION,
            domain=Domain.MEDICINE,
            description="Thyroid disease diagnosis",
            has_missing_values=True,
        ),
        DatasetInfo(
            id=379,
            name="Website Phishing",
            url="https://archive.ics.uci.edu/dataset/379/website+phishing",
            n_instances=1353,
            n_features=9,
            task_type=TaskType.MULTICLASS_CLASSIFICATION,
            domain=Domain.CYBERSECURITY,
            description="Phishing website detection (3 classes)",
        ),
        DatasetInfo(
            id=280,
            name="Higgs",
            url="https://archive.ics.uci.edu/dataset/280/higgs",
            n_instances=11000000,
            n_features=28,
            task_type=TaskType.BINARY_CLASSIFICATION,
            domain=Domain.PHYSICS,
            class_balance={"background": 0.53, "signal": 0.47},
            description="Higgs boson process classification",
            is_imbalanced=False,
        ),
    ]

    # Add all datasets to the database
    success_count = 0
    for i, dataset in enumerate(datasets, 1):
        try:
            manager.add_dataset(dataset)
            success_count += 1
            print(f"  [{i}/{len(datasets)}] Added: {dataset.name}")
        except Exception as e:
            print(f"  [{i}/{len(datasets)}] Failed to add {dataset.name}: {e}")

    print(f"\nSuccessfully added {success_count} out of {len(datasets)} datasets.")

    # Print summary statistics
    stats = manager.get_statistics()
    print("\nDatabase Statistics:")
    print(f"  Total datasets: {stats['total_datasets']}")
    print("  By task type:")
    for task_type, count in stats["by_task_type"].items():
        print(f"    - {task_type}: {count}")
    print("  By domain:")
    for domain, count in sorted(
        stats["by_domain"].items(), key=lambda x: x[1], reverse=True
    )[:5]:
        print(f"    - {domain}: {count}")
    print(f"  Imbalanced datasets: {stats['imbalanced_datasets']}")
    print(
        f"  Average dataset size: {stats['avg_instances']} instances, {stats['avg_features']} features"
    )

    print("\nDatabase initialization complete!")
    print(f"Database location: {manager.db_path}")


if __name__ == "__main__":
    try:
        initialize_database()
    except KeyboardInterrupt:
        print("\n\nInitialization interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during initialization: {e}")
        sys.exit(1)
