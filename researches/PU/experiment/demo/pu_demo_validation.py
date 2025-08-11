"""
Demo script for PU Learning Data Validation Module
Shows different validation scenarios including extreme imbalance
typical for real-world PU Learning applications
"""

import pandas as pd
import numpy as np
import warnings

from researches.PU.experiment.pu_data_validator import (
    DataValidator,
    load_and_validate_data,
)


def create_sample_dataset(
    n_samples=1500, n_features=10, positive_ratio=0.3, add_missing=False
):
    """Create a sample dataset for demonstration"""
    np.random.seed(42)

    # Generate features
    X = np.random.randn(n_samples, n_features)
    feature_names = [f"feature_{i}" for i in range(n_features)]

    # Generate target with specified ratio
    n_positive = int(n_samples * positive_ratio)
    y = np.zeros(n_samples)
    y[:n_positive] = 1
    np.random.shuffle(y)

    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    # Add some missing values if requested
    if add_missing:
        mask = np.random.random((n_samples, n_features)) < 0.05
        df.iloc[:, :n_features][mask] = np.nan

    return df


def demo_successful_validation():
    """Demo: Successful validation with clean data"""
    print("=" * 60)
    print("DEMO 1: Successful Validation")
    print("=" * 60)

    # Create a valid dataset
    df = create_sample_dataset(n_samples=2000, n_features=15, positive_ratio=0.4)

    print(f"Created dataset shape: {df.shape}")
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")

    # Validate
    validator = DataValidator()
    validated_df, metadata = validator.validate(df)

    print("\n✅ Validation successful!")
    print(metadata)

    return validated_df, metadata


def demo_with_imbalance():
    """Demo: Validation with extreme imbalance (natural for PU Learning)"""
    print("\n" + "=" * 60)
    print("DEMO 2: Validation with Extreme Imbalance")
    print("=" * 60)

    # Create severely imbalanced dataset
    df = create_sample_dataset(n_samples=1500, n_features=20, positive_ratio=0.005)

    print(f"Created imbalanced dataset (natural for PU Learning):")
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")
    print(f"Positive ratio: {df['target'].mean():.2%}")

    # Validate - should pass without warnings about imbalance
    validator = DataValidator()
    validated_df, metadata = validator.validate(df)

    print("\n✅ Validation successful!")
    print(f"  No warnings about imbalance - this is expected in PU Learning")
    print(f"  PU Learning methods are designed for such scenarios")
    print(f"  Positive examples: {metadata.n_positive}")
    print(f"  Negative examples: {metadata.n_negative}")

    return validated_df, metadata


def demo_missing_values():
    """Demo: Handling missing values"""
    print("\n" + "=" * 60)
    print("DEMO 3: Dataset with Missing Values")
    print("=" * 60)

    # Create dataset with missing values
    df = create_sample_dataset(
        n_samples=1200, n_features=12, positive_ratio=0.3, add_missing=True
    )

    missing_count = df.isna().sum().sum()
    print(f"Created dataset with {missing_count} missing values")
    print(f"Missing value columns: {df.columns[df.isna().any()].tolist()[:5]}...")

    # Validate
    validator = DataValidator()
    validated_df, metadata = validator.validate(df)

    print("\n✅ Validation successful!")
    print(f"  Has missing: {metadata.has_missing}")
    print(f"  Missing ratio: {metadata.missing_ratio:.2%}")

    return validated_df, metadata


def demo_custom_config():
    """Demo: Using custom configuration"""
    print("\n" + "=" * 60)
    print("DEMO 4: Custom Configuration")
    print("=" * 60)

    # Create dataset
    df = create_sample_dataset(n_samples=800, n_features=8)
    df.rename(columns={"target": "label"}, inplace=True)  # Different target name

    print(f"Created dataset with custom target column 'label'")
    print(f"Dataset shape: {df.shape}")

    # Custom config
    config = {
        "min_samples": 500,  # Lower minimum
        "max_samples": 50000,
        "target_column": "label",  # Custom target name
    }

    print(f"\nUsing custom config: {config}")

    # Validate with custom config
    validated_df, metadata = load_and_validate_data(df, config)

    print("\n✅ Validation successful with custom config!")
    print(f"  Target column identified: {metadata.target_column}")
    print(f"  Samples: {metadata.n_samples}")

    return validated_df, metadata


def demo_validation_failure():
    """Demo: Validation failure scenarios"""
    print("\n" + "=" * 60)
    print("DEMO 5: Validation Failures")
    print("=" * 60)

    # Scenario 1: Wrong target values
    print("\n1. Invalid target values:")
    df1 = create_sample_dataset(n_samples=1000)
    df1["target"] = np.random.randint(0, 5, 1000)  # Multi-class instead of binary

    validator = DataValidator()
    try:
        validator.validate(df1)
    except ValueError as e:
        print(f"❌ Validation failed: {str(e)[:100]}...")

    # Scenario 2: Too few samples (now generates warning, not error)
    print("\n2. Too few samples (warning, not error):")
    df2 = create_sample_dataset(n_samples=100)  # Below minimum

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            validated_df, metadata = validator.validate(df2)
            print(f"✅ Validation passed with {len(w)} warning(s)")
            for warning in w:
                print(f"  ⚠️ {warning.message}")
        except ValueError as e:
            print(f"❌ Validation failed: {str(e)[:100]}...")

    # Scenario 3: Missing target column
    print("\n3. No binary columns (can't identify target):")
    df3 = pd.DataFrame(
        np.random.randn(1000, 5), columns=[f"feat_{i}" for i in range(5)]
    )

    try:
        validator.validate(df3)
    except ValueError as e:
        print(f"❌ Validation failed: {str(e)[:100]}...")


def demo_feature_type_detection():
    """Demo: Feature type detection"""
    print("\n" + "=" * 60)
    print("DEMO 6: Feature Type Detection")
    print("=" * 60)

    # Create dataset with mixed types
    np.random.seed(42)
    n = 1500

    df = pd.DataFrame(
        {
            "numerical_1": np.random.randn(n),
            "numerical_2": np.random.uniform(0, 100, n),
            "categorical_low_card": np.random.choice(["A", "B", "C"], n),
            "categorical_as_int": np.random.choice([1, 2, 3, 4], n),
            "binary_feature": np.random.choice([0, 1], n),
            "text_feature": [f"text_{i%10}" for i in range(n)],
            "target": np.random.choice([0, 1], n, p=[0.6, 0.4]),
        }
    )

    print("Created dataset with mixed feature types")
    print(f"Columns: {df.columns.tolist()}")

    # Validate
    validator = DataValidator()
    validated_df, metadata = validator.validate(df)

    print("\n✅ Detected feature types:")
    for feat, ftype in metadata.feature_types.items():
        print(f"  - {feat}: {ftype}")

    return validated_df, metadata


def demo_small_dataset():
    """Demo: Working with small dataset (warnings only)"""
    print("\n" + "=" * 60)
    print("DEMO 7: Small Dataset Validation")
    print("=" * 60)

    # Create very small dataset
    df = create_sample_dataset(n_samples=50, n_features=3, positive_ratio=0.4)

    print(f"Created small dataset:")
    print(f"  Shape: {df.shape}")
    print(f"  Target distribution: {df['target'].value_counts().to_dict()}")

    # Validate - should pass with warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validator = DataValidator()
        validated_df, metadata = validator.validate(df)

        print("\n✅ Validation successful (despite small size)!")
        if w:
            print("\n⚠️ Warnings generated:")
            for warning in w:
                print(f"  - {warning.message}")

        print(f"\nMetadata:")
        print(f"  Samples: {metadata.n_samples}")
        print(f"  Features: {metadata.n_features}")
        print(f"  The experiment can proceed with this small dataset")

    return validated_df, metadata


def demo_extreme_imbalance():
    """Demo: Extreme imbalance - typical real-world PU scenario"""
    print("\n" + "=" * 60)
    print("DEMO 8: Real-world PU Scenario (Extreme Imbalance)")
    print("=" * 60)

    # Create extremely imbalanced dataset - typical for fraud detection, rare disease, etc.
    np.random.seed(42)
    n_samples = 10000
    n_positive = 20  # Only 0.2% positive - typical for real PU problems

    # Create features
    X = np.random.randn(n_samples, 25)
    feature_names = [f"feature_{i}" for i in range(25)]

    # Create highly imbalanced target
    y = np.zeros(n_samples)
    y[:n_positive] = 1
    np.random.shuffle(y)

    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    print(f"Created real-world PU scenario dataset:")
    print(f"  Total samples: {n_samples:,}")
    print(f"  Positive samples: {n_positive} ({n_positive/n_samples:.2%})")
    print(f"  This extreme imbalance is typical for:")
    print(f"    - Fraud detection (< 1% fraud)")
    print(f"    - Rare disease diagnosis (< 0.1%)")
    print(f"    - Click-through prediction (< 2%)")

    # Validate
    validator = DataValidator()
    validated_df, metadata = validator.validate(df)

    print("\n✅ Validation successful!")
    print(metadata)
    print("\n📌 Key insight: PU Learning methods are specifically")
    print("   designed to handle such extreme imbalance scenarios!")

    return validated_df, metadata


def main():
    """Run all demonstrations"""
    print("\n🚀 PU Learning Data Validation Module - Demonstration\n")

    # Run demos
    demos = [
        ("Successful Validation", demo_successful_validation),
        ("Extreme Imbalance (Natural for PU)", demo_with_imbalance),
        ("Missing Values Handling", demo_missing_values),
        ("Custom Configuration", demo_custom_config),
        ("Validation Failures", demo_validation_failure),
        ("Feature Type Detection", demo_feature_type_detection),
        ("Small Dataset Validation", demo_small_dataset),
        ("Real-world PU Scenario", demo_extreme_imbalance),
    ]

    results = {}
    for name, demo_func in demos:
        try:
            if result := demo_func():
                results[name] = result
        except Exception as e:
            print(f"\nError in {name}: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully validated {len(results)} datasets")
    print("✅ Demonstrated validation failures and warnings")
    print("✅ Showed feature type detection")
    print("✅ Tested custom configurations")
    print("✅ Validated small datasets with warnings only")
    print("✅ Handled extreme class imbalance (natural for PU Learning)")
    print("✅ Demonstrated real-world PU scenarios with < 1% positive rate")

    return results


if __name__ == "__main__":
    results = main()
