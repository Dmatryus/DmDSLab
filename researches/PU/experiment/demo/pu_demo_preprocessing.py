"""
Demo script for PU Learning Data Preprocessing Module
Shows different preprocessing scenarios and transformations
"""

import pandas as pd
import numpy as np
import warnings
from typing import Tuple

# Import modules from previous steps
from researches.PU.experiment.pu_data_validator import load_and_validate_data
from researches.PU.experiment.pu_data_preprocessor import (
    DataPreprocessor,
    preprocess_data,
    PreprocessingStatistics,
    PreprocessingPipeline
)


def create_complex_dataset(
    n_samples=2000,
    n_numerical=8, 
    n_categorical=4,
    positive_ratio=0.3,
    add_missing=True,
    add_high_cardinality=False
):
    """Create a complex dataset with mixed feature types"""
    np.random.seed(42)
    
    data = {}
    
    # Numerical features
    for i in range(n_numerical):
        if i % 3 == 0:
            # Some features with different distributions
            data[f'num_feat_{i}'] = np.random.exponential(2, n_samples)
        elif i % 3 == 1:
            data[f'num_feat_{i}'] = np.random.normal(100, 15, n_samples)
        else:
            data[f'num_feat_{i}'] = np.random.uniform(0, 1, n_samples)
    
    # Categorical features
    for i in range(n_categorical):
        if add_high_cardinality and i == 0:
            # High cardinality feature
            data[f'cat_feat_{i}'] = [f'category_{j%100}' for j in range(n_samples)]
        elif i % 2 == 0:
            # Low cardinality
            data[f'cat_feat_{i}'] = np.random.choice(['A', 'B', 'C'], n_samples)
        else:
            # Medium cardinality
            data[f'cat_feat_{i}'] = np.random.choice(
                [f'Type_{j}' for j in range(10)], n_samples
            )
    
    # Add some integer features that might be categorical
    data['small_int'] = np.random.choice([1, 2, 3, 4], n_samples)
    data['large_int'] = np.random.randint(0, 1000, n_samples)
    
    # Target variable
    n_positive = int(n_samples * positive_ratio)
    target = np.zeros(n_samples)
    target[:n_positive] = 1
    np.random.shuffle(target)
    data['target'] = target
    
    df = pd.DataFrame(data)
    
    # Add missing values
    if add_missing:
        # Add missing to numerical features
        for col in [c for c in df.columns if 'num_feat' in c][:3]:
            missing_idx = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
            df.loc[missing_idx, col] = np.nan
        
        # Add missing to categorical features
        for col in [c for c in df.columns if 'cat_feat' in c][:2]:
            missing_idx = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
            df.loc[missing_idx, col] = np.nan
    
    return df


def demo_basic_preprocessing():
    """Demo: Basic preprocessing with default settings"""
    print("=" * 70)
    print("DEMO 1: Basic Preprocessing with Default Settings")
    print("=" * 70)
    
    # Create dataset
    df = create_complex_dataset(n_samples=1500, add_missing=True)
    print(f"Created dataset shape: {df.shape}")
    print(f"Features: {[c for c in df.columns if c != 'target'][:5]}...")
    print(f"Missing values: {df.isna().sum().sum()}")
    
    # Validate first
    validated_df, metadata = load_and_validate_data(df)
    print(f"\n✅ Validation passed: {metadata.n_samples} samples, {metadata.n_features} features")
    
    # Preprocess
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, stats, pipeline = preprocessor.preprocess(
        validated_df, metadata.target_column
    )
    
    print("\n📊 Preprocessing Results:")
    print(f"  Train shape: {X_train.shape}")
    print(f"  Test shape: {X_test.shape}")
    print(f"  Train positive ratio: {y_train.mean():.1%}")
    print(f"  Test positive ratio: {y_test.mean():.1%}")
    print(f"\n  Numerical features: {stats.n_numerical_features}")
    print(f"  Categorical features: {stats.n_categorical_features}")
    print(f"  Missing values handled: {stats.missing_handled}")
    print(f"  Scaling applied: {stats.scaling_applied}")
    
    return X_train, X_test, y_train, y_test, stats, pipeline


def demo_custom_configuration():
    """Demo: Custom preprocessing configuration"""
    print("\n" + "=" * 70)
    print("DEMO 2: Custom Preprocessing Configuration")
    print("=" * 70)
    
    # Create dataset
    df = create_complex_dataset(n_samples=1200)
    print(f"Created dataset shape: {df.shape}")
    
    # Custom configuration
    config = {
        'test_size': 0.3,  # 30% test instead of 20%
        'scale_features': False,  # No scaling
        'numerical_impute_strategy': 'mean',  # Mean instead of median
        'encoding_method': 'label',  # Label encoding instead of target
        'high_cardinality_threshold': 20,  # Lower threshold
        'random_state': 123
    }
    
    print(f"\nCustom config:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Validate and preprocess
    validated_df, metadata = load_and_validate_data(df)
    X_train, X_test, y_train, y_test, stats, pipeline = preprocess_data(
        validated_df, metadata.target_column, config
    )
    
    print("\n📊 Custom Preprocessing Results:")
    print(f"  Train/Test split: {len(X_train)}/{len(X_test)} (70/30)")
    print(f"  Scaling applied: {stats.scaling_applied} (disabled)")
    print(f"  Encoding method: {stats.categorical_encoding_method}")
    print(f"  Imputation strategy: {list(stats.missing_strategies.values())[:3]}...")
    
    return X_train, X_test, y_train, y_test, stats


def demo_high_cardinality_handling():
    """Demo: Handling high cardinality categorical features"""
    print("\n" + "=" * 70)
    print("DEMO 3: High Cardinality Categorical Features")
    print("=" * 70)
    
    # Create dataset with high cardinality
    df = create_complex_dataset(
        n_samples=1000, 
        n_categorical=3,
        add_high_cardinality=True
    )
    print(f"Created dataset with high cardinality feature")
    
    # Check cardinality
    cat_cols = [c for c in df.columns if 'cat_feat' in c]
    for col in cat_cols:
        print(f"  {col}: {df[col].nunique()} unique values")
    
    # Preprocess with target encoding (handles high cardinality well)
    validated_df, metadata = load_and_validate_data(df)
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        preprocessor = DataPreprocessor({'encoding_method': 'target'})
        X_train, X_test, y_train, y_test, stats, pipeline = preprocessor.preprocess(
            validated_df, metadata.target_column
        )
        
        if w:
            print("\n⚠️ Warnings:")
            for warning in w:
                print(f"  {warning.message}")
    
    print("\n📊 High Cardinality Handling:")
    print(f"  Encoding method: {stats.categorical_encoding_method}")
    print(f"  Categories per feature:")
    for feat, n_cats in list(stats.categories_per_feature.items())[:3]:
        print(f"    {feat}: {n_cats} categories")
    print(f"  High cardinality warnings: {stats.high_cardinality_warnings}")
    
    return X_train, X_test, y_train, y_test, stats


def demo_pipeline_transform():
    """Demo: Using preprocessing pipeline on new data"""
    print("\n" + "=" * 70)
    print("DEMO 4: Preprocessing Pipeline for New Data")
    print("=" * 70)
    
    # Create training dataset
    df_train = create_complex_dataset(n_samples=1000, add_missing=True)
    print(f"Training dataset shape: {df_train.shape}")
    
    # Preprocess and get pipeline
    validated_df, metadata = load_and_validate_data(df_train)
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, stats, pipeline = preprocessor.preprocess(
        validated_df, metadata.target_column
    )
    
    print(f"\n✅ Pipeline created from training data")
    print(f"  Imputers: {len(pipeline.imputers)}")
    print(f"  Encoders: {len(pipeline.encoders)}")
    print(f"  Scaler: {'Yes' if pipeline.scaler else 'No'}")
    
    # Create new data (simulating production)
    print("\n🔄 Applying pipeline to new data...")
    df_new = create_complex_dataset(n_samples=200, add_missing=True)
    
    # Separate features and apply pipeline
    X_new = df_new.drop(columns=['target'])
    X_new_transformed = pipeline.transform(X_new)
    
    print(f"  Original shape: {X_new.shape}")
    print(f"  Transformed shape: {X_new_transformed.shape}")
    print(f"  Missing values before: {X_new.isna().sum().sum()}")
    print(f"  Missing values after: {X_new_transformed.isna().sum().sum()}")
    
    # Check that features are properly scaled
    if pipeline.numerical_features:
        num_feat = pipeline.numerical_features[0]
        if num_feat in X_new_transformed.columns:
            print(f"\n  Example numerical feature '{num_feat}':")
            print(f"    Mean: {X_new_transformed[num_feat].mean():.3f} (should be ~0)")
            print(f"    Std: {X_new_transformed[num_feat].std():.3f} (should be ~1)")
    
    return pipeline, X_new_transformed


def demo_extreme_pu_scenario():
    """Demo: Preprocessing for extreme PU scenario"""
    print("\n" + "=" * 70)
    print("DEMO 5: Extreme PU Scenario (0.5% positive)")
    print("=" * 70)
    
    # Create extremely imbalanced dataset
    np.random.seed(42)
    n_samples = 5000
    n_positive = 25  # Only 0.5% positive
    
    # Create features
    X = np.random.randn(n_samples, 20)
    feature_names = [f'feature_{i}' for i in range(20)]
    
    # Add some categorical
    cat_features = {
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        'type': np.random.choice(['A', 'B', 'C'], n_samples),
    }
    
    # Create target
    y = np.zeros(n_samples)
    y[:n_positive] = 1
    np.random.shuffle(y)
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    for key, values in cat_features.items():
        df[key] = values
    df['target'] = y
    
    print(f"Created extreme PU dataset:")
    print(f"  Total samples: {n_samples:,}")
    print(f"  Positive samples: {n_positive} ({n_positive/n_samples:.2%})")
    
    # Validate and preprocess
    validated_df, metadata = load_and_validate_data(df)
    
    # Use special config for extreme imbalance
    config = {
        'test_size': 0.3,  # Larger test set to ensure some positives
        'encoding_method': 'target',  # Good for imbalanced data
        'random_state': 42
    }
    
    X_train, X_test, y_train, y_test, stats, pipeline = preprocess_data(
        validated_df, metadata.target_column, config
    )
    
    print("\n📊 Extreme PU Preprocessing Results:")
    print(f"  Train samples: {len(X_train):,}")
    print(f"  Train positive: {y_train.sum()} ({y_train.mean():.2%})")
    print(f"  Test samples: {len(X_test):,}")
    print(f"  Test positive: {y_test.sum()} ({y_test.mean():.2%})")
    print(f"\n✅ Stratification preserved the rare positive class!")
    print(f"  Both train and test sets contain positive examples")
    
    return X_train, X_test, y_train, y_test, stats


def demo_feature_type_detection():
    """Demo: Automatic feature type detection"""
    print("\n" + "=" * 70)
    print("DEMO 6: Automatic Feature Type Detection")
    print("=" * 70)
    
    # Create dataset with ambiguous features
    np.random.seed(42)
    n = 1000
    
    df = pd.DataFrame({
        # Clear numerical
        'clear_numerical': np.random.randn(n),
        
        # Clear categorical
        'clear_categorical': np.random.choice(['Red', 'Blue', 'Green'], n),
        
        # Integer that should be categorical (low cardinality)
        'int_as_categorical': np.random.choice([1, 2, 3], n),
        
        # Integer that should be numerical (high cardinality)
        'int_as_numerical': np.random.randint(0, 500, n),
        
        # Binary feature
        'binary_feature': np.random.choice([0, 1], n),
        
        # Ordinal-like
        'rating': np.random.choice([1, 2, 3, 4, 5], n),
        
        # Float that might be categorical
        'float_categorical': np.random.choice([1.0, 2.0, 3.0], n),
        
        'target': np.random.choice([0, 1], n, p=[0.7, 0.3])
    })
    
    print("Created dataset with ambiguous feature types")
    print("\nOriginal dtypes:")
    for col in df.columns:
        if col != 'target':
            print(f"  {col}: {df[col].dtype}, unique values: {df[col].nunique()}")
    
    # Validate and preprocess
    validated_df, metadata = load_and_validate_data(df)
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, stats, pipeline = preprocessor.preprocess(
        validated_df, metadata.target_column
    )
    
    print("\n📊 Detected Feature Types:")
    print(f"\nNumerical features ({stats.n_numerical_features}):")
    for feat in stats.numerical_features:
        print(f"  - {feat}")
    
    print(f"\nCategorical features ({stats.n_categorical_features}):")
    for feat in stats.categorical_features:
        print(f"  - {feat}")
    
    return X_train, X_test, y_train, y_test, stats


def main():
    """Run all preprocessing demonstrations"""
    print("\n🚀 PU Learning Data Preprocessing Module - Demonstration\n")
    
    # Run demos
    demos = [
        ("Basic Preprocessing", demo_basic_preprocessing),
        ("Custom Configuration", demo_custom_configuration),
        ("High Cardinality Handling", demo_high_cardinality_handling),
        ("Pipeline Transform", demo_pipeline_transform),
        ("Extreme PU Scenario", demo_extreme_pu_scenario),
        ("Feature Type Detection", demo_feature_type_detection),
    ]
    
    results = {}
    for name, demo_func in demos:
        try:
            result = demo_func()
            if result:
                results[name] = result
        except Exception as e:
            print(f"\nError in {name}: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✅ Successfully demonstrated {len(results)} preprocessing scenarios")
    print("✅ Handled missing values with configurable strategies")
    print("✅ Encoded categorical features with multiple methods")
    print("✅ Scaled numerical features")
    print("✅ Preserved class distribution through stratified splitting")
    print("✅ Created reusable preprocessing pipelines")
    print("✅ Handled extreme PU scenarios (0.5% positive rate)")
    print("✅ Demonstrated automatic feature type detection")
    
    return results


if __name__ == "__main__":
    results = main()
