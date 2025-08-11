"""
PU Learning Experiment - Step 1: Data Loading and Validation Module

This module provides flexible validation for PU Learning datasets.
Key features:
- Size constraints generate warnings only, allowing experiments with 
  datasets of any size while informing users about potential issues
- Class imbalance is expected and natural for PU Learning, so no
  warnings are generated for extreme imbalance scenarios
- Focus on critical validation: binary targets and presence of both classes
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
import warnings


@dataclass
class DatasetMetadata:
    """Metadata container for validated dataset
    
    Contains comprehensive information about the dataset including
    class distribution. Note that class imbalance is natural and
    expected in PU Learning scenarios.
    """
    n_samples: int
    n_features: int
    n_positive: int
    n_negative: int
    positive_ratio: float  # Natural to be low in PU Learning
    has_missing: bool
    missing_ratio: float
    feature_types: Dict[str, str]
    target_column: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        return {
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'n_positive': self.n_positive,
            'n_negative': self.n_negative,
            'positive_ratio': self.positive_ratio,
            'has_missing': self.has_missing,
            'missing_ratio': self.missing_ratio,
            'feature_types': self.feature_types,
            'target_column': self.target_column
        }
    
    def __str__(self) -> str:
        """String representation of metadata"""
        imbalance_note = ""
        if self.positive_ratio < 0.1:
            imbalance_note = " (natural for PU Learning)"
        
        return f"""
Dataset Metadata:
  Samples: {self.n_samples:,}
  Features: {self.n_features}
  Positive samples: {self.n_positive:,} ({self.positive_ratio:.1%}){imbalance_note}
  Negative samples: {self.n_negative:,} ({1-self.positive_ratio:.1%})
  Missing values: {'Yes' if self.has_missing else 'No'} ({self.missing_ratio:.1%})
  Target column: {self.target_column}
        """


class DataValidator:
    """Validator for PU Learning experiment data"""
    
    # Validation constants (recommendations only, not hard requirements)
    MIN_SAMPLES = 1_000      # Recommended minimum samples
    MAX_SAMPLES = 100_000    # Recommended maximum samples
    MIN_FEATURES = 1         # Recommended minimum features
    MAX_FEATURES = 500       # Recommended maximum features
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data validator
        
        Args:
            config: Configuration dictionary with validation parameters
        """
        self.config = config or {}
        
        # Override defaults with config if provided (now as recommendations)
        self.min_samples = self.config.get('min_samples', self.MIN_SAMPLES)
        self.max_samples = self.config.get('max_samples', self.MAX_SAMPLES)
        self.min_features = self.config.get('min_features', self.MIN_FEATURES)
        self.max_features = self.config.get('max_features', self.MAX_FEATURES)
        
        # Target column name
        self.target_column = self.config.get('target_column', 'target')
        
        # Validation results storage
        self.validation_errors = []
        self.validation_warnings = []
    
    def validate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, DatasetMetadata]:
        """
        Main validation method
        
        Performs comprehensive validation of the input DataFrame.
        Size constraints generate warnings only, allowing flexibility
        in dataset sizes while informing about potential issues.
        
        Args:
            df: Input DataFrame to validate
            
        Returns:
            Tuple of (validated DataFrame, metadata)
            
        Raises:
            ValueError: If critical validation errors are found
                       (missing target, non-binary values, etc.)
                       Note: Dataset size never causes errors, only warnings
        """
        # Reset validation messages
        self.validation_errors = []
        self.validation_warnings = []
        
        # Make a copy to avoid modifying original
        df_validated = df.copy()
        
        # Step 1: Check DataFrame is not empty
        self._check_not_empty(df_validated)
        
        # Step 2: Identify and validate target column
        target_col = self._identify_target_column(df_validated)
        
        # Step 3: Validate target values
        self._validate_target_values(df_validated[target_col])
        
        # Step 4: Check that both classes are present (imbalance is OK for PU)
        self._check_class_balance(df_validated[target_col])
        
        # Step 5: Validate dataset size
        self._validate_dataset_size(df_validated)
        
        # Step 6: Check for missing values
        missing_info = self._check_missing_values(df_validated)
        
        # Step 7: Identify feature types
        feature_types = self._identify_feature_types(df_validated, target_col)
        
        # If we have critical errors, raise exception
        if self.validation_errors:
            error_msg = "Data validation failed:\n" + "\n".join(self.validation_errors)
            raise ValueError(error_msg)
        
        # Show warnings if any
        if self.validation_warnings:
            for warning in self.validation_warnings:
                warnings.warn(warning, UserWarning)
        
        # Calculate metadata
        metadata = self._calculate_metadata(
            df_validated, 
            target_col, 
            missing_info, 
            feature_types
        )
        
        return df_validated, metadata
    
    def _check_not_empty(self, df: pd.DataFrame) -> None:
        """Check that DataFrame is not empty"""
        if df.empty:
            self.validation_errors.append("DataFrame is empty")
        if df.shape[1] < 2:
            self.validation_errors.append(
                f"DataFrame must have at least 2 columns (features + target), got {df.shape[1]}"
            )
    
    def _identify_target_column(self, df: pd.DataFrame) -> str:
        """Identify the target column"""
        # First, check if specified column exists
        if self.target_column in df.columns:
            return self.target_column
        
        # Try common target column names
        common_names = ['target', 'label', 'class', 'y', 'outcome']
        for name in common_names:
            if name in df.columns:
                self.validation_warnings.append(
                    f"Target column '{self.target_column}' not found, using '{name}' instead"
                )
                return name
        
        # If binary column exists, assume it's the target
        binary_cols = []
        for col in df.columns:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                binary_cols.append(col)
        
        if len(binary_cols) == 1:
            self.validation_warnings.append(
                f"Target column not specified, using binary column '{binary_cols[0]}'"
            )
            return binary_cols[0]
        elif len(binary_cols) > 1:
            # Use the last binary column (common pattern)
            self.validation_warnings.append(
                f"Multiple binary columns found: {binary_cols}, using '{binary_cols[-1]}'"
            )
            return binary_cols[-1]
        
        self.validation_errors.append(
            f"Could not identify target column. Please specify it in the config."
        )
        return None
    
    def _validate_target_values(self, target: pd.Series) -> None:
        """Validate that target contains only 0 and 1"""
        unique_values = target.dropna().unique()
        
        # Check if values are binary
        if not set(unique_values).issubset({0, 1, 0.0, 1.0}):
            self.validation_errors.append(
                f"Target column must contain only 0 and 1, found: {sorted(unique_values)}"
            )
        
        # Check for NaN values in target
        if target.isna().any():
            n_missing = target.isna().sum()
            self.validation_errors.append(
                f"Target column contains {n_missing} missing values"
            )
    
    def _check_class_balance(self, target: pd.Series) -> None:
        """Check that both classes are present"""
        value_counts = target.value_counts()
        
        if 0 not in value_counts and 0.0 not in value_counts:
            self.validation_errors.append("No negative samples (0) found in target")
        
        if 1 not in value_counts and 1.0 not in value_counts:
            self.validation_errors.append("No positive samples (1) found in target")
        
        # Note: No warning for class imbalance as it's natural for PU Learning
        # PU Learning is designed to handle scenarios with few positive examples
    
    def _validate_dataset_size(self, df: pd.DataFrame) -> None:
        """Validate dataset size constraints (warnings only, never errors)"""
        n_samples = len(df)
        n_features = len(df.columns) - 1  # Exclude target
        
        # Check sample size - only warnings, never errors
        if n_samples < self.min_samples:
            self.validation_warnings.append(
                f"Small dataset: {n_samples} samples (recommended minimum: {self.min_samples}). "
                f"The experiment will proceed, but results may be less reliable."
            )
        elif n_samples > self.max_samples:
            self.validation_warnings.append(
                f"Large dataset: {n_samples} samples (recommended maximum: {self.max_samples}). "
                f"Processing may be slower, but the experiment will proceed."
            )
        
        # Check feature count - only warnings, never errors
        if n_features < self.min_features:
            self.validation_warnings.append(
                f"Few features: {n_features} (recommended minimum: {self.min_features}). "
                f"The experiment will proceed, but consider adding more features if available."
            )
        elif n_features > self.max_features:
            self.validation_warnings.append(
                f"Many features: {n_features} (recommended maximum: {self.max_features}). "
                f"Consider feature selection to improve performance, but the experiment will proceed."
            )
    
    def _check_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for missing values in the dataset"""
        n_missing = df.isna().sum().sum()
        n_total = df.size
        missing_ratio = n_missing / n_total
        
        missing_info = {
            'has_missing': n_missing > 0,
            'n_missing': n_missing,
            'missing_ratio': missing_ratio,
            'columns_with_missing': df.columns[df.isna().any()].tolist()
        }
        
        if missing_ratio > 0.3:
            self.validation_warnings.append(
                f"High proportion of missing values: {missing_ratio:.1%}"
            )
        
        return missing_info
    
    def _identify_feature_types(self, df: pd.DataFrame, target_col: str) -> Dict[str, str]:
        """Identify types of features"""
        feature_types = {}
        
        for col in df.columns:
            if col == target_col:
                continue
                
            dtype = df[col].dtype
            unique_ratio = df[col].nunique() / len(df)
            
            if dtype in ['int64', 'float64']:
                # Check if it's actually categorical
                if df[col].nunique() < 10 and unique_ratio < 0.05:
                    feature_types[col] = 'categorical'
                else:
                    feature_types[col] = 'numerical'
            elif dtype == 'object':
                feature_types[col] = 'categorical'
            elif dtype == 'bool':
                feature_types[col] = 'binary'
            else:
                feature_types[col] = str(dtype)
        
        return feature_types
    
    def _calculate_metadata(
        self, 
        df: pd.DataFrame, 
        target_col: str,
        missing_info: Dict[str, Any],
        feature_types: Dict[str, str]
    ) -> DatasetMetadata:
        """Calculate and return dataset metadata
        
        Note: The positive_ratio can be very low (e.g., 0.5%) which is
        completely normal and expected for PU Learning scenarios.
        """
        target = df[target_col]
        
        # Convert to int for counting
        target_int = target.astype(int)
        n_positive = (target_int == 1).sum()
        n_negative = (target_int == 0).sum()
        
        metadata = DatasetMetadata(
            n_samples=len(df),
            n_features=len(df.columns) - 1,
            n_positive=n_positive,
            n_negative=n_negative,
            positive_ratio=n_positive / len(df),
            has_missing=missing_info['has_missing'],
            missing_ratio=missing_info['missing_ratio'],
            feature_types=feature_types,
            target_column=target_col
        )
        
        return metadata


def load_and_validate_data(
    df: pd.DataFrame, 
    config: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, DatasetMetadata]:
    """
    Convenience function to load and validate data in one step
    
    This function provides flexible validation that accepts datasets
    of any size, generating warnings for sizes outside recommended
    ranges but never failing due to size constraints.
    
    Args:
        df: Input DataFrame
        config: Optional configuration dictionary with keys:
                - min_samples: recommended minimum samples (default: 1000)
                - max_samples: recommended maximum samples (default: 100000)
                - min_features: recommended minimum features (default: 1)
                - max_features: recommended maximum features (default: 500)
                - target_column: expected target column name (default: 'target')
        
    Returns:
        Tuple of (validated DataFrame, metadata)
        
    Raises:
        ValueError: Only for critical errors (missing target, non-binary values)
                   Never raised for dataset size issues
    """
    validator = DataValidator(config)
    return validator.validate(df)