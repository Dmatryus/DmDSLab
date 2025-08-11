"""
PU Learning Experiment - Step 2: Data Preprocessing Module

This module provides comprehensive preprocessing for PU Learning datasets.
Key features:
- Intelligent handling of missing values with strategy selection
- Automatic detection and encoding of categorical features
- Proper train/test splitting with stratification
- Preservation of preprocessing pipelines for reproducibility
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
import warnings
from copy import deepcopy


@dataclass
class PreprocessingStatistics:
    """Container for preprocessing statistics and information"""
    
    # Split information
    train_size: int
    test_size: int
    train_positive_ratio: float
    test_positive_ratio: float
    
    # Missing value handling
    missing_handled: bool
    missing_strategies: Dict[str, str]
    missing_fill_values: Dict[str, Any]
    
    # Feature processing
    n_numerical_features: int
    n_categorical_features: int
    categorical_features: List[str]
    numerical_features: List[str]
    
    # Encoding information
    categorical_encoding_method: str
    categories_per_feature: Dict[str, int]
    
    # Scaling information  
    scaling_applied: bool
    scaling_stats: Dict[str, Dict[str, float]]  # mean, std per feature
    
    # Data quality
    constant_features_removed: List[str]
    high_cardinality_warnings: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary"""
        return {
            'split_info': {
                'train_size': self.train_size,
                'test_size': self.test_size,
                'train_positive_ratio': self.train_positive_ratio,
                'test_positive_ratio': self.test_positive_ratio
            },
            'missing_values': {
                'handled': self.missing_handled,
                'strategies': self.missing_strategies,
                'fill_values': self.missing_fill_values
            },
            'features': {
                'n_numerical': self.n_numerical_features,
                'n_categorical': self.n_categorical_features,
                'numerical': self.numerical_features,
                'categorical': self.categorical_features
            },
            'encoding': {
                'method': self.categorical_encoding_method,
                'categories_per_feature': self.categories_per_feature
            },
            'scaling': {
                'applied': self.scaling_applied,
                'stats': self.scaling_stats
            },
            'data_quality': {
                'constant_features_removed': self.constant_features_removed,
                'high_cardinality_warnings': self.high_cardinality_warnings
            }
        }
    
    def __str__(self) -> str:
        """String representation of preprocessing statistics"""
        return f"""
Preprocessing Statistics:
  Train/Test Split:
    - Train samples: {self.train_size:,} ({self.train_positive_ratio:.1%} positive)
    - Test samples: {self.test_size:,} ({self.test_positive_ratio:.1%} positive)
  
  Features:
    - Numerical: {self.n_numerical_features} features
    - Categorical: {self.n_categorical_features} features
  
  Missing Values: {'Handled' if self.missing_handled else 'None found'}
  Scaling: {'Applied' if self.scaling_applied else 'Not applied'}
  Encoding: {self.categorical_encoding_method if self.n_categorical_features > 0 else 'Not needed'}
  
  Data Quality:
    - Constant features removed: {len(self.constant_features_removed)}
    - High cardinality warnings: {len(self.high_cardinality_warnings)}
        """


@dataclass
class PreprocessingPipeline:
    """Container for all preprocessing transformers"""
    
    imputers: Dict[str, SimpleImputer] = field(default_factory=dict)
    scaler: Optional[StandardScaler] = None
    encoders: Dict[str, Any] = field(default_factory=dict)
    label_encoder: Optional[LabelEncoder] = None
    feature_order: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    numerical_features: List[str] = field(default_factory=list)
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the preprocessing pipeline to new data"""
        X_transformed = X.copy()
        
        # Apply imputation
        for col, imputer in self.imputers.items():
            if col in X_transformed.columns:
                X_transformed[col] = imputer.transform(X_transformed[[col]])
        
        # Apply encoding
        for col, encoder in self.encoders.items():
            if col in X_transformed.columns:
                X_transformed[col] = encoder.transform(X_transformed[col])
        
        # Apply scaling to numerical features
        if self.scaler and self.numerical_features:
            num_cols = [c for c in self.numerical_features if c in X_transformed.columns]
            if num_cols:
                X_transformed[num_cols] = self.scaler.transform(X_transformed[num_cols])
        
        # Ensure feature order
        cols_present = [c for c in self.feature_order if c in X_transformed.columns]
        X_transformed = X_transformed[cols_present]
        
        return X_transformed


class DataPreprocessor:
    """Main preprocessor for PU Learning experiments"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data preprocessor
        
        Args:
            config: Configuration dictionary with preprocessing parameters
        """
        self.config = config or {}
        
        # Default configuration
        self.test_size = self.config.get('test_size', 0.2)
        self.random_state = self.config.get('random_state', 42)
        self.scale_features = self.config.get('scale_features', True)
        self.handle_missing = self.config.get('handle_missing', True)
        self.encode_categorical = self.config.get('encode_categorical', True)
        self.remove_constant = self.config.get('remove_constant_features', True)
        self.high_cardinality_threshold = self.config.get('high_cardinality_threshold', 50)
        
        # Missing value strategies
        self.numerical_impute_strategy = self.config.get('numerical_impute_strategy', 'median')
        self.categorical_impute_strategy = self.config.get('categorical_impute_strategy', 'constant')
        self.categorical_fill_value = self.config.get('categorical_fill_value', 'missing')
        
        # Encoding method
        self.encoding_method = self.config.get('encoding_method', 'target')  # 'target', 'onehot', 'label'
        
        # Storage for preprocessing objects
        self.pipeline = PreprocessingPipeline()
        self.statistics = None
        self.warnings_list = []
    
    def preprocess(
        self, 
        df: pd.DataFrame, 
        target_column: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, PreprocessingStatistics, PreprocessingPipeline]:
        """
        Main preprocessing method
        
        Args:
            df: Validated DataFrame
            target_column: Name of target column
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, statistics, pipeline)
        """
        # Reset warnings
        self.warnings_list = []
        
        # Make a copy
        df_processed = df.copy()
        
        # Step 1: Separate features and target
        X, y = self._separate_features_target(df_processed, target_column)
        
        # Step 2: Remove constant features
        if self.remove_constant:
            X = self._remove_constant_features(X)
        
        # Step 3: Identify feature types
        categorical_features, numerical_features = self._identify_feature_types(X)
        self.pipeline.categorical_features = categorical_features
        self.pipeline.numerical_features = numerical_features
        
        # Step 4: Split data with stratification
        X_train, X_test, y_train, y_test = self._split_data(X, y)
        
        # Step 5: Handle missing values
        if self.handle_missing:
            X_train, X_test = self._handle_missing_values(
                X_train, X_test, categorical_features, numerical_features
            )
        
        # Step 6: Encode categorical features
        if self.encode_categorical and categorical_features:
            X_train, X_test = self._encode_categorical_features(
                X_train, X_test, y_train, categorical_features
            )
        
        # Step 7: Scale numerical features
        if self.scale_features and numerical_features:
            X_train, X_test = self._scale_features(
                X_train, X_test, numerical_features
            )
        
        # Step 8: Calculate statistics
        self.statistics = self._calculate_statistics(
            X_train, X_test, y_train, y_test, 
            categorical_features, numerical_features
        )
        
        # Show warnings if any
        if self.warnings_list:
            for warning in self.warnings_list:
                warnings.warn(warning, UserWarning)
        
        # Store feature order for pipeline
        self.pipeline.feature_order = X_train.columns.tolist()
        
        return X_train, X_test, y_train, y_test, self.statistics, self.pipeline
    
    def _separate_features_target(
        self, 
        df: pd.DataFrame, 
        target_column: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate features and target variable"""
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
        X = df.drop(columns=[target_column])
        y = df[target_column].astype(int)  # Ensure integer type for classification
        
        return X, y
    
    def _remove_constant_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove features with constant values"""
        constant_features = []
        for col in X.columns:
            if X[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            X = X.drop(columns=constant_features)
            self.warnings_list.append(
                f"Removed {len(constant_features)} constant features: {constant_features[:5]}..."
            )
        
        self.pipeline.feature_order = X.columns.tolist()
        return X
    
    def _identify_feature_types(
        self, 
        X: pd.DataFrame
    ) -> Tuple[List[str], List[str]]:
        """Identify categorical and numerical features"""
        categorical_features = []
        numerical_features = []
        high_cardinality_features = []
        
        for col in X.columns:
            if X[col].dtype == 'object':
                categorical_features.append(col)
                # Check for high cardinality
                n_unique = X[col].nunique()
                if n_unique > self.high_cardinality_threshold:
                    high_cardinality_features.append(f"{col} ({n_unique} categories)")
            elif X[col].dtype in ['int64', 'float64']:
                # Check if it's actually categorical
                n_unique = X[col].nunique()
                if n_unique < 10 and n_unique / len(X) < 0.05:
                    categorical_features.append(col)
                else:
                    numerical_features.append(col)
            else:
                numerical_features.append(col)
        
        if high_cardinality_features:
            self.warnings_list.append(
                f"High cardinality categorical features detected: {high_cardinality_features[:3]}..."
            )
        
        return categorical_features, numerical_features
    
    def _split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets with stratification"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y  # Maintain class distribution
        )
        
        # Report if stratification worked well
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        original_ratio = y.mean()
        
        if abs(train_ratio - original_ratio) > 0.05 or abs(test_ratio - original_ratio) > 0.05:
            self.warnings_list.append(
                f"Class distribution changed after split. "
                f"Original: {original_ratio:.1%}, Train: {train_ratio:.1%}, Test: {test_ratio:.1%}"
            )
        
        return X_train, X_test, y_train, y_test
    
    def _handle_missing_values(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        categorical_features: List[str],
        numerical_features: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Handle missing values in features"""
        X_train_imputed = X_train.copy()
        X_test_imputed = X_test.copy()
        
        # Impute numerical features
        if numerical_features:
            num_cols_with_missing = [
                col for col in numerical_features 
                if col in X_train.columns and X_train[col].isna().any()
            ]
            
            if num_cols_with_missing:
                for col in num_cols_with_missing:
                    imputer = SimpleImputer(strategy=self.numerical_impute_strategy)
                    X_train_imputed[col] = imputer.fit_transform(X_train[[col]])
                    X_test_imputed[col] = imputer.transform(X_test[[col]])
                    self.pipeline.imputers[col] = imputer
        
        # Impute categorical features
        if categorical_features:
            cat_cols_with_missing = [
                col for col in categorical_features
                if col in X_train.columns and X_train[col].isna().any()
            ]
            
            if cat_cols_with_missing:
                for col in cat_cols_with_missing:
                    imputer = SimpleImputer(
                        strategy=self.categorical_impute_strategy,
                        fill_value=self.categorical_fill_value
                    )
                    X_train_imputed[col] = imputer.fit_transform(X_train[[col]])
                    X_test_imputed[col] = imputer.transform(X_test[[col]])
                    self.pipeline.imputers[col] = imputer
        
        return X_train_imputed, X_test_imputed
    
    def _encode_categorical_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        categorical_features: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Encode categorical features"""
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()
        
        cat_cols_present = [col for col in categorical_features if col in X_train.columns]
        
        if not cat_cols_present:
            return X_train_encoded, X_test_encoded
        
        if self.encoding_method == 'target':
            # Use target encoding for PU Learning (handles high cardinality well)
            for col in cat_cols_present:
                encoder = TargetEncoder(cols=[col], smoothing=1.0)
                X_train_encoded[col] = encoder.fit_transform(X_train[col], y_train)
                X_test_encoded[col] = encoder.transform(X_test[col])
                self.pipeline.encoders[col] = encoder
                
        elif self.encoding_method == 'label':
            # Simple label encoding
            for col in cat_cols_present:
                encoder = LabelEncoder()
                # Fit on combined data to handle unseen categories
                combined = pd.concat([X_train[col], X_test[col]])
                encoder.fit(combined)
                X_train_encoded[col] = encoder.transform(X_train[col])
                X_test_encoded[col] = encoder.transform(X_test[col])
                self.pipeline.encoders[col] = encoder
        
        elif self.encoding_method == 'onehot':
            # One-hot encoding (be careful with high cardinality)
            X_train_encoded = pd.get_dummies(X_train_encoded, columns=cat_cols_present)
            X_test_encoded = pd.get_dummies(X_test_encoded, columns=cat_cols_present)
            
            # Align columns
            X_train_encoded, X_test_encoded = X_train_encoded.align(
                X_test_encoded, join='left', axis=1, fill_value=0
            )
        
        return X_train_encoded, X_test_encoded
    
    def _scale_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        numerical_features: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Scale numerical features"""
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        num_cols_present = [col for col in numerical_features if col in X_train.columns]
        
        if num_cols_present:
            scaler = StandardScaler()
            X_train_scaled[num_cols_present] = scaler.fit_transform(X_train[num_cols_present])
            X_test_scaled[num_cols_present] = scaler.transform(X_test[num_cols_present])
            self.pipeline.scaler = scaler
        
        return X_train_scaled, X_test_scaled
    
    def _calculate_statistics(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        categorical_features: List[str],
        numerical_features: List[str]
    ) -> PreprocessingStatistics:
        """Calculate preprocessing statistics"""
        
        # Missing value strategies
        missing_strategies = {}
        missing_fill_values = {}
        for col, imputer in self.pipeline.imputers.items():
            missing_strategies[col] = imputer.strategy
            if hasattr(imputer, 'fill_value'):
                missing_fill_values[col] = imputer.fill_value
            else:
                missing_fill_values[col] = imputer.statistics_[0]
        
        # Categories per feature
        categories_per_feature = {}
        for col in categorical_features:
            if col in X_train.columns:
                categories_per_feature[col] = X_train[col].nunique()
        
        # Scaling statistics
        scaling_stats = {}
        if self.pipeline.scaler:
            for i, col in enumerate(numerical_features):
                if col in X_train.columns:
                    col_idx = X_train.columns.get_loc(col)
                    if col_idx < len(self.pipeline.scaler.mean_):
                        scaling_stats[col] = {
                            'mean': self.pipeline.scaler.mean_[col_idx],
                            'std': self.pipeline.scaler.scale_[col_idx]
                        }
        
        # High cardinality warnings
        high_cardinality_warnings = []
        for col in categorical_features:
            if col in categories_per_feature:
                if categories_per_feature[col] > self.high_cardinality_threshold:
                    high_cardinality_warnings.append(col)
        
        stats = PreprocessingStatistics(
            train_size=len(X_train),
            test_size=len(X_test),
            train_positive_ratio=y_train.mean(),
            test_positive_ratio=y_test.mean(),
            missing_handled=len(self.pipeline.imputers) > 0,
            missing_strategies=missing_strategies,
            missing_fill_values=missing_fill_values,
            n_numerical_features=len([c for c in numerical_features if c in X_train.columns]),
            n_categorical_features=len([c for c in categorical_features if c in X_train.columns]),
            categorical_features=[c for c in categorical_features if c in X_train.columns],
            numerical_features=[c for c in numerical_features if c in X_train.columns],
            categorical_encoding_method=self.encoding_method if categorical_features else 'none',
            categories_per_feature=categories_per_feature,
            scaling_applied=self.pipeline.scaler is not None,
            scaling_stats=scaling_stats,
            constant_features_removed=[],  # Filled if remove_constant is True
            high_cardinality_warnings=high_cardinality_warnings
        )
        
        return stats


def preprocess_data(
    df: pd.DataFrame,
    target_column: str,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, PreprocessingStatistics, PreprocessingPipeline]:
    """
    Convenience function to preprocess data in one step
    
    Args:
        df: Validated DataFrame
        target_column: Name of target column
        config: Optional configuration dictionary
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, statistics, pipeline)
    """
    preprocessor = DataPreprocessor(config)
    return preprocessor.preprocess(df, target_column)
