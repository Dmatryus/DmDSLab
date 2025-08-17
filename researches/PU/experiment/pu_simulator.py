"""
PU Learning Experiment - Step 3: PU Scenario Simulation Module

This module simulates realistic PU Learning scenarios from fully labeled datasets.
Key features:
- Multiple simulation strategies (SCAR, SAR, etc.)
- Configurable selection probabilities and bias patterns
- Ground truth preservation for evaluation
- Comprehensive simulation statistics
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass
from scipy import stats
import warnings


@dataclass
class SimulationStatistics:
    """Container for PU simulation statistics"""
    
    # Original data info
    original_positive_count: int
    original_negative_count: int
    original_positive_ratio: float
    
    # Simulation parameters
    simulation_strategy: str
    alpha_value: float  # Selection probability for positive examples
    
    # Simulated data info
    labeled_positive_count: int
    unlabeled_count: int
    hidden_positive_count: int
    hidden_negative_count: int
    
    # Ratios after simulation
    simulated_positive_ratio: float  # labeled_pos / total
    hidden_positive_ratio: float    # hidden_pos / unlabeled
    
    # Quality metrics
    label_completeness: float  # labeled_pos / original_pos
    pu_bias_score: float      # Measure of selection bias
    
    # Distribution metrics
    kl_divergence: float      # Distribution shift measurement
    wasserstein_distance: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary"""
        return {
            'original_data': {
                'positive_count': self.original_positive_count,
                'negative_count': self.original_negative_count,
                'positive_ratio': self.original_positive_ratio
            },
            'simulation_params': {
                'strategy': self.simulation_strategy,
                'alpha': self.alpha_value
            },
            'simulated_data': {
                'labeled_positive_count': self.labeled_positive_count,
                'unlabeled_count': self.unlabeled_count,
                'hidden_positive_count': self.hidden_positive_count,
                'hidden_negative_count': self.hidden_negative_count
            },
            'ratios': {
                'simulated_positive_ratio': self.simulated_positive_ratio,
                'hidden_positive_ratio': self.hidden_positive_ratio,
                'label_completeness': self.label_completeness
            },
            'quality_metrics': {
                'pu_bias_score': self.pu_bias_score,
                'kl_divergence': self.kl_divergence,
                'wasserstein_distance': self.wasserstein_distance
            }
        }
    
    def __str__(self) -> str:
        """String representation of simulation statistics"""
        return f"""
PU Simulation Statistics:
  Original Data:
    - Total samples: {self.original_positive_count + self.original_negative_count:,}
    - Positive: {self.original_positive_count:,} ({self.original_positive_ratio:.1%})
    - Negative: {self.original_negative_count:,}
  
  Simulation ({self.simulation_strategy}, α={self.alpha_value}):
    - Labeled positive: {self.labeled_positive_count:,}
    - Unlabeled total: {self.unlabeled_count:,}
    - Hidden positive: {self.hidden_positive_count:,}
    - Hidden negative: {self.hidden_negative_count:,}
  
  Quality Metrics:
    - Label completeness: {self.label_completeness:.1%}
    - PU positive ratio: {self.simulated_positive_ratio:.2%}
    - Hidden positive ratio: {self.hidden_positive_ratio:.1%}
    - KL divergence: {self.kl_divergence:.4f}
        """


class PUSimulator:
    """Simulator for creating PU Learning scenarios"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize PU simulator
        
        Args:
            config: Configuration dictionary with simulation parameters
        """
        self.config = config or {}
        
        # Default simulation parameters
        self.default_alpha = self.config.get('default_alpha', 0.3)
        self.random_state = self.config.get('random_state', 42)
        self.min_labeled_positive = self.config.get('min_labeled_positive', 5)
        self.warn_low_alpha = self.config.get('warn_low_alpha', True)
        
        # Strategy-specific parameters
        self.sar_bias_strength = self.config.get('sar_bias_strength', 0.5)
        self.sar_feature_indices = self.config.get('sar_feature_indices', None)
        
        self.warnings_list = []
    
    def simulate_pu_scenario(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        alpha: Optional[float] = None,
        strategy: str = 'scar'
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, SimulationStatistics]:
        """
        Main simulation method
        
        Args:
            X_train, X_test: Feature datasets
            y_train, y_test: True labels (0 and 1)
            alpha: Selection probability for positive examples
            strategy: Simulation strategy ('scar', 'sar', 'prior_shift')
            
        Returns:
            Tuple of (y_train_pu, y_test_pu, y_train_true, y_test_true, statistics)
        """
        # Reset warnings
        self.warnings_list = []
        
        # Set random seed
        np.random.seed(self.random_state)
        
        # Use default alpha if not specified
        if alpha is None:
            alpha = self.default_alpha
        
        # Validate inputs
        self._validate_inputs(X_train, X_test, y_train, y_test, alpha)
        
        # Simulate train set
        y_train_pu, y_train_true = self._simulate_split(
            X_train, y_train, alpha, strategy
        )
        
        # Simulate test set with same strategy
        y_test_pu, y_test_true = self._simulate_split(
            X_test, y_test, alpha, strategy
        )
        
        # Calculate statistics
        statistics = self._calculate_statistics(
            y_train, y_test, y_train_pu, y_test_pu, alpha, strategy
        )
        
        # Show warnings if any
        if self.warnings_list:
            for warning in self.warnings_list:
                warnings.warn(warning, UserWarning)
        
        return y_train_pu, y_test_pu, y_train_true, y_test_true, statistics
    
    def _validate_inputs(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        alpha: float
    ) -> None:
        """Validate simulation inputs"""
        
        # Check alpha range
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"Alpha must be in (0, 1], got {alpha}")
        
        # Check for binary labels
        unique_train = set(y_train.unique())
        unique_test = set(y_test.unique())
        if not unique_train.issubset({0, 1}) or not unique_test.issubset({0, 1}):
            raise ValueError("Labels must be binary (0 and 1)")
        
        # Check for presence of both classes
        if 1 not in unique_train or 1 not in unique_test:
            raise ValueError("Both train and test must contain positive examples")
        
        # Warn about low alpha
        if alpha < 0.1 and self.warn_low_alpha:
            self.warnings_list.append(
                f"Very low alpha ({alpha:.1%}) may result in too few labeled positive examples"
            )
        
        # Check minimum positive examples
        n_pos_train = (y_train == 1).sum()
        n_pos_test = (y_test == 1).sum()
        expected_labeled_train = int(n_pos_train * alpha)
        expected_labeled_test = int(n_pos_test * alpha)
        
        if expected_labeled_train < self.min_labeled_positive:
            self.warnings_list.append(
                f"Train set may have too few labeled positives: {expected_labeled_train} expected"
            )
        
        if expected_labeled_test < 2:  # Need at least 2 for test
            self.warnings_list.append(
                f"Test set may have too few labeled positives: {expected_labeled_test} expected"
            )
    
    def _simulate_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        alpha: float,
        strategy: str
    ) -> Tuple[pd.Series, pd.Series]:
        """Simulate PU scenario for a single data split"""
        
        # Keep true labels for evaluation
        y_true = y.copy()
        
        # Start with all examples as unlabeled (0)
        y_pu = pd.Series(0, index=y.index, dtype=int)
        
        # Get positive indices
        positive_indices = y[y == 1].index
        
        if strategy == 'scar':
            # Selected Completely At Random
            selected_indices = self._scar_selection(positive_indices, alpha)
            
        elif strategy == 'sar':
            # Selected At Random (with feature bias)
            selected_indices = self._sar_selection(X, positive_indices, alpha)
            
        elif strategy == 'prior_shift':
            # Prior probability shift
            selected_indices = self._prior_shift_selection(X, positive_indices, alpha)
            
        else:
            raise ValueError(f"Unknown simulation strategy: {strategy}")
        
        # Mark selected positive examples as labeled (1)
        y_pu.loc[selected_indices] = 1
        
        return y_pu, y_true
    
    def _scar_selection(
        self,
        positive_indices: pd.Index,
        alpha: float
    ) -> pd.Index:
        """SCAR: Selected Completely At Random"""
        n_to_select = max(1, int(len(positive_indices) * alpha))
        selected = np.random.choice(
            positive_indices, 
            size=min(n_to_select, len(positive_indices)), 
            replace=False
        )
        return pd.Index(selected)
    
    def _sar_selection(
        self,
        X: pd.DataFrame,
        positive_indices: pd.Index,
        alpha: float
    ) -> pd.Index:
        """SAR: Selected At Random with feature-dependent bias"""
        
        # Get positive examples features
        X_pos = X.loc[positive_indices]
        
        # Determine which features to use for bias
        if self.sar_feature_indices is None:
            # Use first few numerical features
            numerical_cols = X.select_dtypes(include=[np.number]).columns
            bias_features = numerical_cols[:min(3, len(numerical_cols))]
        else:
            bias_features = [X.columns[i] for i in self.sar_feature_indices]
        
        if len(bias_features) == 0:
            # Fallback to SCAR if no suitable features
            return self._scar_selection(positive_indices, alpha)
        
        # Calculate selection probabilities based on features
        # Higher values in selected features -> higher probability
        feature_scores = X_pos[bias_features].mean(axis=1)
        feature_scores = (feature_scores - feature_scores.min()) / (feature_scores.max() - feature_scores.min() + 1e-8)
        
        # Apply bias strength
        selection_probs = alpha + self.sar_bias_strength * (feature_scores - 0.5)
        selection_probs = np.clip(selection_probs, 0.01, 0.99)
        
        # Select based on probabilities
        selected_mask = np.random.random(len(selection_probs)) < selection_probs
        selected_indices = positive_indices[selected_mask]
        
        # Ensure we have at least one selection
        if len(selected_indices) == 0:
            selected_indices = positive_indices[:1]
        
        return selected_indices
    
    def _prior_shift_selection(
        self,
        X: pd.DataFrame,
        positive_indices: pd.Index,
        alpha: float
    ) -> pd.Index:
        """Prior shift: Selection probability changes over time/features"""
        
        # Simple implementation: bias based on sample order (temporal shift)
        n_pos = len(positive_indices)
        
        # Create time-dependent selection probability
        time_bias = np.linspace(0.5, 1.5, n_pos)  # Increasing over time
        selection_probs = alpha * time_bias
        selection_probs = np.clip(selection_probs, 0.01, 0.99)
        
        # Select based on probabilities
        selected_mask = np.random.random(n_pos) < selection_probs
        selected_indices = positive_indices[selected_mask]
        
        # Ensure minimum selection
        if len(selected_indices) == 0:
            selected_indices = positive_indices[:1]
        
        return selected_indices
    
    def _calculate_statistics(
        self,
        y_train_true: pd.Series,
        y_test_true: pd.Series,
        y_train_pu: pd.Series,
        y_test_pu: pd.Series,
        alpha: float,
        strategy: str
    ) -> SimulationStatistics:
        """Calculate comprehensive simulation statistics"""
        
        # Combine train and test for overall statistics
        y_true_combined = pd.concat([y_train_true, y_test_true])
        y_pu_combined = pd.concat([y_train_pu, y_test_pu])
        
        # Original data stats
        original_pos = (y_true_combined == 1).sum()
        original_neg = (y_true_combined == 0).sum()
        original_ratio = original_pos / len(y_true_combined)
        
        # Simulated data stats
        labeled_pos = (y_pu_combined == 1).sum()
        unlabeled_total = (y_pu_combined == 0).sum()
        
        # Hidden stats (what's in the unlabeled set)
        unlabeled_mask = y_pu_combined == 0
        hidden_pos = ((y_true_combined == 1) & unlabeled_mask).sum()
        hidden_neg = ((y_true_combined == 0) & unlabeled_mask).sum()
        
        # Calculated ratios
        simulated_ratio = labeled_pos / len(y_pu_combined)
        hidden_pos_ratio = hidden_pos / unlabeled_total if unlabeled_total > 0 else 0
        label_completeness = labeled_pos / original_pos if original_pos > 0 else 0
        
        # Quality metrics
        pu_bias_score = self._calculate_bias_score(y_true_combined, y_pu_combined)
        kl_div = self._calculate_kl_divergence(y_true_combined, y_pu_combined)
        wasserstein_dist = self._calculate_wasserstein_distance(y_true_combined, y_pu_combined)
        
        return SimulationStatistics(
            original_positive_count=original_pos,
            original_negative_count=original_neg,
            original_positive_ratio=original_ratio,
            simulation_strategy=strategy,
            alpha_value=alpha,
            labeled_positive_count=labeled_pos,
            unlabeled_count=unlabeled_total,
            hidden_positive_count=hidden_pos,
            hidden_negative_count=hidden_neg,
            simulated_positive_ratio=simulated_ratio,
            hidden_positive_ratio=hidden_pos_ratio,
            label_completeness=label_completeness,
            pu_bias_score=pu_bias_score,
            kl_divergence=kl_div,
            wasserstein_distance=wasserstein_dist
        )
    
    def _calculate_bias_score(
        self, 
        y_true: pd.Series, 
        y_pu: pd.Series
    ) -> float:
        """Calculate PU bias score (higher = more biased selection)"""
        
        # Compare the ratio of selected vs non-selected positives
        pos_indices = y_true == 1
        selected_pos = (y_pu == 1) & pos_indices
        
        if pos_indices.sum() == 0:
            return 0.0
        
        selection_rate = selected_pos.sum() / pos_indices.sum()
        
        # Bias score: deviation from uniform random selection
        # 0 = no bias, 1 = maximum bias
        uniform_expected = 0.5  # Expected selection rate for uniform random
        bias_score = abs(selection_rate - uniform_expected) * 2
        
        return min(bias_score, 1.0)
    
    def _calculate_kl_divergence(
        self, 
        y_true: pd.Series, 
        y_pu: pd.Series
    ) -> float:
        """Calculate KL divergence between true and observed distributions"""
        
        # True distribution
        p_true = [
            (y_true == 0).mean(),  # P(y=0)
            (y_true == 1).mean()   # P(y=1)
        ]
        
        # Observed distribution (treating unlabeled as negative)
        q_observed = [
            (y_pu == 0).mean(),   # P(observed=0)
            (y_pu == 1).mean()    # P(observed=1)
        ]
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        p_true = [p + epsilon for p in p_true]
        q_observed = [q + epsilon for q in q_observed]
        
        # Normalize
        p_sum = sum(p_true)
        q_sum = sum(q_observed)
        p_true = [p / p_sum for p in p_true]
        q_observed = [q / q_sum for q in q_observed]
        
        # KL divergence
        kl_div = sum(p * np.log(p / q) for p, q in zip(p_true, q_observed))
        
        return float(kl_div)
    
    def _calculate_wasserstein_distance(
        self, 
        y_true: pd.Series, 
        y_pu: pd.Series
    ) -> float:
        """Calculate Wasserstein distance between distributions"""
        
        # For binary case, Wasserstein distance is simply the absolute difference in means
        return abs(y_true.mean() - y_pu.mean())


def simulate_pu_scenario(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    alpha: Optional[float] = None,
    strategy: str = 'scar',
    config: Optional[Dict[str, Any]] = None
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, SimulationStatistics]:
    """
    Convenience function to simulate PU scenario in one step
    
    Args:
        X_train, X_test: Feature datasets
        y_train, y_test: True binary labels
        alpha: Selection probability for positive examples (default: 0.3)
        strategy: Simulation strategy ('scar', 'sar', 'prior_shift')
        config: Optional configuration dictionary
        
    Returns:
        Tuple of (y_train_pu, y_test_pu, y_train_true, y_test_true, statistics)
    """
    simulator = PUSimulator(config)
    return simulator.simulate_pu_scenario(
        X_train, X_test, y_train, y_test, alpha, strategy
    )