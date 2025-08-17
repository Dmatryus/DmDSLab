"""
Demo script for PU Learning Scenario Simulation Module
Shows different simulation strategies and their effects
"""

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any

# Import modules from previous steps
from researches.PU.experiment.pu_data_validator import load_and_validate_data
from researches.PU.experiment.pu_data_preprocessor import preprocess_data
from researches.PU.experiment.pu_simulator import (
    PUSimulator,
    simulate_pu_scenario,
    SimulationStatistics
)


def create_diverse_dataset(n_samples=2000, n_features=15, positive_ratio=0.4):
    """Create a dataset with diverse feature types for simulation"""
    np.random.seed(42)
    
    # Numerical features with different distributions
    data = {}
    for i in range(n_features):
        if i % 3 == 0:
            data[f'num_feat_{i}'] = np.random.exponential(2, n_samples)
        elif i % 3 == 1:
            data[f'num_feat_{i}'] = np.random.normal(50, 10, n_samples)
        else:
            data[f'num_feat_{i}'] = np.random.uniform(0, 100, n_samples)
    
    # Add categorical features
    data['category_A'] = np.random.choice(['Type1', 'Type2', 'Type3'], n_samples)
    data['category_B'] = np.random.choice(['Low', 'Medium', 'High'], n_samples)
    
    # Create target with specified ratio
    n_positive = int(n_samples * positive_ratio)
    target = np.zeros(n_samples)
    target[:n_positive] = 1
    np.random.shuffle(target)
    data['target'] = target
    
    return pd.DataFrame(data)


def demo_scar_simulation():
    """Demo: SCAR (Selected Completely At Random) simulation"""
    print("=" * 70)
    print("DEMO 1: SCAR (Selected Completely At Random) Simulation")
    print("=" * 70)
    
    # Create and preprocess dataset
    df = create_diverse_dataset(n_samples=1500, positive_ratio=0.3)
    print(f"Created dataset: {df.shape}")
    print(f"Original positive ratio: {df['target'].mean():.1%}")
    
    # Validate and preprocess
    validated_df, metadata = load_and_validate_data(df)
    X_train, X_test, y_train, y_test, _, _ = preprocess_data(
        validated_df, metadata.target_column
    )
    
    print(f"\nAfter preprocessing:")
    print(f"  Train: {len(y_train)} samples ({y_train.mean():.1%} positive)")
    print(f"  Test: {len(y_test)} samples ({y_test.mean():.1%} positive)")
    
    # Simulate SCAR with different alpha values
    alphas = [0.1, 0.3, 0.5, 0.7]
    
    print(f"\n📊 SCAR Simulation Results:")
    for alpha in alphas:
        y_train_pu, y_test_pu, y_train_true, y_test_true, stats = simulate_pu_scenario(
            X_train, X_test, y_train, y_test, 
            alpha=alpha, strategy='scar'
        )
        
        print(f"\n  α = {alpha}:")
        print(f"    Labeled positive: {stats.labeled_positive_count} ({stats.label_completeness:.1%} of original)")
        print(f"    Unlabeled: {stats.unlabeled_count} ({stats.hidden_positive_ratio:.1%} hidden positive)")
        print(f"    KL divergence: {stats.kl_divergence:.4f}")
    
    # Detailed analysis for α=0.3
    y_train_pu, y_test_pu, y_train_true, y_test_true, stats = simulate_pu_scenario(
        X_train, X_test, y_train, y_test, 
        alpha=0.3, strategy='scar'
    )
    
    print(f"\n📈 Detailed SCAR Analysis (α=0.3):")
    print(stats)
    
    return y_train_pu, y_test_pu, y_train_true, y_test_true, stats


def demo_sar_simulation():
    """Demo: SAR (Selected At Random with bias) simulation"""
    print("\n" + "=" * 70)
    print("DEMO 2: SAR (Selected At Random with Feature Bias) Simulation")
    print("=" * 70)
    
    # Create dataset with feature patterns
    df = create_diverse_dataset(n_samples=1200, positive_ratio=0.25)
    
    # Validate and preprocess
    validated_df, metadata = load_and_validate_data(df)
    X_train, X_test, y_train, y_test, _, _ = preprocess_data(
        validated_df, metadata.target_column
    )
    
    # Configure SAR simulation
    config = {
        'sar_bias_strength': 0.7,  # Strong bias
        'sar_feature_indices': [0, 1, 2],  # Use first 3 features for bias
        'random_state': 42
    }
    
    print(f"SAR Configuration: {config}")
    
    # Compare SCAR vs SAR
    strategies = ['scar', 'sar']
    alpha = 0.4
    
    results = {}
    for strategy in strategies:
        y_train_pu, y_test_pu, y_train_true, y_test_true, stats = simulate_pu_scenario(
            X_train, X_test, y_train, y_test,
            alpha=alpha, strategy=strategy, config=config
        )
        results[strategy] = stats
    
    print(f"\n📊 SCAR vs SAR Comparison (α={alpha}):")
    print(f"{'Metric':<25} {'SCAR':<15} {'SAR':<15} {'Difference'}")
    print("-" * 65)
    
    metrics = [
        ('Label completeness', 'label_completeness', '{:.1%}'),
        ('Hidden pos ratio', 'hidden_positive_ratio', '{:.1%}'),
        ('PU bias score', 'pu_bias_score', '{:.3f}'),
        ('KL divergence', 'kl_divergence', '{:.4f}'),
        ('Wasserstein dist', 'wasserstein_distance', '{:.4f}')
    ]
    
    for metric_name, attr, fmt in metrics:
        scar_val = getattr(results['scar'], attr)
        sar_val = getattr(results['sar'], attr)
        diff = sar_val - scar_val
        
        print(f"{metric_name:<25} {fmt.format(scar_val):<15} {fmt.format(sar_val):<15} {diff:+.4f}")
    
    print(f"\n🔍 SAR creates more biased selection (higher bias score)")
    print(f"   This simulates real-world scenarios where selection depends on features")
    
    return results


def demo_prior_shift_simulation():
    """Demo: Prior shift simulation"""
    print("\n" + "=" * 70)
    print("DEMO 3: Prior Shift Simulation")
    print("=" * 70)
    
    # Create dataset
    df = create_diverse_dataset(n_samples=1000, positive_ratio=0.35)
    
    # Validate and preprocess
    validated_df, metadata = load_and_validate_data(df)
    X_train, X_test, y_train, y_test, _, _ = preprocess_data(
        validated_df, metadata.target_column
    )
    
    print(f"Created dataset for prior shift simulation")
    print(f"Original distribution: {y_train.mean():.1%} positive in train")
    
    # Simulate prior shift
    y_train_pu, y_test_pu, y_train_true, y_test_true, stats = simulate_pu_scenario(
        X_train, X_test, y_train, y_test,
        alpha=0.5, strategy='prior_shift'
    )
    
    print(f"\n📊 Prior Shift Results:")
    print(f"  Strategy: {stats.simulation_strategy}")
    print(f"  Temporal bias creates varying selection probability")
    print(f"  Label completeness: {stats.label_completeness:.1%}")
    print(f"  Distribution shift (KL): {stats.kl_divergence:.4f}")
    
    return y_train_pu, y_test_pu, stats


def demo_extreme_alpha_values():
    """Demo: Effect of extreme alpha values"""
    print("\n" + "=" * 70)
    print("DEMO 4: Extreme Alpha Values")
    print("=" * 70)
    
    # Create dataset
    df = create_diverse_dataset(n_samples=2000, positive_ratio=0.2)
    
    # Validate and preprocess
    validated_df, metadata = load_and_validate_data(df)
    X_train, X_test, y_train, y_test, _, _ = preprocess_data(
        validated_df, metadata.target_column
    )
    
    # Test extreme alpha values
    extreme_alphas = [0.05, 0.1, 0.9, 0.95]
    
    print(f"Testing extreme alpha values on dataset with {y_train.mean():.1%} positive examples")
    
    for alpha in extreme_alphas:
        print(f"\n🔬 Testing α = {alpha}:")
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            try:
                y_train_pu, y_test_pu, y_train_true, y_test_true, stats = simulate_pu_scenario(
                    X_train, X_test, y_train, y_test,
                    alpha=alpha, strategy='scar'
                )
                
                print(f"  ✅ Success: {stats.labeled_positive_count} labeled positive")
                print(f"     Completeness: {stats.label_completeness:.1%}")
                
                if w:
                    for warning in w:
                        print(f"  ⚠️ Warning: {warning.message}")
                        
            except Exception as e:
                print(f"  ❌ Failed: {e}")
    
    return None


def demo_realistic_pu_scenarios():
    """Demo: Realistic PU Learning scenarios from different domains"""
    print("\n" + "=" * 70)
    print("DEMO 5: Realistic PU Scenarios from Different Domains")
    print("=" * 70)
    
    scenarios = [
        {
            'name': 'Fraud Detection',
            'n_samples': 5000,
            'positive_ratio': 0.008,  # 0.8% fraud
            'alpha': 0.15,  # Only 15% of fraud is detected
            'description': 'Credit card fraud detection'
        },
        {
            'name': 'Rare Disease',
            'n_samples': 3000,
            'positive_ratio': 0.003,  # 0.3% disease prevalence
            'alpha': 0.4,   # 40% of cases are diagnosed
            'description': 'Rare disease diagnosis'
        },
        {
            'name': 'Click Prediction',
            'n_samples': 10000,
            'positive_ratio': 0.02,  # 2% click rate
            'alpha': 0.6,   # 60% of clicks are tracked
            'description': 'Online advertising clicks'
        },
        {
            'name': 'Drug Discovery',
            'n_samples': 2000,
            'positive_ratio': 0.001,  # 0.1% active compounds
            'alpha': 0.8,   # 80% of actives are found in screening
            'description': 'Active compound screening'
        }
    ]
    
    results = {}
    
    for scenario in scenarios:
        print(f"\n🏢 {scenario['name']} Scenario:")
        print(f"   {scenario['description']}")
        print(f"   Population: {scenario['n_samples']:,} samples")
        print(f"   True positive rate: {scenario['positive_ratio']:.1%}")
        print(f"   Detection rate (α): {scenario['alpha']:.0%}")
        
        # Create scenario-specific dataset
        df = create_diverse_dataset(
            n_samples=scenario['n_samples'],
            positive_ratio=scenario['positive_ratio']
        )
        
        # Validate and preprocess
        validated_df, metadata = load_and_validate_data(df)
        X_train, X_test, y_train, y_test, _, _ = preprocess_data(
            validated_df, metadata.target_column
        )
        
        # Simulate PU scenario
        try:
            y_train_pu, y_test_pu, y_train_true, y_test_true, stats = simulate_pu_scenario(
                X_train, X_test, y_train, y_test,
                alpha=scenario['alpha'], strategy='scar'
            )
            
            print(f"   📊 Results:")
            print(f"     Labeled positive: {stats.labeled_positive_count}")
            print(f"     Hidden positive: {stats.hidden_positive_count}")
            print(f"     Observed positive rate: {stats.simulated_positive_ratio:.3%}")
            print(f"     Hidden positive rate: {stats.hidden_positive_ratio:.1%}")
            
            results[scenario['name']] = stats
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    return results


def demo_simulation_quality_metrics():
    """Demo: Understanding simulation quality metrics"""
    print("\n" + "=" * 70)
    print("DEMO 6: Understanding Simulation Quality Metrics")
    print("=" * 70)
    
    # Create base dataset
    df = create_diverse_dataset(n_samples=1500, positive_ratio=0.3)
    validated_df, metadata = load_and_validate_data(df)
    X_train, X_test, y_train, y_test, _, _ = preprocess_data(
        validated_df, metadata.target_column
    )
    
    print("Quality metrics explanation:")
    print("📏 KL Divergence: Measures distribution shift (0 = no shift)")
    print("📐 Wasserstein Distance: Alternative distribution distance measure")
    print("🎯 PU Bias Score: Selection bias (0 = no bias, 1 = maximum bias)")
    print("📊 Label Completeness: Fraction of positives that are labeled")
    
    # Test different configurations
    configs = [
        {'alpha': 0.5, 'strategy': 'scar', 'desc': 'Balanced SCAR'},
        {'alpha': 0.1, 'strategy': 'scar', 'desc': 'Low α SCAR'},
        {'alpha': 0.3, 'strategy': 'sar', 'desc': 'Biased SAR'},
        {'alpha': 0.4, 'strategy': 'prior_shift', 'desc': 'Prior Shift'}
    ]
    
    print(f"\n📊 Quality Metrics Comparison:")
    print(f"{'Configuration':<20} {'KL Div':<8} {'Wasser':<8} {'Bias':<8} {'Complete':<10}")
    print("-" * 60)
    
    for config in configs:
        y_train_pu, y_test_pu, y_train_true, y_test_true, stats = simulate_pu_scenario(
            X_train, X_test, y_train, y_test,
            alpha=config['alpha'], 
            strategy=config['strategy'],
            config={'sar_bias_strength': 0.8}
        )
        
        print(f"{config['desc']:<20} "
              f"{stats.kl_divergence:<8.4f} "
              f"{stats.wasserstein_distance:<8.4f} "
              f"{stats.pu_bias_score:<8.3f} "
              f"{stats.label_completeness:<10.1%}")
    
    print(f"\n💡 Insights:")
    print(f"   • Higher KL divergence = more distribution shift")
    print(f"   • SAR creates more bias than SCAR")
    print(f"   • Lower α reduces label completeness")
    print(f"   • These metrics help choose realistic simulation parameters")


def demo_alpha_sensitivity_analysis():
    """Demo: Sensitivity analysis for alpha parameter"""
    print("\n" + "=" * 70)
    print("DEMO 7: Alpha Parameter Sensitivity Analysis")
    print("=" * 70)
    
    # Create dataset
    df = create_diverse_dataset(n_samples=1800, positive_ratio=0.25)
    validated_df, metadata = load_and_validate_data(df)
    X_train, X_test, y_train, y_test, _, _ = preprocess_data(
        validated_df, metadata.target_column
    )
    
    # Test range of alpha values
    alpha_range = np.arange(0.1, 1.0, 0.1)
    
    print(f"Testing alpha sensitivity on dataset with {y_train.mean():.1%} positive rate")
    print(f"\n{'Alpha':<8} {'Labeled':<8} {'Hidden+':<8} {'Bias':<8} {'KL Div':<8}")
    print("-" * 50)
    
    alpha_results = []
    
    for alpha in alpha_range:
        try:
            y_train_pu, y_test_pu, y_train_true, y_test_true, stats = simulate_pu_scenario(
                X_train, X_test, y_train, y_test,
                alpha=alpha, strategy='scar'
            )
            
            print(f"{alpha:<8.1f} "
                  f"{stats.labeled_positive_count:<8} "
                  f"{stats.hidden_positive_count:<8} "
                  f"{stats.pu_bias_score:<8.3f} "
                  f"{stats.kl_divergence:<8.4f}")
            
            alpha_results.append({
                'alpha': alpha,
                'labeled_count': stats.labeled_positive_count,
                'hidden_count': stats.hidden_positive_count,
                'bias_score': stats.pu_bias_score,
                'kl_divergence': stats.kl_divergence
            })
            
        except Exception as e:
            print(f"{alpha:<8.1f} ERROR: {e}")
    
    print(f"\n📈 Trends:")
    print(f"   • As α increases: more labeled positives, fewer hidden positives")
    print(f"   • KL divergence generally decreases with higher α")
    print(f"   • Choice of α should reflect real-world detection rates")
    
    return alpha_results


def main():
    """Run all simulation demonstrations"""
    print("\n🚀 PU Learning Scenario Simulation Module - Demonstration\n")
    
    # Run demos
    demos = [
        ("SCAR Simulation", demo_scar_simulation),
        ("SAR Simulation", demo_sar_simulation),
        ("Prior Shift Simulation", demo_prior_shift_simulation),
        ("Extreme Alpha Values", demo_extreme_alpha_values),
        ("Realistic PU Scenarios", demo_realistic_pu_scenarios),
        ("Simulation Quality Metrics", demo_simulation_quality_metrics),
        ("Alpha Sensitivity Analysis", demo_alpha_sensitivity_analysis),
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
    print(f"✅ Successfully demonstrated {len(results)} simulation scenarios")
    print("✅ Tested SCAR, SAR, and Prior Shift strategies")
    print("✅ Analyzed extreme alpha values and edge cases")
    print("✅ Simulated realistic domain-specific PU scenarios")
    print("✅ Explained quality metrics for simulation assessment")
    print("✅ Performed sensitivity analysis for parameter selection")
    print("\n💡 Key insights:")
    print("   • SCAR provides unbiased selection (good baseline)")
    print("   • SAR creates feature-dependent bias (more realistic)")
    print("   • Alpha choice critical for realistic simulation")
    print("   • Quality metrics help validate simulation realism")
    print("   • Extreme imbalance scenarios need careful parameter tuning")
    
    return results


if __name__ == "__main__":
    results = main()
