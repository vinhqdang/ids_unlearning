"""
Comprehensive comparison test of AMDU vs existing methods
"""
import sys
sys.path.append('src')

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import time

from data_loader import NSLKDDDataLoader
from data_poisoner import DataPoisoner
from unlearning_algorithms import SISAUnlearner, GradientAscentUnlearner
from amdu_unlearning import AMDUUnlearner

def run_amdu_comparison():
    print("=" * 80)
    print("AMDU vs EXISTING METHODS - COMPREHENSIVE COMPARISON")
    print("=" * 80)
    
    # Load data
    data_loader = NSLKDDDataLoader()
    train_df, test_df = data_loader.load_data()
    X_train_full, X_test, y_train_full, y_test = data_loader.preprocess_data(
        train_df, test_df, binary_classification=True
    )
    
    # Use manageable subset
    subset_size = 3000
    test_size = 1000
    
    indices = np.random.choice(len(X_train_full), subset_size, replace=False)
    test_indices = np.random.choice(len(X_test), test_size, replace=False)
    
    X_train = X_train_full[indices]
    y_train = y_train_full[indices]
    X_test_subset = X_test[test_indices]
    y_test_subset = y_test[test_indices]
    
    print(f"Dataset: {X_train.shape[0]} train, {X_test_subset.shape[0]} test")
    
    # Test scenarios
    scenarios = [
        {'noise_type': 'label_noise', 'noise_ratio': 0.10, 'name': '10% Label Noise'},
        {'noise_type': 'label_noise', 'noise_ratio': 0.20, 'name': '20% Label Noise'},
        {'noise_type': 'feature_noise', 'noise_ratio': 0.15, 'name': '15% Feature Noise'}
    ]
    
    poisoner = DataPoisoner(random_state=42)
    base_model = LogisticRegression(random_state=42, max_iter=500)
    
    results = []
    
    for scenario in scenarios:
        print(f"\\n{'='*60}")
        print(f"SCENARIO: {scenario['name']}")
        print(f"{'='*60}")
        
        # Create poisoned data
        if scenario['noise_type'] == 'label_noise':
            X_poisoned, y_poisoned, poison_indices = poisoner.add_label_noise(
                X_train, y_train, noise_ratio=scenario['noise_ratio']
            )
        else:
            X_poisoned, y_poisoned, poison_indices = poisoner.add_feature_noise(
                X_train, y_train, noise_ratio=scenario['noise_ratio'], noise_strength=1.0
            )
        
        remaining_indices = [i for i in range(len(X_poisoned)) if i not in poison_indices]
        
        # Baseline (clean)
        base_model.fit(X_train, y_train)
        baseline_pred = base_model.predict(X_test_subset)
        baseline_f1 = f1_score(y_test_subset, baseline_pred)
        baseline_auc = roc_auc_score(y_test_subset, base_model.predict_proba(X_test_subset)[:, 1])
        
        # Poisoned
        poisoned_model = LogisticRegression(random_state=42, max_iter=500)
        poisoned_model.fit(X_poisoned, y_poisoned)
        poisoned_pred = poisoned_model.predict(X_test_subset)
        poisoned_f1 = f1_score(y_test_subset, poisoned_pred)
        poisoned_auc = roc_auc_score(y_test_subset, poisoned_model.predict_proba(X_test_subset)[:, 1])
        
        print(f"Baseline F1: {baseline_f1:.4f}, AUC: {baseline_auc:.4f}")
        print(f"Poisoned F1: {poisoned_f1:.4f}, AUC: {poisoned_auc:.4f}")
        print(f"Performance Drop: {baseline_f1 - poisoned_f1:.4f}")
        
        unlearning_results = {}
        
        # 1. Simple Retraining
        print("\\nTesting Simple Retraining...")
        start_time = time.time()
        retrain_model = LogisticRegression(random_state=42, max_iter=500)
        retrain_model.fit(X_poisoned[remaining_indices], y_poisoned[remaining_indices])
        retrain_time = time.time() - start_time
        
        retrain_pred = retrain_model.predict(X_test_subset)
        retrain_f1 = f1_score(y_test_subset, retrain_pred)
        retrain_auc = roc_auc_score(y_test_subset, retrain_model.predict_proba(X_test_subset)[:, 1])
        
        unlearning_results['Retraining'] = {
            'f1': retrain_f1, 'auc': retrain_auc, 'time': retrain_time,
            'recovery': retrain_f1 - poisoned_f1
        }
        
        # 2. SISA Unlearning
        print("Testing SISA Unlearning...")
        try:
            start_time = time.time()
            sisa_model = SISAUnlearner(base_model, n_shards=3, random_state=42)
            sisa_model.fit(X_poisoned, y_poisoned)
            sisa_model.unlearn(X_poisoned, y_poisoned, poison_indices)
            sisa_time = time.time() - start_time
            
            sisa_pred = sisa_model.predict(X_test_subset)
            sisa_f1 = f1_score(y_test_subset, sisa_pred)
            sisa_proba = sisa_model.predict_proba(X_test_subset)
            sisa_auc = roc_auc_score(y_test_subset, sisa_proba[:, 1])
            
            unlearning_results['SISA'] = {
                'f1': sisa_f1, 'auc': sisa_auc, 'time': sisa_time,
                'recovery': sisa_f1 - poisoned_f1
            }
        except Exception as e:
            print(f"SISA failed: {e}")
            unlearning_results['SISA'] = {'f1': 0, 'auc': 0, 'time': 0, 'recovery': 0}
        
        # 3. Gradient Ascent
        print("Testing Gradient Ascent...")
        try:
            start_time = time.time()
            ga_model = GradientAscentUnlearner(base_model, learning_rate=0.01, max_iterations=20)
            ga_model.fit(X_poisoned, y_poisoned)
            
            X_forget = X_poisoned[poison_indices]
            y_forget = y_poisoned[poison_indices]
            X_retain = X_poisoned[remaining_indices] if remaining_indices else None
            y_retain = y_poisoned[remaining_indices] if remaining_indices else None
            
            ga_model.unlearn(X_forget, y_forget, X_retain, y_retain)
            ga_time = time.time() - start_time
            
            ga_pred = ga_model.predict(X_test_subset)
            ga_f1 = f1_score(y_test_subset, ga_pred)
            ga_auc = roc_auc_score(y_test_subset, ga_model.predict_proba(X_test_subset)[:, 1])
            
            unlearning_results['Gradient Ascent'] = {
                'f1': ga_f1, 'auc': ga_auc, 'time': ga_time,
                'recovery': ga_f1 - poisoned_f1
            }
        except Exception as e:
            print(f"Gradient Ascent failed: {e}")
            unlearning_results['Gradient Ascent'] = {'f1': 0, 'auc': 0, 'time': 0, 'recovery': 0}
        
        # 4. AMDU (Our Novel Algorithm)
        print("Testing AMDU (Novel Algorithm)...")
        try:
            start_time = time.time()
            amdu_model = AMDUUnlearner(
                input_dim=X_poisoned.shape[1],
                hidden_dim=64,
                memory_dim=32,
                learning_rate=0.002,
                device='cpu'
            )
            
            amdu_model.fit(X_poisoned, y_poisoned, teacher_model=poisoned_model)
            amdu_model.unlearn(X_poisoned, y_poisoned, poison_indices, remaining_indices)
            amdu_time = time.time() - start_time
            
            amdu_pred = amdu_model.predict(X_test_subset)
            amdu_proba = amdu_model.predict_proba(X_test_subset)
            amdu_f1 = f1_score(y_test_subset, amdu_pred)
            amdu_auc = roc_auc_score(y_test_subset, amdu_proba[:, 1])
            
            # Additional AMDU metrics
            forgetting_effectiveness = 0
            if len(poison_indices) > 10:
                X_forget_sample = X_poisoned[poison_indices[:10]]
                forgetting_effectiveness = amdu_model.evaluate_forgetting_effectiveness(X_forget_sample)
            
            unlearning_results['AMDU'] = {
                'f1': amdu_f1, 'auc': amdu_auc, 'time': amdu_time,
                'recovery': amdu_f1 - poisoned_f1,
                'forgetting_effectiveness': forgetting_effectiveness
            }
        except Exception as e:
            print(f"AMDU failed: {e}")
            import traceback
            traceback.print_exc()
            unlearning_results['AMDU'] = {'f1': 0, 'auc': 0, 'time': 0, 'recovery': 0}
        
        # Print results for this scenario
        print(f"\\nResults for {scenario['name']}:")
        print(f"{'Method':<18} {'F1-Score':<10} {'Recovery':<10} {'AUC':<10} {'Time(s)':<10}")
        print("-" * 65)
        
        for method, metrics in unlearning_results.items():
            print(f"{method:<18} {metrics['f1']:<10.4f} {metrics['recovery']:<+10.4f} "
                  f"{metrics['auc']:<10.4f} {metrics['time']:<10.2f}")
        
        # Store for overall analysis
        for method, metrics in unlearning_results.items():
            results.append({
                'Scenario': scenario['name'],
                'Method': method,
                'Baseline_F1': baseline_f1,
                'Poisoned_F1': poisoned_f1,
                'F1_Score': metrics['f1'],
                'Recovery': metrics['recovery'],
                'AUC': metrics['auc'],
                'Time': metrics['time'],
                'Forgetting_Effectiveness': metrics.get('forgetting_effectiveness', None)
            })
    
    # Overall Analysis
    print(f"\\n{'='*80}")
    print("OVERALL ANALYSIS")
    print(f"{'='*80}")
    
    results_df = pd.DataFrame(results)
    
    # Compare methods across all scenarios
    method_summary = results_df.groupby('Method').agg({
        'F1_Score': ['mean', 'std'],
        'Recovery': ['mean', 'std'],
        'AUC': ['mean', 'std'],
        'Time': ['mean', 'std']
    }).round(4)
    
    print("\\nMethod Performance Summary (Mean ± Std):")
    print(method_summary)
    
    # Find best methods
    avg_f1 = results_df.groupby('Method')['F1_Score'].mean().sort_values(ascending=False)
    avg_recovery = results_df.groupby('Method')['Recovery'].mean().sort_values(ascending=False)
    
    print(f"\\nBest Methods by F1-Score:")
    for i, (method, f1) in enumerate(avg_f1.head(3).items(), 1):
        print(f"{i}. {method}: {f1:.4f}")
    
    print(f"\\nBest Methods by Recovery:")
    for i, (method, recovery) in enumerate(avg_recovery.head(3).items(), 1):
        print(f"{i}. {method}: {recovery:+.4f}")
    
    # AMDU specific analysis
    if 'AMDU' in results_df['Method'].values:
        amdu_results = results_df[results_df['Method'] == 'AMDU']
        amdu_wins = sum(amdu_results['F1_Score'] > amdu_results['Poisoned_F1'])
        total_tests = len(amdu_results)
        
        print(f"\\nAMDU Performance:")
        print(f"- Scenarios where AMDU > Poisoned: {amdu_wins}/{total_tests}")
        print(f"- Average Recovery: {amdu_results['Recovery'].mean():+.4f}")
        print(f"- Average F1-Score: {amdu_results['F1_Score'].mean():.4f}")
        
        if 'Forgetting_Effectiveness' in amdu_results.columns:
            avg_forgetting = amdu_results['Forgetting_Effectiveness'].dropna().mean()
            if not np.isnan(avg_forgetting):
                print(f"- Average Forgetting Effectiveness: {avg_forgetting:.4f}")
    
    # Save results
    results_df.to_csv('results/amdu_comparison_results.csv', index=False)
    print(f"\\nDetailed results saved to: results/amdu_comparison_results.csv")
    
    # Create visualization
    create_comparison_plot(results_df)
    
    return results_df

def create_comparison_plot(results_df):
    """Create comparison visualization"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: F1-Score comparison
    ax1 = axes[0, 0]
    methods = results_df['Method'].unique()
    scenarios = results_df['Scenario'].unique()
    
    x_pos = np.arange(len(scenarios))
    width = 0.15
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    
    for i, method in enumerate(methods):
        method_data = results_df[results_df['Method'] == method]
        f1_scores = [method_data[method_data['Scenario'] == s]['F1_Score'].iloc[0] if len(method_data[method_data['Scenario'] == s]) > 0 else 0 for s in scenarios]
        
        ax1.bar(x_pos + i * width, f1_scores, width, label=method, color=colors[i], alpha=0.8)
    
    ax1.set_xlabel('Scenarios')
    ax1.set_ylabel('F1-Score')
    ax1.set_title('F1-Score Comparison Across Methods')
    ax1.set_xticks(x_pos + width * len(methods) / 2)
    ax1.set_xticklabels([s.replace(' ', '\\n') for s in scenarios])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Recovery comparison
    ax2 = axes[0, 1]
    for i, method in enumerate(methods):
        method_data = results_df[results_df['Method'] == method]
        recoveries = [method_data[method_data['Scenario'] == s]['Recovery'].iloc[0] if len(method_data[method_data['Scenario'] == s]) > 0 else 0 for s in scenarios]
        
        ax2.bar(x_pos + i * width, recoveries, width, label=method, color=colors[i], alpha=0.8)
    
    ax2.set_xlabel('Scenarios')
    ax2.set_ylabel('Performance Recovery')
    ax2.set_title('Recovery Comparison Across Methods')
    ax2.set_xticks(x_pos + width * len(methods) / 2)
    ax2.set_xticklabels([s.replace(' ', '\\n') for s in scenarios])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Plot 3: Time comparison
    ax3 = axes[1, 0]
    avg_times = results_df.groupby('Method')['Time'].mean()
    colors_dict = dict(zip(methods, colors))
    
    bars = ax3.bar(avg_times.index, avg_times.values, 
                   color=[colors_dict[method] for method in avg_times.index], alpha=0.8)
    
    ax3.set_ylabel('Average Time (seconds)')
    ax3.set_title('Training Time Comparison')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: AMDU specific metrics
    ax4 = axes[1, 1]
    if 'AMDU' in results_df['Method'].values:
        amdu_data = results_df[results_df['Method'] == 'AMDU']
        
        scenarios_amdu = amdu_data['Scenario'].values
        f1_scores_amdu = amdu_data['F1_Score'].values
        recoveries_amdu = amdu_data['Recovery'].values
        
        x_pos_amdu = np.arange(len(scenarios_amdu))
        
        ax4_twin = ax4.twinx()
        
        bars1 = ax4.bar(x_pos_amdu - 0.2, f1_scores_amdu, 0.4, 
                       label='F1-Score', color='blue', alpha=0.7)
        bars2 = ax4_twin.bar(x_pos_amdu + 0.2, recoveries_amdu, 0.4, 
                            label='Recovery', color='red', alpha=0.7)
        
        ax4.set_xlabel('Scenarios')
        ax4.set_ylabel('F1-Score', color='blue')
        ax4_twin.set_ylabel('Recovery', color='red')
        ax4.set_title('AMDU Performance Analysis')
        ax4.set_xticks(x_pos_amdu)
        ax4.set_xticklabels([s.replace(' ', '\\n') for s in scenarios_amdu])
        
        # Add legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('results/amdu_vs_existing_methods.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Comparison visualization saved to: results/amdu_vs_existing_methods.png")

if __name__ == "__main__":
    results = run_amdu_comparison()