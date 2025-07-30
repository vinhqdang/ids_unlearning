"""
Test more aggressive poisoning scenarios to demonstrate unlearning effectiveness
"""
import sys
sys.path.append('src')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

from data_loader import NSLKDDDataLoader
from data_poisoner import DataPoisoner
from unlearning_algorithms import SISAUnlearner, GradientAscentUnlearner

def test_aggressive_scenarios():
    print("=" * 70)
    print("AGGRESSIVE POISONING SCENARIOS - UNLEARNING EFFECTIVENESS")
    print("=" * 70)
    
    # Load smaller dataset for speed
    data_loader = NSLKDDDataLoader()
    train_df, test_df = data_loader.load_data()
    X_train_full, X_test, y_train_full, y_test = data_loader.preprocess_data(
        train_df, test_df, binary_classification=True
    )
    
    # Use subset
    subset_size = 5000
    indices = np.random.choice(len(X_train_full), subset_size, replace=False)
    X_train = X_train_full[indices]
    y_train = y_train_full[indices]
    
    print(f"Using {subset_size} training samples for aggressive testing")
    
    # Test with high noise ratios
    noise_ratios = [0.20, 0.30, 0.40]  # 20%, 30%, 40% noise
    model = LogisticRegression(random_state=42, max_iter=500)
    poisoner = DataPoisoner(random_state=42)
    
    results = []
    
    # Baseline
    model.fit(X_train, y_train)
    baseline_f1 = f1_score(y_test, model.predict(X_test))
    baseline_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    print(f"\nBaseline Performance:")
    print(f"  F1-Score: {baseline_f1:.4f}")
    print(f"  ROC-AUC: {baseline_auc:.4f}")
    
    for noise_ratio in noise_ratios:
        print(f"\n{'='*50}")
        print(f"TESTING WITH {noise_ratio*100:.0f}% LABEL NOISE")
        print(f"{'='*50}")
        
        # Create heavily poisoned data
        X_train_poisoned, y_train_poisoned, poison_indices = poisoner.add_label_noise(
            X_train, y_train, noise_ratio=noise_ratio
        )
        
        # Train on poisoned data
        poisoned_model = LogisticRegression(random_state=42, max_iter=500)
        poisoned_model.fit(X_train_poisoned, y_train_poisoned)
        
        poisoned_f1 = f1_score(y_test, poisoned_model.predict(X_test))
        poisoned_auc = roc_auc_score(y_test, poisoned_model.predict_proba(X_test)[:, 1])
        
        print(f"Poisoned Performance:")
        print(f"  F1-Score: {poisoned_f1:.4f} (drop: {baseline_f1-poisoned_f1:.4f})")
        print(f"  ROC-AUC: {poisoned_auc:.4f} (drop: {baseline_auc-poisoned_auc:.4f})")
        
        # Test unlearning methods
        unlearning_results = {}
        
        # 1. Simple Retraining
        remaining_indices = np.setdiff1d(np.arange(len(X_train_poisoned)), poison_indices)
        retrain_model = LogisticRegression(random_state=42, max_iter=500)
        retrain_model.fit(X_train_poisoned[remaining_indices], y_train_poisoned[remaining_indices])
        
        retrain_f1 = f1_score(y_test, retrain_model.predict(X_test))
        retrain_auc = roc_auc_score(y_test, retrain_model.predict_proba(X_test)[:, 1])
        unlearning_results['Retraining'] = {'f1': retrain_f1, 'auc': retrain_auc}
        
        # 2. SISA Unlearning
        try:
            sisa_model = SISAUnlearner(model, n_shards=3, random_state=42)
            sisa_model.fit(X_train_poisoned, y_train_poisoned)
            sisa_model.unlearn(X_train_poisoned, y_train_poisoned, poison_indices)
            
            sisa_f1 = f1_score(y_test, sisa_model.predict(X_test))
            sisa_auc = roc_auc_score(y_test, sisa_model.predict_proba(X_test)[:, 1])
            unlearning_results['SISA'] = {'f1': sisa_f1, 'auc': sisa_auc}
        except:
            unlearning_results['SISA'] = {'f1': 0, 'auc': 0}
        
        # 3. Gradient Ascent
        try:
            ga_model = GradientAscentUnlearner(model, learning_rate=0.02, max_iterations=30)
            ga_model.fit(X_train_poisoned, y_train_poisoned)
            
            X_forget = X_train_poisoned[poison_indices]
            y_forget = y_train_poisoned[poison_indices]
            X_retain = X_train_poisoned[remaining_indices]
            y_retain = y_train_poisoned[remaining_indices]
            
            ga_model.unlearn(X_forget, y_forget, X_retain, y_retain)
            
            ga_f1 = f1_score(y_test, ga_model.predict(X_test))
            ga_auc = roc_auc_score(y_test, ga_model.predict_proba(X_test)[:, 1])
            unlearning_results['Gradient Ascent'] = {'f1': ga_f1, 'auc': ga_auc}
        except:
            unlearning_results['Gradient Ascent'] = {'f1': 0, 'auc': 0}
        
        print(f"\nUnlearning Results:")
        print(f"{'Method':<15} {'F1-Score':<10} {'Recovery':<12} {'ROC-AUC':<10} {'AUC Recovery':<12}")
        print("-" * 65)
        
        for method, metrics in unlearning_results.items():
            f1_recovery = metrics['f1'] - poisoned_f1
            auc_recovery = metrics['auc'] - poisoned_auc
            
            print(f"{method:<15} {metrics['f1']:<10.4f} {f1_recovery:<+12.4f} "
                  f"{metrics['auc']:<10.4f} {auc_recovery:<+12.4f}")
        
        # Store for plotting
        results.append({
            'noise_ratio': noise_ratio,
            'baseline_f1': baseline_f1,
            'poisoned_f1': poisoned_f1,
            'unlearning': unlearning_results
        })
    
    # Create visualization
    create_aggressive_visualization(results)
    
    return results

def create_aggressive_visualization(results):
    """Create visualization for aggressive poisoning results"""
    
    noise_ratios = [r['noise_ratio'] * 100 for r in results]
    baseline_f1s = [r['baseline_f1'] for r in results]
    poisoned_f1s = [r['poisoned_f1'] for r in results]
    
    # Extract unlearning results
    retrain_f1s = [r['unlearning']['Retraining']['f1'] for r in results]
    sisa_f1s = [r['unlearning']['SISA']['f1'] for r in results]
    ga_f1s = [r['unlearning']['Gradient Ascent']['f1'] for r in results]
    
    plt.figure(figsize=(12, 8))
    
    # Plot 1: F1-Score comparison
    plt.subplot(2, 2, 1)
    plt.plot(noise_ratios, baseline_f1s, 'k--', label='Baseline', linewidth=2, marker='s')
    plt.plot(noise_ratios, poisoned_f1s, 'r-', label='Poisoned', linewidth=2, marker='o')
    plt.plot(noise_ratios, retrain_f1s, 'b-', label='Retraining', linewidth=2, marker='^')
    plt.plot(noise_ratios, sisa_f1s, 'g-', label='SISA', linewidth=2, marker='D')
    plt.plot(noise_ratios, ga_f1s, 'm-', label='Gradient Ascent', linewidth=2, marker='*')
    
    plt.xlabel('Noise Ratio (%)')
    plt.ylabel('F1-Score')
    plt.title('F1-Score vs Noise Ratio')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Plot 2: Recovery effectiveness
    plt.subplot(2, 2, 2)
    retrain_recovery = [(retrain_f1s[i] - poisoned_f1s[i]) / (baseline_f1s[i] - poisoned_f1s[i]) * 100 
                       if baseline_f1s[i] != poisoned_f1s[i] else 0 for i in range(len(results))]
    sisa_recovery = [(sisa_f1s[i] - poisoned_f1s[i]) / (baseline_f1s[i] - poisoned_f1s[i]) * 100 
                    if baseline_f1s[i] != poisoned_f1s[i] else 0 for i in range(len(results))]
    ga_recovery = [(ga_f1s[i] - poisoned_f1s[i]) / (baseline_f1s[i] - poisoned_f1s[i]) * 100 
                  if baseline_f1s[i] != poisoned_f1s[i] else 0 for i in range(len(results))]
    
    plt.plot(noise_ratios, retrain_recovery, 'b-', label='Retraining', linewidth=2, marker='^')
    plt.plot(noise_ratios, sisa_recovery, 'g-', label='SISA', linewidth=2, marker='D')
    plt.plot(noise_ratios, ga_recovery, 'm-', label='Gradient Ascent', linewidth=2, marker='*')
    
    plt.axhline(y=100, color='black', linestyle='--', alpha=0.5, label='Full Recovery')
    plt.xlabel('Noise Ratio (%)')
    plt.ylabel('Recovery Rate (%)')
    plt.title('Unlearning Recovery Effectiveness')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Plot 3: Performance drop
    plt.subplot(2, 2, 3)
    performance_drops = [(baseline_f1s[i] - poisoned_f1s[i]) * 100 for i in range(len(results))]
    
    plt.bar(noise_ratios, performance_drops, alpha=0.7, color='red')
    plt.xlabel('Noise Ratio (%)')
    plt.ylabel('Performance Drop (%)')
    plt.title('Performance Impact of Poisoning')
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 4: Best unlearning method comparison
    plt.subplot(2, 2, 4)
    methods = ['Retraining', 'SISA', 'Gradient Ascent']
    colors = ['blue', 'green', 'magenta']
    
    x_pos = np.arange(len(methods))
    
    # Average performance across all noise levels
    avg_retrain = np.mean(retrain_f1s)
    avg_sisa = np.mean(sisa_f1s)
    avg_ga = np.mean(ga_f1s)
    
    avg_performances = [avg_retrain, avg_sisa, avg_ga]
    
    plt.bar(x_pos, avg_performances, color=colors, alpha=0.7)
    plt.axhline(y=np.mean(baseline_f1s), color='black', linestyle='--', label='Baseline')
    plt.axhline(y=np.mean(poisoned_f1s), color='red', linestyle='--', label='Poisoned')
    
    plt.xlabel('Unlearning Method')
    plt.ylabel('Average F1-Score')
    plt.title('Average Performance Comparison')
    plt.xticks(x_pos, methods)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/aggressive_poisoning_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved to: results/aggressive_poisoning_analysis.png")

if __name__ == "__main__":
    results = test_aggressive_scenarios()