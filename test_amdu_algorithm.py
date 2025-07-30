"""
Test and Compare AMDU Algorithm against existing unlearning methods
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

def comprehensive_amdu_test():
    print("=" * 80)
    print("AMDU ALGORITHM COMPREHENSIVE EVALUATION")
    print("=" * 80)
    
    # Load data (smaller subset for detailed analysis)
    print("\n1. Loading NSL-KDD Dataset...")
    data_loader = NSLKDDDataLoader()
    train_df, test_df = data_loader.load_data()
    X_train_full, X_test, y_train_full, y_test = data_loader.preprocess_data(
        train_df, test_df, binary_classification=True
    )
    
    # Use subset for comprehensive analysis
    subset_size = 8000
    test_subset_size = 3000
    
    indices = np.random.choice(len(X_train_full), subset_size, replace=False)
    X_train = X_train_full[indices]
    y_train = y_train_full[indices]
    
    test_indices = np.random.choice(len(X_test), test_subset_size, replace=False)
    X_test = X_test[test_indices]
    y_test = y_test[test_indices]
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Initialize base models and poisoner
    base_models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=500),
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    }
    
    poisoner = DataPoisoner(random_state=42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test different noise scenarios
    noise_scenarios = [
        {'type': 'label_noise', 'ratio': 0.10, 'name': '10% Label Noise'},
        {'type': 'label_noise', 'ratio': 0.20, 'name': '20% Label Noise'},
        {'type': 'feature_noise', 'ratio': 0.15, 'name': '15% Feature Noise'}
    ]
    
    results = {}
    
    for model_name, base_model in base_models.items():
        print(f"\n{'='*60}")
        print(f"TESTING WITH BASE MODEL: {model_name}")
        print(f"{'='*60}")
        
        results[model_name] = {}
        
        # Baseline performance
        print("\\n2. Baseline Performance...")
        base_model.fit(X_train, y_train)
        baseline_pred = base_model.predict(X_test)
        baseline_metrics = {
            'accuracy': accuracy_score(y_test, baseline_pred),
            'f1_score': f1_score(y_test, baseline_pred),
        }
        
        if hasattr(base_model, 'predict_proba'):
            baseline_proba = base_model.predict_proba(X_test)
            baseline_metrics['roc_auc'] = roc_auc_score(y_test, baseline_proba[:, 1])
        
        print(f"   Baseline - Accuracy: {baseline_metrics['accuracy']:.4f}, "
              f"F1: {baseline_metrics['f1_score']:.4f}, "
              f"ROC-AUC: {baseline_metrics.get('roc_auc', 'N/A')}")
        
        for scenario in noise_scenarios:
            scenario_name = scenario['name']
            print(f"\\n3. Testing {scenario_name}...")
            
            # Create poisoned data
            if scenario['type'] == 'label_noise':
                X_train_poisoned, y_train_poisoned, poison_indices = poisoner.add_label_noise(
                    X_train, y_train, noise_ratio=scenario['ratio']
                )
            else:  # feature_noise
                X_train_poisoned, y_train_poisoned, poison_indices = poisoner.add_feature_noise(
                    X_train, y_train, noise_ratio=scenario['ratio'], noise_strength=1.0
                )
            
            # Train on poisoned data
            poisoned_model = base_model.__class__(**base_model.get_params())
            poisoned_model.fit(X_train_poisoned, y_train_poisoned)
            
            poisoned_pred = poisoned_model.predict(X_test)
            poisoned_metrics = {
                'accuracy': accuracy_score(y_test, poisoned_pred),
                'f1_score': f1_score(y_test, poisoned_pred)
            }
            
            if hasattr(poisoned_model, 'predict_proba'):
                poisoned_proba = poisoned_model.predict_proba(X_test)
                poisoned_metrics['roc_auc'] = roc_auc_score(y_test, poisoned_proba[:, 1])
            
            print(f"   Poisoned - Accuracy: {poisoned_metrics['accuracy']:.4f}, "
                  f"F1: {poisoned_metrics['f1_score']:.4f}")
            
            # Test unlearning methods
            unlearning_results = test_all_unlearning_methods(
                base_model, poisoned_model, X_train_poisoned, y_train_poisoned, 
                X_test, y_test, poison_indices, device
            )
            
            # Store results
            results[model_name][scenario_name] = {
                'baseline': baseline_metrics,
                'poisoned': poisoned_metrics,
                'unlearning': unlearning_results
            }
    
    # Generate comprehensive analysis
    print(f"\\n{'='*80}")
    print("COMPREHENSIVE RESULTS ANALYSIS")
    print(f"{'='*80}")
    
    analyze_amdu_performance(results)
    create_amdu_visualizations(results)
    
    return results

def test_all_unlearning_methods(base_model, poisoned_model, X_train_poisoned, y_train_poisoned, 
                               X_test, y_test, poison_indices, device):
    """Test all unlearning methods including AMDU"""
    
    remaining_indices = [i for i in range(len(X_train_poisoned)) if i not in poison_indices]
    unlearning_results = {}
    
    print("   Testing Unlearning Methods:")
    
    # 1. Simple Retraining
    print("     - Simple Retraining...")
    start_time = time.time()
    
    retrain_model = base_model.__class__(**base_model.get_params())
    retrain_model.fit(X_train_poisoned[remaining_indices], y_train_poisoned[remaining_indices])
    
    retrain_time = time.time() - start_time
    retrain_pred = retrain_model.predict(X_test)
    
    unlearning_results['Retraining'] = {
        'accuracy': accuracy_score(y_test, retrain_pred),
        'f1_score': f1_score(y_test, retrain_pred),
        'time': retrain_time
    }
    
    if hasattr(retrain_model, 'predict_proba'):
        retrain_proba = retrain_model.predict_proba(X_test)
        unlearning_results['Retraining']['roc_auc'] = roc_auc_score(y_test, retrain_proba[:, 1])
    
    # 2. SISA Unlearning
    print("     - SISA Unlearning...")
    start_time = time.time()
    
    try:
        sisa_model = SISAUnlearner(base_model, n_shards=3, random_state=42)
        sisa_model.fit(X_train_poisoned, y_train_poisoned)
        sisa_model.unlearn(X_train_poisoned, y_train_poisoned, poison_indices)
        
        sisa_time = time.time() - start_time
        sisa_pred = sisa_model.predict(X_test)
        
        unlearning_results['SISA'] = {
            'accuracy': accuracy_score(y_test, sisa_pred),
            'f1_score': f1_score(y_test, sisa_pred),
            'time': sisa_time
        }
        
        try:
            sisa_proba = sisa_model.predict_proba(X_test)
            unlearning_results['SISA']['roc_auc'] = roc_auc_score(y_test, sisa_proba[:, 1])
        except:
            pass
            
    except Exception as e:
        print(f"       SISA failed: {str(e)}")
        unlearning_results['SISA'] = {'accuracy': 0, 'f1_score': 0, 'time': 0}
    
    # 3. Gradient Ascent (for LogisticRegression only)
    if isinstance(base_model, LogisticRegression):
        print("     - Gradient Ascent Unlearning...")
        start_time = time.time()
        
        try:
            ga_model = GradientAscentUnlearner(base_model, learning_rate=0.01, max_iterations=30)
            ga_model.fit(X_train_poisoned, y_train_poisoned)
            
            X_forget = X_train_poisoned[poison_indices]
            y_forget = y_train_poisoned[poison_indices]
            X_retain = X_train_poisoned[remaining_indices] if remaining_indices else None
            y_retain = y_train_poisoned[remaining_indices] if remaining_indices else None
            
            ga_model.unlearn(X_forget, y_forget, X_retain, y_retain)
            
            ga_time = time.time() - start_time
            ga_pred = ga_model.predict(X_test)
            
            unlearning_results['Gradient Ascent'] = {
                'accuracy': accuracy_score(y_test, ga_pred),
                'f1_score': f1_score(y_test, ga_pred),
                'time': ga_time
            }
            
            try:
                ga_proba = ga_model.predict_proba(X_test)
                unlearning_results['Gradient Ascent']['roc_auc'] = roc_auc_score(y_test, ga_proba[:, 1])
            except:
                pass
                
        except Exception as e:
            print(f"       Gradient Ascent failed: {str(e)}")
            unlearning_results['Gradient Ascent'] = {'accuracy': 0, 'f1_score': 0, 'time': 0}
    
    # 4. AMDU (Our Novel Algorithm)
    print("     - AMDU (Novel Algorithm)...")
    start_time = time.time()
    
    try:
        # Initialize AMDU
        amdu_model = AMDUUnlearner(
            input_dim=X_train_poisoned.shape[1],
            hidden_dim=128,
            memory_dim=64,
            learning_rate=0.001,
            device=device
        )
        
        # Fit AMDU with teacher model
        amdu_model.fit(X_train_poisoned, y_train_poisoned, teacher_model=poisoned_model)
        
        # Perform unlearning
        amdu_model.unlearn(X_train_poisoned, y_train_poisoned, poison_indices, remaining_indices)
        
        amdu_time = time.time() - start_time
        amdu_pred = amdu_model.predict(X_test)
        amdu_proba = amdu_model.predict_proba(X_test)
        
        # Additional AMDU-specific metrics
        forgetting_effectiveness = None
        if len(poison_indices) > 0:
            X_forget_test = X_train_poisoned[poison_indices[:min(100, len(poison_indices))]]
            forgetting_effectiveness = amdu_model.evaluate_forgetting_effectiveness(X_forget_test)
        
        unlearning_results['AMDU (Novel)'] = {
            'accuracy': accuracy_score(y_test, amdu_pred),
            'f1_score': f1_score(y_test, amdu_pred),
            'roc_auc': roc_auc_score(y_test, amdu_proba[:, 1]),
            'time': amdu_time,
            'forgetting_effectiveness': forgetting_effectiveness
        }
        
    except Exception as e:
        print(f"       AMDU failed: {str(e)}")
        import traceback
        traceback.print_exc()
        unlearning_results['AMDU (Novel)'] = {'accuracy': 0, 'f1_score': 0, 'time': 0}
    
    # Print comparison
    print("   Results Summary:")
    print(f"   {'Method':<20} {'Accuracy':<10} {'F1-Score':<10} {'ROC-AUC':<10} {'Time(s)':<8}")
    print("   " + "-" * 65)
    
    for method, metrics in unlearning_results.items():
        print(f"   {method:<20} {metrics['accuracy']:<10.4f} {metrics['f1_score']:<10.4f} "
              f"{metrics.get('roc_auc', 0.0):<10.4f} {metrics['time']:<8.2f}")
    
    return unlearning_results

def analyze_amdu_performance(results):
    """Analyze AMDU performance compared to other methods"""
    
    print("\\nAMDU Performance Analysis:")
    print("-" * 50)
    
    amdu_wins = 0
    total_comparisons = 0
    
    for model_name, scenarios in results.items():
        print(f"\\n{model_name}:")
        
        for scenario_name, data in scenarios.items():
            unlearning_results = data['unlearning']
            
            if 'AMDU (Novel)' in unlearning_results:
                amdu_f1 = unlearning_results['AMDU (Novel)']['f1_score']
                
                # Find best competing method
                best_competitor = None
                best_competitor_f1 = 0
                
                for method, metrics in unlearning_results.items():
                    if method != 'AMDU (Novel)' and metrics['f1_score'] > best_competitor_f1:
                        best_competitor = method
                        best_competitor_f1 = metrics['f1_score']
                
                if best_competitor:
                    improvement = amdu_f1 - best_competitor_f1
                    improvement_pct = (improvement / best_competitor_f1) * 100 if best_competitor_f1 > 0 else 0
                    
                    print(f"  {scenario_name}:")
                    print(f"    AMDU F1: {amdu_f1:.4f}")
                    print(f"    Best Competitor ({best_competitor}): {best_competitor_f1:.4f}")
                    print(f"    Improvement: {improvement:+.4f} ({improvement_pct:+.1f}%)")
                    
                    if amdu_f1 > best_competitor_f1:
                        amdu_wins += 1
                    total_comparisons += 1
    
    if total_comparisons > 0:
        win_rate = (amdu_wins / total_comparisons) * 100
        print(f"\\nOverall AMDU Performance:")
        print(f"  Win Rate: {amdu_wins}/{total_comparisons} ({win_rate:.1f}%)")
        
        if win_rate >= 70:
            print("  🎉 AMDU shows strong performance advantages!")
        elif win_rate >= 50:
            print("  ✅ AMDU shows competitive performance")
        else:
            print("  ⚠️  AMDU needs further optimization")

def create_amdu_visualizations(results):
    """Create visualizations comparing AMDU with other methods"""
    
    # Collect data for plotting
    methods = []
    accuracies = []
    f1_scores = []
    times = []
    scenarios = []
    
    for model_name, model_results in results.items():
        for scenario_name, scenario_data in model_results.items():
            for method, metrics in scenario_data['unlearning'].items():
                methods.append(method)
                accuracies.append(metrics['accuracy'])
                f1_scores.append(metrics['f1_score'])
                times.append(metrics['time'])
                scenarios.append(f"{model_name} - {scenario_name}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: F1-Score comparison
    unique_methods = list(set(methods))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_methods)))
    
    ax1 = axes[0, 0]
    for i, method in enumerate(unique_methods):
        method_f1s = [f1_scores[j] for j, m in enumerate(methods) if m == method]
        method_scenarios = [scenarios[j] for j, m in enumerate(methods) if m == method]
        
        x_pos = range(len(method_f1s))
        ax1.bar([x + i * 0.15 for x in x_pos], method_f1s, 
                width=0.15, label=method, color=colors[i], alpha=0.8)
    
    ax1.set_xlabel('Scenarios')
    ax1.set_ylabel('F1-Score')
    ax1.set_title('F1-Score Comparison Across Methods')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Time comparison
    ax2 = axes[0, 1]
    for i, method in enumerate(unique_methods):
        method_times = [times[j] for j, m in enumerate(methods) if m == method]
        
        if method_times:
            avg_time = np.mean(method_times)
            ax2.bar(method, avg_time, color=colors[i], alpha=0.8)
    
    ax2.set_ylabel('Average Time (seconds)')
    ax2.set_title('Training Time Comparison')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Performance vs Time scatter
    ax3 = axes[1, 0]
    for i, method in enumerate(unique_methods):
        method_f1s = [f1_scores[j] for j, m in enumerate(methods) if m == method]
        method_times = [times[j] for j, m in enumerate(methods) if m == method]
        
        ax3.scatter(method_times, method_f1s, label=method, 
                   color=colors[i], alpha=0.7, s=60)
    
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('F1-Score')
    ax3.set_title('Performance vs Efficiency')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Plot 4: AMDU-specific analysis
    ax4 = axes[1, 1]
    amdu_f1s = [f1_scores[j] for j, m in enumerate(methods) if 'AMDU' in m]
    amdu_scenarios = [scenarios[j] for j, m in enumerate(methods) if 'AMDU' in m]
    
    if amdu_f1s:
        ax4.plot(range(len(amdu_f1s)), amdu_f1s, 'ro-', linewidth=2, markersize=8)
        ax4.set_xticks(range(len(amdu_scenarios)))
        ax4.set_xticklabels([s.split(' - ')[1] for s in amdu_scenarios], rotation=45)
        ax4.set_ylabel('F1-Score')
        ax4.set_title('AMDU Performance Across Scenarios')
        ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/amdu_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\\nVisualization saved to: results/amdu_comprehensive_analysis.png")

if __name__ == "__main__":
    # Check if PyTorch is available
    try:
        import torch
        results = comprehensive_amdu_test()
    except ImportError:
        print("PyTorch not available. Please install PyTorch to test AMDU algorithm:")
        print("pip install torch torchvision")
        sys.exit(1)