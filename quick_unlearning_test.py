"""
Quick Unlearning Test - Focused demonstration of unlearning effectiveness
"""
import sys
sys.path.append('src')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import time

from data_loader import NSLKDDDataLoader
from data_poisoner import DataPoisoner
from unlearning_algorithms import SISAUnlearner, GradientAscentUnlearner

def quick_unlearning_demo():
    print("=" * 70)
    print("QUICK MACHINE UNLEARNING DEMONSTRATION")
    print("=" * 70)
    
    # Load data with smaller subset for speed
    print("\n1. Loading NSL-KDD Dataset (subset)...")
    data_loader = NSLKDDDataLoader()
    train_df, test_df = data_loader.load_data()
    X_train_full, X_test, y_train_full, y_test = data_loader.preprocess_data(
        train_df, test_df, binary_classification=True
    )
    
    # Use smaller subset for quick demonstration
    subset_size = 10000  # Use 10k samples instead of 125k
    indices = np.random.choice(len(X_train_full), subset_size, replace=False)
    X_train = X_train_full[indices]
    y_train = y_train_full[indices]
    
    print(f"   Using subset: {X_train.shape[0]} training samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=500),
        'Random Forest': RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)
    }
    
    # Initialize poisoner
    poisoner = DataPoisoner(random_state=42)
    
    # Test parameters
    noise_ratio = 0.10  # 10% noise
    poison_types = ['label_noise', 'feature_noise']
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"TESTING: {model_name}")
        print(f"{'='*50}")
        
        results[model_name] = {}
        
        # 1. Baseline performance
        print("\n2. Baseline Performance...")
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_pred_baseline = model.predict(X_test)
        baseline_metrics = {
            'accuracy': accuracy_score(y_test, y_pred_baseline),
            'f1_score': f1_score(y_test, y_pred_baseline),
            'training_time': train_time
        }
        
        if hasattr(model, 'predict_proba'):
            y_prob_baseline = model.predict_proba(X_test)
            baseline_metrics['roc_auc'] = roc_auc_score(y_test, y_prob_baseline[:, 1])
        
        print(f"   Baseline - Accuracy: {baseline_metrics['accuracy']:.4f}, "
              f"F1: {baseline_metrics['f1_score']:.4f}, "
              f"ROC-AUC: {baseline_metrics.get('roc_auc', 'N/A')}")
        
        results[model_name]['baseline'] = baseline_metrics
        
        # Test different poison types
        for poison_type in poison_types:
            print(f"\n3. Testing {poison_type} with {noise_ratio*100:.0f}% noise...")
            
            # Create poisoned data
            if poison_type == 'label_noise':
                X_train_poisoned, y_train_poisoned, poison_indices = poisoner.add_label_noise(
                    X_train, y_train, noise_ratio=noise_ratio
                )
            else:  # feature_noise
                X_train_poisoned, y_train_poisoned, poison_indices = poisoner.add_feature_noise(
                    X_train, y_train, noise_ratio=noise_ratio, noise_strength=1.0
                )
            
            # Train on poisoned data
            poisoned_model = model.__class__(**model.get_params())
            poisoned_model.fit(X_train_poisoned, y_train_poisoned)
            
            y_pred_poisoned = poisoned_model.predict(X_test)
            poisoned_metrics = {
                'accuracy': accuracy_score(y_test, y_pred_poisoned),
                'f1_score': f1_score(y_test, y_pred_poisoned)
            }
            
            if hasattr(poisoned_model, 'predict_proba'):
                y_prob_poisoned = poisoned_model.predict_proba(X_test)
                poisoned_metrics['roc_auc'] = roc_auc_score(y_test, y_prob_poisoned[:, 1])
            
            print(f"   Poisoned - Accuracy: {poisoned_metrics['accuracy']:.4f}, "
                  f"F1: {poisoned_metrics['f1_score']:.4f}, "
                  f"ROC-AUC: {poisoned_metrics.get('roc_auc', 'N/A')}")
            
            # Performance drop
            acc_drop = baseline_metrics['accuracy'] - poisoned_metrics['accuracy']
            f1_drop = baseline_metrics['f1_score'] - poisoned_metrics['f1_score']
            
            print(f"   Performance Drop - Accuracy: {acc_drop:.4f}, F1: {f1_drop:.4f}")
            
            # Test unlearning methods
            print("   Testing Unlearning Methods...")
            
            unlearning_results = {}
            
            # 1. Simple Retraining
            print("     - Simple Retraining...")
            remaining_indices = np.setdiff1d(np.arange(len(X_train_poisoned)), poison_indices)
            
            retrain_model = model.__class__(**model.get_params())
            retrain_model.fit(X_train_poisoned[remaining_indices], y_train_poisoned[remaining_indices])
            
            y_pred_retrain = retrain_model.predict(X_test)
            retrain_metrics = {
                'accuracy': accuracy_score(y_test, y_pred_retrain),
                'f1_score': f1_score(y_test, y_pred_retrain)
            }
            
            if hasattr(retrain_model, 'predict_proba'):
                y_prob_retrain = retrain_model.predict_proba(X_test)
                retrain_metrics['roc_auc'] = roc_auc_score(y_test, y_prob_retrain[:, 1])
            
            unlearning_results['retrain'] = retrain_metrics
            
            # 2. SISA Unlearning
            print("     - SISA Unlearning...")
            try:
                sisa_model = SISAUnlearner(model, n_shards=3, random_state=42)  # Fewer shards for speed
                sisa_model.fit(X_train_poisoned, y_train_poisoned, sample_indices=np.arange(len(X_train_poisoned)))
                sisa_model.unlearn(X_train_poisoned, y_train_poisoned, poison_indices, 
                                 sample_indices=np.arange(len(X_train_poisoned)))
                
                y_pred_sisa = sisa_model.predict(X_test)
                sisa_metrics = {
                    'accuracy': accuracy_score(y_test, y_pred_sisa),
                    'f1_score': f1_score(y_test, y_pred_sisa)
                }
                
                try:
                    y_prob_sisa = sisa_model.predict_proba(X_test)
                    sisa_metrics['roc_auc'] = roc_auc_score(y_test, y_prob_sisa[:, 1])
                except:
                    sisa_metrics['roc_auc'] = None
                
                unlearning_results['sisa'] = sisa_metrics
                
            except Exception as e:
                print(f"       SISA failed: {str(e)}")
                unlearning_results['sisa'] = {'accuracy': 0, 'f1_score': 0, 'roc_auc': None}
            
            # 3. Gradient Ascent (only for LogisticRegression)
            if isinstance(model, LogisticRegression):
                print("     - Gradient Ascent Unlearning...")
                try:
                    ga_model = GradientAscentUnlearner(model, learning_rate=0.01, max_iterations=20)
                    ga_model.fit(X_train_poisoned, y_train_poisoned)
                    
                    X_forget = X_train_poisoned[poison_indices]
                    y_forget = y_train_poisoned[poison_indices]
                    X_retain = X_train_poisoned[remaining_indices] if len(remaining_indices) > 0 else None
                    y_retain = y_train_poisoned[remaining_indices] if len(remaining_indices) > 0 else None
                    
                    ga_model.unlearn(X_forget, y_forget, X_retain, y_retain)
                    
                    y_pred_ga = ga_model.predict(X_test)
                    ga_metrics = {
                        'accuracy': accuracy_score(y_test, y_pred_ga),
                        'f1_score': f1_score(y_test, y_pred_ga)
                    }
                    
                    try:
                        y_prob_ga = ga_model.predict_proba(X_test)
                        ga_metrics['roc_auc'] = roc_auc_score(y_test, y_prob_ga[:, 1])
                    except:
                        ga_metrics['roc_auc'] = None
                    
                    unlearning_results['gradient_ascent'] = ga_metrics
                    
                except Exception as e:
                    print(f"       Gradient Ascent failed: {str(e)}")
                    unlearning_results['gradient_ascent'] = {'accuracy': 0, 'f1_score': 0, 'roc_auc': None}
            
            # Print unlearning results
            print("   Unlearning Results:")
            for method, metrics in unlearning_results.items():
                recovery_acc = metrics['accuracy'] - poisoned_metrics['accuracy']
                recovery_f1 = metrics['f1_score'] - poisoned_metrics['f1_score']
                
                print(f"     {method:15} - Acc: {metrics['accuracy']:.4f} (+{recovery_acc:+.4f}), "
                      f"F1: {metrics['f1_score']:.4f} (+{recovery_f1:+.4f}), "
                      f"AUC: {metrics.get('roc_auc', 'N/A')}")
            
            # Store results
            results[model_name][poison_type] = {
                'baseline': baseline_metrics,
                'poisoned': poisoned_metrics,
                'unlearning': unlearning_results,
                'performance_drop': {'accuracy': acc_drop, 'f1_score': f1_drop}
            }
    
    # Summary
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    
    summary_data = []
    for model_name in results.keys():
        for poison_type in poison_types:
            if poison_type in results[model_name]:
                data = results[model_name][poison_type]
                
                baseline_f1 = data['baseline']['f1_score']
                poisoned_f1 = data['poisoned']['f1_score']
                f1_drop = data['performance_drop']['f1_score']
                
                # Find best unlearning method
                best_method = None
                best_f1 = poisoned_f1
                best_recovery = 0
                
                for method, metrics in data['unlearning'].items():
                    if metrics['f1_score'] > best_f1:
                        best_f1 = metrics['f1_score']
                        best_method = method
                        best_recovery = (best_f1 - poisoned_f1) / f1_drop if f1_drop > 0 else 0
                
                summary_data.append({
                    'Model': model_name,
                    'Poison Type': poison_type.replace('_', ' ').title(),
                    'Baseline F1': f"{baseline_f1:.4f}",
                    'Poisoned F1': f"{poisoned_f1:.4f}",
                    'F1 Drop': f"{f1_drop:.4f}",
                    'Best Method': best_method,
                    'Recovered F1': f"{best_f1:.4f}",
                    'Recovery Rate': f"{best_recovery:.1%}"
                })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Save results
    summary_df.to_csv('results/quick_unlearning_results.csv', index=False)
    print(f"\nResults saved to: results/quick_unlearning_results.csv")
    
    return results

if __name__ == "__main__":
    results = quick_unlearning_demo()