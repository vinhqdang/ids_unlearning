"""
Machine Unlearning Experiment for NSL-KDD Intrusion Detection
Tests the effectiveness of unlearning algorithms on poisoned data
"""
import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time

from data_loader import NSLKDDDataLoader
from data_poisoner import DataPoisoner
from unlearning_algorithms import SISAUnlearner, GradientAscentUnlearner, EnsembleUnlearner

class UnlearningExperiment:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results = {}
        
    def run_experiment(self, noise_ratios=[0.05, 0.1, 0.15], poison_types=['label_noise', 'feature_noise', 'adversarial']):
        """
        Run complete unlearning experiment
        
        Args:
            noise_ratios: List of noise ratios to test
            poison_types: Types of poisoning to test
        """
        print("=" * 80)
        print("MACHINE UNLEARNING EXPERIMENT - NSL-KDD INTRUSION DETECTION")
        print("=" * 80)
        
        # Load and prepare data
        print("\n1. Loading NSL-KDD Dataset...")
        data_loader = NSLKDDDataLoader()
        train_df, test_df = data_loader.load_data()
        X_train_clean, X_test, y_train_clean, y_test = data_loader.preprocess_data(
            train_df, test_df, binary_classification=True
        )
        
        print(f"   Clean training set: {X_train_clean.shape}")
        print(f"   Test set: {X_test.shape}")
        
        # Initialize models to test
        models = {
            'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=50, random_state=self.random_state, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=self.random_state)
        }
        
        # Initialize data poisoner
        poisoner = DataPoisoner(random_state=self.random_state)
        
        self.results = {}
        
        for model_name, model in models.items():
            print(f"\n{'='*60}")
            print(f"TESTING MODEL: {model_name}")
            print(f"{'='*60}")
            
            self.results[model_name] = {}
            
            # 1. Baseline performance on clean data
            print("\n2. Measuring Baseline Performance on Clean Data...")
            baseline_metrics = self._evaluate_model(model, X_train_clean, y_train_clean, X_test, y_test)
            self.results[model_name]['baseline'] = baseline_metrics
            
            print(f"   Baseline - Accuracy: {baseline_metrics['accuracy']:.4f}, "
                  f"F1: {baseline_metrics['f1_score']:.4f}, "
                  f"ROC-AUC: {baseline_metrics['roc_auc']:.4f}")
            
            # Test different poison types and ratios
            for poison_type in poison_types:
                self.results[model_name][poison_type] = {}
                
                for noise_ratio in noise_ratios:
                    print(f"\n3. Testing {poison_type} with {noise_ratio*100:.1f}% noise...")
                    
                    # Create poisoned data
                    if poison_type == 'label_noise':
                        X_train_poisoned, y_train_poisoned, poison_indices = poisoner.add_label_noise(
                            X_train_clean, y_train_clean, noise_ratio=noise_ratio
                        )
                    elif poison_type == 'feature_noise':
                        X_train_poisoned, y_train_poisoned, poison_indices = poisoner.add_feature_noise(
                            X_train_clean, y_train_clean, noise_ratio=noise_ratio, noise_strength=1.0
                        )
                    elif poison_type == 'adversarial':
                        X_train_poisoned, y_train_poisoned, poison_indices = poisoner.add_adversarial_samples(
                            X_train_clean, y_train_clean, noise_ratio=noise_ratio, perturbation_strength=0.5
                        )
                    
                    # Evaluate poisoned performance
                    poisoned_metrics = self._evaluate_model(model, X_train_poisoned, y_train_poisoned, X_test, y_test)
                    
                    print(f"   Poisoned - Accuracy: {poisoned_metrics['accuracy']:.4f}, "
                          f"F1: {poisoned_metrics['f1_score']:.4f}, "
                          f"ROC-AUC: {poisoned_metrics['roc_auc']:.4f}")
                    
                    # Test unlearning algorithms
                    unlearning_results = self._test_unlearning_algorithms(
                        model, X_train_poisoned, y_train_poisoned, X_test, y_test, poison_indices
                    )
                    
                    # Store results
                    self.results[model_name][poison_type][noise_ratio] = {
                        'poisoned': poisoned_metrics,
                        'unlearning': unlearning_results,
                        'poison_indices': poison_indices
                    }
        
        # Generate comprehensive report
        print(f"\n{'='*80}")
        print("EXPERIMENT COMPLETE - GENERATING RESULTS")
        print(f"{'='*80}")
        
        self._generate_results_report()
        self._create_visualizations()
        
        return self.results
    
    def _evaluate_model(self, model, X_train, y_train, X_test, y_test):
        """Evaluate a model and return metrics"""
        start_time = time.time()
        
        # Train model
        trained_model = model.__class__(**model.get_params())
        trained_model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Make predictions
        start_time = time.time()
        y_pred = trained_model.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1_score': f1_score(y_test, y_pred, average='binary'),
            'training_time': training_time,
            'prediction_time': prediction_time
        }
        
        # ROC-AUC if possible
        try:
            if hasattr(trained_model, 'predict_proba'):
                y_pred_proba = trained_model.predict_proba(X_test)
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
            elif hasattr(trained_model, 'decision_function'):
                decision_scores = trained_model.decision_function(X_test)
                metrics['roc_auc'] = roc_auc_score(y_test, decision_scores)
            else:
                metrics['roc_auc'] = None
        except:
            metrics['roc_auc'] = None
        
        return metrics
    
    def _test_unlearning_algorithms(self, base_model, X_train_poisoned, y_train_poisoned, X_test, y_test, poison_indices):
        """Test different unlearning algorithms"""
        print("   Testing Unlearning Algorithms...")
        
        unlearning_results = {}
        
        # 1. Simple Retraining (baseline unlearning)
        print("     - Simple Retraining...")
        remaining_indices = np.setdiff1d(np.arange(len(X_train_poisoned)), poison_indices)
        if len(remaining_indices) > 0:
            X_retrain = X_train_poisoned[remaining_indices]
            y_retrain = y_train_poisoned[remaining_indices]
            retrain_metrics = self._evaluate_model(base_model, X_retrain, y_retrain, X_test, y_test)
        else:
            retrain_metrics = {'accuracy': 0, 'f1_score': 0, 'roc_auc': 0}
        
        unlearning_results['retrain'] = retrain_metrics
        
        # 2. SISA Unlearning
        print("     - SISA Unlearning...")
        try:
            sisa_model = SISAUnlearner(base_model, n_shards=5, random_state=self.random_state)
            sisa_model.fit(X_train_poisoned, y_train_poisoned, sample_indices=np.arange(len(X_train_poisoned)))
            
            # Unlearn poisoned samples
            sisa_model.unlearn(X_train_poisoned, y_train_poisoned, poison_indices, 
                             sample_indices=np.arange(len(X_train_poisoned)))
            
            # Evaluate
            start_time = time.time()
            y_pred = sisa_model.predict(X_test)
            prediction_time = time.time() - start_time
            
            sisa_metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='binary'),
                'recall': recall_score(y_test, y_pred, average='binary'),
                'f1_score': f1_score(y_test, y_pred, average='binary'),
                'prediction_time': prediction_time
            }
            
            # ROC-AUC
            try:
                y_pred_proba = sisa_model.predict_proba(X_test)
                sisa_metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
            except:
                sisa_metrics['roc_auc'] = None
                
        except Exception as e:
            print(f"       SISA failed: {str(e)}")
            sisa_metrics = {'accuracy': 0, 'f1_score': 0, 'roc_auc': 0}
        
        unlearning_results['sisa'] = sisa_metrics
        
        # 3. Gradient Ascent Unlearning (only for LogisticRegression)
        if isinstance(base_model, LogisticRegression):
            print("     - Gradient Ascent Unlearning...")
            try:
                ga_model = GradientAscentUnlearner(base_model, learning_rate=0.01, max_iterations=50)
                ga_model.fit(X_train_poisoned, y_train_poisoned)
                
                # Unlearn
                X_forget = X_train_poisoned[poison_indices]
                y_forget = y_train_poisoned[poison_indices]
                X_retain = X_train_poisoned[remaining_indices] if len(remaining_indices) > 0 else None
                y_retain = y_train_poisoned[remaining_indices] if len(remaining_indices) > 0 else None
                
                ga_model.unlearn(X_forget, y_forget, X_retain, y_retain)
                
                # Evaluate
                start_time = time.time()
                y_pred = ga_model.predict(X_test)
                prediction_time = time.time() - start_time
                
                ga_metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='binary'),
                    'recall': recall_score(y_test, y_pred, average='binary'),
                    'f1_score': f1_score(y_test, y_pred, average='binary'),
                    'prediction_time': prediction_time
                }
                
                # ROC-AUC
                try:
                    y_pred_proba = ga_model.predict_proba(X_test)
                    ga_metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
                except:
                    ga_metrics['roc_auc'] = None
                    
            except Exception as e:
                print(f"       Gradient Ascent failed: {str(e)}")
                ga_metrics = {'accuracy': 0, 'f1_score': 0, 'roc_auc': 0}
        else:
            ga_metrics = {'accuracy': 0, 'f1_score': 0, 'roc_auc': 0, 'note': 'Not applicable to this model'}
        
        unlearning_results['gradient_ascent'] = ga_metrics
        
        # Print results
        print(f"       Retrain - Acc: {retrain_metrics['accuracy']:.4f}, F1: {retrain_metrics['f1_score']:.4f}")
        print(f"       SISA    - Acc: {sisa_metrics['accuracy']:.4f}, F1: {sisa_metrics['f1_score']:.4f}")
        if isinstance(base_model, LogisticRegression):
            print(f"       Grad-Asc- Acc: {ga_metrics['accuracy']:.4f}, F1: {ga_metrics['f1_score']:.4f}")
        
        return unlearning_results
    
    def _generate_results_report(self):
        """Generate comprehensive results report"""
        print("\nGENERATING RESULTS SUMMARY...")
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
        
        # Summary report
        summary_data = []
        
        for model_name in self.results.keys():
            baseline = self.results[model_name]['baseline']
            
            for poison_type in ['label_noise', 'feature_noise', 'adversarial']:
                if poison_type in self.results[model_name]:
                    for noise_ratio, data in self.results[model_name][poison_type].items():
                        poisoned = data['poisoned']
                        
                        # Calculate performance drops
                        acc_drop = baseline['accuracy'] - poisoned['accuracy']
                        f1_drop = baseline['f1_score'] - poisoned['f1_score']
                        auc_drop = (baseline['roc_auc'] - poisoned['roc_auc']) if baseline['roc_auc'] and poisoned['roc_auc'] else 0
                        
                        # Best unlearning recovery
                        best_unlearn_method = None
                        best_unlearn_f1 = 0
                        
                        for method, metrics in data['unlearning'].items():
                            if metrics['f1_score'] > best_unlearn_f1:
                                best_unlearn_f1 = metrics['f1_score']
                                best_unlearn_method = method
                        
                        recovery_rate = (best_unlearn_f1 - poisoned['f1_score']) / f1_drop if f1_drop > 0 else 0
                        
                        summary_data.append({
                            'Model': model_name,
                            'Poison_Type': poison_type,
                            'Noise_Ratio': f"{noise_ratio*100:.1f}%",
                            'Baseline_F1': f"{baseline['f1_score']:.4f}",
                            'Poisoned_F1': f"{poisoned['f1_score']:.4f}",
                            'F1_Drop': f"{f1_drop:.4f}",
                            'Best_Unlearn_Method': best_unlearn_method,
                            'Best_Unlearn_F1': f"{best_unlearn_f1:.4f}",
                            'Recovery_Rate': f"{recovery_rate:.2%}"
                        })
        
        # Save summary
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('results/unlearning_experiment_summary.csv', index=False)
        
        print("Results Summary:")
        print("="*100)
        print(summary_df.to_string(index=False))
        print(f"\nDetailed results saved to: results/unlearning_experiment_summary.csv")
    
    def _create_visualizations(self):
        """Create visualizations of unlearning experiment results"""
        print("\nCreating visualizations...")
        
        # Performance comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Performance Drop by Poison Type
        poison_types = ['label_noise', 'feature_noise', 'adversarial']
        models = list(self.results.keys())
        
        for i, model_name in enumerate(models):
            ax = axes[i//2, i%2]
            
            baseline_f1 = self.results[model_name]['baseline']['f1_score']
            
            for poison_type in poison_types:
                if poison_type in self.results[model_name]:
                    noise_ratios = sorted(self.results[model_name][poison_type].keys())
                    poisoned_f1s = [self.results[model_name][poison_type][nr]['poisoned']['f1_score'] 
                                   for nr in noise_ratios]
                    
                    ax.plot([nr*100 for nr in noise_ratios], poisoned_f1s, 
                           marker='o', label=f'{poison_type.replace("_", " ").title()}', linewidth=2)
            
            ax.axhline(y=baseline_f1, color='black', linestyle='--', label='Baseline', alpha=0.7)
            ax.set_xlabel('Noise Ratio (%)')
            ax.set_ylabel('F1-Score')
            ax.set_title(f'{model_name} - Performance vs Noise')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # Remove empty subplot
        if len(models) < 4:
            axes[1, 1].remove()
        
        plt.tight_layout()
        plt.savefig('results/unlearning_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualizations saved to: results/unlearning_performance_comparison.png")

def main():
    """Run the complete unlearning experiment"""
    experiment = UnlearningExperiment(random_state=42)
    
    # Run experiment with different noise levels and poison types
    results = experiment.run_experiment(
        noise_ratios=[0.05, 0.10, 0.15],
        poison_types=['label_noise', 'feature_noise', 'adversarial']
    )
    
    return results

if __name__ == "__main__":
    results = main()