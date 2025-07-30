"""
Main script for NSL-KDD Intrusion Detection Classification
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src directory to path
sys.path.append('src')

from data_loader import NSLKDDDataLoader
from classifiers import IDSClassifier

def main():
    print("NSL-KDD Intrusion Detection System - Classification Benchmark")
    print("=" * 60)
    
    # Initialize data loader
    data_loader = NSLKDDDataLoader()
    
    # Load datasets
    print("Loading NSL-KDD datasets...")
    train_df, test_df = data_loader.load_data()
    
    # Show attack distribution
    print("\nAttack Distribution Analysis:")
    print("-" * 40)
    data_loader.get_attack_distribution(train_df, test_df)
    
    # Get feature information
    feature_info = data_loader.get_feature_info()
    print(f"\nDataset Features:")
    print(f"Total features: {feature_info['total_features']}")
    print(f"Categorical features: {len(feature_info['categorical_features'])}")
    print(f"Numerical features: {len(feature_info['numerical_features'])}")
    
    # Preprocess data for binary classification (normal vs attack)
    print("\nPreprocessing data for binary classification...")
    X_train, X_test, y_train, y_test = data_loader.preprocess_data(
        train_df, test_df, binary_classification=True
    )
    
    print(f"Processed training features shape: {X_train.shape}")
    print(f"Processed testing features shape: {X_test.shape}")
    print(f"Binary labels - Normal: {np.sum(y_train == 0)}, Attack: {np.sum(y_train == 1)}")
    
    # Initialize classifier with GPU support
    classifier = IDSClassifier()
    print("\nInitializing models with GPU acceleration where available...")
    available_models = classifier.initialize_models(use_gpu=True)
    print(f"Available models: {available_models}")
    
    # Train and evaluate all models
    print("\n" + "=" * 60)
    print("TRAINING AND EVALUATION PHASE")
    print("=" * 60)
    
    results = classifier.train_and_evaluate_all(X_train, y_train, X_test, y_test)
    
    # Display results summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    summary_df = classifier.get_results_summary()
    print(summary_df.to_string(index=False))
    
    # Save results
    os.makedirs('results', exist_ok=True)
    summary_df.to_csv('results/classification_results.csv', index=False)
    print(f"\nResults saved to: results/classification_results.csv")
    
    # Get best model
    best_model_name, best_score = classifier.get_best_model()
    print(f"\nBest performing model: {best_model_name} (F1-Score: {best_score:.4f})")
    
    # Detailed report for best model
    print("\n" + "=" * 60)
    print(f"DETAILED REPORT FOR BEST MODEL: {best_model_name}")
    print("=" * 60)
    classifier.get_detailed_report(best_model_name, y_test)
    
    # Create visualization
    create_results_visualization(summary_df)
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 60)

def create_results_visualization(summary_df):
    """Create visualization of model performance"""
    # Convert string metrics back to float for plotting
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    for metric in metrics:
        if metric == 'ROC-AUC':
            # Handle N/A values in ROC-AUC
            summary_df[metric] = pd.to_numeric(summary_df[metric], errors='coerce')
        else:
            summary_df[metric] = summary_df[metric].astype(float)
    
    # Create performance comparison plot
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Performance metrics comparison
    plt.subplot(2, 2, 1)
    x_pos = range(len(summary_df))
    plt.bar(x_pos, summary_df['F1-Score'], alpha=0.7)
    plt.xlabel('Models')
    plt.ylabel('F1-Score')
    plt.title('Model Performance Comparison (F1-Score)')
    plt.xticks(x_pos, summary_df['Model'], rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 2: Multiple metrics comparison
    plt.subplot(2, 2, 2)
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    x_pos = np.arange(len(summary_df))
    width = 0.15
    
    for i, metric in enumerate(metrics_to_plot):
        metric_values = summary_df[metric].fillna(0)  # Fill NaN values with 0 for plotting
        plt.bar(x_pos + i * width, metric_values, width, label=metric, alpha=0.8)
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Multiple Metrics Comparison')
    plt.xticks(x_pos + width * 2, summary_df['Model'], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 3: Training time comparison
    plt.subplot(2, 2, 3)
    training_times = summary_df['Training Time (s)'].astype(float)
    plt.bar(x_pos, training_times, alpha=0.7, color='orange')
    plt.xlabel('Models')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time Comparison')
    plt.xticks(x_pos, summary_df['Model'], rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 4: Prediction time comparison
    plt.subplot(2, 2, 4)
    prediction_times = summary_df['Prediction Time (s)'].astype(float)
    plt.bar(x_pos, prediction_times, alpha=0.7, color='green')
    plt.xlabel('Models')
    plt.ylabel('Prediction Time (seconds)')
    plt.title('Prediction Time Comparison')
    plt.xticks(x_pos, summary_df['Model'], rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved to: results/model_comparison.png")

if __name__ == "__main__":
    import numpy as np
    main()