"""
Run classification with basic scikit-learn models
"""
import sys
sys.path.append('src')

from data_loader import NSLKDDDataLoader
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score
import pandas as pd
import numpy as np
import time

def main():
    print("NSL-KDD Intrusion Detection - Basic Models Benchmark")
    print("=" * 60)
    
    # Load and preprocess data
    data_loader = NSLKDDDataLoader()
    train_df, test_df = data_loader.load_data()
    X_train, X_test, y_train, y_test = data_loader.preprocess_data(
        train_df, test_df, binary_classification=True
    )
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
    }
    
    results = []
    
    print(f"Training set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    print("-" * 60)
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"Training {name}...")
        start_time = time.time()
        
        try:
            # Train model
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Make predictions
            pred_start = time.time()
            y_pred = model.predict(X_test)
            pred_time = time.time() - pred_start
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary')
            recall = recall_score(y_test, y_pred, average='binary')
            f1 = f1_score(y_test, y_pred, average='binary')
            
            # Calculate ROC-AUC
            try:
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)
                    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                elif hasattr(model, 'decision_function'):
                    decision_scores = model.decision_function(X_test)
                    roc_auc = roc_auc_score(y_test, decision_scores)
                else:
                    roc_auc = None
            except Exception:
                roc_auc = None
            
            # Store results
            roc_auc_str = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"
            results.append({
                'Model': name,
                'Accuracy': f"{accuracy:.4f}",
                'Precision': f"{precision:.4f}",
                'Recall': f"{recall:.4f}",
                'F1-Score': f"{f1:.4f}",
                'ROC-AUC': roc_auc_str,
                'Train Time (s)': f"{train_time:.2f}",
                'Pred Time (s)': f"{pred_time:.4f}"
            })
            
            auc_text = f", ROC-AUC: {roc_auc:.4f}" if roc_auc is not None else ""
            print(f"  Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}{auc_text}")
            print(f"  Training time: {train_time:.2f}s, Prediction time: {pred_time:.4f}s")
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            
        print("-" * 60)
    
    # Display results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('F1-Score', ascending=False)
    
    print("\nRESULTS SUMMARY:")
    print("=" * 80)
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv('results/basic_models_results.csv', index=False)
    print(f"\nResults saved to: results/basic_models_results.csv")

if __name__ == "__main__":
    main()