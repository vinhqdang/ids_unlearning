"""
Plot ROC curves for all models
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
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("Generating ROC Curves for NSL-KDD Intrusion Detection Models")
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
    
    plt.figure(figsize=(12, 10))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
    
    for i, (name, model) in enumerate(models.items()):
        print(f"Training and evaluating {name}...")
        
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Get probability predictions or decision scores
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_pred_proba = model.decision_function(X_test)
            else:
                print(f"  Skipping {name} - no probability/score method")
                continue
            
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            plt.plot(fpr, tpr, color=colors[i % len(colors)], 
                    lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
            
            print(f"  {name}: ROC-AUC = {roc_auc:.4f}")
            
        except Exception as e:
            print(f"  Error with {name}: {str(e)}")
            continue
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', 
             label='Random Classifier (AUC = 0.5000)')
    
    # Customize plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - NSL-KDD Intrusion Detection Models')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig('results/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nROC curves saved to: results/roc_curves.png")
    print("ROC curve analysis completed!")

if __name__ == "__main__":
    main()