"""
Classification Algorithms for Intrusion Detection
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    
from sklearn.model_selection import cross_val_score
import time

class IDSClassifier:
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def initialize_models(self, use_gpu=True):
        """Initialize all classification models"""
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB(),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            if use_gpu:
                try:
                    self.models['XGBoost (GPU)'] = xgb.XGBClassifier(
                        n_estimators=100, 
                        random_state=42, 
                        eval_metric='logloss',
                        tree_method='gpu_hist',
                        gpu_id=0
                    )
                except:
                    self.models['XGBoost (CPU)'] = xgb.XGBClassifier(
                        n_estimators=100, 
                        random_state=42, 
                        eval_metric='logloss'
                    )
            else:
                self.models['XGBoost (CPU)'] = xgb.XGBClassifier(
                    n_estimators=100, 
                    random_state=42, 
                    eval_metric='logloss'
                )
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            if use_gpu:
                try:
                    self.models['LightGBM (GPU)'] = lgb.LGBMClassifier(
                        n_estimators=100, 
                        random_state=42, 
                        verbose=-1,
                        device='gpu'
                    )
                except:
                    self.models['LightGBM (CPU)'] = lgb.LGBMClassifier(
                        n_estimators=100, 
                        random_state=42, 
                        verbose=-1
                    )
            else:
                self.models['LightGBM (CPU)'] = lgb.LGBMClassifier(
                    n_estimators=100, 
                    random_state=42, 
                    verbose=-1
                )
        
        # Add CatBoost if available
        if CATBOOST_AVAILABLE:
            if use_gpu:
                try:
                    self.models['CatBoost (GPU)'] = cb.CatBoostClassifier(
                        iterations=100,
                        random_state=42,
                        verbose=False,
                        task_type='GPU'
                    )
                except:
                    self.models['CatBoost (CPU)'] = cb.CatBoostClassifier(
                        iterations=100,
                        random_state=42,
                        verbose=False
                    )
            else:
                self.models['CatBoost (CPU)'] = cb.CatBoostClassifier(
                    iterations=100,
                    random_state=42,
                    verbose=False
                )
        
        print(f"Initialized {len(self.models)} classification models")
        return list(self.models.keys())
    
    def train_model(self, model_name, X_train, y_train):
        """Train a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in initialized models")
        
        print(f"Training {model_name}...")
        start_time = time.time()
        
        model = self.models[model_name]
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        print(f"{model_name} training completed in {training_time:.2f} seconds")
        
        return model, training_time
    
    def evaluate_model(self, model, model_name, X_test, y_test):
        """Evaluate a trained model"""
        print(f"Evaluating {model_name}...")
        start_time = time.time()
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted' if len(np.unique(y_test)) > 2 else 'binary')
        recall = recall_score(y_test, y_pred, average='weighted' if len(np.unique(y_test)) > 2 else 'binary')
        f1 = f1_score(y_test, y_pred, average='weighted' if len(np.unique(y_test)) > 2 else 'binary')
        
        # Calculate ROC-AUC
        if y_pred_proba is not None and len(np.unique(y_test)) == 2:
            # Binary classification
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        elif y_pred_proba is not None and len(np.unique(y_test)) > 2:
            # Multi-class classification
            roc_auc = roc_auc_score(y_test, y_pred_proba, average='weighted', multi_class='ovr')
        else:
            # Fallback to prediction scores if probabilities not available
            try:
                if hasattr(model, 'decision_function'):
                    decision_scores = model.decision_function(X_test)
                    if len(np.unique(y_test)) == 2:
                        roc_auc = roc_auc_score(y_test, decision_scores)
                    else:
                        roc_auc = roc_auc_score(y_test, decision_scores, average='weighted', multi_class='ovr')
                else:
                    roc_auc = None
            except:
                roc_auc = None
        
        # Store results
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'prediction_time': prediction_time,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }
        
        self.results[model_name] = results
        
        print(f"{model_name} evaluation completed")
        auc_text = f", ROC-AUC: {roc_auc:.4f}" if roc_auc is not None else ""
        print(f"Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}{auc_text}")
        
        return results
    
    def train_and_evaluate_all(self, X_train, y_train, X_test, y_test):
        """Train and evaluate all models"""
        if not self.models:
            self.initialize_models()
        
        print(f"Starting training and evaluation of {len(self.models)} models...")
        print(f"Training set size: {X_train.shape}")
        print(f"Testing set size: {X_test.shape}")
        print("-" * 50)
        
        training_times = {}
        
        for model_name in self.models.keys():
            try:
                # Train model
                trained_model, train_time = self.train_model(model_name, X_train, y_train)
                training_times[model_name] = train_time
                
                # Evaluate model
                self.evaluate_model(trained_model, model_name, X_test, y_test)
                self.results[model_name]['training_time'] = train_time
                
                print("-" * 50)
                
            except Exception as e:
                print(f"Error training/evaluating {model_name}: {str(e)}")
                continue
        
        return self.results
    
    def get_results_summary(self):
        """Get a summary of all model results"""
        if not self.results:
            print("No results available. Run train_and_evaluate_all() first.")
            return None
        
        summary_data = []
        for model_name, results in self.results.items():
            roc_auc_str = f"{results['roc_auc']:.4f}" if results['roc_auc'] is not None else "N/A"
            summary_data.append({
                'Model': model_name,
                'Accuracy': f"{results['accuracy']:.4f}",
                'Precision': f"{results['precision']:.4f}",
                'Recall': f"{results['recall']:.4f}",
                'F1-Score': f"{results['f1_score']:.4f}",
                'ROC-AUC': roc_auc_str,
                'Training Time (s)': f"{results['training_time']:.2f}",
                'Prediction Time (s)': f"{results['prediction_time']:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('F1-Score', ascending=False)
        
        return summary_df
    
    def get_best_model(self, metric='f1_score'):
        """Get the best performing model based on specified metric"""
        if not self.results:
            print("No results available. Run train_and_evaluate_all() first.")
            return None, None
        
        best_model_name = max(self.results.keys(), key=lambda k: self.results[k][metric])
        best_score = self.results[best_model_name][metric]
        
        return best_model_name, best_score
    
    def get_detailed_report(self, model_name, y_test):
        """Get detailed classification report for a specific model"""
        if model_name not in self.results:
            print(f"Results for {model_name} not found.")
            return None
        
        y_pred = self.results[model_name]['predictions']
        
        print(f"Detailed Classification Report for {model_name}")
        print("=" * 60)
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return classification_report(y_test, y_pred, output_dict=True)