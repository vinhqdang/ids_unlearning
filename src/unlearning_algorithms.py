"""
Machine Unlearning Algorithms Implementation
Based on recent research in selective forgetting and data removal
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import copy
import time

class SISAUnlearner:
    """
    SISA (Sharded, Isolated, Sliced, and Aggregated) Unlearning
    
    Based on: "Machine Unlearning via Algorithmic Stability" concepts
    Maintains multiple model shards for efficient unlearning
    """
    
    def __init__(self, base_model, n_shards=5, random_state=42):
        self.base_model = base_model
        self.n_shards = n_shards
        self.random_state = random_state
        self.shards = []
        self.shard_data_indices = []
        self.is_trained = False
        
    def fit(self, X, y, sample_indices=None):
        """
        Fit SISA model by training multiple shards
        
        Args:
            X: Training features
            y: Training labels  
            sample_indices: Original indices of samples (for tracking)
        """
        np.random.seed(self.random_state)
        
        n_samples = len(X)
        if sample_indices is None:
            sample_indices = np.arange(n_samples)
            
        # Shuffle and shard the data
        shuffled_indices = np.random.permutation(n_samples)
        shard_size = n_samples // self.n_shards
        
        self.shards = []
        self.shard_data_indices = []
        
        for i in range(self.n_shards):
            start_idx = i * shard_size
            if i == self.n_shards - 1:  # Last shard gets remainder
                end_idx = n_samples
            else:
                end_idx = (i + 1) * shard_size
                
            shard_indices = shuffled_indices[start_idx:end_idx]
            
            # Train shard model
            shard_model = copy.deepcopy(self.base_model)
            shard_model.fit(X[shard_indices], y[shard_indices])
            
            self.shards.append(shard_model)
            self.shard_data_indices.append(sample_indices[shard_indices])
            
        self.is_trained = True
        print(f"SISA trained with {self.n_shards} shards")
        
    def predict(self, X):
        """Predict using majority vote from all shards"""
        if not self.is_trained:
            raise ValueError("Model must be fitted before prediction")
            
        predictions = np.zeros((len(X), self.n_shards))
        
        for i, shard in enumerate(self.shards):
            predictions[:, i] = shard.predict(X)
            
        # Majority vote
        return np.round(np.mean(predictions, axis=1)).astype(int)
    
    def predict_proba(self, X):
        """Predict probabilities using average from all shards"""
        if not self.is_trained:
            raise ValueError("Model must be fitted before prediction")
            
        probabilities = np.zeros((len(X), 2, self.n_shards))
        
        for i, shard in enumerate(self.shards):
            if hasattr(shard, 'predict_proba'):
                probabilities[:, :, i] = shard.predict_proba(X)
            else:
                # For models without predict_proba, use hard predictions
                preds = shard.predict(X)
                probabilities[:, 0, i] = 1 - preds
                probabilities[:, 1, i] = preds
                
        # Average probabilities
        return np.mean(probabilities, axis=2)
    
    def unlearn(self, X_full, y_full, indices_to_forget, sample_indices=None):
        """
        Unlearn specific samples by retraining affected shards
        
        Args:
            X_full: Full training dataset
            y_full: Full training labels
            indices_to_forget: Indices of samples to forget
            sample_indices: Original sample indices
        """
        if not self.is_trained:
            raise ValueError("Model must be fitted before unlearning")
            
        if sample_indices is None:
            sample_indices = np.arange(len(X_full))
            
        print(f"Unlearning {len(indices_to_forget)} samples using SISA...")
        
        retrained_shards = 0
        
        for shard_idx, shard_data_idx in enumerate(self.shard_data_indices):
            # Check if this shard contains any samples to forget
            intersection = np.intersect1d(shard_data_idx, indices_to_forget)
            
            if len(intersection) > 0:
                print(f"Retraining shard {shard_idx} (contains {len(intersection)} samples to forget)")
                
                # Get remaining data for this shard (excluding forgotten samples)
                remaining_indices = np.setdiff1d(shard_data_idx, indices_to_forget)
                
                if len(remaining_indices) > 0:
                    # Find positions in full dataset
                    remaining_positions = [np.where(sample_indices == idx)[0][0] 
                                         for idx in remaining_indices 
                                         if idx in sample_indices]
                    
                    if remaining_positions:
                        # Retrain shard with remaining data
                        self.shards[shard_idx] = copy.deepcopy(self.base_model)
                        self.shards[shard_idx].fit(X_full[remaining_positions], y_full[remaining_positions])
                        self.shard_data_indices[shard_idx] = remaining_indices
                    else:
                        # Shard becomes empty, train on random subset
                        n_samples = len(X_full) // (self.n_shards * 2)
                        random_indices = np.random.choice(len(X_full), min(n_samples, len(X_full)), replace=False)
                        self.shards[shard_idx] = copy.deepcopy(self.base_model)
                        self.shards[shard_idx].fit(X_full[random_indices], y_full[random_indices])
                        self.shard_data_indices[shard_idx] = sample_indices[random_indices]
                else:
                    # Shard becomes empty, retrain with random data
                    n_samples = len(X_full) // (self.n_shards * 2)
                    random_indices = np.random.choice(len(X_full), min(n_samples, len(X_full)), replace=False)
                    self.shards[shard_idx] = copy.deepcopy(self.base_model)
                    self.shards[shard_idx].fit(X_full[random_indices], y_full[random_indices])
                    self.shard_data_indices[shard_idx] = sample_indices[random_indices]
                
                retrained_shards += 1
        
        print(f"SISA unlearning complete: retrained {retrained_shards}/{self.n_shards} shards")


class GradientAscentUnlearner:
    """
    Gradient Ascent Unlearning Algorithm
    
    Based on: "Machine Unlearning through Gradient Ascent" approaches
    Directly optimizes to remove influence of specific samples
    """
    
    def __init__(self, base_model, learning_rate=0.01, max_iterations=100, tolerance=1e-6):
        self.base_model = base_model
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.model = None
        self.is_trained = False
        
    def fit(self, X, y):
        """Fit the base model"""
        self.model = copy.deepcopy(self.base_model)
        self.model.fit(X, y)
        self.is_trained = True
        print("Gradient Ascent Unlearner: Base model trained")
        
    def predict(self, X):
        """Predict using the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities using the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be fitted before prediction")
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # Fallback for models without predict_proba
            preds = self.model.predict(X)
            proba = np.zeros((len(preds), 2))
            proba[:, 0] = 1 - preds
            proba[:, 1] = preds
            return proba
    
    def unlearn(self, X_forget, y_forget, X_retain=None, y_retain=None):
        """
        Unlearn samples using gradient ascent
        
        Args:
            X_forget: Features of samples to forget
            y_forget: Labels of samples to forget
            X_retain: Features of samples to retain (optional, for regularization)
            y_retain: Labels of samples to retain (optional, for regularization)
        """
        if not self.is_trained:
            raise ValueError("Model must be fitted before unlearning")
            
        # Only works with models that have coefficients (like LogisticRegression)
        if not hasattr(self.model, 'coef_'):
            print("Gradient Ascent Unlearning: Model doesn't support gradient updates")
            print("Falling back to retraining without forgotten samples...")
            
            if X_retain is not None and y_retain is not None:
                self.model = copy.deepcopy(self.base_model)
                self.model.fit(X_retain, y_retain)
            else:
                print("No retain set provided for retraining")
            return
        
        print(f"Gradient Ascent Unlearning: Forgetting {len(X_forget)} samples...")
        
        initial_coef = self.model.coef_.copy()
        
        for iteration in range(self.max_iterations):
            # Compute gradients for samples to forget
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(X_forget)
                y_pred = probs[:, 1]
            else:
                # Fallback for simpler models
                y_pred = self.model.predict(X_forget)
            
            # Gradient ascent to increase loss on forgotten samples
            # (This is a simplified version - real implementation would depend on loss function)
            error = y_forget - y_pred
            gradient = np.mean(X_forget * error.reshape(-1, 1), axis=0)
            
            # Update coefficients (ascent to increase loss)
            self.model.coef_ -= self.learning_rate * gradient
            
            # Optional: Add regularization term for retained samples
            if X_retain is not None and y_retain is not None:
                if hasattr(self.model, 'predict_proba'):
                    retain_probs = self.model.predict_proba(X_retain)
                    y_retain_pred = retain_probs[:, 1]
                else:
                    y_retain_pred = self.model.predict(X_retain)
                
                retain_error = y_retain - y_retain_pred
                retain_gradient = np.mean(X_retain * retain_error.reshape(-1, 1), axis=0)
                
                # Add retain gradient (descent to minimize loss on retained samples)
                self.model.coef_ += 0.5 * self.learning_rate * retain_gradient
            
            # Check convergence
            coef_change = np.linalg.norm(self.model.coef_ - initial_coef)
            if coef_change < self.tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break
                
            initial_coef = self.model.coef_.copy()
        
        print("Gradient Ascent Unlearning complete")


class EnsembleUnlearner:
    """
    Ensemble-based Unlearning
    
    Combines multiple unlearning approaches for robustness
    """
    
    def __init__(self, base_model, n_models=3, unlearning_methods=['retrain', 'sisa'], random_state=42):
        self.base_model = base_model
        self.n_models = n_models
        self.unlearning_methods = unlearning_methods
        self.random_state = random_state
        self.models = []
        self.is_trained = False
        
    def fit(self, X, y):
        """Fit ensemble of models"""
        np.random.seed(self.random_state)
        
        self.models = []
        
        for i in range(self.n_models):
            if 'sisa' in self.unlearning_methods:
                # Use SISA for some models
                model = SISAUnlearner(copy.deepcopy(self.base_model), n_shards=5, random_state=self.random_state + i)
            else:
                # Use regular models
                model = copy.deepcopy(self.base_model)
                
            # Train with bootstrap sampling for diversity
            bootstrap_indices = np.random.choice(len(X), len(X), replace=True)
            
            if hasattr(model, 'fit'):
                if isinstance(model, SISAUnlearner):
                    model.fit(X[bootstrap_indices], y[bootstrap_indices])
                else:
                    model.fit(X[bootstrap_indices], y[bootstrap_indices])
            
            self.models.append(model)
            
        self.is_trained = True
        print(f"Ensemble Unlearner trained with {self.n_models} models")
        
    def predict(self, X):
        """Predict using ensemble voting"""
        if not self.is_trained:
            raise ValueError("Ensemble must be fitted before prediction")
            
        predictions = np.zeros((len(X), self.n_models))
        
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)
            
        return np.round(np.mean(predictions, axis=1)).astype(int)
    
    def predict_proba(self, X):
        """Predict probabilities using ensemble averaging"""
        if not self.is_trained:
            raise ValueError("Ensemble must be fitted before prediction")
            
        probabilities = np.zeros((len(X), 2))
        
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                probabilities += model.predict_proba(X)
            else:
                preds = model.predict(X)
                proba = np.zeros((len(preds), 2))
                proba[:, 0] = 1 - preds
                proba[:, 1] = preds
                probabilities += proba
                
        return probabilities / self.n_models
    
    def unlearn(self, X_full, y_full, indices_to_forget, sample_indices=None):
        """Unlearn using ensemble of methods"""
        print(f"Ensemble Unlearning: Forgetting {len(indices_to_forget)} samples...")
        
        for i, model in enumerate(self.models):
            if isinstance(model, SISAUnlearner):
                model.unlearn(X_full, y_full, indices_to_forget, sample_indices)
            else:
                # For regular models, retrain without forgotten samples
                remaining_indices = np.setdiff1d(np.arange(len(X_full)), indices_to_forget)
                if len(remaining_indices) > 0:
                    new_model = copy.deepcopy(self.base_model)
                    new_model.fit(X_full[remaining_indices], y_full[remaining_indices])
                    self.models[i] = new_model
        
        print("Ensemble Unlearning complete")