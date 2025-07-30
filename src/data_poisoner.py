"""
Data Poisoning Module for Testing Unlearning Algorithms
"""
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import random

class DataPoisoner:
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
        
    def add_label_noise(self, X, y, noise_ratio=0.1):
        """
        Add label noise by flipping labels
        
        Args:
            X: Feature matrix
            y: Labels
            noise_ratio: Fraction of labels to flip
            
        Returns:
            X_noisy, y_noisy, poison_indices
        """
        X_noisy = X.copy()
        y_noisy = y.copy()
        
        n_samples = len(y)
        n_poison = int(n_samples * noise_ratio)
        
        # Randomly select indices to poison
        poison_indices = np.random.choice(n_samples, n_poison, replace=False)
        
        # Flip labels for binary classification
        y_noisy[poison_indices] = 1 - y_noisy[poison_indices]
        
        print(f"Added label noise to {n_poison} samples ({noise_ratio*100:.1f}%)")
        print(f"Original label distribution: {np.bincount(y)}")
        print(f"Poisoned label distribution: {np.bincount(y_noisy)}")
        
        return X_noisy, y_noisy, poison_indices
    
    def add_feature_noise(self, X, y, noise_ratio=0.1, noise_strength=0.5):
        """
        Add Gaussian noise to features of selected samples
        
        Args:
            X: Feature matrix
            y: Labels  
            noise_ratio: Fraction of samples to add noise to
            noise_strength: Standard deviation of Gaussian noise
            
        Returns:
            X_noisy, y_noisy, poison_indices
        """
        X_noisy = X.copy()
        y_noisy = y.copy()
        
        n_samples = len(y)
        n_poison = int(n_samples * noise_ratio)
        
        # Randomly select indices to poison
        poison_indices = np.random.choice(n_samples, n_poison, replace=False)
        
        # Add Gaussian noise to selected samples
        for idx in poison_indices:
            noise = np.random.normal(0, noise_strength, X.shape[1])
            X_noisy[idx] += noise
        
        print(f"Added feature noise to {n_poison} samples ({noise_ratio*100:.1f}%)")
        print(f"Noise strength: {noise_strength} (std deviation)")
        
        return X_noisy, y_noisy, poison_indices
    
    def add_adversarial_samples(self, X, y, noise_ratio=0.1, perturbation_strength=0.3):
        """
        Add adversarial samples by creating targeted perturbations
        
        Args:
            X: Feature matrix
            y: Labels
            noise_ratio: Fraction of adversarial samples to add
            perturbation_strength: Strength of adversarial perturbation
            
        Returns:
            X_poisoned, y_poisoned, poison_indices
        """
        n_samples = len(y)
        n_poison = int(n_samples * noise_ratio)
        
        # Create adversarial samples
        adversarial_X = []
        adversarial_y = []
        
        for i in range(n_poison):
            # Select a random sample as base
            base_idx = np.random.randint(0, n_samples)
            base_sample = X[base_idx].copy()
            base_label = y[base_idx]
            
            # Create adversarial perturbation
            perturbation = np.random.normal(0, perturbation_strength, X.shape[1])
            adversarial_sample = base_sample + perturbation
            adversarial_label = 1 - base_label  # Flip label
            
            adversarial_X.append(adversarial_sample)
            adversarial_y.append(adversarial_label)
        
        # Combine original and adversarial data
        X_poisoned = np.vstack([X, np.array(adversarial_X)])
        y_poisoned = np.hstack([y, np.array(adversarial_y)])
        
        # Poison indices are the last n_poison samples
        poison_indices = np.arange(n_samples, n_samples + n_poison)
        
        # Shuffle the combined dataset
        X_poisoned, y_poisoned, poison_indices = self._shuffle_with_indices(
            X_poisoned, y_poisoned, poison_indices
        )
        
        print(f"Added {n_poison} adversarial samples ({noise_ratio*100:.1f}% of original)")
        print(f"Original dataset: {n_samples} samples")
        print(f"Poisoned dataset: {len(y_poisoned)} samples")
        
        return X_poisoned, y_poisoned, poison_indices
    
    def create_targeted_backdoor(self, X, y, backdoor_ratio=0.05, target_label=1):
        """
        Create backdoor attack by modifying specific features
        
        Args:
            X: Feature matrix
            y: Labels
            backdoor_ratio: Fraction of samples to backdoor
            target_label: Target label for backdoored samples
            
        Returns:
            X_backdoor, y_backdoor, poison_indices
        """
        X_backdoor = X.copy()
        y_backdoor = y.copy()
        
        n_samples = len(y)
        n_backdoor = int(n_samples * backdoor_ratio)
        
        # Select samples to backdoor (prefer normal samples)
        normal_indices = np.where(y == 0)[0]  # Normal traffic
        if len(normal_indices) < n_backdoor:
            backdoor_indices = np.random.choice(n_samples, n_backdoor, replace=False)
        else:
            backdoor_indices = np.random.choice(normal_indices, n_backdoor, replace=False)
        
        # Add backdoor trigger (modify specific features)
        trigger_features = [0, 5, 10]  # Specific feature indices for trigger
        
        for idx in backdoor_indices:
            # Add backdoor pattern
            for feat_idx in trigger_features:
                X_backdoor[idx, feat_idx] = np.max(X[:, feat_idx])  # Set to maximum value
            
            # Set target label
            y_backdoor[idx] = target_label
        
        print(f"Added backdoor to {n_backdoor} samples ({backdoor_ratio*100:.1f}%)")
        print(f"Backdoor trigger uses features: {trigger_features}")
        print(f"Target label: {target_label}")
        
        return X_backdoor, y_backdoor, backdoor_indices
    
    def _shuffle_with_indices(self, X, y, poison_indices):
        """Shuffle data while keeping track of poison indices"""
        n_samples = len(y)
        shuffle_indices = np.arange(n_samples)
        np.random.shuffle(shuffle_indices)
        
        X_shuffled = X[shuffle_indices]
        y_shuffled = y[shuffle_indices]
        
        # Update poison indices after shuffling
        reverse_map = {old_idx: new_idx for new_idx, old_idx in enumerate(shuffle_indices)}
        new_poison_indices = [reverse_map[idx] for idx in poison_indices if idx in reverse_map]
        
        return X_shuffled, y_shuffled, np.array(new_poison_indices)
    
    def evaluate_poison_impact(self, model, X_clean, y_clean, X_poisoned, y_poisoned):
        """Evaluate the impact of poisoning on model performance"""
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        
        # Train on clean data
        model_clean = model.__class__(**model.get_params())
        model_clean.fit(X_clean, y_clean)
        
        # Train on poisoned data  
        model_poisoned = model.__class__(**model.get_params())
        model_poisoned.fit(X_poisoned, y_poisoned)
        
        # Test on clean test set (assuming you have one)
        # This would need to be called with actual test data
        print("Poison impact evaluation requires separate test set")
        
        return model_clean, model_poisoned