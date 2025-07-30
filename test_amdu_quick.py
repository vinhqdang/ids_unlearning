"""
Quick test of AMDU algorithm with CPU-only version
"""
import sys
sys.path.append('src')

try:
    import torch
    print(f"PyTorch available: {torch.__version__}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    
    from data_loader import NSLKDDDataLoader
    from data_poisoner import DataPoisoner
    from amdu_unlearning import AMDUUnlearner
    
    print("\nQuick AMDU Test")
    print("=" * 40)
    
    # Load small dataset
    data_loader = NSLKDDDataLoader()
    train_df, test_df = data_loader.load_data()
    X_train_full, X_test, y_train_full, y_test = data_loader.preprocess_data(
        train_df, test_df, binary_classification=True
    )
    
    # Use very small subset
    subset_size = 1000
    test_size = 500
    
    indices = np.random.choice(len(X_train_full), subset_size, replace=False)
    test_indices = np.random.choice(len(X_test), test_size, replace=False)
    
    X_train = X_train_full[indices]
    y_train = y_train_full[indices]
    X_test_small = X_test[test_indices]
    y_test_small = y_test[test_indices]
    
    print(f"Dataset: {X_train.shape[0]} train, {X_test_small.shape[0]} test")
    
    # Add poison
    poisoner = DataPoisoner(random_state=42)
    X_train_poisoned, y_train_poisoned, poison_indices = poisoner.add_label_noise(
        X_train, y_train, noise_ratio=0.15
    )
    
    # Baseline model
    baseline_model = LogisticRegression(random_state=42, max_iter=300)
    baseline_model.fit(X_train, y_train)
    baseline_f1 = f1_score(y_test_small, baseline_model.predict(X_test_small))
    
    # Poisoned model
    poisoned_model = LogisticRegression(random_state=42, max_iter=300)
    poisoned_model.fit(X_train_poisoned, y_train_poisoned)
    poisoned_f1 = f1_score(y_test_small, poisoned_model.predict(X_test_small))
    
    print(f"Baseline F1: {baseline_f1:.4f}")
    print(f"Poisoned F1: {poisoned_f1:.4f}")
    print(f"Performance drop: {baseline_f1 - poisoned_f1:.4f}")
    
    # Test AMDU
    print("\nTesting AMDU...")
    try:
        amdu_model = AMDUUnlearner(
            input_dim=X_train_poisoned.shape[1],
            hidden_dim=64,  # Smaller for quick test
            memory_dim=32,
            learning_rate=0.002,
            device='cpu'  # Force CPU
        )
        
        # Fit AMDU
        amdu_model.fit(X_train_poisoned, y_train_poisoned, teacher_model=poisoned_model)
        
        # Unlearn
        remaining_indices = [i for i in range(len(X_train_poisoned)) if i not in poison_indices]
        amdu_model.unlearn(X_train_poisoned, y_train_poisoned, poison_indices, remaining_indices)
        
        # Test
        amdu_pred = amdu_model.predict(X_test_small)
        amdu_f1 = f1_score(y_test_small, amdu_pred)
        
        print(f"AMDU F1: {amdu_f1:.4f}")
        print(f"Recovery: {amdu_f1 - poisoned_f1:.4f}")
        print(f"vs Baseline: {amdu_f1 - baseline_f1:.4f}")
        
        # Test forgetting effectiveness
        if len(poison_indices) > 10:
            X_forget_sample = X_train_poisoned[poison_indices[:10]]
            effectiveness = amdu_model.evaluate_forgetting_effectiveness(X_forget_sample)
            print(f"Forgetting effectiveness: {effectiveness:.4f}")
        
        print("\n✅ AMDU algorithm working successfully!")
        
    except Exception as e:
        print(f"❌ AMDU failed: {str(e)}")
        import traceback
        traceback.print_exc()

except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install PyTorch: pip install torch")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()