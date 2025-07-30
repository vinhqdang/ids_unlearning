"""
Basic test of data loading functionality
"""
import sys
sys.path.append('src')

from data_loader import NSLKDDDataLoader

def main():
    print("Testing NSL-KDD Data Loader...")
    
    # Initialize data loader
    data_loader = NSLKDDDataLoader()
    
    # Load datasets
    print("Loading datasets...")
    train_df, test_df = data_loader.load_data()
    
    # Show basic info
    print(f"\nTraining data shape: {train_df.shape}")
    print(f"Testing data shape: {test_df.shape}")
    
    # Show attack distribution
    print("\nAttack types in training set:")
    print(train_df['attack_type'].value_counts().head())
    
    # Test preprocessing
    print("\nTesting preprocessing...")
    X_train, X_test, y_train, y_test = data_loader.preprocess_data(
        train_df, test_df, binary_classification=True
    )
    
    print(f"Preprocessed X_train shape: {X_train.shape}")
    print(f"Preprocessed X_test shape: {X_test.shape}")
    print(f"Labels - Normal: {sum(y_train == 0)}, Attack: {sum(y_train == 1)}")
    
    print("Data loading test completed successfully!")

if __name__ == "__main__":
    main()