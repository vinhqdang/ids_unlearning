"""
NSL-KDD Dataset Loading and Preprocessing Module
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class NSLKDDDataLoader:
    def __init__(self):
        # NSL-KDD feature names based on the original KDD Cup 99 dataset
        self.feature_names = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
            'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
            'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
            'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
            'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
            'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 
            'attack_type', 'difficulty_level'
        ]
        
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self, train_path='data/KDDTrain_plus.txt', test_path='data/KDDTest_plus.txt'):
        """Load training and testing datasets"""
        
        # Load training data
        train_df = pd.read_csv(train_path, header=None, names=self.feature_names)
        
        # Load testing data  
        test_df = pd.read_csv(test_path, header=None, names=self.feature_names)
        
        print(f"Training data shape: {train_df.shape}")
        print(f"Testing data shape: {test_df.shape}")
        
        return train_df, test_df
    
    def preprocess_data(self, train_df, test_df, binary_classification=True):
        """
        Preprocess the datasets for machine learning
        
        Args:
            train_df: Training dataframe
            test_df: Testing dataframe  
            binary_classification: If True, convert to binary (normal vs attack)
                                  If False, keep multi-class classification
        """
        
        # Combine data for consistent preprocessing
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        
        # Separate features and labels
        X_combined = combined_df.drop(['attack_type', 'difficulty_level'], axis=1)
        y_combined = combined_df['attack_type'].copy()
        
        # Encode categorical features
        categorical_features = ['protocol_type', 'service', 'flag']
        
        for feature in categorical_features:
            le = LabelEncoder()
            X_combined[feature] = le.fit_transform(X_combined[feature])
            self.label_encoders[feature] = le
        
        # Process labels
        if binary_classification:
            # Convert to binary: normal=0, attack=1
            y_combined = (y_combined != 'normal').astype(int)
        else:
            # Multi-class: encode all attack types
            le_label = LabelEncoder()
            y_combined = le_label.fit_transform(y_combined)
            self.label_encoders['attack_type'] = le_label
        
        # Split back to train/test
        train_size = len(train_df)
        X_train = X_combined.iloc[:train_size]
        X_test = X_combined.iloc[train_size:]
        y_train = y_combined.iloc[:train_size]
        y_test = y_combined.iloc[train_size:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train.values, y_test.values
    
    def get_attack_distribution(self, train_df, test_df):
        """Get distribution of attack types in the datasets"""
        
        print("Training set attack distribution:")
        train_dist = train_df['attack_type'].value_counts()
        print(train_dist)
        
        print("\nTesting set attack distribution:")  
        test_dist = test_df['attack_type'].value_counts()
        print(test_dist)
        
        return train_dist, test_dist
    
    def get_feature_info(self):
        """Get information about the features"""
        feature_info = {
            'total_features': len(self.feature_names) - 2,  # Excluding attack_type and difficulty_level
            'categorical_features': ['protocol_type', 'service', 'flag'],
            'numerical_features': [f for f in self.feature_names[:-2] if f not in ['protocol_type', 'service', 'flag']]
        }
        
        return feature_info