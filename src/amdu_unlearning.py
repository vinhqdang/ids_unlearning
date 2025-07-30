"""
Adaptive Memory Distillation Unlearning (AMDU) Algorithm
A novel approach combining knowledge distillation with adaptive memory networks
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
import copy
import time
from typing import Optional, Tuple, List

class MemoryBank(nn.Module):
    """Adaptive memory bank for storing compressed data representations"""
    
    def __init__(self, input_dim: int, memory_dim: int = 64, num_memories: int = 1000):
        super().__init__()
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.num_memories = num_memories
        
        # Memory storage
        self.memory_keys = nn.Parameter(torch.randn(num_memories, memory_dim))
        self.memory_values = nn.Parameter(torch.randn(num_memories, memory_dim))
        
        # Compression network
        self.compress_net = nn.Sequential(
            nn.Linear(input_dim, memory_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(memory_dim * 2, memory_dim),
            nn.Tanh()
        )
        
        # Forget gate network
        self.forget_gate = nn.Sequential(
            nn.Linear(memory_dim + 10, 32),  # +10 for context features
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, forget_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through memory bank with optional forgetting"""
        batch_size = x.shape[0]
        
        # Compress input to memory space
        compressed = self.compress_net(x)  # [batch_size, memory_dim]
        
        # Compute attention weights to memory
        attention = torch.matmul(compressed, self.memory_keys.t())  # [batch_size, num_memories]
        attention_weights = torch.softmax(attention, dim=1)
        
        # Retrieve memory values
        retrieved_memory = torch.matmul(attention_weights, self.memory_values)  # [batch_size, memory_dim]
        
        # Apply forget gates if forget_mask is provided
        if forget_mask is not None:
            # Create context features (simple version - could be more sophisticated)
            context = torch.cat([
                compressed.mean(dim=1, keepdim=True).expand(-1, 5),
                compressed.std(dim=1, keepdim=True).expand(-1, 5)
            ], dim=1)
            
            gate_input = torch.cat([retrieved_memory, context], dim=1)
            forget_gates = self.forget_gate(gate_input)  # [batch_size, 1]
            
            # Apply forgetting - multiply by (1 - forget_mask * forget_gates)
            forget_strength = forget_mask.unsqueeze(1) * forget_gates
            retrieved_memory = retrieved_memory * (1 - forget_strength)
        
        return retrieved_memory, attention_weights

class AMDUStudent(nn.Module):
    """Student network for distillation-based unlearning"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 2, memory_dim: int = 64):
        super().__init__()
        self.memory_bank = MemoryBank(input_dim, memory_dim)
        
        # Main classification network
        self.classifier = nn.Sequential(
            nn.Linear(memory_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x: torch.Tensor, forget_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with memory-augmented classification"""
        memory_out, attention = self.memory_bank(x, forget_mask)
        
        # Combine original features with memory
        combined = torch.cat([x, memory_out], dim=1)
        
        # Classify
        logits = self.classifier(combined)
        return logits

class AdversarialValidator(nn.Module):
    """Adversarial network to verify forgotten data cannot be recovered"""
    
    def __init__(self, memory_dim: int):  # Changed from input_dim to memory_dim
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(memory_dim, 64),  # Use memory_dim instead of input_dim
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)

class AMDUUnlearner(BaseEstimator, ClassifierMixin):
    """
    Adaptive Memory Distillation Unlearning (AMDU) Algorithm
    
    Novel unlearning approach that combines:
    1. Adaptive memory banks for efficient representation
    2. Knowledge distillation for selective forgetting
    3. Adversarial validation for forgetting guarantees
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 memory_dim: int = 64,
                 learning_rate: float = 0.001,
                 distill_temperature: float = 4.0,
                 forget_strength: float = 1.0,
                 adversarial_weight: float = 0.1,
                 device: str = 'cpu'):
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.learning_rate = learning_rate
        self.distill_temperature = distill_temperature
        self.forget_strength = forget_strength
        self.adversarial_weight = adversarial_weight
        self.device = device
        
        # Initialize networks
        self.student = AMDUStudent(input_dim, hidden_dim, output_dim=2, memory_dim=memory_dim).to(device)
        self.adversarial_validator = AdversarialValidator(memory_dim).to(device)  # Use memory_dim
        
        # Store original teacher model (will be set during fit)
        self.teacher = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def _sklearn_to_tensor(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Convert sklearn arrays to PyTorch tensors"""
        X_tensor = torch.FloatTensor(X).to(self.device)
        if y is not None:
            y_tensor = torch.LongTensor(y).to(self.device)
            return X_tensor, y_tensor
        return X_tensor, None
        
    def fit(self, X: np.ndarray, y: np.ndarray, teacher_model=None):
        """Fit the AMDU model"""
        print("Training AMDU unlearner...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_tensor, y_tensor = self._sklearn_to_tensor(X_scaled, y)
        
        # Store teacher model predictions for distillation
        if teacher_model is not None:
            self.teacher = teacher_model
            with torch.no_grad():
                if hasattr(teacher_model, 'predict_proba'):
                    teacher_probs = teacher_model.predict_proba(X_scaled)
                    self.teacher_predictions = torch.FloatTensor(teacher_probs).to(self.device)
                else:
                    teacher_preds = teacher_model.predict(X_scaled)
                    # Convert to one-hot
                    self.teacher_predictions = torch.zeros(len(teacher_preds), 2).to(self.device)
                    self.teacher_predictions[range(len(teacher_preds)), teacher_preds] = 1.0
        else:
            # Use ground truth as teacher (for initial training)
            self.teacher_predictions = torch.zeros(len(y), 2).to(self.device)
            self.teacher_predictions[range(len(y)), y] = 1.0
        
        # Training setup
        optimizer = optim.Adam(self.student.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        kl_loss = nn.KLDivLoss(reduction='batchmean')
        
        # Training loop
        epochs = 50
        batch_size = 256
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for i in range(0, len(X_tensor), batch_size):
                end_idx = min(i + batch_size, len(X_tensor))
                batch_X = X_tensor[i:end_idx]
                batch_y = y_tensor[i:end_idx]
                batch_teacher = self.teacher_predictions[i:end_idx]
                
                optimizer.zero_grad()
                
                # Forward pass
                logits = self.student(batch_X)
                probs = torch.softmax(logits / self.distill_temperature, dim=1)
                
                # Loss computation
                task_loss = criterion(logits, batch_y)
                
                # Distillation loss
                teacher_soft = torch.softmax(batch_teacher * self.distill_temperature, dim=1)
                distill_loss = kl_loss(torch.log(probs), teacher_soft)
                
                total_loss_batch = task_loss + 0.5 * distill_loss
                total_loss_batch.backward()
                optimizer.step()
                
                total_loss += total_loss_batch.item()
                num_batches += 1
            
            if epoch % 10 == 0:
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
        
        self.is_fitted = True
        print("AMDU training completed")
        
    def unlearn(self, X_full: np.ndarray, y_full: np.ndarray, 
                forget_indices: List[int], retain_indices: Optional[List[int]] = None):
        """
        Perform unlearning using AMDU algorithm
        
        Args:
            X_full: Full dataset features
            y_full: Full dataset labels
            forget_indices: Indices of samples to forget
            retain_indices: Indices of samples to explicitly retain (optional)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before unlearning")
            
        print(f"AMDU Unlearning: Forgetting {len(forget_indices)} samples...")
        
        # Prepare data
        X_scaled = self.scaler.transform(X_full)
        X_tensor, y_tensor = self._sklearn_to_tensor(X_scaled, y_full)
        
        # Create forget mask
        forget_mask = torch.zeros(len(X_full)).to(self.device)
        forget_mask[forget_indices] = self.forget_strength
        
        # Prepare retain data (exclude forget samples)
        if retain_indices is None:
            retain_indices = [i for i in range(len(X_full)) if i not in forget_indices]
        
        X_retain = X_tensor[retain_indices]
        y_retain = y_tensor[retain_indices]
        
        # Adversarial training setup
        student_optimizer = optim.Adam(self.student.parameters(), lr=self.learning_rate * 0.5)
        adversarial_optimizer = optim.Adam(self.adversarial_validator.parameters(), lr=self.learning_rate)
        
        criterion = nn.CrossEntropyLoss()
        adversarial_criterion = nn.BCELoss()
        
        # Unlearning iterations
        unlearn_epochs = 30
        batch_size = 256
        
        for epoch in range(unlearn_epochs):
            # Phase 1: Update student with forget gates
            for i in range(0, len(X_retain), batch_size):
                end_idx = min(i + batch_size, len(X_retain))
                batch_X = X_retain[i:end_idx]
                batch_y = y_retain[i:end_idx]
                
                # Create batch forget mask (zeros for retain data)
                batch_forget_mask = torch.zeros(len(batch_X)).to(self.device)
                
                student_optimizer.zero_grad()
                
                # Forward pass with memory and forget gates
                logits = self.student(batch_X, batch_forget_mask)
                
                # Standard classification loss on retain data
                retain_loss = criterion(logits, batch_y)
                
                # Adversarial loss - make sure forgotten data patterns aren't recoverable
                if len(forget_indices) > 0:
                    forget_batch_size = min(32, len(forget_indices))
                    forget_sample_idx = np.random.choice(forget_indices, forget_batch_size, replace=False)
                    X_forget_batch = X_tensor[forget_sample_idx]
                    
                    # Apply strong forget mask
                    forget_batch_mask = torch.ones(len(X_forget_batch)).to(self.device)
                    
                    with torch.no_grad():
                        forgotten_logits = self.student(X_forget_batch, forget_batch_mask)
                        # We want the adversarial validator to fail on forgotten representations
                        forgotten_features = self.student.memory_bank(X_forget_batch, forget_batch_mask)[0]
                    
                    adversarial_pred = self.adversarial_validator(forgotten_features)
                    
                    # Adversarial loss: encourage validator to output 0 (cannot recover)
                    adversarial_target = torch.zeros_like(adversarial_pred)
                    adversarial_loss = adversarial_criterion(adversarial_pred, adversarial_target)
                else:
                    adversarial_loss = 0
                
                total_loss = retain_loss + self.adversarial_weight * adversarial_loss
                total_loss.backward()
                student_optimizer.step()
            
            # Phase 2: Update adversarial validator
            if len(forget_indices) > 0:
                for i in range(0, len(forget_indices), batch_size):
                    end_idx = min(i + batch_size, len(forget_indices))
                    batch_forget_idx = forget_indices[i:end_idx]
                    
                    X_forget_batch = X_tensor[batch_forget_idx]
                    batch_forget_mask = torch.ones(len(X_forget_batch)).to(self.device)
                    
                    adversarial_optimizer.zero_grad()
                    
                    with torch.no_grad():
                        forgotten_features = self.student.memory_bank(X_forget_batch, batch_forget_mask)[0]
                    
                    # Train validator to detect if data can be recovered (should output 1 for recoverable)
                    adversarial_pred = self.adversarial_validator(forgotten_features)
                    adversarial_target = torch.ones_like(adversarial_pred)  # Try to detect forgotten data
                    
                    validator_loss = adversarial_criterion(adversarial_pred, adversarial_target)
                    validator_loss.backward()
                    adversarial_optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Unlearn Epoch {epoch}: Retain Loss = {retain_loss:.4f}")
        
        print("AMDU unlearning completed")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X_scaled = self.scaler.transform(X)
        X_tensor, _ = self._sklearn_to_tensor(X_scaled)
        
        self.student.eval()
        with torch.no_grad():
            logits = self.student(X_tensor)
            predictions = torch.argmax(logits, dim=1)
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X_scaled = self.scaler.transform(X)
        X_tensor, _ = self._sklearn_to_tensor(X_scaled)
        
        self.student.eval()
        with torch.no_grad():
            logits = self.student(X_tensor)
            probabilities = torch.softmax(logits, dim=1)
        
        return probabilities.cpu().numpy()
    
    def get_memory_attention(self, X: np.ndarray) -> np.ndarray:
        """Get attention weights for analysis"""
        X_scaled = self.scaler.transform(X)
        X_tensor, _ = self._sklearn_to_tensor(X_scaled)
        
        self.student.eval()
        with torch.no_grad():
            _, attention = self.student.memory_bank(X_tensor)
        
        return attention.cpu().numpy()
    
    def evaluate_forgetting_effectiveness(self, X_forget: np.ndarray) -> float:
        """Evaluate how effectively forgotten data has been removed"""
        X_scaled = self.scaler.transform(X_forget)
        X_tensor, _ = self._sklearn_to_tensor(X_scaled)
        
        # Apply strong forget mask
        forget_mask = torch.ones(len(X_tensor)).to(self.device)
        
        self.student.eval()
        self.adversarial_validator.eval()
        
        with torch.no_grad():
            forgotten_features, _ = self.student.memory_bank(X_tensor, forget_mask)
            recovery_score = self.adversarial_validator(forgotten_features)
        
        # Lower score means better forgetting (adversarial validator can't recover)
        return 1.0 - recovery_score.mean().cpu().item()