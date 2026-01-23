"""
LSTM Deep Learning Model for Price Prediction

Phase 15: Sequential pattern recognition using LSTM with attention.
Optimized for short-term (intraday) predictions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
import os
import json

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TimeSeriesDataset(Dataset):
    """Dataset for LSTM time series prediction."""
    
    def __init__(
        self, 
        features: np.ndarray, 
        targets: Optional[np.ndarray] = None,
        sequence_length: int = 30
    ):
        """
        Args:
            features: Feature array of shape (samples, features)
            targets: Target array of shape (samples,)
            sequence_length: Number of time steps in each sequence
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        self.seq_len = sequence_length
        
    def __len__(self):
        return len(self.features) - self.seq_len
    
    def __getitem__(self, idx):
        # Get sequence of features
        x = self.features[idx:idx + self.seq_len]
        
        if self.targets is not None:
            # Target is the value at the end of the sequence
            y = self.targets[idx + self.seq_len]
            return x, y
        return x


class Attention(nn.Module):
    """Simple attention mechanism for LSTM outputs."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_size)
        attention_weights = torch.softmax(
            self.attention(lstm_output).squeeze(-1), dim=1
        )
        # Weighted sum of hidden states
        context = torch.bmm(
            attention_weights.unsqueeze(1), 
            lstm_output
        ).squeeze(1)
        return context, attention_weights


class LSTMPredictor(nn.Module):
    """
    Bidirectional LSTM with Attention for price prediction.
    
    Architecture:
    - Input normalization (BatchNorm)
    - Bidirectional LSTM (2 layers)
    - Attention mechanism
    - Fully connected layers
    - Sigmoid output (probability of price increase)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_size)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention layer
        self.attention = Attention(hidden_size * 2)  # *2 for bidirectional
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        batch_size, seq_len, features = x.shape
        
        # Apply batch normalization per feature
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        x = self.input_norm(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, features)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_size * 2)
        
        # Attention
        context, _ = self.attention(lstm_out)
        
        # Classification
        out = self.fc(context)
        
        return out.squeeze(-1)


class LSTMModel:
    """
    High-level wrapper for LSTM training and prediction.
    
    Example:
        >>> model = LSTMModel(input_size=39)
        >>> model.train(X_train, y_train, X_val, y_val, epochs=50)
        >>> predictions = model.predict(X_test)
    """
    
    MODEL_PATH = "models/lstm_model.pt"
    CONFIG_PATH = "models/lstm_config.json"
    
    def __init__(
        self,
        input_size: int = 39,
        hidden_size: int = 128,
        num_layers: int = 2,
        sequence_length: int = 30,
        dropout: float = 0.3,
        learning_rate: float = 0.001
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        self.model = LSTMPredictor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(DEVICE)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        self.is_trained = False
        self.training_history = []
        
        # Try to load existing model
        self.load()
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 10
    ) -> dict:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training features (samples, features)
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            early_stopping_patience: Stop if no improvement for N epochs
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training LSTM on {DEVICE}...")
        logger.info(f"Input: {X_train.shape}, Sequence length: {self.sequence_length}")
        
        # Create datasets
        train_dataset = TimeSeriesDataset(X_train, y_train, self.sequence_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TimeSeriesDataset(X_val, y_val, self.sequence_length)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                
                train_loss += loss.item()
                predictions = (outputs > 0.5).float()
                train_correct += (predictions == batch_y).sum().item()
                train_total += batch_y.size(0)
            
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            
            # Validation phase
            val_loss = 0
            val_accuracy = 0
            
            if val_loader:
                self.model.eval()
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(DEVICE)
                        batch_y = batch_y.to(DEVICE)
                        
                        outputs = self.model(batch_x)
                        loss = self.criterion(outputs, batch_y)
                        val_loss += loss.item()
                        
                        predictions = (outputs > 0.5).float()
                        val_correct += (predictions == batch_y).sum().item()
                        val_total += batch_y.size(0)
                
                val_loss = val_loss / len(val_loader)
                val_accuracy = val_correct / val_total
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save()  # Save best model
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break
            
            # Log progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Train Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.2%} | "
                    f"Val Loss: {val_loss:.4f}, Acc: {val_accuracy:.2%}"
                )
            
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            })
        
        self.is_trained = True
        self.save()
        
        return {
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'best_val_loss': best_val_loss,
            'epochs_trained': len(self.training_history)
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for new data.
        
        Args:
            X: Features (samples, features)
            
        Returns:
            Array of probabilities
        """
        if not self.is_trained:
            logger.warning("Model not trained, returning 0.5")
            return np.full(len(X) - self.sequence_length, 0.5)
        
        self.model.eval()
        dataset = TimeSeriesDataset(X, sequence_length=self.sequence_length)
        loader = DataLoader(dataset, batch_size=64)
        
        predictions = []
        with torch.no_grad():
            for batch_x in loader:
                batch_x = batch_x.to(DEVICE)
                outputs = self.model(batch_x)
                predictions.extend(outputs.cpu().numpy())
        
        return np.array(predictions)
    
    def predict_single(self, X_sequence: np.ndarray) -> float:
        """
        Predict for a single sequence.
        
        Args:
            X_sequence: Feature sequence (seq_len, features)
            
        Returns:
            Probability of price increase
        """
        if not self.is_trained:
            return 0.5
        
        self.model.eval()
        x = torch.FloatTensor(X_sequence).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = self.model(x)
        
        return output.item()
    
    def save(self):
        """Save model and config to disk."""
        os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
        
        # Save model weights
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'is_trained': self.is_trained,
            'training_history': self.training_history
        }, self.MODEL_PATH)
        
        # Save config
        config = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'sequence_length': self.sequence_length,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate
        }
        with open(self.CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"LSTM model saved to {self.MODEL_PATH}")
    
    def load(self):
        """Load model from disk if exists."""
        if not os.path.exists(self.MODEL_PATH):
            return
        
        try:
            checkpoint = torch.load(self.MODEL_PATH, map_location=DEVICE, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.is_trained = checkpoint.get('is_trained', False)
            self.training_history = checkpoint.get('training_history', [])
            logger.info(f"LSTM model loaded from {self.MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")


# Singleton instance
lstm_model = LSTMModel(input_size=39)
