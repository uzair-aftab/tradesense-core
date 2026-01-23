"""
Transformer Model for Time Series Prediction

Uses Transformer Encoder architecture to capture long-range dependencies
in price data. Attention is All You Need.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import math

from src.ml.lstm_model import TimeSeriesDataset, DEVICE
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PositionalEncoding(nn.Module):
    """Injects some information about the relative or absolute position of the tokens."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerPredictor(nn.Module):
    """
    Transformer Encoder for time series classification.
    """
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        dim_feedforward: int = 128
    ):
        super().__init__()
        
        # Project input features to d_model dimension
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.input_projection(x)
        x = x * math.sqrt(x.size(-1))
        x = self.pos_encoder(x)
        
        # Transformer encoder
        output = self.transformer_encoder(x)
        
        # Global average pooling
        output = output.mean(dim=1)
        
        return self.classifier(output).squeeze(-1)


class TransformerModel:
    """Wrapper for Transformer training and prediction."""
    
    MODEL_PATH = "models/transformer_model.pt"
    
    def __init__(
        self,
        input_size: int = 39,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        sequence_length: int = 30,
        dropout: float = 0.1,
        learning_rate: float = 0.0005
    ):
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        
        self.model = TransformerPredictor(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        ).to(DEVICE)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        self.is_trained = False
        
        self.load()
        
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32
    ):
        """Train Transformer model."""
        logger.info(f"Training Transformer on {DEVICE}...")
        
        train_dataset = TimeSeriesDataset(X_train, y_train, self.sequence_length)
        val_dataset = TimeSeriesDataset(X_val, y_val, self.sequence_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                
            # Validation
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    preds = (outputs > 0.5).float()
                    val_correct += (preds == batch_y).sum().item()
                    val_total += batch_y.size(0)
            
            val_accuracy = val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"TF Epoch {epoch+1}: Val Acc {val_accuracy:.2%}, Loss {avg_val_loss:.4f}")
                
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save()
        
        self.is_trained = True
        return {"val_accuracy": val_accuracy}

    def predict_single(self, X_sequence: np.ndarray) -> float:
        if not self.is_trained:
            return 0.5
        self.model.eval()
        x = torch.FloatTensor(X_sequence).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            return self.model(x).item()
            
    def save(self):
        torch.save(self.model.state_dict(), self.MODEL_PATH)
        self.is_trained = True
        
    def load(self):
        if os.path.exists(self.MODEL_PATH):
            try:
                self.model.load_state_dict(torch.load(self.MODEL_PATH, weights_only=True))
                self.is_trained = True
            except Exception as e:
                logger.error(f"Transformer load failed: {e}")


# Singleton
transformer_model = TransformerModel()
