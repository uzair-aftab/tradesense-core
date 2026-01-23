"""
Advanced Hyperparameter Optimization with Optuna.

Uses Bayesian optimization (TPE) to find optimal hyperparameters 
for XGBoost and LSTM models much faster than GridSearch.
Includes pruning of unpromising trials.
"""

import optuna
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json

from src.ml import FeatureEngineer, signal_model, lstm_model
from src.ml.lstm_model import LSTMPredictor, TimeSeriesDataset, DEVICE
from src.data import stock_fetcher
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Reduce Optuna verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ModelOptimizer:
    """
    Manages hyperparameter optimization for all models using Optuna.
    """
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        
    def load_data(self, tickers: list = None, period: str = "2y"):
        """Load and prepare multi-ticker data for optimization."""
        if tickers is None:
            tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "AMD", "META", "AMZN"]
            
        logger.info(f"Loading data for optimization: {len(tickers)} tickers, {period}")
        
        all_features = []
        for ticker in tickers:
            try:
                df = stock_fetcher.get_historical(ticker, period=period)
                features_df = self.feature_engineer.calculate_features(
                    df, include_targets=True, prediction_horizon=3
                )
                if not features_df.empty:
                    all_features.append(features_df)
            except Exception as e:
                logger.warning(f"Failed to load {ticker}: {e}")
                
        if not all_features:
            raise ValueError("No data loaded for optimization")
            
        combined = pd.concat(all_features, ignore_index=True)
        
        # Prepare X and y
        feature_cols = self.feature_engineer.feature_columns
        X = combined[feature_cols].values
        y = combined['target'].values
        
        # Train/val split (time-ordered)
        split_idx = int(len(X) * 0.8)
        self.X_train, self.X_val = X[:split_idx], X[split_idx:]
        self.y_train, self.y_val = y[:split_idx], y[split_idx:]
        
        logger.info(f"Data prepared: {len(self.X_train)} train, {len(self.X_val)} val samples")
        
    def optimize_xgboost(self, n_trials: int = 50, save_model: bool = True) -> dict:
        """
        Optimize XGBoost hyperparameters.
        
        Args:
            n_trials: Number of optimization trials
            save_model: Whether to save the best model
            
        Returns:
            Dict of best parameters
        """
        if self.X_train is None:
            self.load_data()
            
        logger.info(f"Starting XGBoost optimization ({n_trials} trials)...")
        
        def objective(trial):
            param = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
                'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
                'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'eta': trial.suggest_float('eta', 0.01, 0.3),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            }
            
            if param['booster'] == 'dart':
                param['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
                param['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
                param['rate_drop'] = trial.suggest_float('rate_drop', 1e-8, 1.0, log=True)
                param['skip_drop'] = trial.suggest_float('skip_drop', 1e-8, 1.0, log=True)
            
            # Create DMatrix
            dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
            dval = xgb.DMatrix(self.X_val, label=self.y_val)
            
            # Pruning callback
            pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-error")
            
            # Train model
            bst = xgb.train(
                param, dtrain, 
                num_boost_round=100, 
                evals=[(dval, "validation")],
                callbacks=[pruning_callback],
                early_stopping_rounds=10,
                verbose_eval=False
            )
            
            # Prediction
            preds = bst.predict(dval)
            pred_labels = np.rint(preds)
            accuracy = accuracy_score(self.y_val, pred_labels)
            
            return accuracy

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best XGBoost trial: {study.best_trial.value:.4f}")
        logger.info(f"Best params: {study.best_trial.params}")
        
        if save_model:
            # Retrain with best params and save
            best_params = study.best_trial.params
            best_params['objective'] = 'binary:logistic'
            best_params['eval_metric'] = 'logloss'
            
            clf = xgb.XGBClassifier(**best_params, n_estimators=100)
            clf.fit(self.X_train, self.y_train, eval_set=[(self.X_val, self.y_val)], verbose=False)
            
            # Update production model
            signal_model.model = clf
            signal_model.save()
            logger.info("Saved optimized XGBoost model to disk")
            
        return study.best_trial.params

    def optimize_lstm(self, n_trials: int = 30, save_model: bool = True) -> dict:
        """
        Optimize LSTM hyperparameters.
        
        Args:
            n_trials: Number of optimization trials
            save_model: Whether to save the best model
            
        Returns:
            Dict of best parameters
        """
        if self.X_train is None:
            self.load_data()
            
        logger.info(f"Starting LSTM optimization ({n_trials} trials)...")
        
        def objective(trial):
            # Sample parameters
            sequence_length = trial.suggest_int("sequence_length", 20, 60, step=10)
            hidden_size = trial.suggest_int("hidden_size", 64, 256, step=32)
            num_layers = trial.suggest_int("num_layers", 1, 3)
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            dropout = trial.suggest_float("dropout", 0.1, 0.5)
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
            
            # Prepare data with proposed sequence length
            train_dataset = TimeSeriesDataset(self.X_train, self.y_train, sequence_length)
            val_dataset = TimeSeriesDataset(self.X_val, self.y_val, sequence_length)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Model setup
            model = LSTMPredictor(
                input_size=self.X_train.shape[1],
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
            ).to(DEVICE)
            
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.BCELoss()
            
            # Smaller epoch count for tuning
            epochs = 10
            
            # Training loop
            for epoch in range(epochs):
                model.train()
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                # Validation
                model.eval()
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                        outputs = model(batch_x)
                        preds = (outputs > 0.5).float()
                        val_correct += (preds == batch_y).sum().item()
                        val_total += batch_y.size(0)
                        
                accuracy = val_correct / val_total
                
                # Report intermediate result for pruning
                trial.report(accuracy, epoch)
                
                # Handle pruning
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            
            return accuracy
            
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best LSTM trial: {study.best_trial.value:.4f}")
        logger.info(f"Best params: {study.best_trial.params}")
        
        if save_model:
            # Retrain best model fully
            best_params = study.best_trial.params
            
            # Update singleton instance settings
            lstm_model.hidden_size = best_params['hidden_size']
            lstm_model.num_layers = best_params['num_layers']
            lstm_model.sequence_length = best_params['sequence_length']
            lstm_model.learning_rate = best_params['lr']
            lstm_model.dropout = best_params['dropout']
            
            # Rebuild model architecture
            lstm_model.model = LSTMPredictor(
                input_size=self.X_train.shape[1],
                hidden_size=best_params['hidden_size'],
                num_layers=best_params['num_layers'],
                dropout=best_params['dropout']
            ).to(DEVICE)
            
            lstm_model.optimizer = optim.Adam(lstm_model.model.parameters(), lr=best_params['lr'])
            
            # Full training
            metrics = lstm_model.train(
                self.X_train, self.y_train,
                self.X_val, self.y_val,
                epochs=50,  # Full training
                batch_size=best_params['batch_size'],
                early_stopping_patience=10
            )
            
            logger.info(f"Saved optimized LSTM model (Acc: {metrics['val_accuracy']:.2%})")
            
        return study.best_trial.params


# Singleton
optimizer = ModelOptimizer()
