"""
TradeSense Reinforcement Learning Environment

A gymnasium-compatible environment for training RL trading agents.
Simulates market mechanics, fees, and observation state.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict

from src.ml import FeatureEngineer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TradingEnv(gym.Env):
    """
    Trading Environment that follows gymnasium interface.
    
    Actions:
        0: HOLD
        1: BUY (Enter Long / Add)
        2: SELL (Close Long / Short?) -> For now, just Close Long or Sell
        
    Observation Space:
        Box(low=-inf, high=inf, shape=(feature_dim + account_state,))
    """
    
    metadata = {'render_modes': ['human']}

    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0, fee_pct: float = 0.001):
        super(TradingEnv, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.fee_pct = fee_pct
        self.feature_engineer = FeatureEngineer()
        
        # Precompute features
        # We need to ensure features are numeric and clean
        self.features_df = self.feature_engineer.calculate_features(df, include_targets=False)
        self.features_df = self.features_df.fillna(0)
        
        # Align features with price data
        self.prices = df['Close'].loc[self.features_df.index].values
        self.feature_array = self.features_df[self.feature_engineer.feature_columns].values
        self.dates = self.features_df.index
        
        # Action Space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation Space
        # Features + [Balance, Net Worth, Position Size, Held Price, Unrealized PnL]
        self.n_features = self.feature_array.shape[1]
        self.n_account_state = 5
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.n_features + self.n_account_state,), 
            dtype=np.float32
        )
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.position = 0.0  # Fraction of portfolio? Or number of shares? 
                             # Let's simple: 0 = No Position, 1 = Full Position
                             # Or shares count. Let's use shares count for realism.
        self.shares_held = 0
        self.cost_basis = 0.0
        
        self.history = []
        
        return self._get_observation(), {}
        
    def _get_observation(self):
        # Current market features
        market_obs = self.feature_array[self.current_step]
        
        # Account state
        current_price = self.prices[self.current_step]
        unrealized_pnl = 0.0
        if self.shares_held > 0:
            unrealized_pnl = (current_price - self.cost_basis) / self.cost_basis
            
        account_obs = np.array([
            self.balance / self.initial_balance,    # Normalized balance
            self.net_worth / self.initial_balance,  # Normalized net worth
            1.0 if self.shares_held > 0 else 0.0,   # Has position
            unrealized_pnl,                         # PnL pct
            self.current_step / len(self.df)        # Time progress
        ], dtype=np.float32)
        
        return np.concatenate((market_obs, account_obs))
        
    def step(self, action):
        current_price = self.prices[self.current_step]
        reward = 0
        terminated = False
        truncated = False
        
        prev_net_worth = self.net_worth
        
        # Execute Action
        if action == 1: # BUY
            if self.shares_held == 0:
                # Buy as much as possible
                # Fee is taken from balance
                max_shares = int(self.balance / (current_price * (1 + self.fee_pct)))
                if max_shares > 0:
                    cost = max_shares * current_price
                    fee = cost * self.fee_pct
                    self.balance -= (cost + fee)
                    self.shares_held = max_shares
                    self.cost_basis = current_price
                    # Log buy?
            else:
                pass # Already holding, do nothing (or could add to position)
                
        elif action == 2: # SELL
            if self.shares_held > 0:
                # Sell all
                proceeds = self.shares_held * current_price
                fee = proceeds * self.fee_pct
                self.balance += (proceeds - fee)
                self.shares_held = 0
                self.cost_basis = 0.0
                
        elif action == 0: # HOLD
            pass
            
        # Update Net Worth
        portfolio_value = self.shares_held * current_price
        self.net_worth = self.balance + portfolio_value
        
        # Reward Function
        # 1. Log return of net worth
        step_return = (self.net_worth - prev_net_worth) / prev_net_worth if prev_net_worth > 0 else 0
        
        # 2. Add minimal reward for holding if trend is up? No, keep it pure.
        reward = step_return * 100 # Scale up for better gradient
        
        # 3. Penalty for drawdown? (Optional, handled by PPO value function mostly)
        
        # Step forward
        self.current_step += 1
        
        # Check termination
        # We need to stop before we run out of data
        if self.current_step >= len(self.prices) - 1:
            terminated = True
            # Clamp for final observation to avoid OOB
            self.current_step = len(self.prices) - 1
            
        # Check if blown up
        if self.net_worth < self.initial_balance * 0.5:
            terminated = True
            reward -= 10
            
        obs = self._get_observation()
        info = {
            "net_worth": self.net_worth, 
            "shares": self.shares_held, 
            "price": current_price
        }
        
        return obs, reward, terminated, truncated, info
        
    def render(self):
        print(f"Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, Held: {self.shares_held}")
