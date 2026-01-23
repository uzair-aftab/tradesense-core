"""
PPO Trading Agent

Wrapper for Stable Baselines 3 PPO agent.
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd
import numpy as np

from src.rl.environment import TradingEnv
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TradingAgent:
    """
    Reinforcement Learning Agent using PPO.
    """
    
    MODEL_PATH = "models/ppo_agent.zip"
    
    def __init__(self, verbose: int = 1):
        self.model = None
        self.verbose = verbose
        
    def train(self, df: pd.DataFrame, timesteps: int = 100000):
        """
        Train the agent on historical data.
        
        Args:
            df: OHLCV DataFrame
            timesteps: Total training steps
        """
        logger.info(f"Setting up PPO training ({timesteps} steps)...")
        
        # vectorized environment
        train_env = DummyVecEnv([lambda: TradingEnv(df)])
        
        self.model = PPO(
            "MlpPolicy", 
            train_env, 
            verbose=self.verbose,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01, # Encourage exploration
        )
        
        self.model.learn(total_timesteps=timesteps)
        self.save()
        logger.info("Agent training complete and saved.")
        
    def predict(self, observation: np.ndarray) -> int:
        """
        Get action for a single observation.
        """
        if self.model is None:
            self.load()
            
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)
        
    def save(self):
        if self.model:
            self.model.save(self.MODEL_PATH)
            
    def load(self):
        if os.path.exists(self.MODEL_PATH):
            try:
                self.model = PPO.load(self.MODEL_PATH)
            except Exception as e:
                logger.error(f"Failed to load PPO agent: {e}")
                
# Singleton
trading_agent = TradingAgent()
