"""
Paper Trading Engine

Allows users to trade with virtual money to test signals before risking real capital.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TradeAction(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Trade:
    """Represents a single trade."""
    id: str
    ticker: str
    action: str
    quantity: int
    price: float
    timestamp: str
    signal_confidence: float = 0.0
    pnl: float = 0.0
    closed: bool = False
    close_price: float = 0.0
    close_timestamp: str = ""


@dataclass
class Position:
    """Current position in a stock."""
    ticker: str
    quantity: int
    avg_price: float
    current_price: float = 0.0
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_price
    
    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_basis
    
    @property
    def unrealized_pnl_pct(self) -> float:
        if self.cost_basis == 0:
            return 0
        return (self.unrealized_pnl / self.cost_basis) * 100


@dataclass
class Portfolio:
    """Paper trading portfolio."""
    cash: float = 10000.0
    starting_balance: float = 10000.0
    positions: Dict[str, Position] = field(default_factory=dict)
    trades: List[Trade] = field(default_factory=list)
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    @property
    def total_value(self) -> float:
        positions_value = sum(p.market_value for p in self.positions.values())
        return self.cash + positions_value
    
    @property
    def total_return(self) -> float:
        return ((self.total_value - self.starting_balance) / self.starting_balance) * 100
    
    @property
    def realized_pnl(self) -> float:
        return sum(t.pnl for t in self.trades if t.closed)
    
    @property
    def win_rate(self) -> float:
        closed_trades = [t for t in self.trades if t.closed]
        if not closed_trades:
            return 0.0
        wins = sum(1 for t in closed_trades if t.pnl > 0)
        return (wins / len(closed_trades)) * 100


class PaperTradingEngine:
    """
    Engine for paper trading with virtual money.
    
    Example:
        >>> engine = PaperTradingEngine()
        >>> engine.buy("AAPL", 10, 150.00, 0.8)
        >>> engine.update_prices({"AAPL": 160.00})
        >>> print(engine.portfolio.total_return)
    """
    
    SAVE_PATH = "data/paper_portfolio.json"
    
    def __init__(self, starting_balance: float = 10000.0, load_existing: bool = True):
        """
        Initialize paper trading engine.
        
        Args:
            starting_balance: Starting virtual cash
            load_existing: Load existing portfolio if available
        """
        self.portfolio = Portfolio(cash=starting_balance, starting_balance=starting_balance)
        self._trade_counter = 0
        
        if load_existing:
            self.load()
    
    def buy(
        self,
        ticker: str,
        quantity: int,
        price: float,
        signal_confidence: float = 0.0
    ) -> Optional[Trade]:
        """
        Execute a buy order.
        
        Args:
            ticker: Stock symbol
            quantity: Number of shares
            price: Price per share
            signal_confidence: Confidence of the signal (0-1)
            
        Returns:
            Trade object if successful, None otherwise
        """
        cost = quantity * price
        
        if cost > self.portfolio.cash:
            logger.warning(f"Insufficient funds: need ${cost:.2f}, have ${self.portfolio.cash:.2f}")
            return None
        
        # Deduct cash
        self.portfolio.cash -= cost
        
        # Update or create position
        if ticker in self.portfolio.positions:
            pos = self.portfolio.positions[ticker]
            total_quantity = pos.quantity + quantity
            pos.avg_price = (pos.cost_basis + cost) / total_quantity
            pos.quantity = total_quantity
        else:
            self.portfolio.positions[ticker] = Position(
                ticker=ticker,
                quantity=quantity,
                avg_price=price,
                current_price=price
            )
        
        # Create trade record
        self._trade_counter += 1
        trade = Trade(
            id=f"T{self._trade_counter:05d}",
            ticker=ticker,
            action=TradeAction.BUY.value,
            quantity=quantity,
            price=price,
            timestamp=datetime.now().isoformat(),
            signal_confidence=signal_confidence
        )
        self.portfolio.trades.append(trade)
        
        logger.info(f"BUY {quantity} {ticker} @ ${price:.2f} = ${cost:.2f}")
        self.save()
        
        return trade
    
    def sell(
        self,
        ticker: str,
        quantity: int,
        price: float,
        signal_confidence: float = 0.0
    ) -> Optional[Trade]:
        """
        Execute a sell order.
        
        Args:
            ticker: Stock symbol
            quantity: Number of shares to sell
            price: Price per share
            signal_confidence: Confidence of the signal (0-1)
            
        Returns:
            Trade object if successful, None otherwise
        """
        if ticker not in self.portfolio.positions:
            logger.warning(f"No position in {ticker}")
            return None
        
        pos = self.portfolio.positions[ticker]
        
        if quantity > pos.quantity:
            logger.warning(f"Cannot sell {quantity} shares, only have {pos.quantity}")
            return None
        
        # Calculate PnL
        proceeds = quantity * price
        cost_basis = quantity * pos.avg_price
        pnl = proceeds - cost_basis
        
        # Add cash
        self.portfolio.cash += proceeds
        
        # Update position
        pos.quantity -= quantity
        if pos.quantity == 0:
            del self.portfolio.positions[ticker]
        
        # Create trade record
        self._trade_counter += 1
        trade = Trade(
            id=f"T{self._trade_counter:05d}",
            ticker=ticker,
            action=TradeAction.SELL.value,
            quantity=quantity,
            price=price,
            timestamp=datetime.now().isoformat(),
            signal_confidence=signal_confidence,
            pnl=pnl,
            closed=True,
            close_price=price,
            close_timestamp=datetime.now().isoformat()
        )
        self.portfolio.trades.append(trade)
        
        logger.info(f"SELL {quantity} {ticker} @ ${price:.2f} = ${proceeds:.2f}, PnL: ${pnl:.2f}")
        self.save()
        
        return trade
    
    def update_prices(self, prices: Dict[str, float]):
        """Update current prices for all positions."""
        for ticker, price in prices.items():
            if ticker in self.portfolio.positions:
                self.portfolio.positions[ticker].current_price = price
    
    def get_position(self, ticker: str) -> Optional[Position]:
        """Get position for a ticker."""
        return self.portfolio.positions.get(ticker)
    
    def get_summary(self) -> Dict:
        """Get portfolio summary."""
        return {
            "cash": self.portfolio.cash,
            "total_value": self.portfolio.total_value,
            "total_return": self.portfolio.total_return,
            "realized_pnl": self.portfolio.realized_pnl,
            "win_rate": self.portfolio.win_rate,
            "positions": len(self.portfolio.positions),
            "trades": len(self.portfolio.trades),
            "starting_balance": self.portfolio.starting_balance,
        }
    
    def reset(self, starting_balance: float = 10000.0):
        """Reset portfolio to starting state."""
        self.portfolio = Portfolio(cash=starting_balance, starting_balance=starting_balance)
        self._trade_counter = 0
        self.save()
        logger.info(f"Portfolio reset with ${starting_balance:.2f}")
    
    def save(self):
        """Save portfolio to disk."""
        os.makedirs(os.path.dirname(self.SAVE_PATH), exist_ok=True)
        
        data = {
            "cash": self.portfolio.cash,
            "starting_balance": self.portfolio.starting_balance,
            "created_at": self.portfolio.created_at,
            "positions": {
                ticker: asdict(pos) 
                for ticker, pos in self.portfolio.positions.items()
            },
            "trades": [asdict(t) for t in self.portfolio.trades],
            "trade_counter": self._trade_counter
        }
        
        with open(self.SAVE_PATH, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self):
        """Load portfolio from disk."""
        if not os.path.exists(self.SAVE_PATH):
            return
        
        try:
            with open(self.SAVE_PATH, 'r') as f:
                data = json.load(f)
            
            self.portfolio.cash = data.get("cash", 10000.0)
            self.portfolio.starting_balance = data.get("starting_balance", 10000.0)
            self.portfolio.created_at = data.get("created_at", "")
            self._trade_counter = data.get("trade_counter", 0)
            
            # Load positions
            self.portfolio.positions = {}
            for ticker, pos_data in data.get("positions", {}).items():
                self.portfolio.positions[ticker] = Position(**pos_data)
            
            # Load trades
            self.portfolio.trades = []
            for trade_data in data.get("trades", []):
                self.portfolio.trades.append(Trade(**trade_data))
            
            logger.info(f"Loaded portfolio: ${self.portfolio.total_value:.2f} value, {len(self.portfolio.positions)} positions")
            
        except Exception as e:
            logger.error(f"Error loading portfolio: {e}")


# Singleton instance
paper_trader = PaperTradingEngine()
