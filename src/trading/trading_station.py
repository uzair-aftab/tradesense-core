"""
Trading Station

Professional paper trading interface with:
- One-click trade execution
- Automatic position sizing (2% risk rule)
- Real-time P&L tracking
- Auto-close on target/stop hit
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import numpy as np

from src.signals.timeframe_generator import TradeSetup
from src.utils.logger import get_logger

logger = get_logger(__name__)

PORTFOLIO_FILE = "data/trading_portfolio.json"


@dataclass
class Position:
    """An open position in the portfolio."""
    id: str
    ticker: str
    direction: str  # LONG or SHORT
    quantity: int
    entry_price: float
    target_price: float
    stop_price: float
    entry_time: str
    timeframe: str
    confidence: float
    reasons: List[str] = field(default_factory=list)
    
    # Updated on refresh
    current_price: float = 0.0
    
    @property
    def unrealized_pnl(self) -> float:
        """Current unrealized P&L."""
        if self.direction == "LONG":
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """P&L as percentage."""
        cost = self.entry_price * self.quantity
        if cost == 0:
            return 0
        return (self.unrealized_pnl / cost) * 100
    
    @property
    def target_distance(self) -> float:
        """Distance to target as %."""
        return ((self.target_price - self.current_price) / self.current_price) * 100
    
    @property
    def stop_distance(self) -> float:
        """Distance to stop as %."""
        return ((self.stop_price - self.current_price) / self.current_price) * 100
    
    def should_close(self) -> Optional[str]:
        """Check if position should auto-close."""
        if self.current_price == 0:
            return None
            
        if self.direction == "LONG":
            if self.current_price >= self.target_price:
                return "TARGET_HIT"
            if self.current_price <= self.stop_price:
                return "STOP_HIT"
        else:  # SHORT
            if self.current_price <= self.target_price:
                return "TARGET_HIT"
            if self.current_price >= self.stop_price:
                return "STOP_HIT"
        return None
    
    def to_dict(self) -> Dict:
        """Convert to dict with native Python types."""
        return {
            "id": self.id,
            "ticker": self.ticker,
            "direction": self.direction,
            "quantity": int(self.quantity),
            "entry_price": float(self.entry_price),
            "target_price": float(self.target_price),
            "stop_price": float(self.stop_price),
            "entry_time": self.entry_time,
            "timeframe": str(self.timeframe),
            "confidence": float(self.confidence),
            "reasons": [str(r) for r in self.reasons],
            "current_price": float(self.current_price),
        }


@dataclass
class ClosedTrade:
    """A completed trade."""
    id: str
    ticker: str
    direction: str
    quantity: int
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    exit_reason: str  # TARGET_HIT, STOP_HIT, MANUAL
    realized_pnl: float
    realized_pnl_pct: float
    timeframe: str
    
    def to_dict(self) -> Dict:
        """Convert to dict with native Python types."""
        return {
            "id": self.id,
            "ticker": self.ticker,
            "direction": self.direction,
            "quantity": int(self.quantity),
            "entry_price": float(self.entry_price),
            "exit_price": float(self.exit_price),
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "exit_reason": self.exit_reason,
            "realized_pnl": float(self.realized_pnl),
            "realized_pnl_pct": float(self.realized_pnl_pct),
            "timeframe": str(self.timeframe),
        }


class TradingStation:
    """
    Professional paper trading interface.
    
    Features:
    - One-click trade execution from TradeSetup
    - Automatic position sizing (2% risk per trade)
    - Real-time P&L tracking
    - Auto-close on target/stop hit
    - Performance analytics
    """
    
    def __init__(
        self, 
        starting_capital: float = 10000.0,
        risk_per_trade: float = 0.02,
        load_existing: bool = True
    ):
        self.starting_capital = starting_capital
        self.risk_per_trade = risk_per_trade
        self.cash = starting_capital
        self.positions: List[Position] = []
        self.closed_trades: List[ClosedTrade] = []
        self._trade_counter = 0
        
        if load_existing:
            self.load()
    
    @property
    def total_value(self) -> float:
        """Total portfolio value including open positions."""
        positions_value = sum(
            p.current_price * p.quantity if p.current_price > 0 else p.entry_price * p.quantity
            for p in self.positions
        )
        return self.cash + positions_value
    
    @property
    def open_pnl(self) -> float:
        """Total unrealized P&L from open positions."""
        return sum(p.unrealized_pnl for p in self.positions)
    
    @property
    def closed_pnl(self) -> float:
        """Total realized P&L from closed trades."""
        return sum(t.realized_pnl for t in self.closed_trades)
    
    @property
    def total_return_pct(self) -> float:
        """Total return as percentage of starting capital."""
        return ((self.total_value - self.starting_capital) / self.starting_capital) * 100
    
    def execute(self, setup: TradeSetup) -> Optional[Position]:
        """
        Execute a trade from TradeSetup.
        
        Uses 2% risk rule for position sizing.
        Returns the created Position or None if not executed.
        """
        if setup.direction == "HOLD":
            logger.info(f"Not executing HOLD signal for {setup.ticker}")
            return None
        
        # Calculate position size based on risk
        risk_amount = self.total_value * self.risk_per_trade
        stop_distance = abs(setup.entry_price - setup.stop_loss)
        
        if stop_distance == 0:
            logger.warning(f"Invalid stop distance for {setup.ticker}")
            return None
        
        position_size = int(risk_amount / stop_distance)
        
        # Check available cash
        cost = position_size * setup.entry_price
        if cost > self.cash:
            position_size = int(self.cash / setup.entry_price)
            cost = position_size * setup.entry_price
        
        if position_size <= 0:
            logger.warning(f"Insufficient funds for {setup.ticker}")
            return None
        
        # Create position
        self._trade_counter += 1
        position = Position(
            id=f"P{self._trade_counter:05d}",
            ticker=setup.ticker,
            direction=setup.direction,
            quantity=position_size,
            entry_price=setup.entry_price,
            target_price=setup.target_price,
            stop_price=setup.stop_loss,
            entry_time=datetime.now().isoformat(),
            timeframe=setup.timeframe,
            confidence=setup.confidence,
            reasons=setup.reasons,
            current_price=setup.entry_price,
        )
        
        # Deduct cash
        self.cash -= cost
        self.positions.append(position)
        
        logger.info(
            f"OPENED: {position.direction} {position.quantity} {position.ticker} "
            f"@ ${position.entry_price:.2f} | Target: ${position.target_price:.2f} | "
            f"Stop: ${position.stop_price:.2f}"
        )
        
        self.save()
        return position
    
    def close_position(self, position_id: str, reason: str = "MANUAL") -> Optional[ClosedTrade]:
        """Close a position by ID."""
        position = next((p for p in self.positions if p.id == position_id), None)
        if not position:
            logger.warning(f"Position {position_id} not found")
            return None
        
        exit_price = position.current_price if position.current_price > 0 else position.entry_price
        
        # Calculate P&L
        if position.direction == "LONG":
            pnl = (exit_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - exit_price) * position.quantity
        
        pnl_pct = (pnl / (position.entry_price * position.quantity)) * 100
        
        # Create closed trade record
        closed = ClosedTrade(
            id=position.id,
            ticker=position.ticker,
            direction=position.direction,
            quantity=position.quantity,
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_time=position.entry_time,
            exit_time=datetime.now().isoformat(),
            exit_reason=reason,
            realized_pnl=pnl,
            realized_pnl_pct=pnl_pct,
            timeframe=position.timeframe,
        )
        
        # Update cash and remove position
        self.cash += exit_price * position.quantity
        self.positions.remove(position)
        self.closed_trades.append(closed)
        
        logger.info(
            f"CLOSED: {position.ticker} | Reason: {reason} | "
            f"P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)"
        )
        
        self.save()
        return closed
    
    def update_prices(self, prices: Dict[str, float]):
        """Update all position prices and check for auto-close."""
        for position in self.positions[:]:  # Copy list for safe iteration
            if position.ticker in prices:
                position.current_price = prices[position.ticker]
                
                # Check for auto-close
                reason = position.should_close()
                if reason:
                    self.close_position(position.id, reason)
    
    def get_analytics(self) -> Dict:
        """Get portfolio analytics."""
        wins = [t for t in self.closed_trades if t.realized_pnl > 0]
        losses = [t for t in self.closed_trades if t.realized_pnl < 0]
        
        win_rate = len(wins) / len(self.closed_trades) * 100 if self.closed_trades else 0
        avg_win = sum(t.realized_pnl for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.realized_pnl for t in losses) / len(losses) if losses else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        return {
            "total_value": self.total_value,
            "cash": self.cash,
            "open_pnl": self.open_pnl,
            "closed_pnl": self.closed_pnl,
            "total_return_pct": self.total_return_pct,
            "open_positions": len(self.positions),
            "total_trades": len(self.closed_trades),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
        }
    
    def reset(self, starting_capital: float = None):
        """Reset portfolio to starting state."""
        self.cash = starting_capital or self.starting_capital
        self.starting_capital = self.cash
        self.positions = []
        self.closed_trades = []
        self._trade_counter = 0
        self.save()
        logger.info(f"Portfolio reset to ${self.cash:.2f}")
    
    def save(self):
        """Save portfolio to disk."""
        os.makedirs(os.path.dirname(PORTFOLIO_FILE), exist_ok=True)
        
        data = {
            "starting_capital": float(self.starting_capital),
            "cash": float(self.cash),
            "trade_counter": int(self._trade_counter),
            "positions": [p.to_dict() for p in self.positions],
            "closed_trades": [t.to_dict() for t in self.closed_trades],
        }
        
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load(self):
        """Load portfolio from disk."""
        if not os.path.exists(PORTFOLIO_FILE):
            return
        
        try:
            with open(PORTFOLIO_FILE, 'r') as f:
                data = json.load(f)
            
            self.starting_capital = data.get("starting_capital", 10000)
            self.cash = data.get("cash", self.starting_capital)
            self._trade_counter = data.get("trade_counter", 0)
            
            self.positions = []
            for p_data in data.get("positions", []):
                self.positions.append(Position(**p_data))
            
            self.closed_trades = []
            for t_data in data.get("closed_trades", []):
                self.closed_trades.append(ClosedTrade(**t_data))
            
            logger.info(
                f"Loaded portfolio: ${self.total_value:.2f} value, "
                f"{len(self.positions)} open positions"
            )
        except Exception as e:
            logger.error(f"Error loading portfolio: {e}")


# Singleton instance
trading_station = TradingStation()
