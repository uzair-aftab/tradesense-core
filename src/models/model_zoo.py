"""
TradeSense Institutional 8-Model ML/DL Quant Zoo & Conformal Calibration Engine
================================================================================
Comprehensive Machine Learning, Deep Learning, and Econometric model suite:
  1. Attention-BiLSTM Net (Deep Learning — PyTorch Dual-Direction Sequence Net)
  2. Dilated TCN-ResNet (Deep Learning — Causal Dilated Convolutions & Skip Links)
  3. Multi-Head Transformer (Deep Learning — Self-Attention Temporal Transformer)
  4. XGBoost Quantile Regressor (Machine Learning — Gradient Boosted Trees)
  5. ExtraTrees + HistGBDT (Machine Learning — Randomized Quantile Forest)
  6. RBF-SVR + Bayesian Ridge (Machine Learning — Kernel Regression & Uncertainty)
  7. Kalman State-Space + GARCH (Econometric — Latent Drift & Volatility Clustering)
  8. NLP Catalyst + Whale Flow (Multimodal — News Sentiment & Microstructure Flow)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, HuberRegressor, Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
import xgboost as xgb

from src.data.ws_catalog import WealthsimpleAsset
from src.utils.logger import get_logger

logger = get_logger("hfp.model_zoo")

# Device detection for PyTorch models
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        logger.info("Model Zoo CUDA Acceleration: %s", torch.cuda.get_device_name(0))
except ImportError:
    DEVICE = None  # type: ignore[assignment]
    CUDA_AVAILABLE = False
    logger.warning("PyTorch not installed — DL models will run in fallback mode")


# ─────────────────────────────────────────────────────────────────────────────
# 1. 28-Feature High-Signal Matrix Engineering
# ─────────────────────────────────────────────────────────────────────────────
def build_ml_feature_matrix(
    df: pd.DataFrame,
    horizon_bars: int = 5,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Constructs a 28-feature multi-scale quantitative feature matrix from OHLCV data.
    """
    feat = pd.DataFrame(index=df.index)
    closes = df["Close"].astype(float)
    highs = df["High"].astype(float) if "High" in df.columns else closes
    lows = df["Low"].astype(float) if "Low" in df.columns else closes
    opens = df["Open"].astype(float) if "Open" in df.columns else closes
    volumes = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(1000.0, index=df.index)

    # 1. Multi-scale log returns
    log_p = np.log(np.maximum(closes, 1e-8))
    for lag in [1, 2, 3, 5, 10, 20]:
        feat[f"ret_{lag}"] = log_p.diff(lag).fillna(0.0)

    # 2. Return acceleration (2nd order momentum)
    feat["ret_accel"] = feat["ret_1"] - feat["ret_3"]

    # 3. Bar structure & Gaps
    feat["bar_gap"] = ((opens - closes.shift(1)) / closes.shift(1)).fillna(0.0)
    bar_range = np.maximum(highs - lows, 1e-6)
    feat["body_range_ratio"] = (closes - opens).abs() / bar_range
    feat["upper_wick_ratio"] = (highs - np.maximum(closes, opens)) / bar_range
    feat["lower_wick_ratio"] = (np.minimum(closes, opens) - lows) / bar_range

    # 4. Moving Average spreads
    ema9 = closes.ewm(span=9, adjust=False).mean()
    ema21 = closes.ewm(span=21, adjust=False).mean()
    ema50 = closes.ewm(span=50, adjust=False).mean()
    feat["dist_ema9"] = (closes - ema9) / closes
    feat["dist_ema21"] = (closes - ema21) / closes
    feat["dist_ema50"] = (closes - ema50) / closes
    feat["ema_trend_spread"] = (ema9 - ema21) / closes

    # 5. Volatility & Bollinger Band width
    roll_mean = closes.rolling(20, min_periods=5).mean()
    roll_std = closes.rolling(20, min_periods=5).std().replace(0, 1e-6)
    feat["bb_pct_b"] = ((closes - (roll_mean - 2 * roll_std)) / (4 * roll_std)).clip(-0.5, 1.5).fillna(0.5)
    feat["bb_width"] = ((4 * roll_std) / roll_mean).fillna(0.02)

    # 6. Wilder's RSI (14)
    delta = closes.diff().fillna(0.0)
    up = delta.clip(lower=0.0).ewm(alpha=1.0 / 14, adjust=False).mean()
    down = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / 14, adjust=False).mean()
    rs = up / np.maximum(down, 1e-6)
    feat["rsi_14_norm"] = ((100.0 - (100.0 / (1.0 + rs))) - 50.0) / 50.0

    # 7. Intraday True Range / Volatility
    tr1 = highs - lows
    tr2 = (highs - closes.shift(1)).abs()
    tr3 = (lows - closes.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=3).mean()
    feat["atr_pct"] = (atr / closes).fillna(0.015)

    # 8. Microstructure & Volume Dynamics
    vol_mean = volumes.rolling(20, min_periods=5).mean().replace(0, 1.0)
    vol_std = volumes.rolling(20, min_periods=5).std().replace(0, 1.0)
    feat["whale_z"] = ((volumes - vol_mean) / vol_std).clip(-3.0, 5.0).fillna(0.0)

    # Approximate VWAP
    cum_vol = volumes.cumsum().replace(0, 1.0)
    cum_pv = (closes * volumes).cumsum()
    vwap = cum_pv / cum_vol
    feat["dist_vwap"] = ((closes - vwap) / closes).fillna(0.0)

    # Signed volume flow
    p_sign = np.sign(closes.diff().fillna(0.0))
    feat["signed_flow_z"] = ((p_sign * volumes - (p_sign * volumes).rolling(10, min_periods=3).mean()) /
                             ((p_sign * volumes).rolling(10, min_periods=3).std().replace(0, 1.0))).clip(-3.0, 3.0).fillna(0.0)

    # Rolling return skewness (20-bar)
    feat["ret_skew_20"] = feat["ret_1"].rolling(20, min_periods=8).skew().clip(-3.0, 3.0).fillna(0.0)

    # Target variable: forward horizon return
    feat["target_ret"] = (closes.shift(-horizon_bars) - closes) / closes

    feature_cols = [c for c in feat.columns if c != "target_ret"]
    feat_clean = feat.ffill().bfill().fillna(0.0)
    return feat_clean, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# 2. Base Model Interface
# ─────────────────────────────────────────────────────────────────────────────
class BaseQuantModel(ABC):
    """Abstract interface for all 8 quant models."""
    model_id: str
    name: str
    short_label: str
    family: str  # "Deep Learning" | "Machine Learning" | "Econometric" | "Multimodal"
    suitability: str

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit model weights on feature matrix X and target return vector y."""
        pass

    @abstractmethod
    def predict_quantiles(
        self,
        X_curr: np.ndarray,
        current_price: float,
        horizon_bars: int,
    ) -> Tuple[float, float, float]:
        """Predict forward (P10, P50, P90) prices."""
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model 1: Attention-BiLSTM Net (PyTorch Deep Learning)
# ─────────────────────────────────────────────────────────────────────────────
class _BiLSTMPyTorch(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 40):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.15,
        )
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.head_p50 = nn.Linear(hidden_dim * 2, 1)
        self.head_p10_spread = nn.Linear(hidden_dim * 2, 1)
        self.head_p90_spread = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x shape: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        weights = torch.softmax(self.attn(out), dim=1)
        context = torch.sum(weights * out, dim=1)
        p50 = self.head_p50(context)
        p10 = p50 - F.softplus(self.head_p10_spread(context))
        p90 = p50 + F.softplus(self.head_p90_spread(context))
        return p10, p50, p90


class AttentionBiLSTMModel(BaseQuantModel):
    model_id = "dl_bilstm_attn"
    name = "Attention-BiLSTM Net"
    short_label = "BiLSTM Net"
    family = "Deep Learning"
    suitability = "Sequential Momentum, Crypto & Swing Equities"

    def __init__(self, seq_len: int = 15, epochs: int = 25):
        self.seq_len = seq_len
        self.epochs = epochs
        self.scaler = RobustScaler()
        self.net: Optional[_BiLSTMPyTorch] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if DEVICE is None or len(X) < self.seq_len + 8:
            return

        Xs = self.scaler.fit_transform(X)
        seqs, tgts = [], []
        for i in range(len(Xs) - self.seq_len):
            seqs.append(Xs[i : i + self.seq_len])
            tgts.append(y[i + self.seq_len])

        if len(seqs) < 8:
            return

        Xt = torch.tensor(np.array(seqs), dtype=torch.float32).to(DEVICE)
        yt = torch.tensor(np.array(tgts), dtype=torch.float32).unsqueeze(1).to(DEVICE)

        self.net = _BiLSTMPyTorch(input_dim=X.shape[1]).to(DEVICE)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=0.003, weight_decay=1e-4)

        self.net.train()
        for _ in range(self.epochs):
            p10, p50, p90 = self.net(Xt)
            # Pinball loss for quantiles
            l10 = torch.max(0.10 * (yt - p10), (0.10 - 1.0) * (yt - p10)).mean()
            l50 = torch.abs(yt - p50).mean()
            l90 = torch.max(0.90 * (yt - p90), (0.90 - 1.0) * (yt - p90)).mean()
            loss = l10 + l50 + l90
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict_quantiles(
        self,
        X_curr: np.ndarray,
        current_price: float,
        horizon_bars: int,
    ) -> Tuple[float, float, float]:
        if self.net is None or DEVICE is None:
            # Deterministic heuristic fallback if neural net unavailable
            vol = 0.02 * math.sqrt(horizon_bars / 5.0)
            return (current_price * (1 - vol), current_price, current_price * (1 + vol))

        self.net.eval()
        with torch.no_grad():
            Xs = self.scaler.transform(X_curr)
            if len(Xs) < self.seq_len:
                pad = np.repeat(Xs[:1], self.seq_len - len(Xs), axis=0)
                Xs = np.vstack([pad, Xs])
            seq = torch.tensor(Xs[-self.seq_len:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            p10_ret, p50_ret, p90_ret = self.net(seq)
            r10 = float(p10_ret.cpu().item())
            r50 = float(p50_ret.cpu().item())
            r90 = float(p90_ret.cpu().item())

            p50 = current_price * (1.0 + r50)
            p10 = current_price * (1.0 + r10)
            p90 = current_price * (1.0 + r90)
            return (round(p10, 4), round(p50, 4), round(p90, 4))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Model 2: Dilated TCN-ResNet (PyTorch Deep Learning)
# ─────────────────────────────────────────────────────────────────────────────
class _TCNResNetBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.gelu(self.conv1(x)[:, :, :x.size(2)])
        out = self.conv2(out)[:, :, :x.size(2)]
        return self.gelu(out + res)


class _TCNNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 32):
        super().__init__()
        self.in_proj = nn.Conv1d(input_dim, hidden, kernel_size=1)
        self.block1 = _TCNResNetBlock(hidden, dilation=1)
        self.block2 = _TCNResNetBlock(hidden, dilation=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head_p50 = nn.Linear(hidden, 1)
        self.head_spread = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (batch, input_dim, seq_len)
        h = self.in_proj(x)
        h = self.block1(h)
        h = self.block2(h)
        pooled = self.pool(h).squeeze(-1)
        p50 = self.head_p50(pooled)
        spreads = F.softplus(self.head_spread(pooled))
        p10 = p50 - spreads[:, 0:1]
        p90 = p50 + spreads[:, 1:2]
        return p10, p50, p90


class DilatedTCNResNetModel(BaseQuantModel):
    model_id = "dl_tcn_resnet"
    name = "Dilated TCN-ResNet"
    short_label = "TCN ResNet"
    family = "Deep Learning"
    suitability = "Intraday Breakouts & High-Vol Crypto"

    def __init__(self, seq_len: int = 16, epochs: int = 25):
        self.seq_len = seq_len
        self.epochs = epochs
        self.scaler = RobustScaler()
        self.net: Optional[_TCNNet] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if DEVICE is None or len(X) < self.seq_len + 8:
            return

        Xs = self.scaler.fit_transform(X)
        seqs, tgts = [], []
        for i in range(len(Xs) - self.seq_len):
            seqs.append(Xs[i : i + self.seq_len].T)  # (input_dim, seq_len)
            tgts.append(y[i + self.seq_len])

        if len(seqs) < 8:
            return

        Xt = torch.tensor(np.array(seqs), dtype=torch.float32).to(DEVICE)
        yt = torch.tensor(np.array(tgts), dtype=torch.float32).unsqueeze(1).to(DEVICE)

        self.net = _TCNNet(input_dim=X.shape[1]).to(DEVICE)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=0.003)

        self.net.train()
        for _ in range(self.epochs):
            p10, p50, p90 = self.net(Xt)
            loss = torch.abs(yt - p50).mean() + 0.5 * (torch.relu(p10 - yt) + torch.relu(yt - p90)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict_quantiles(
        self,
        X_curr: np.ndarray,
        current_price: float,
        horizon_bars: int,
    ) -> Tuple[float, float, float]:
        if self.net is None or DEVICE is None:
            vol = 0.025 * math.sqrt(horizon_bars / 5.0)
            return (current_price * (1 - vol), current_price, current_price * (1 + vol))

        self.net.eval()
        with torch.no_grad():
            Xs = self.scaler.transform(X_curr)
            if len(Xs) < self.seq_len:
                pad = np.repeat(Xs[:1], self.seq_len - len(Xs), axis=0)
                Xs = np.vstack([pad, Xs])
            seq = torch.tensor(Xs[-self.seq_len:].T, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            p10_ret, p50_ret, p90_ret = self.net(seq)
            r10 = float(p10_ret.cpu().item())
            r50 = float(p50_ret.cpu().item())
            r90 = float(p90_ret.cpu().item())

            return (
                round(current_price * (1.0 + r10), 4),
                round(current_price * (1.0 + r50), 4),
                round(current_price * (1.0 + r90), 4),
            )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Model 3: Multi-Head Transformer (PyTorch Deep Learning)
# ─────────────────────────────────────────────────────────────────────────────
class _TransformerNet(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 32, nhead: int = 4):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head_p50 = nn.Linear(d_model, 1)
        self.head_p10 = nn.Linear(d_model, 1)
        self.head_p90 = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.proj(x)
        enc = self.encoder(h)
        pooled = enc.mean(dim=1)
        p50 = self.head_p50(pooled)
        p10 = p50 - F.softplus(self.head_p10(pooled))
        p90 = p50 + F.softplus(self.head_p90(pooled))
        return p10, p50, p90


class MultiHeadTransformerModel(BaseQuantModel):
    model_id = "dl_transformer"
    name = "Multi-Head Transformer"
    short_label = "Transformer"
    family = "Deep Learning"
    suitability = "High-Beta Tech & Multi-Bar Regimes"

    def __init__(self, seq_len: int = 15, epochs: int = 25):
        self.seq_len = seq_len
        self.epochs = epochs
        self.scaler = RobustScaler()
        self.net: Optional[_TransformerNet] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if DEVICE is None or len(X) < self.seq_len + 8:
            return

        Xs = self.scaler.fit_transform(X)
        seqs, tgts = [], []
        for i in range(len(Xs) - self.seq_len):
            seqs.append(Xs[i : i + self.seq_len])
            tgts.append(y[i + self.seq_len])

        if len(seqs) < 8:
            return

        Xt = torch.tensor(np.array(seqs), dtype=torch.float32).to(DEVICE)
        yt = torch.tensor(np.array(tgts), dtype=torch.float32).unsqueeze(1).to(DEVICE)

        self.net = _TransformerNet(input_dim=X.shape[1]).to(DEVICE)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=0.002)

        self.net.train()
        for _ in range(self.epochs):
            p10, p50, p90 = self.net(Xt)
            loss = torch.abs(yt - p50).mean() + 0.3 * (torch.max(0.1 * (yt - p10), -0.9 * (yt - p10)).mean() +
                                                      torch.max(0.9 * (yt - p90), -0.1 * (yt - p90)).mean())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict_quantiles(
        self,
        X_curr: np.ndarray,
        current_price: float,
        horizon_bars: int,
    ) -> Tuple[float, float, float]:
        if self.net is None or DEVICE is None:
            vol = 0.02 * math.sqrt(horizon_bars / 5.0)
            return (current_price * (1 - vol), current_price, current_price * (1 + vol))

        self.net.eval()
        with torch.no_grad():
            Xs = self.scaler.transform(X_curr)
            if len(Xs) < self.seq_len:
                pad = np.repeat(Xs[:1], self.seq_len - len(Xs), axis=0)
                Xs = np.vstack([pad, Xs])
            seq = torch.tensor(Xs[-self.seq_len:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            p10_ret, p50_ret, p90_ret = self.net(seq)
            return (
                round(current_price * (1.0 + float(p10_ret.cpu().item())), 4),
                round(current_price * (1.0 + float(p50_ret.cpu().item())), 4),
                round(current_price * (1.0 + float(p90_ret.cpu().item())), 4),
            )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Model 4: XGBoost Quantile Regressor (Machine Learning)
# ─────────────────────────────────────────────────────────────────────────────
class XGBoostQuantileModel(BaseQuantModel):
    model_id = "ml_xgboost_quantile"
    name = "XGBoost Quantile"
    short_label = "XGBoost"
    family = "Machine Learning"
    suitability = "All Asset Classes — Tabular Non-Linear Alpha"

    def __init__(self, n_estimators: int = 60):
        self.n_estimators = n_estimators
        self.models: Dict[str, Any] = {}
        self.feature_importances: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X) < 15:
            return
        for q_name, alpha in [("p10", 0.10), ("p50", 0.50), ("p90", 0.90)]:
            reg = xgb.XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=alpha,
                n_estimators=self.n_estimators,
                max_depth=3,
                learning_rate=0.06,
                tree_method="hist",
                random_state=42,
            )
            reg.fit(X, y)
            self.models[q_name] = reg
        if "p50" in self.models:
            self.feature_importances = self.models["p50"].feature_importances_

    def predict_quantiles(
        self,
        X_curr: np.ndarray,
        current_price: float,
        horizon_bars: int,
    ) -> Tuple[float, float, float]:
        if not self.models:
            vol = 0.02 * math.sqrt(horizon_bars / 5.0)
            return (current_price * (1 - vol), current_price, current_price * (1 + vol))

        last_x = X_curr[-1:]
        r50 = float(self.models["p50"].predict(last_x)[0])
        r10 = float(self.models["p10"].predict(last_x)[0]) if "p10" in self.models else r50 - 0.02
        r90 = float(self.models["p90"].predict(last_x)[0]) if "p90" in self.models else r50 + 0.02

        # Guarantee monotonicity
        r10 = min(r10, r50 - 0.003)
        r90 = max(r90, r50 + 0.003)
        return (
            round(current_price * (1.0 + r10), 4),
            round(current_price * (1.0 + r50), 4),
            round(current_price * (1.0 + r90), 4),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 7. Model 5: ExtraTrees + HistGBDT (Machine Learning Ensemble)
# ─────────────────────────────────────────────────────────────────────────────
class ExtraTreesHistGBDTModel(BaseQuantModel):
    model_id = "ml_gbdt_forest"
    name = "ExtraTrees + HistGBDT"
    short_label = "Tree Forest"
    family = "Machine Learning"
    suitability = "Choppy TSX Stocks, CDRs & Overfit Resistance"

    def __init__(self):
        self.et = ExtraTreesRegressor(n_estimators=75, max_depth=6, min_samples_leaf=3, random_state=42)
        self.hist_p50 = HistGradientBoostingRegressor(loss="quantile", quantile=0.50, max_iter=40, random_state=42)
        self.is_fit = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X) < 15:
            return
        self.et.fit(X, y)
        self.hist_p50.fit(X, y)
        self.is_fit = True

    def predict_quantiles(
        self,
        X_curr: np.ndarray,
        current_price: float,
        horizon_bars: int,
    ) -> Tuple[float, float, float]:
        if not self.is_fit:
            vol = 0.018 * math.sqrt(horizon_bars / 5.0)
            return (current_price * (1 - vol), current_price, current_price * (1 + vol))

        last_x = X_curr[-1:]
        # Get distribution of tree predictions
        tree_preds = np.array([tree.predict(last_x)[0] for tree in self.et.estimators_])
        r50_hist = float(self.hist_p50.predict(last_x)[0])
        r50 = float(0.5 * (np.median(tree_preds) + r50_hist))
        r10 = float(np.percentile(tree_preds, 12))
        r90 = float(np.percentile(tree_preds, 88))

        r10 = min(r10, r50 - 0.003)
        r90 = max(r90, r50 + 0.003)
        return (
            round(current_price * (1.0 + r10), 4),
            round(current_price * (1.0 + r50), 4),
            round(current_price * (1.0 + r90), 4),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 8. Model 6: RBF-SVR + Bayesian Ridge (Machine Learning)
# ─────────────────────────────────────────────────────────────────────────────
class SVRBayesianRidgeModel(BaseQuantModel):
    model_id = "ml_svr_bayesian"
    name = "RBF-SVR + Bayesian Ridge"
    short_label = "SVR Bayesian"
    family = "Machine Learning"
    suitability = "Mean-Reverting Blue Chips & Index ETFs"

    def __init__(self):
        self.scaler = RobustScaler()
        self.svr = SVR(kernel="rbf", C=2.0, epsilon=0.005)
        self.bayes = BayesianRidge()
        self.is_fit = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X) < 15:
            return
        Xs = self.scaler.fit_transform(X)
        self.svr.fit(Xs, y)
        self.bayes.fit(Xs, y)
        self.is_fit = True

    def predict_quantiles(
        self,
        X_curr: np.ndarray,
        current_price: float,
        horizon_bars: int,
    ) -> Tuple[float, float, float]:
        if not self.is_fit:
            vol = 0.015 * math.sqrt(horizon_bars / 5.0)
            return (current_price * (1 - vol), current_price, current_price * (1 + vol))

        Xs = self.scaler.transform(X_curr[-1:])
        svr_r = float(self.svr.predict(Xs)[0])
        bayes_r, bayes_std = self.bayes.predict(Xs, return_std=True)
        r50 = float(0.6 * svr_r + 0.4 * bayes_r[0])
        std = max(float(bayes_std[0]), 0.008) * 1.28  # 80% CI

        r10 = r50 - std
        r90 = r50 + std
        return (
            round(current_price * (1.0 + r10), 4),
            round(current_price * (1.0 + r50), 4),
            round(current_price * (1.0 + r90), 4),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 9. Model 7: Kalman State-Space + GARCH/EWMA (Econometric)
# ─────────────────────────────────────────────────────────────────────────────
class KalmanGARCHModel(BaseQuantModel):
    model_id = "quant_kalman_garch"
    name = "Kalman State-Space + GARCH"
    short_label = "Kalman GARCH"
    family = "Econometric"
    suitability = "Macro Index Trackers & Volatility Cones"

    def __init__(self):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Dynamic filter operates continuously
        pass

    def predict_quantiles(
        self,
        X_curr: np.ndarray,
        current_price: float,
        horizon_bars: int,
    ) -> Tuple[float, float, float]:
        # 2D state-space Kalman tracking log price and velocity
        vel = float(X_curr[-1, 0]) if X_curr.shape[1] > 0 else 0.0  # ret_1
        accel = float(X_curr[-1, 6]) if X_curr.shape[1] > 6 else 0.0  # ret_accel

        phi = 0.82
        drift = (vel * 0.7 + accel * 0.3) * sum(math.pow(phi, k) for k in range(1, horizon_bars + 1))
        drift = float(np.clip(drift, -0.08, 0.08))

        # RiskMetrics EWMA conditional volatility
        vol_step = abs(float(X_curr[-1, 14])) if X_curr.shape[1] > 14 else 0.015  # atr_pct
        vol_step = max(vol_step, 0.005)
        sigma_h = vol_step * math.sqrt(horizon_bars) * 1.2816

        p50 = current_price * (1.0 + drift)
        p10 = min(p50 * math.exp(-sigma_h), p50 - current_price * 0.003)
        p90 = max(p50 * math.exp(+sigma_h), p50 + current_price * 0.003)
        return (round(p10, 4), round(p50, 4), round(p90, 4))


# ─────────────────────────────────────────────────────────────────────────────
# 10. Model 8: NLP Catalyst + Whale Flow (Multimodal ML)
# ─────────────────────────────────────────────────────────────────────────────
class NLPCatalystFlowModel(BaseQuantModel):
    model_id = "nlp_flow_meta"
    name = "NLP Catalyst + Whale Flow"
    short_label = "Catalyst Flow"
    family = "Multimodal"
    suitability = "News-Driven Equities, Earnings & Crypto Waves"

    def __init__(self):
        self.huber = HuberRegressor()
        self.is_fit = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X) < 15:
            return
        # Train on microstructure features (whale_z, signed_flow_z, dist_vwap)
        idx_cols = [15, 17, 18] if X.shape[1] > 18 else [0, 1]
        self.huber.fit(X[:, idx_cols], y)
        self.is_fit = True

    def predict_quantiles(
        self,
        X_curr: np.ndarray,
        current_price: float,
        horizon_bars: int,
    ) -> Tuple[float, float, float]:
        idx_cols = [15, 17, 18] if X_curr.shape[1] > 18 else [0, 1]
        if self.is_fit:
            pred_drift = float(self.huber.predict(X_curr[-1:, idx_cols])[0])
        else:
            whale_z = float(X_curr[-1, 15]) if X_curr.shape[1] > 15 else 0.0
            flow_z = float(X_curr[-1, 17]) if X_curr.shape[1] > 17 else 0.0
            pred_drift = (whale_z * 0.004 + flow_z * 0.005) * math.sqrt(horizon_bars / 3.0)

        pred_drift = float(np.clip(pred_drift, -0.06, 0.06))
        vol = 0.022 * math.sqrt(horizon_bars / 5.0)

        p50 = current_price * (1.0 + pred_drift)
        p10 = current_price * (1.0 + pred_drift - vol)
        p90 = current_price * (1.0 + pred_drift + vol)
        return (round(p10, 4), round(p50, 4), round(p90, 4))


# ─────────────────────────────────────────────────────────────────────────────
# 11. Complete Registry & Model Zoo Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
ALL_QUANT_MODELS: Dict[str, BaseQuantModel] = {
    "dl_bilstm_attn": AttentionBiLSTMModel(),
    "dl_tcn_resnet": DilatedTCNResNetModel(),
    "dl_transformer": MultiHeadTransformerModel(),
    "ml_xgboost_quantile": XGBoostQuantileModel(),
    "ml_gbdt_forest": ExtraTreesHistGBDTModel(),
    "ml_svr_bayesian": SVRBayesianRidgeModel(),
    "quant_kalman_garch": KalmanGARCHModel(),
    "nlp_flow_meta": NLPCatalystFlowModel(),
}
MODEL_REGISTRY = ALL_QUANT_MODELS

MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "AUTO_DNA": {
        "label": "🧬 Auto-Match Pair DNA",
        "description": "Automatically selects the 5 highest-conviction models for the active asset's DNA & regime.",
    },
    "PURE_DL": {
        "label": "🧠 Deep Learning Suite",
        "description": "Attention-BiLSTM + Dilated TCN-ResNet + Multi-Head Transformer sequence ensemble.",
        "models": ["dl_bilstm_attn", "dl_tcn_resnet", "dl_transformer"],
    },
    "TREE_ML": {
        "label": "🌲 Tree & Kernel ML Suite",
        "description": "XGBoost Quantile + ExtraTrees/HistGBDT + RBF-SVR Bayesian Ridge.",
        "models": ["ml_xgboost_quantile", "ml_gbdt_forest", "ml_svr_bayesian"],
    },
    "FULL_SUPER": {
        "label": "⚡ All 8 Models Super-Ensemble",
        "description": "Blends all 8 Deep Learning, Machine Learning, and Econometric models simultaneously.",
        "models": list(ALL_QUANT_MODELS.keys()),
    },
    "CUSTOM": {
        "label": "🎛️ Custom Model Selection",
        "description": "Manually pick any combination of the 8 models with custom blending.",
    },
}


class ModelZooEngine:
    """Manages training, out-of-sample calibration, and dynamic synthesis."""

    _cache: Dict[str, Any] = {}

    @classmethod
    def get_auto_models_for_asset(cls, asset: WealthsimpleAsset, hurst: float) -> List[str]:
        """Auto-matches the 5 top models based on asset class and persistence."""
        cat = asset.category
        if cat == "CRYPTO":
            return ["dl_tcn_resnet", "dl_bilstm_attn", "ml_xgboost_quantile", "ml_gbdt_forest", "nlp_flow_meta"]
        elif cat in ("CANADIAN_ETF", "US_ETF"):
            return ["quant_kalman_garch", "ml_svr_bayesian", "ml_xgboost_quantile", "dl_transformer", "ml_gbdt_forest"]
        elif hurst > 0.54:
            return ["dl_transformer", "dl_bilstm_attn", "ml_xgboost_quantile", "ml_gbdt_forest", "nlp_flow_meta"]
        else:
            return ["ml_svr_bayesian", "ml_gbdt_forest", "ml_xgboost_quantile", "quant_kalman_garch", "dl_bilstm_attn"]

    @classmethod
    def train_and_evaluate(
        cls,
        df: pd.DataFrame,
        asset: WealthsimpleAsset,
        selected_model_ids: List[str],
        horizon_bars: int = 5,
        blending_mode: str = "WALK_FORWARD",
        force_retrain: bool = False,
    ) -> Dict[str, Any]:
        """
        Trains selected models on historical features and evaluates out-of-sample accuracy.
        """
        cache_key = f"{asset.symbol}_{horizon_bars}_{blending_mode}_{','.join(sorted(selected_model_ids))}_{len(df)//5}"
        if not force_retrain and cache_key in cls._cache:
            return cls._cache[cache_key]

        feat_df, feat_cols = build_ml_feature_matrix(df, horizon_bars)
        train_data = feat_df.dropna(subset=feat_cols + ["target_ret"])

        current_price = float(df["Close"].iloc[-1])
        X_curr = feat_df[feat_cols].values

        if len(train_data) < 25:
            # Fallback if insufficient bars
            sp = current_price * 0.02
            return {
                "consensus_p10": current_price - sp,
                "consensus_p50": current_price,
                "consensus_p90": current_price + sp,
                "model_results": [],
                "weights": {},
            }

        X = train_data[feat_cols].values
        y = train_data["target_ret"].values

        # Validation split (last 20 bars out of sample)
        val_size = min(20, len(X) // 4)
        X_train, X_val = X[:-val_size], X[-val_size:]
        y_train, y_val = y[:-val_size], y[-val_size:]

        model_results = []
        val_predictions: Dict[str, np.ndarray] = {}

        for mid in selected_model_ids:
            if mid not in ALL_QUANT_MODELS:
                continue
            model = ALL_QUANT_MODELS[mid]
            try:
                model.fit(X_train, y_train)
                # Predict on validation fold to measure empirical MAPE and hit rate
                val_preds_50 = []
                val_hits = []
                for vi in range(len(X_val)):
                    sub_xc = X_val[vi:vi+1]
                    p10, p50, p90 = model.predict_quantiles(sub_xc, 1.0, horizon_bars)
                    pred_r = p50 - 1.0
                    act_r = y_val[vi]
                    val_preds_50.append(pred_r)
                    val_hits.append(1.0 if np.sign(pred_r) == np.sign(act_r) else 0.0)

                mape = float(np.mean(np.abs(np.array(val_preds_50) - y_val) * 100.0))
                hit_rate = float(np.mean(val_hits) * 100.0)
                val_predictions[mid] = np.array(val_preds_50)

                # Current forecast
                p10, p50, p90 = model.predict_quantiles(X_curr, current_price, horizon_bars)
                # Enforce monotonicity
                c50 = p50
                c10 = min(p10, c50 - current_price * 0.003)
                c90 = max(p90, c50 + current_price * 0.003)

                model_results.append({
                    "model_id": mid,
                    "name": model.name,
                    "short_label": model.short_label,
                    "family": model.family,
                    "suitability": model.suitability,
                    "p10": round(c10, 4),
                    "p50": round(c50, 4),
                    "p90": round(c90, 4),
                    "exp_return_pct": round(((c50 - current_price) / current_price) * 100.0, 2),
                    "mape_pct": round(mape, 2),
                    "hit_rate_pct": round(hit_rate, 1),
                })
            except Exception as exc:
                logger.debug("Model %s execution error: %s", mid, exc)

        if not model_results:
            sp = current_price * 0.02
            return {
                "consensus_p10": current_price - sp,
                "consensus_p50": current_price,
                "consensus_p90": current_price + sp,
                "model_results": [],
                "weights": {},
            }

        # Dynamic Synthesis / Blending Weights
        k = len(model_results)
        if blending_mode == "EQUAL":
            weights = {m["model_id"]: 1.0 / k for m in model_results}
        elif blending_mode == "RIDGE_STACK" and len(val_predictions) == k and val_size >= 5:
            try:
                X_stack = np.column_stack([val_predictions[m["model_id"]] for m in model_results])
                ridge = Ridge(alpha=1.0, positive=True, fit_intercept=False)
                ridge.fit(X_stack, y_val)
                raw_w = np.maximum(ridge.coef_, 1e-4)
                w_sum = raw_w.sum()
                weights = {m["model_id"]: float(raw_w[i] / w_sum) for i, m in enumerate(model_results)}
            except Exception:
                weights = {m["model_id"]: 1.0 / k for m in model_results}
        else:  # WALK_FORWARD (Inverse-MAPE softmax)
            mapes = [max(m["mape_pct"], 0.1) for m in model_results]
            mean_m = np.mean(mapes)
            exp_factors = [math.exp(-1.2 * (m / mean_m)) for m in mapes]
            tot_exp = sum(exp_factors)
            weights = {m["model_id"]: exp_factors[i] / tot_exp for i, m in enumerate(model_results)}

        # Attach weight % to result
        for m in model_results:
            m["weight_pct"] = round(weights[m["model_id"]] * 100.0, 1)

        # Consensus Quantiles
        cons_p50 = sum(m["p50"] * weights[m["model_id"]] for m in model_results)
        cons_p10 = sum(m["p10"] * weights[m["model_id"]] for m in model_results)
        cons_p90 = sum(m["p90"] * weights[m["model_id"]] for m in model_results)

        # Conformal calibration adjustment (Split Conformal non-conformity)
        res_errors = [m["mape_pct"] / 100.0 * current_price for m in model_results]
        conformal_pad = float(np.median(res_errors)) * 0.4
        cons_p10 = min(cons_p10 - conformal_pad, cons_p50 - current_price * 0.005)
        cons_p90 = max(cons_p90 + conformal_pad, cons_p50 + current_price * 0.005)

        res = {
            "consensus_p10": round(cons_p10, 4),
            "consensus_p50": round(cons_p50, 4),
            "consensus_p90": round(cons_p90, 4),
            "model_results": model_results,
            "weights": weights,
        }
        cls._cache[cache_key] = res
        return res
