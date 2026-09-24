"""
HFP AI Engine — Hybrid Stacking Ensemble with Quantile Output
==============================================================

Institutional-grade prediction engine that combines:
    1. **LSTM** — temporal sequence modelling (price patterns over time)
    2. **XGBoost** — tabular feature importance (indicator + volume signals)
    3. **Stacking Meta-Learner** — ridge regression that blends base outputs

Output Contract
---------------
Every prediction is a **Probability Range**, not a point estimate::

    P10  — Conservative Low   (90 % chance price stays above)
    P50  — Most Likely price
    P90  — Aggressive High    (90 % chance price stays below)

GPU Acceleration
----------------
Detects ``torch.cuda.is_available()`` and auto-routes LSTM training/inference
to RTX 4070 CUDA cores.  XGBoost uses ``device="cuda"`` when available.

Rolling-Window Retraining
-------------------------
``AIEngine.retrain()`` implements walk-forward expanding-window training so the
model always learns from the freshest data without look-ahead bias.

Usage
-----
    from src.models.ai_engine import AIEngine
    from src.data.data_feed import DataLake

    lake   = DataLake()
    engine = AIEngine()

    feat = lake.pipeline("AAPL", timeframe="1d", lookback="2y", normalise=False)
    engine.train(feat, target_col="Close")

    pred = engine.predict(feat.iloc[-30:])
    print(pred)   # QuantilePrediction(p10=148.3, p50=152.7, p90=157.1, ...)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

# Ensure project root is on path
_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from src.utils.logger import get_logger

logger = get_logger("hfp.ai_engine")

# ---------------------------------------------------------------------------
# Device detection (shared by LSTM + XGBoost)
# ---------------------------------------------------------------------------
try:
    import torch

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        logger.info("CUDA detected: %s", torch.cuda.get_device_name(0))
except ImportError:
    DEVICE = None  # type: ignore[assignment]
    CUDA_AVAILABLE = False
    logger.warning("PyTorch not installed — LSTM disabled, XGBoost-only mode")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "hfp"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Data Contracts                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝


@dataclass
class QuantilePrediction:
    """Single-step quantile prediction output."""

    p10: float  # 10th percentile (conservative low)
    p50: float  # 50th percentile (most likely)
    p90: float  # 90th percentile (aggressive high)
    confidence: float  # 0–1 derived from spread width
    current_price: float = 0.0  # reference price for direction
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def spread(self) -> float:
        """Width of the prediction cloud (P90 − P10)."""
        return max(0.0, self.p90 - self.p10)

    @property
    def direction(self) -> str:
        """Predicted direction relative to current price or baseline."""
        ref = self.current_price if self.current_price > 0 else 0.0
        if ref > 0:
            diff_pct = (self.p50 - ref) / ref
            if diff_pct > 0.002:
                return "BULLISH"
            elif diff_pct < -0.002:
                return "BEARISH"
            return "NEUTRAL"
        elif self.p50 > 0.002:
            return "BULLISH"
        elif self.p50 < -0.002:
            return "BEARISH"
        return "NEUTRAL"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainResult:
    """Result container from a training session."""

    lstm_loss: Optional[float] = None
    xgb_rmse: Optional[float] = None
    meta_mae: Optional[float] = None
    relative_mae_pct: Optional[float] = None
    duration_sec: float = 0.0
    train_rows: int = 0
    test_rows: int = 0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def health_color(self) -> str:
        """Green / Yellow / Red health indicator for UI."""
        if self.relative_mae_pct is not None:
            if self.relative_mae_pct < 1.5:
                return "green"
            if self.relative_mae_pct < 3.5:
                return "yellow"
            return "red"
        # Fallback if relative_mae_pct not set
        if self.meta_mae is not None and self.meta_mae < 2.0:
            return "green"
        if self.meta_mae is not None and self.meta_mae < 5.0:
            return "yellow"
        return "red"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  LSTM Base Learner                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class _LSTMNet(torch.nn.Module if DEVICE is not None else object):  # type: ignore[misc]
    """
    Multi-layer LSTM with attention → 3-head quantile output (P10/P50/P90).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Attention
        self.attn = torch.nn.Linear(hidden_dim, 1)
        # Three heads for quantile output
        self.head_p10 = torch.nn.Linear(hidden_dim, 1)
        self.head_p50 = torch.nn.Linear(hidden_dim, 1)
        self.head_p90 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)
        # Attention weights
        attn_w = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1)  # (batch, seq_len)
        context = torch.bmm(attn_w.unsqueeze(1), lstm_out).squeeze(1)  # (batch, hidden)
        p50 = self.head_p50(context)
        # Guarantee monotonic non-crossing quantiles: P10 <= P50 <= P90
        p10 = p50 - torch.nn.functional.softplus(self.head_p10(context))
        p90 = p50 + torch.nn.functional.softplus(self.head_p90(context))
        return p10, p50, p90


class _QuantileLoss(torch.nn.Module if DEVICE is not None else object):  # type: ignore[misc]
    """Pinball loss for quantile regression (asymmetric penalty)."""

    def __init__(self, quantiles: List[float] = None):
        super().__init__()
        self.quantiles = quantiles or [0.10, 0.50, 0.90]

    def forward(
        self,
        preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        for q, p in zip(self.quantiles, preds):
            err = target - p.squeeze(-1)
            losses.append(torch.max(q * err, (q - 1) * err).mean())
        return sum(losses)


class LSTMBaseLeaner:
    """Wrapper around _LSTMNet for train / predict lifecycle."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        seq_len: int = 30,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 64,
    ):
        if DEVICE is None:
            raise RuntimeError("PyTorch required for LSTM — install torch with CUDA")

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = _LSTMNet(input_dim, hidden_dim, num_layers).to(DEVICE)
        self.criterion = _QuantileLoss().to(DEVICE)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5
        )
        self.scaler_y = MinMaxScaler()
        self._trained = False

    # ── helpers ────────────────────────────────────────────────────────

    def _make_sequences(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Sliding-window sequencing."""
        seqs, targets = [], []
        for i in range(len(X) - self.seq_len):
            seqs.append(X[i : i + self.seq_len])
            if y is not None:
                targets.append(y[i + self.seq_len])
        Xt = torch.FloatTensor(np.array(seqs)).to(DEVICE)
        yt = torch.FloatTensor(np.array(targets)).to(DEVICE) if y is not None else None
        return Xt, yt

    # ── train ──────────────────────────────────────────────────────────

    def train(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Train LSTM on feature matrix *X* and target vector *y*.

        Returns final epoch loss.
        """
        # Scale target to [0, 1]
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        Xt, yt = self._make_sequences(X, y_scaled)
        if Xt.shape[0] == 0:
            logger.warning("Not enough data for LSTM sequences (need > %d rows)", self.seq_len)
            return float("inf")

        dataset = torch.utils.data.TensorDataset(Xt, yt)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        self.model.train()
        final_loss = float("inf")

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for xb, yb in loader:
                self.optimizer.zero_grad()
                preds = self.model(xb)
                loss = self.criterion(preds, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(len(loader), 1)
            self.scheduler.step(avg_loss)

            if epoch % 10 == 0 or epoch == 1:
                logger.info("LSTM epoch %3d/%d  loss=%.6f", epoch, self.epochs, avg_loss)
            final_loss = avg_loss

        self.model.eval()
        self._trained = True
        return final_loss

    # ── predict ────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict quantiles from feature matrix *X*.

        Returns (p10, p50, p90) arrays in original scale.
        """
        if not self._trained:
            raise RuntimeError("LSTM not trained yet — call .train() first")

        Xt, _ = self._make_sequences(X)
        if Xt.shape[0] == 0:
            empty = np.array([])
            return empty, empty, empty

        self.model.eval()
        with torch.no_grad():
            p10_s, p50_s, p90_s = self.model(Xt)

        # Inverse-scale back to original price domain
        p10 = self.scaler_y.inverse_transform(p10_s.cpu().numpy()).ravel()
        p50 = self.scaler_y.inverse_transform(p50_s.cpu().numpy()).ravel()
        p90 = self.scaler_y.inverse_transform(p90_s.cpu().numpy()).ravel()
        return p10, p50, p90

    # ── persistence ────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or MODELS_DIR / "lstm_quantile.pt"
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "scaler_y": self.scaler_y,
                "input_dim": self.input_dim,
                "seq_len": self.seq_len,
            },
            path,
        )
        logger.info("LSTM saved → %s", path)
        return path

    def load(self, path: Optional[Path] = None) -> None:
        path = path or MODELS_DIR / "lstm_quantile.pt"
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        if "scaler_y" in ckpt:
            self.scaler_y = ckpt["scaler_y"]
        else:
            self.scaler_y.data_min_ = np.array(ckpt["scaler_y_min"])
            self.scaler_y.data_max_ = np.array(ckpt["scaler_y_max"])
            self.scaler_y.data_range_ = self.scaler_y.data_max_ - self.scaler_y.data_min_
            self.scaler_y.scale_ = 1.0 / self.scaler_y.data_range_
            self.scaler_y.min_ = -self.scaler_y.data_min_ * self.scaler_y.scale_
        self._trained = True
        logger.info("LSTM loaded ← %s", path)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  XGBoost Base Learner                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class XGBBaseLearner:
    """Three XGBoost regressors — one per quantile (P10, P50, P90)."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        early_stopping_rounds: int = 20,
    ):
        import xgboost as xgb

        base_params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )

        # Use GPU if available
        if CUDA_AVAILABLE:
            base_params["device"] = "cuda"
            base_params["tree_method"] = "hist"
            logger.info("XGBoost: GPU acceleration enabled")

        self.models: Dict[str, xgb.XGBRegressor] = {
            "p10": xgb.XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=0.10,
                **base_params,
            ),
            "p50": xgb.XGBRegressor(
                objective="reg:squarederror",
                **base_params,
            ),
            "p90": xgb.XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=0.90,
                **base_params,
            ),
        }
        self.early_stopping_rounds = early_stopping_rounds
        self._trained = False

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Train all three quantile regressors. Returns per-quantile RMSE."""
        scores: Dict[str, float] = {}
        for name, model in self.models.items():
            fit_params: Dict[str, Any] = {}
            if X_val is not None and y_val is not None:
                fit_params["eval_set"] = [(X_val, y_val)]
                fit_params["verbose"] = False
            model.fit(X_train, y_train, **fit_params)

            preds = model.predict(X_train)
            rmse = float(np.sqrt(mean_squared_error(y_train, preds)))
            scores[name] = rmse
            logger.info("XGBoost %s train RMSE: %.6f", name, rmse)

        self._trained = True
        return scores

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._trained:
            raise RuntimeError("XGBoost not trained — call .train() first")
        p10 = self.models["p10"].predict(X)
        p50 = self.models["p50"].predict(X)
        p90 = self.models["p90"].predict(X)
        # Guarantee non-crossing quantiles: P10 <= P50 <= P90
        p10 = np.minimum(p10, p50)
        p90 = np.maximum(p90, p50)
        return p10, p50, p90

    def feature_importance(self) -> pd.DataFrame:
        """Return P50 model feature importance scores."""
        imp = self.models["p50"].feature_importances_
        return pd.DataFrame({"importance": imp}).sort_values(
            "importance", ascending=False
        )

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or MODELS_DIR / "xgb_quantile"
        path.mkdir(parents=True, exist_ok=True)
        for name, model in self.models.items():
            model.save_model(str(path / f"{name}.json"))
        logger.info("XGBoost saved → %s", path)
        return path

    def load(self, path: Optional[Path] = None) -> None:
        import xgboost as xgb

        path = path or MODELS_DIR / "xgb_quantile"
        for name in self.models:
            fpath = path / f"{name}.json"
            if fpath.exists():
                self.models[name].load_model(str(fpath))
        self._trained = True
        logger.info("XGBoost loaded ← %s", path)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Stacking Meta-Learner                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class _StackingMeta:
    """
    Meta-learner that blends LSTM + XGBoost predictions into calibrated,
    non-crossing P10 / P50 / P90 quantile outputs.
    """

    def __init__(self, alpha: float = 1.0):
        self.meta_p50 = Ridge(alpha=alpha)
        self.calibrated_p10_spread: float = 0.5
        self.calibrated_p90_spread: float = 0.5
        self._trained: bool = False

    def train(
        self,
        lstm_preds: Tuple[np.ndarray, np.ndarray, np.ndarray],
        xgb_preds: Tuple[np.ndarray, np.ndarray, np.ndarray],
        y_true: np.ndarray,
    ) -> float:
        """Train meta-learner on stacked base predictions and calibrate quantile spread."""
        # Stack: [lstm_p50, xgb_p50]
        X_meta = np.column_stack([lstm_preds[1], xgb_preds[1]])

        self.meta_p50.fit(X_meta, y_true)
        meta_pred = self.meta_p50.predict(X_meta)

        # Residuals to calibrate quantile spread on holdout fold
        residuals = y_true - meta_pred
        q10_res = abs(float(np.percentile(residuals, 10))) if len(residuals) > 5 else 0.5
        q90_res = abs(float(np.percentile(residuals, 90))) if len(residuals) > 5 else 0.5

        # Base model spreads
        lstm_s10 = np.maximum(0.01, lstm_preds[1] - lstm_preds[0])
        lstm_s90 = np.maximum(0.01, lstm_preds[2] - lstm_preds[1])
        xgb_s10 = np.maximum(0.01, xgb_preds[1] - xgb_preds[0])
        xgb_s90 = np.maximum(0.01, xgb_preds[2] - xgb_preds[1])

        mean_base_10 = float(np.mean(0.5 * (lstm_s10 + xgb_s10)))
        mean_base_90 = float(np.mean(0.5 * (lstm_s90 + xgb_s90)))

        self.calibrated_p10_spread = max(q10_res, mean_base_10, 0.05)
        self.calibrated_p90_spread = max(q90_res, mean_base_90, 0.05)
        self._trained = True

        mae = float(mean_absolute_error(y_true, meta_pred))
        logger.info(
            "Meta-learner MAE: %.6f (Calibrated Spread: P10=-%.4f, P90=+%.4f)",
            mae, self.calibrated_p10_spread, self.calibrated_p90_spread
        )
        return mae

    def predict(
        self,
        lstm_preds: Tuple[np.ndarray, np.ndarray, np.ndarray],
        xgb_preds: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_meta = np.column_stack([lstm_preds[1], xgb_preds[1]])
        p50 = self.meta_p50.predict(X_meta)

        lstm_s10 = np.maximum(0.01, lstm_preds[1] - lstm_preds[0])
        lstm_s90 = np.maximum(0.01, lstm_preds[2] - lstm_preds[1])
        xgb_s10 = np.maximum(0.01, xgb_preds[1] - xgb_preds[0])
        xgb_s90 = np.maximum(0.01, xgb_preds[2] - xgb_preds[1])

        blend_s10 = 0.5 * (lstm_s10 + xgb_s10)
        blend_s90 = 0.5 * (lstm_s90 + xgb_s90)

        spread_10 = np.maximum(blend_s10, self.calibrated_p10_spread * 0.5)
        spread_90 = np.maximum(blend_s90, self.calibrated_p90_spread * 0.5)

        p10 = p50 - spread_10
        p90 = p50 + spread_90
        return p10, p50, p90


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  AIEngine — Primary Interface                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class AIEngine:
    """
    Stacking ensemble: LSTM + XGBoost → Ridge meta-learner → P10/P50/P90.

    Parameters
    ----------
    seq_len : int
        LSTM look-back window (bars).
    lstm_hidden : int
        LSTM hidden dimension.
    lstm_layers : int
        Number of stacked LSTM layers.
    lstm_epochs : int
        Training epochs for LSTM.
    xgb_estimators : int
        Number of boosted trees per XGBoost quantile head.
    test_ratio : float
        Fraction of data held out for meta-learner training.
    """

    def __init__(
        self,
        seq_len: int = 30,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        lstm_epochs: int = 50,
        xgb_estimators: int = 500,
        test_ratio: float = 0.2,
    ):
        self.seq_len = seq_len
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.lstm_epochs = lstm_epochs
        self.xgb_estimators = xgb_estimators
        self.test_ratio = test_ratio

        self._lstm: Optional[LSTMBaseLeaner] = None
        self._xgb: Optional[XGBBaseLearner] = None
        self._meta: Optional[_StackingMeta] = None
        self._feature_cols: List[str] = []
        self._target_col: str = "Close"
        self._trained = False
        self._last_result: Optional[TrainResult] = None

    # ══════════════════════════════════════════════════════════════════════
    #  Training
    # ══════════════════════════════════════════════════════════════════════

    def train(
        self,
        df: pd.DataFrame,
        target_col: str = "Close",
        *,
        feature_cols: Optional[List[str]] = None,
    ) -> TrainResult:
        """
        Full training pipeline:
            1. Split chronologically into train / validation.
            2. Train LSTM on sequences.
            3. Train XGBoost on flat features.
            4. Collect base predictions on the validation fold.
            5. Train Ridge meta-learner on stacked base outputs.

        Parameters
        ----------
        df : pd.DataFrame
            Feature-engineered DataFrame (from ``DataLake``).
        target_col : str
            Column to predict (default ``"Close"``).
        feature_cols : list[str] | None
            Explicit feature column list.  If None, auto-selects all
            numeric columns except *target_col*.

        Returns
        -------
        TrainResult
        """
        t0 = time.time()
        result = TrainResult()

        # ── Resolve feature columns ───────────────────────────────────
        if feature_cols:
            self._feature_cols = feature_cols
        else:
            self._feature_cols = [
                c
                for c in df.select_dtypes(include=[np.number]).columns
                if c != target_col
            ]
        self._target_col = target_col

        logger.info(
            "Training on %d rows, %d features, target='%s'",
            len(df), len(self._feature_cols), target_col,
        )

        # ── Prepare arrays ────────────────────────────────────────────
        df_clean = df[self._feature_cols + [target_col]].dropna().copy()
        # Shift target by -1 so that features at row t predict the NEXT bar's target (t+1)
        df_clean["Forward_Target"] = df_clean[target_col].shift(-1)
        df_clean = df_clean.dropna()

        if len(df_clean) < self.seq_len + 10:
            logger.error("Not enough data to train (%d rows, need %d+)", len(df_clean), self.seq_len + 10)
            result.duration_sec = time.time() - t0
            return result

        X = df_clean[self._feature_cols].values.astype(np.float32)
        y = df_clean["Forward_Target"].values.astype(np.float32)

        # Chronological split
        split = int(len(X) * (1 - self.test_ratio))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        result.train_rows = len(X_train)
        result.test_rows = len(X_val)

        # ── 1. Train LSTM ─────────────────────────────────────────────
        if DEVICE is not None:
            self._lstm = LSTMBaseLeaner(
                input_dim=X_train.shape[1],
                hidden_dim=self.lstm_hidden,
                num_layers=self.lstm_layers,
                seq_len=self.seq_len,
                epochs=self.lstm_epochs,
            )
            result.lstm_loss = self._lstm.train(X_train, y_train)
        else:
            logger.warning("PyTorch unavailable — skipping LSTM, XGBoost-only mode")

        # ── 2. Train XGBoost ──────────────────────────────────────────
        self._xgb = XGBBaseLearner(n_estimators=self.xgb_estimators)
        xgb_scores = self._xgb.train(X_train, y_train, X_val, y_val)
        result.xgb_rmse = xgb_scores.get("p50")

        # ── 3. Collect base predictions on validation set ─────────────
        xgb_val = self._xgb.predict(X_val)

        if self._lstm is not None and self._lstm._trained:
            # LSTM needs seq_len context, so feed full data and slice
            lstm_full_p10, lstm_full_p50, lstm_full_p90 = self._lstm.predict(X)
            # Align with validation indices
            n_lstm = len(lstm_full_p10)
            val_start = n_lstm - len(X_val)
            if val_start < 0:
                val_start = 0
            lstm_val = (
                lstm_full_p10[val_start : val_start + len(X_val)],
                lstm_full_p50[val_start : val_start + len(X_val)],
                lstm_full_p90[val_start : val_start + len(X_val)],
            )
        else:
            # Duplicate XGBoost preds as placeholder
            lstm_val = xgb_val

        # Ensure same length
        min_len = min(len(lstm_val[0]), len(xgb_val[0]), len(y_val))
        lstm_val = tuple(a[:min_len] for a in lstm_val)
        xgb_val = tuple(a[:min_len] for a in xgb_val)
        y_val_aligned = y_val[:min_len]

        # ── 4. Train meta-learner ─────────────────────────────────────
        self._meta = _StackingMeta()
        result.meta_mae = self._meta.train(lstm_val, xgb_val, y_val_aligned)
        avg_target = float(np.mean(y_val_aligned)) if len(y_val_aligned) > 0 else 1.0
        result.relative_mae_pct = (result.meta_mae / avg_target) * 100.0 if avg_target > 0 else 0.0

        result.duration_sec = time.time() - t0
        self._trained = True
        self._last_result = result

        logger.info(
            "Training complete in %.1fs  — LSTM loss=%.6f  XGB RMSE=%.6f  Meta MAE=%.6f (Rel=%.2f%%)  Health=%s",
            result.duration_sec,
            result.lstm_loss or 0,
            result.xgb_rmse or 0,
            result.meta_mae or 0,
            result.relative_mae_pct,
            result.health_color,
        )
        return result

    # ══════════════════════════════════════════════════════════════════════
    #  Rolling-Window Retrain
    # ══════════════════════════════════════════════════════════════════════

    def retrain(
        self,
        df: pd.DataFrame,
        target_col: str = "Close",
        window_size: int = 252,
        step: int = 21,
    ) -> List[TrainResult]:
        """
        Walk-forward rolling-window retraining.

        Re-trains the full ensemble every *step* bars using a sliding
        window of *window_size* bars.  Returns a list of results
        for each fold.
        """
        results: List[TrainResult] = []
        total = len(df)
        fold = 0

        for start in range(0, total - window_size, step):
            fold += 1
            chunk = df.iloc[start : start + window_size]
            logger.info("Retrain fold %d  rows %d–%d", fold, start, start + window_size)
            r = self.train(chunk, target_col)
            results.append(r)

        logger.info("Rolling retrain complete: %d folds", len(results))
        return results

    # ══════════════════════════════════════════════════════════════════════
    #  Prediction
    # ══════════════════════════════════════════════════════════════════════

    def predict(self, df: pd.DataFrame) -> QuantilePrediction:
        """
        Generate a single QuantilePrediction from the latest data window.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain all feature columns used during training.
            Should be at least ``seq_len`` rows long.

        Returns
        -------
        QuantilePrediction
        """
        if not self._trained:
            raise RuntimeError("Engine not trained — call .train() first")

        clean_df = df[self._feature_cols].dropna()
        X = clean_df.values.astype(np.float32)
        if len(X) == 0:
            raise ValueError("No valid feature rows in DataFrame for prediction")

        # Reference price for direction calculation
        current_price = 0.0
        if self._target_col in df.columns:
            target_series = df[self._target_col].dropna()
            if not target_series.empty:
                current_price = float(target_series.iloc[-1])

        # XGBoost predicts from the last row
        xgb_preds = self._xgb.predict(X[-1:])

        # LSTM predicts from the last seq_len rows
        if self._lstm is not None and self._lstm._trained and len(X) >= self.seq_len:
            self._lstm.model.eval()
            last_seq = X[-self.seq_len:][np.newaxis, :, :]  # (1, seq_len, features)
            with torch.no_grad():
                seq_t = torch.FloatTensor(last_seq).to(DEVICE)
                p10_s, p50_s, p90_s = self._lstm.model(seq_t)
            p10 = self._lstm.scaler_y.inverse_transform(p10_s.cpu().numpy()).ravel()
            p50 = self._lstm.scaler_y.inverse_transform(p50_s.cpu().numpy()).ravel()
            p90 = self._lstm.scaler_y.inverse_transform(p90_s.cpu().numpy()).ravel()
            lstm_preds = (p10, p50, p90)
        else:
            lstm_preds = xgb_preds

        # Meta-learner blend
        p10, p50, p90 = self._meta.predict(lstm_preds, xgb_preds)

        val_p50 = float(p50[0])
        val_p10 = float(min(p10[0], val_p50 - 0.01))
        val_p90 = float(max(p90[0], val_p50 + 0.01))

        # Confidence: inversely proportional to relative spread
        spread = val_p90 - val_p10
        mid = abs(val_p50) if val_p50 != 0 else 1.0
        rel_spread = spread / mid
        confidence = float(np.clip(1.0 / (1.0 + 20.0 * rel_spread), 0.05, 0.99))

        return QuantilePrediction(
            p10=val_p10,
            p50=val_p50,
            p90=val_p90,
            confidence=confidence,
            current_price=current_price,
        )

    def predict_series(
        self,
        df: pd.DataFrame,
        horizon: int = 5,
    ) -> List[QuantilePrediction]:
        """
        Generate a series of predictions by sliding forward.

        Useful for plotting a predicted path on the chart.
        """
        predictions: List[QuantilePrediction] = []
        clean_df = df.dropna(subset=self._feature_cols)
        n = len(clean_df)
        if n == 0:
            return predictions

        start = max(1, n - horizon)
        for i in range(start, n):
            chunk = clean_df.iloc[: i + 1]
            try:
                pred = self.predict(chunk)
                predictions.append(pred)
            except Exception as exc:
                logger.debug("predict_series skip idx %d: %s", i, exc)

        return predictions

    # ══════════════════════════════════════════════════════════════════════
    #  Persistence
    # ══════════════════════════════════════════════════════════════════════

    def save(self, tag: str = "latest") -> Path:
        """Save all sub-models to ``models/hfp/<tag>/``."""
        save_dir = MODELS_DIR / tag
        save_dir.mkdir(parents=True, exist_ok=True)

        if self._lstm is not None:
            self._lstm.save(save_dir / "lstm_quantile.pt")

        if self._xgb is not None:
            self._xgb.save(save_dir / "xgb_quantile")

        # Meta-learner coefficients
        if self._meta is not None:
            import pickle

            with open(save_dir / "meta_stack.pkl", "wb") as f:
                pickle.dump(self._meta, f)

        # Metadata
        meta = {
            "feature_cols": self._feature_cols,
            "target_col": self._target_col,
            "seq_len": self.seq_len,
            "lstm_hidden": self.lstm_hidden,
            "lstm_layers": self.lstm_layers,
            "xgb_estimators": self.xgb_estimators,
            "trained": self._trained,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        if self._last_result:
            meta["last_train"] = self._last_result.to_dict()

        with open(save_dir / "engine_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("AIEngine saved → %s", save_dir)
        return save_dir

    def load(self, tag: str = "latest") -> None:
        """Load all sub-models from ``models/hfp/<tag>/``."""
        load_dir = MODELS_DIR / tag
        if not load_dir.exists():
            raise FileNotFoundError(f"No saved model at {load_dir}")

        # Metadata
        meta_path = load_dir / "engine_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self._feature_cols = meta.get("feature_cols", [])
            self._target_col = meta.get("target_col", "Close")
            self.seq_len = meta.get("seq_len", self.seq_len)
            self.lstm_hidden = meta.get("lstm_hidden", self.lstm_hidden)
            self.lstm_layers = meta.get("lstm_layers", self.lstm_layers)
            self.xgb_estimators = meta.get("xgb_estimators", self.xgb_estimators)

        # LSTM
        lstm_path = load_dir / "lstm_quantile.pt"
        if lstm_path.exists() and DEVICE is not None:
            n_features = len(self._feature_cols) or 1
            self._lstm = LSTMBaseLeaner(
                input_dim=n_features,
                hidden_dim=self.lstm_hidden,
                num_layers=self.lstm_layers,
                seq_len=self.seq_len,
            )
            self._lstm.load(lstm_path)

        # XGBoost
        xgb_path = load_dir / "xgb_quantile"
        if xgb_path.exists():
            self._xgb = XGBBaseLearner()
            self._xgb.load(xgb_path)

        # Meta-learner
        meta_pkl = load_dir / "meta_stack.pkl"
        if meta_pkl.exists():
            import pickle

            with open(meta_pkl, "rb") as f:
                self._meta = pickle.load(f)

        self._trained = True
        logger.info("AIEngine loaded ← %s", load_dir)

    # ══════════════════════════════════════════════════════════════════════
    #  Diagnostics
    # ══════════════════════════════════════════════════════════════════════

    @property
    def health(self) -> str:
        """Colour-coded health indicator for the UI."""
        if self._last_result:
            return self._last_result.health_color
        return "red"

    @property
    def last_train_result(self) -> Optional[TrainResult]:
        return self._last_result

    def feature_importance(self) -> Optional[pd.DataFrame]:
        """Delegate to XGBoost P50 importance."""
        if self._xgb and self._xgb._trained:
            imp = self._xgb.feature_importance()
            imp.index = self._feature_cols[: len(imp)]
            return imp
        return None

    def __repr__(self) -> str:
        return (
            f"AIEngine(trained={self._trained}, health={self.health!r}, "
            f"seq_len={self.seq_len}, features={len(self._feature_cols)})"
        )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Smoke test  (python -m src.models.ai_engine)                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    import sys

    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from src.data.data_feed import DataLake

    print("=" * 60)
    print("  HFP AI Engine — Smoke Test")
    print("=" * 60)

    lake = DataLake(sentiment_backend="vader")
    engine = AIEngine(lstm_epochs=10, xgb_estimators=100, seq_len=20)

    # Fetch & engineer (skip heavy sentiment for speed)
    df = lake.fetch("AAPL", timeframe="1d", lookback="1y")
    feat = lake.engineer_features(df, include_sentiment=False)
    feat.ffill(inplace=True)
    feat.bfill(inplace=True)
    feat.dropna(inplace=True)

    print(f"\n📊 Feature matrix: {feat.shape}")

    # Train
    result = engine.train(feat, target_col="Close")
    print(f"\n🧠 Train result: {result}")
    print(f"   Health: {result.health_color}")

    # Predict
    pred = engine.predict(feat)
    print(f"\n🎯 Prediction: P10={pred.p10:.2f}  P50={pred.p50:.2f}  P90={pred.p90:.2f}")
    print(f"   Confidence: {pred.confidence:.2%}")
    print(f"   Direction:  {pred.direction}")
    print(f"   Spread:     {pred.spread:.2f}")

    # Feature importance
    imp = engine.feature_importance()
    if imp is not None:
        print(f"\n📈 Top 5 features:")
        print(imp.head())

    # Save
    path = engine.save("smoke_test")
    print(f"\n💾 Saved to: {path}")

    # Load
    engine2 = AIEngine()
    engine2.load("smoke_test")
    pred2 = engine2.predict(feat)
    print(f"\n🔄 Reload prediction: P50={pred2.p50:.2f}  (match={abs(pred.p50 - pred2.p50) < 0.01})")

    print("\n✅ AI Engine smoke test passed.")
