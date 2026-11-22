"""
TradeSense v8 — GPU FinBERT Transformer & Macro Polarity Engine
===============================================================
Provides high-throughput transformer financial sentiment scoring powered by
ProsusAI/finbert on PyTorch CUDA (RTX 4070) with graceful VADER fallback,
SHA-256 headline caching, and domain-aware macro polarity mapping so hawkish
inflation news correctly moves Fed/CPI event contracts in the right direction.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import threading
from typing import Any, Dict, List, Optional

import numpy as np

from src.config import config
from src.utils.logger import get_logger

logger = get_logger("hfp.finbert")

# Macro lexicon weights to resolve FinBERT directional ambiguity on rate/CPI/jobs contracts
HAWKISH_CUES: Dict[str, float] = {
    "hawkish": 0.45,
    "rate hike": 0.55,
    "hike rates": 0.55,
    "hikes": 0.35,
    "higher for longer": 0.50,
    "sticky inflation": 0.45,
    "hot inflation": 0.50,
    "hot cpi": 0.50,
    "inflation accelerates": 0.45,
    "inflation surges": 0.45,
    "reaccelerate": 0.40,
    "tightening": 0.35,
    "hold rates": 0.25,
    "keeps rates steady": 0.25,
    "pause cuts": 0.35,
    "delay rate cut": 0.45,
    "delays rate cuts": 0.45,
    "above forecast": 0.30,
    "above expectations": 0.30,
    "tariff": 0.20,
    "tariffs": 0.20,
}

DOVISH_CUES: Dict[str, float] = {
    "dovish": 0.45,
    "rate cut": 0.55,
    "cut rates": 0.55,
    "rate cuts": 0.50,
    "easing": 0.40,
    "pivot": 0.30,
    "cooling inflation": 0.45,
    "inflation cools": 0.45,
    "disinflation": 0.45,
    "softening": 0.30,
    "recession": 0.45,
    "layoffs": 0.35,
    "unemployment rises": 0.45,
    "weak jobs": 0.45,
    "payrolls miss": 0.45,
    "slowdown": 0.35,
    "below forecast": 0.30,
    "below expectations": 0.30,
}

GROWTH_BULL_CUES: Dict[str, float] = {
    "beats estimates": 0.45,
    "strong payrolls": 0.50,
    "hiring surges": 0.45,
    "gdp accelerates": 0.45,
    "economic boom": 0.45,
    "expansion": 0.35,
    "low unemployment": 0.35,
    "resilient economy": 0.35,
}

GROWTH_BEAR_CUES: Dict[str, float] = {
    "contraction": 0.50,
    "recession": 0.50,
    "negative gdp": 0.55,
    "job losses": 0.45,
    "payrolls drop": 0.45,
    "layoffs surge": 0.40,
    "unemployment spikes": 0.45,
    "slump": 0.35,
}


@dataclass
class FinBERTResult:
    """Tri-class sentiment probabilities and signed score from FinBERT (or VADER fallback)."""

    positive: float
    negative: float
    neutral: float
    net_score: float  # positive - negative in [-1.0, +1.0]
    label: str        # "POSITIVE" | "NEGATIVE" | "NEUTRAL"
    confidence: float
    backend: str      # "finbert_cuda" | "finbert_cpu" | "vader_fallback"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "positive": round(self.positive, 4),
            "negative": round(self.negative, 4),
            "neutral": round(self.neutral, 4),
            "net_score": round(self.net_score, 4),
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "backend": self.backend,
        }


class FinBERTEngine:
    """Thread-safe GPU FinBERT inference engine with SHA-256 cache & VADER fallback."""

    MODEL_ID = "ProsusAI/finbert"
    MAX_CACHE_SIZE = 5000

    def __init__(self):
        self._lock = threading.Lock()
        self._loaded = False
        self._fallback_mode = False
        self._tokenizer = None
        self._model = None
        self._device = "cpu"
        self._backend_name = "uninitialized"
        self._idx_pos = 0
        self._idx_neg = 1
        self._idx_neu = 2
        self._vader = None
        self._text_cache: Dict[str, FinBERTResult] = {}

    def _init_vader(self):
        if self._vader is None:
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self._vader = SentimentIntensityAnalyzer()
            except Exception:
                self._vader = False
        return self._vader if self._vader is not False else None

    def _ensure_loaded(self) -> None:
        if self._loaded or self._fallback_mode:
            return
        with self._lock:
            if self._loaded or self._fallback_mode:
                return

            pref = (config.SENTIMENT_BACKEND or "finbert").lower().strip()
            if pref == "vader" or os.getenv("TRADESENSE_OFFLINE_TEST") == "1":
                self._fallback_mode = True
                self._backend_name = "vader_fallback"
                return

            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                is_ci = os.getenv("CI", "").lower() == "true"
                # Prefer local cache first for zero network stall
                try:
                    tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID, local_files_only=True)
                    model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_ID, local_files_only=True)
                except Exception:
                    if is_ci:
                        raise RuntimeError("Skipping HF download in CI when model is not locally cached")
                    tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
                    model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_ID)

                device = "cuda" if torch.cuda.is_available() else "cpu"
                model.to(device)
                model.eval()

                id2label = getattr(model.config, "id2label", {0: "positive", 1: "negative", 2: "neutral"})
                for idx, lbl in id2label.items():
                    low = str(lbl).lower()
                    if "pos" in low:
                        self._idx_pos = int(idx)
                    elif "neg" in low:
                        self._idx_neg = int(idx)
                    else:
                        self._idx_neu = int(idx)

                self._tokenizer = tokenizer
                self._model = model
                self._device = device
                self._backend_name = "finbert_cuda" if device == "cuda" else "finbert_cpu"
                self._loaded = True
                logger.info(
                    "ProsusAI/finbert loaded on %s (labels pos=%d neg=%d neu=%d)",
                    self._device.upper(),
                    self._idx_pos,
                    self._idx_neg,
                    self._idx_neu,
                )
            except Exception as exc:
                logger.warning("FinBERT unavailable (%s) — falling back to VADER lexicon", exc)
                self._fallback_mode = True
                self._backend_name = "vader_fallback"

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.strip().encode("utf-8", errors="ignore")).hexdigest()

    def _vader_score(self, text: str) -> FinBERTResult:
        analyzer = self._init_vader()
        if not analyzer or not text.strip():
            return FinBERTResult(0.0, 0.0, 1.0, 0.0, "NEUTRAL", 1.0, "vader_fallback")
        scores = analyzer.polarity_scores(text)
        p_pos = float(scores.get("pos", 0.0))
        p_neg = float(scores.get("neg", 0.0))
        p_neu = float(scores.get("neu", 1.0))
        total = max(1e-6, p_pos + p_neg + p_neu)
        p_pos, p_neg, p_neu = p_pos / total, p_neg / total, p_neu / total
        net = float(np.clip(scores.get("compound", p_pos - p_neg), -1.0, 1.0))
        if net >= 0.15:
            lbl = "POSITIVE"
        elif net <= -0.15:
            lbl = "NEGATIVE"
        else:
            lbl = "NEUTRAL"
        return FinBERTResult(
            positive=p_pos,
            negative=p_neg,
            neutral=p_neu,
            net_score=net,
            label=lbl,
            confidence=max(p_pos, p_neg, p_neu),
            backend="vader_fallback",
        )

    def _set_cache(self, hkey: str, res: FinBERTResult) -> None:
        if len(self._text_cache) >= self.MAX_CACHE_SIZE:
            # Evict oldest 10%
            drop_n = self.MAX_CACHE_SIZE // 10
            for k in list(self._text_cache.keys())[:drop_n]:
                self._text_cache.pop(k, None)
        self._text_cache[hkey] = res

    def score_texts(self, texts: List[str], batch_size: int = 32) -> List[FinBERTResult]:
        """Batch-score multiple financial headlines with SHA-256 caching."""
        if not texts:
            return []

        self._ensure_loaded()

        results: List[Optional[FinBERTResult]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        for idx, raw in enumerate(texts):
            txt = (raw or "").strip()
            if not txt:
                results[idx] = FinBERTResult(0.0, 0.0, 1.0, 0.0, "NEUTRAL", 1.0, self._backend_name)
                continue
            hkey = self._hash_text(txt)
            cached = self._text_cache.get(hkey)
            if cached is not None:
                results[idx] = cached
            else:
                uncached_indices.append(idx)
                uncached_texts.append(txt)

        if uncached_texts:
            if self._fallback_mode or not self._loaded:
                for pos, txt in zip(uncached_indices, uncached_texts):
                    res = self._vader_score(txt)
                    self._set_cache(self._hash_text(txt), res)
                    results[pos] = res
            else:
                import torch

                for start in range(0, len(uncached_texts), batch_size):
                    batch = uncached_texts[start : start + batch_size]
                    batch_idx = uncached_indices[start : start + batch_size]
                    enc = self._tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=128,
                        return_tensors="pt",
                    ).to(self._device)

                    with torch.no_grad():
                        if self._device == "cuda":
                            with torch.amp.autocast("cuda", dtype=torch.float16):
                                logits = self._model(**enc).logits
                        else:
                            logits = self._model(**enc).logits
                        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()

                    for j, orig_idx in enumerate(batch_idx):
                        row = probs[j]
                        p_pos = float(row[self._idx_pos])
                        p_neg = float(row[self._idx_neg])
                        p_neu = float(row[self._idx_neu])
                        net = float(np.clip(p_pos - p_neg, -1.0, 1.0))
                        top_idx = int(np.argmax([p_pos, p_neg, p_neu]))
                        lbl = ["POSITIVE", "NEGATIVE", "NEUTRAL"][top_idx]
                        res = FinBERTResult(
                            positive=p_pos,
                            negative=p_neg,
                            neutral=p_neu,
                            net_score=net,
                            label=lbl,
                            confidence=float(max(p_pos, p_neg, p_neu)),
                            backend=self._backend_name,
                        )
                        self._set_cache(self._hash_text(batch[j]), res)
                        results[orig_idx] = res

        return [r if r is not None else FinBERTResult(0.0, 0.0, 1.0, 0.0, "NEUTRAL", 1.0, self._backend_name) for r in results]

    def score_text(self, text: str) -> FinBERTResult:
        """Score a single headline or text snippet."""
        res = self.score_texts([text])
        return res[0] if res else FinBERTResult(0.0, 0.0, 1.0, 0.0, "NEUTRAL", 1.0, self._backend_name)

    def get_status(self) -> Dict[str, Any]:
        """Report current engine readiness without blocking if unloaded."""
        device_label = "CPU"
        if self._loaded and self._device == "cuda":
            try:
                import torch
                device_label = torch.cuda.get_device_name(0)
            except Exception:
                device_label = "CUDA GPU"
        elif not self._loaded and not self._fallback_mode:
            try:
                import torch
                if torch.cuda.is_available():
                    device_label = f"{torch.cuda.get_device_name(0)} (Lazy)"
            except Exception:
                pass
        return {
            "loaded": self._loaded,
            "fallback": self._fallback_mode,
            "backend": self._backend_name if (self._loaded or self._fallback_mode) else "finbert_ready",
            "device": device_label,
            "cache_size": len(self._text_cache),
        }

    @staticmethod
    def _lexicon_balance(text: str, pos_map: Dict[str, float], neg_map: Dict[str, float]) -> float:
        low = text.lower()
        pos_w = sum(w for k, w in pos_map.items() if k in low)
        neg_w = sum(w for k, w in neg_map.items() if k in low)
        if pos_w == 0.0 and neg_w == 0.0:
            return 0.0
        return float(np.clip(pos_w - neg_w, -1.0, 1.0))

    def score_macro_polarity(
        self,
        texts: List[str],
        domain: str,
        outcome_hint: str = "",
        recency_weights: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Compute directional probability shift [-1.0, +1.0] aligned to a binary
        prediction contract's YES resolution semantics.

        Fixes raw FinBERT polarity inversion on macro markets:
        e.g., 'Hot CPI keeps Fed hawkish' scores negative on equities in raw
        FinBERT, yet pushes YES probability UP on 'Fed Hike' / 'Fed Hold' and
        DOWN on 'Fed Cut'.
        """
        if not texts:
            return {
                "raw_finbert_mean": 0.0,
                "directional_shift_score": 0.0,
                "hawkish_dovish_index": 0.0,
                "article_scores": [],
                "backend": self.get_status()["backend"],
            }

        fb_results = self.score_texts(texts)
        weights = recency_weights if (recency_weights and len(recency_weights) == len(texts)) else [1.0] * len(texts)

        dom = (domain or "FINANCIALS").upper().strip()
        hint_low = (outcome_hint or "").lower()

        raw_scores: List[float] = []
        hd_scores: List[float] = []
        dir_scores: List[float] = []
        article_details: List[Dict[str, Any]] = []

        for txt, fb, w in zip(texts, fb_results, weights):
            raw_scores.append(fb.net_score * w)

            if dom in ("FED_RATES", "BOC_RATES"):
                # Positive hd = hawkish, negative hd = dovish
                lex_hd = self._lexicon_balance(txt, HAWKISH_CUES, DOVISH_CUES)
                # Negative equity/bond reaction on rate news often reinforces hawkishness when cues exist
                if abs(lex_hd) > 0.05:
                    hd = float(np.clip(0.75 * lex_hd + 0.25 * (-fb.net_score * np.sign(lex_hd) * 0.5 + lex_hd * 0.5), -1.0, 1.0))
                else:
                    hd = float(np.clip(-fb.net_score * 0.25, -1.0, 1.0))
                hd_scores.append(hd * w)

                # Align hd with the specific contract bucket
                if any(k in hint_low for k in ["cut", "lower", "decrease", "below"]):
                    d_shift = -hd
                elif any(k in hint_low for k in ["hike", "raise", "increase", "above"]):
                    d_shift = hd
                elif any(k in hint_low for k in ["maintain", "hold", "unchanged", "no change"]):
                    # Extreme dovish or extreme hawkish surprises both reduce hold odds;
                    # mild hawkish/pause cues support hold.
                    d_shift = float(np.clip(0.35 - 0.55 * abs(hd) + (0.20 if hd > 0 else 0.0), -0.6, 0.5))
                else:
                    d_shift = hd

            elif dom in ("CPI_INFLATION", "PCE"):
                hot_cool = self._lexicon_balance(txt, HAWKISH_CUES, DOVISH_CUES)
                hd = float(np.clip(0.8 * hot_cool + 0.2 * (-fb.net_score), -1.0, 1.0)) if hot_cool != 0 else float(-fb.net_score * 0.3)
                hd_scores.append(hd * w)
                if any(k in hint_low for k in ["below", "under", "less than", "<"]):
                    d_shift = -hd
                else:
                    d_shift = hd

            elif dom == "GROWTH_JOBS":
                gr = self._lexicon_balance(txt, GROWTH_BULL_CUES, GROWTH_BEAR_CUES)
                g_score = float(np.clip(0.6 * gr + 0.4 * fb.net_score, -1.0, 1.0)) if gr != 0 else fb.net_score
                hd_scores.append(g_score * w)
                # Unemployment rate moves inversely to growth/payrolls!
                if "unemployment" in hint_low or "u3" in hint_low or "jobless" in hint_low:
                    d_shift = -g_score if not any(b in hint_low for b in ["below", "under", "<"]) else g_score
                elif any(b in hint_low for b in ["below", "under", "<"]):
                    d_shift = -g_score
                else:
                    d_shift = g_score

            else:  # FINANCIALS, CRYPTO, CLIMATE
                hd_scores.append(fb.net_score * w)
                if any(b in hint_low for b in ["below", "under", "<"]):
                    d_shift = -fb.net_score
                else:
                    d_shift = fb.net_score

            d_shift = float(np.clip(d_shift, -1.0, 1.0))
            dir_scores.append(d_shift * w)
            article_details.append({
                **fb.to_dict(),
                "directional_shift": round(d_shift, 4),
            })

        w_sum = max(1e-6, float(sum(weights)))
        raw_mean = float(np.clip(sum(raw_scores) / w_sum, -1.0, 1.0))
        hd_mean = float(np.clip(sum(hd_scores) / w_sum, -1.0, 1.0))
        dir_mean = float(np.clip(sum(dir_scores) / w_sum, -1.0, 1.0))

        return {
            "raw_finbert_mean": round(raw_mean, 4),
            "directional_shift_score": round(dir_mean, 4),
            "hawkish_dovish_index": round(hd_mean, 4),
            "article_scores": article_details,
            "backend": fb_results[0].backend if fb_results else self._backend_name,
        }


finbert_engine = FinBERTEngine()
