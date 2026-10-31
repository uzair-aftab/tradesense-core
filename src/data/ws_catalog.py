"""
TradeSense Wealthsimple Universal Asset Catalog & Exact Fee Schedule Engine
=============================================================================
Provides comprehensive coverage of Wealthsimple-tradeable assets in Ontario, Canada:
  - 150+ Curated Instruments (TSX Stocks, Canadian ETFs, NEO CDRs, US Equities, Crypto CAD)
  - Live Yahoo Finance Search / Discovery for any eligible TSX/US/NEO/Crypto ticker
  - Accurate Wealthsimple Fee Schedule (Core, Premium, Generation, USD Accounts, and CDR proxies)
"""

from __future__ import annotations

import time
import urllib.parse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import requests

from src.utils.logger import get_logger

logger = get_logger("hfp.ws_catalog")


@dataclass(frozen=True)
class WealthsimpleAsset:
    """Represents a tradeable asset on Wealthsimple Trade / Crypto."""
    symbol: str                 # DataLake / Yahoo symbol (e.g., "SHOP.TO", "NVDA", "BTC-CAD", "NVDA.NE")
    ws_display_symbol: str      # Symbol shown inside Wealthsimple mobile app (e.g., "SHOP", "NVDA", "BTC", "NVDA.NE")
    name: str                   # Full human-readable name (e.g., "Shopify Inc.", "Solana", "NVIDIA CDR (CAD Hedged)")
    category: str               # "TSX_STOCK" | "CANADIAN_ETF" | "NEO_CDR" | "US_STOCK" | "US_ETF" | "CRYPTO"
    currency: str               # "CAD" | "USD"
    exchange: str               # "TSX" | "TSXV" | "NEO" | "NASDAQ" | "NYSE" | "WS_CRYPTO"
    is_24_7: bool               # True for CRYPTO, False for equities/ETFs/CDRs
    fractional_supported: bool  # True on Wealthsimple for major TSX/US/CDR/Crypto
    alt_cad_proxy: Optional[str] = None  # e.g., "NVDA.NE" for "NVDA", "BTCC.TO" for "BTC-CAD", "VFV.TO" for "SPY"


@dataclass
class FeeEstimate:
    """Calculated Wealthsimple execution fee and friction estimate."""
    one_way_fee_pct: float        # Commission + FX or Crypto fee per side (%)
    one_way_slippage_pct: float   # Realistic market bid-ask half-spread (%)
    round_trip_total_pct: float   # Total round-trip hurdle (%) a trade must beat to break even
    fee_cad_on_notional: float    # Dollar fee on given CAD notional (round-trip)
    badge_text: str               # Short UI badge (e.g., "$0 Comm • ~0.08% Spread")
    optimization_tip: Optional[str] = None  # Actionable fee saving suggestion


# ─────────────────────────────────────────────────────────────────────────────
# 150+ Curated Wealthsimple Assets Registry
# ─────────────────────────────────────────────────────────────────────────────
CURATED_WS_ASSETS: Dict[str, WealthsimpleAsset] = {
    # ─── TSX & TSX-V STOCKS (~45+) ──────────────────────────────────────────
    "RY.TO": WealthsimpleAsset("RY.TO", "RY", "Royal Bank of Canada", "TSX_STOCK", "CAD", "TSX", False, True),
    "TD.TO": WealthsimpleAsset("TD.TO", "TD", "Toronto-Dominion Bank", "TSX_STOCK", "CAD", "TSX", False, True),
    "BMO.TO": WealthsimpleAsset("BMO.TO", "BMO", "Bank of Montreal", "TSX_STOCK", "CAD", "TSX", False, True),
    "BNS.TO": WealthsimpleAsset("BNS.TO", "BNS", "Bank of Nova Scotia", "TSX_STOCK", "CAD", "TSX", False, True),
    "CM.TO": WealthsimpleAsset("CM.TO", "CM", "Canadian Imperial Bank of Commerce", "TSX_STOCK", "CAD", "TSX", False, True),
    "NA.TO": WealthsimpleAsset("NA.TO", "NA", "National Bank of Canada", "TSX_STOCK", "CAD", "TSX", False, True),
    "SHOP.TO": WealthsimpleAsset("SHOP.TO", "SHOP", "Shopify Inc.", "TSX_STOCK", "CAD", "TSX", False, True),
    "ENB.TO": WealthsimpleAsset("ENB.TO", "ENB", "Enbridge Inc.", "TSX_STOCK", "CAD", "TSX", False, True),
    "CNQ.TO": WealthsimpleAsset("CNQ.TO", "CNQ", "Canadian Natural Resources", "TSX_STOCK", "CAD", "TSX", False, True),
    "SU.TO": WealthsimpleAsset("SU.TO", "SU", "Suncor Energy Inc.", "TSX_STOCK", "CAD", "TSX", False, True),
    "TRP.TO": WealthsimpleAsset("TRP.TO", "TRP", "TC Energy Corp.", "TSX_STOCK", "CAD", "TSX", False, True),
    "CP.TO": WealthsimpleAsset("CP.TO", "CP", "Canadian Pacific Kansas City", "TSX_STOCK", "CAD", "TSX", False, True),
    "CNR.TO": WealthsimpleAsset("CNR.TO", "CNR", "Canadian National Railway", "TSX_STOCK", "CAD", "TSX", False, True),
    "CSU.TO": WealthsimpleAsset("CSU.TO", "CSU", "Constellation Software Inc.", "TSX_STOCK", "CAD", "TSX", False, True),
    "DOL.TO": WealthsimpleAsset("DOL.TO", "DOL", "Dollarama Inc.", "TSX_STOCK", "CAD", "TSX", False, True),
    "ATD.TO": WealthsimpleAsset("ATD.TO", "ATD", "Alimentation Couche-Tard Inc.", "TSX_STOCK", "CAD", "TSX", False, True),
    "L.TO": WealthsimpleAsset("L.TO", "L", "Loblaw Companies Ltd.", "TSX_STOCK", "CAD", "TSX", False, True),
    "WCN.TO": WealthsimpleAsset("WCN.TO", "WCN", "Waste Connections Inc.", "TSX_STOCK", "CAD", "TSX", False, True),
    "BN.TO": WealthsimpleAsset("BN.TO", "BN", "Brookfield Corporation", "TSX_STOCK", "CAD", "TSX", False, True),
    "BAM.TO": WealthsimpleAsset("BAM.TO", "BAM", "Brookfield Asset Management", "TSX_STOCK", "CAD", "TSX", False, True),
    "MFC.TO": WealthsimpleAsset("MFC.TO", "MFC", "Manulife Financial Corp.", "TSX_STOCK", "CAD", "TSX", False, True),
    "SLF.TO": WealthsimpleAsset("SLF.TO", "SLF", "Sun Life Financial Inc.", "TSX_STOCK", "CAD", "TSX", False, True),
    "IFC.TO": WealthsimpleAsset("IFC.TO", "IFC", "Intact Financial Corp.", "TSX_STOCK", "CAD", "TSX", False, True),
    "FFH.TO": WealthsimpleAsset("FFH.TO", "FFH", "Fairfax Financial Holdings", "TSX_STOCK", "CAD", "TSX", False, True),
    "BCE.TO": WealthsimpleAsset("BCE.TO", "BCE", "BCE Inc.", "TSX_STOCK", "CAD", "TSX", False, True),
    "T.TO": WealthsimpleAsset("T.TO", "T", "Telus Corp.", "TSX_STOCK", "CAD", "TSX", False, True),
    "RCI-B.TO": WealthsimpleAsset("RCI-B.TO", "RCI.B", "Rogers Communications B", "TSX_STOCK", "CAD", "TSX", False, True),
    "ABX.TO": WealthsimpleAsset("ABX.TO", "ABX", "Barrick Gold Corp.", "TSX_STOCK", "CAD", "TSX", False, True),
    "AEM.TO": WealthsimpleAsset("AEM.TO", "AEM", "Agnico Eagle Mines Ltd.", "TSX_STOCK", "CAD", "TSX", False, True),
    "WPM.TO": WealthsimpleAsset("WPM.TO", "WPM", "Wheaton Precious Metals", "TSX_STOCK", "CAD", "TSX", False, True),
    "CCO.TO": WealthsimpleAsset("CCO.TO", "CCO", "Cameco Corporation", "TSX_STOCK", "CAD", "TSX", False, True),
    "NTR.TO": WealthsimpleAsset("NTR.TO", "NTR", "Nutrien Ltd.", "TSX_STOCK", "CAD", "TSX", False, True),
    "TECK-B.TO": WealthsimpleAsset("TECK-B.TO", "TECK.B", "Teck Resources Ltd. B", "TSX_STOCK", "CAD", "TSX", False, True),
    "CLS.TO": WealthsimpleAsset("CLS.TO", "CLS", "Celestica Inc.", "TSX_STOCK", "CAD", "TSX", False, True),
    "GIB-A.TO": WealthsimpleAsset("GIB-A.TO", "GIB.A", "CGI Inc.", "TSX_STOCK", "CAD", "TSX", False, True),
    "OTEX.TO": WealthsimpleAsset("OTEX.TO", "OTEX", "Open Text Corporation", "TSX_STOCK", "CAD", "TSX", False, True),
    "LSPD.TO": WealthsimpleAsset("LSPD.TO", "LSPD", "Lightspeed Commerce Inc.", "TSX_STOCK", "CAD", "TSX", False, True),
    "AC.TO": WealthsimpleAsset("AC.TO", "AC", "Air Canada", "TSX_STOCK", "CAD", "TSX", False, True),
    "FTS.TO": WealthsimpleAsset("FTS.TO", "FTS", "Fortis Inc.", "TSX_STOCK", "CAD", "TSX", False, True),
    "EMA.TO": WealthsimpleAsset("EMA.TO", "EMA", "Emera Inc.", "TSX_STOCK", "CAD", "TSX", False, True),
    "H.TO": WealthsimpleAsset("H.TO", "H", "Hydro One Ltd.", "TSX_STOCK", "CAD", "TSX", False, True),
    "HUT.TO": WealthsimpleAsset("HUT.TO", "HUT", "Hut 8 Corp.", "TSX_STOCK", "CAD", "TSX", False, True),
    "BITF.TO": WealthsimpleAsset("BITF.TO", "BITF", "Bitfarms Ltd.", "TSX_STOCK", "CAD", "TSX", False, True),
    "HIVE.TO": WealthsimpleAsset("HIVE.TO", "HIVE", "HIVE Digital Technologies", "TSX_STOCK", "CAD", "TSX", False, True),
    "GLXY.TO": WealthsimpleAsset("GLXY.TO", "GLXY", "Galaxy Digital Holdings", "TSX_STOCK", "CAD", "TSX", False, True),

    # ─── CANADIAN ETFS (~25+) ───────────────────────────────────────────────
    "VFV.TO": WealthsimpleAsset("VFV.TO", "VFV", "Vanguard S&P 500 Index ETF (CAD)", "CANADIAN_ETF", "CAD", "TSX", False, True),
    "ZSP.TO": WealthsimpleAsset("ZSP.TO", "ZSP", "BMO S&P 500 Index ETF", "CANADIAN_ETF", "CAD", "TSX", False, True),
    "XEQT.TO": WealthsimpleAsset("XEQT.TO", "XEQT", "iShares Core Equity ETF Portfolio", "CANADIAN_ETF", "CAD", "TSX", False, True),
    "VEQT.TO": WealthsimpleAsset("VEQT.TO", "VEQT", "Vanguard All-Equity ETF", "CANADIAN_ETF", "CAD", "TSX", False, True),
    "XGRO.TO": WealthsimpleAsset("XGRO.TO", "XGRO", "iShares Core Growth ETF Portfolio", "CANADIAN_ETF", "CAD", "TSX", False, True),
    "VGRO.TO": WealthsimpleAsset("VGRO.TO", "VGRO", "Vanguard Growth ETF Portfolio", "CANADIAN_ETF", "CAD", "TSX", False, True),
    "XIU.TO": WealthsimpleAsset("XIU.TO", "XIU", "iShares S&P/TSX 60 Index ETF", "CANADIAN_ETF", "CAD", "TSX", False, True),
    "XIC.TO": WealthsimpleAsset("XIC.TO", "XIC", "iShares Core S&P/TSX Capped ETF", "CANADIAN_ETF", "CAD", "TSX", False, True),
    "VCN.TO": WealthsimpleAsset("VCN.TO", "VCN", "Vanguard FTSE Canada All Cap ETF", "CANADIAN_ETF", "CAD", "TSX", False, True),
    "XQQ.TO": WealthsimpleAsset("XQQ.TO", "XQQ", "iShares NASDAQ 100 Index ETF (CAD-Hedged)", "CANADIAN_ETF", "CAD", "TSX", False, True),
    "HXQ.TO": WealthsimpleAsset("HXQ.TO", "HXQ", "Global X NASDAQ-100 Index ETF", "CANADIAN_ETF", "CAD", "TSX", False, True),
    "TEC.TO": WealthsimpleAsset("TEC.TO", "TEC", "TD Global Technology Leaders ETF", "CANADIAN_ETF", "CAD", "TSX", False, True),
    "VDY.TO": WealthsimpleAsset("VDY.TO", "VDY", "Vanguard Canadian High Dividend Yield ETF", "CANADIAN_ETF", "CAD", "TSX", False, True),
    "XEI.TO": WealthsimpleAsset("XEI.TO", "XEI", "iShares S&P/TSX Composite High Dividend", "CANADIAN_ETF", "CAD", "TSX", False, True),
    "ZEB.TO": WealthsimpleAsset("ZEB.TO", "ZEB", "BMO Equal Weight Banks Index ETF", "CANADIAN_ETF", "CAD", "TSX", False, True),
    "CASH.TO": WealthsimpleAsset("CASH.TO", "CASH", "Global X High Interest Savings ETF", "CANADIAN_ETF", "CAD", "TSX", False, True),
    "CBIL.TO": WealthsimpleAsset("CBIL.TO", "CBIL", "Global X 0-3 Month T-Bill ETF", "CANADIAN_ETF", "CAD", "TSX", False, True),
    "PSA.TO": WealthsimpleAsset("PSA.TO", "PSA", "Purpose High Interest Savings Fund", "CANADIAN_ETF", "CAD", "TSX", False, True),
    "XBB.TO": WealthsimpleAsset("XBB.TO", "XBB", "iShares Core Canadian Universe Bond", "CANADIAN_ETF", "CAD", "TSX", False, True),
    "BTCC.TO": WealthsimpleAsset("BTCC.TO", "BTCC.B", "Purpose Bitcoin ETF CAD", "CANADIAN_ETF", "CAD", "TSX", False, True),
    "BTCX-B.TO": WealthsimpleAsset("BTCX-B.TO", "BTCX.B", "CI Galaxy Bitcoin ETF (CAD)", "CANADIAN_ETF", "CAD", "TSX", False, True),
    "ETHX-B.TO": WealthsimpleAsset("ETHX-B.TO", "ETHX.B", "CI Galaxy Ethereum ETF (CAD)", "CANADIAN_ETF", "CAD", "TSX", False, True),
    "SOLQ.TO": WealthsimpleAsset("SOLQ.TO", "SOLQ", "3iQ Solana Staking ETF (CAD)", "CANADIAN_ETF", "CAD", "TSX", False, True),
    "XGD.TO": WealthsimpleAsset("XGD.TO", "XGD", "iShares S&P/TSX Global Gold Index ETF", "CANADIAN_ETF", "CAD", "TSX", False, True),
    "XEG.TO": WealthsimpleAsset("XEG.TO", "XEG", "iShares S&P/TSX Capped Energy Index ETF", "CANADIAN_ETF", "CAD", "TSX", False, True),

    # ─── NEO / CBOE CAD-HEDGED CDRs (~25+) ──────────────────────────────────
    "NVDA.NE": WealthsimpleAsset("NVDA.NE", "NVDA.NE", "NVIDIA CDR (CAD Hedged)", "NEO_CDR", "CAD", "NEO", False, True),
    "AAPL.NE": WealthsimpleAsset("AAPL.NE", "AAPL.NE", "Apple CDR (CAD Hedged)", "NEO_CDR", "CAD", "NEO", False, True),
    "MSFT.NE": WealthsimpleAsset("MSFT.NE", "MSFT.NE", "Microsoft CDR (CAD Hedged)", "NEO_CDR", "CAD", "NEO", False, True),
    "TSLA.NE": WealthsimpleAsset("TSLA.NE", "TSLA.NE", "Tesla CDR (CAD Hedged)", "NEO_CDR", "CAD", "NEO", False, True),
    "AMZN.NE": WealthsimpleAsset("AMZN.NE", "AMZN.NE", "Amazon CDR (CAD Hedged)", "NEO_CDR", "CAD", "NEO", False, True),
    "GOOG.NE": WealthsimpleAsset("GOOG.NE", "GOOG.NE", "Alphabet CDR (CAD Hedged)", "NEO_CDR", "CAD", "NEO", False, True),
    "META.NE": WealthsimpleAsset("META.NE", "META.NE", "Meta Platforms CDR (CAD Hedged)", "NEO_CDR", "CAD", "NEO", False, True),
    "AMD.NE": WealthsimpleAsset("AMD.NE", "AMD.NE", "Advanced Micro Devices CDR (CAD)", "NEO_CDR", "CAD", "NEO", False, True),
    "NFLX.NE": WealthsimpleAsset("NFLX.NE", "NFLX.NE", "Netflix CDR (CAD Hedged)", "NEO_CDR", "CAD", "NEO", False, True),
    "COST.NE": WealthsimpleAsset("COST.NE", "COST.NE", "Costco CDR (CAD Hedged)", "NEO_CDR", "CAD", "NEO", False, True),
    "AVGO.NE": WealthsimpleAsset("AVGO.NE", "AVGO.NE", "Broadcom CDR (CAD Hedged)", "NEO_CDR", "CAD", "NEO", False, True),
    "PLTR.NE": WealthsimpleAsset("PLTR.NE", "PLTR.NE", "Palantir CDR (CAD Hedged)", "NEO_CDR", "CAD", "NEO", False, True),
    "UBER.NE": WealthsimpleAsset("UBER.NE", "UBER.NE", "Uber CDR (CAD Hedged)", "NEO_CDR", "CAD", "NEO", False, True),
    "DIS.NE": WealthsimpleAsset("DIS.NE", "DIS.NE", "Walt Disney CDR (CAD Hedged)", "NEO_CDR", "CAD", "NEO", False, True),
    "JPM.NE": WealthsimpleAsset("JPM.NE", "JPM.NE", "JPMorgan Chase CDR (CAD Hedged)", "NEO_CDR", "CAD", "NEO", False, True),
    "V.NE": WealthsimpleAsset("V.NE", "V.NE", "Visa CDR (CAD Hedged)", "NEO_CDR", "CAD", "NEO", False, True),
    "MA.NE": WealthsimpleAsset("MA.NE", "MA.NE", "Mastercard CDR (CAD Hedged)", "NEO_CDR", "CAD", "NEO", False, True),
    "BRK.NE": WealthsimpleAsset("BRK.NE", "BRK.NE", "Berkshire Hathaway CDR (CAD)", "NEO_CDR", "CAD", "NEO", False, True),
    "LLY.NE": WealthsimpleAsset("LLY.NE", "LLY.NE", "Eli Lilly CDR (CAD Hedged)", "NEO_CDR", "CAD", "NEO", False, True),
    "UNH.NE": WealthsimpleAsset("UNH.NE", "UNH.NE", "UnitedHealth Group CDR (CAD)", "NEO_CDR", "CAD", "NEO", False, True),
    "KO.NE": WealthsimpleAsset("KO.NE", "KO.NE", "Coca-Cola CDR (CAD Hedged)", "NEO_CDR", "CAD", "NEO", False, True),
    "WMT.NE": WealthsimpleAsset("WMT.NE", "WMT.NE", "Walmart CDR (CAD Hedged)", "NEO_CDR", "CAD", "NEO", False, True),
    "CRM.NE": WealthsimpleAsset("CRM.NE", "CRM.NE", "Salesforce CDR (CAD Hedged)", "NEO_CDR", "CAD", "NEO", False, True),
    "PYPL.NE": WealthsimpleAsset("PYPL.NE", "PYPL.NE", "PayPal CDR (CAD Hedged)", "NEO_CDR", "CAD", "NEO", False, True),
    "COIN.NE": WealthsimpleAsset("COIN.NE", "COIN.NE", "Coinbase CDR (CAD Hedged)", "NEO_CDR", "CAD", "NEO", False, True),

    # ─── US STOCKS & ETFS (~40+) ─────────────────────────────────────────────
    "NVDA": WealthsimpleAsset("NVDA", "NVDA", "NVIDIA Corporation", "US_STOCK", "USD", "NASDAQ", False, True, alt_cad_proxy="NVDA.NE"),
    "AAPL": WealthsimpleAsset("AAPL", "AAPL", "Apple Inc.", "US_STOCK", "USD", "NASDAQ", False, True, alt_cad_proxy="AAPL.NE"),
    "MSFT": WealthsimpleAsset("MSFT", "MSFT", "Microsoft Corporation", "US_STOCK", "USD", "NASDAQ", False, True, alt_cad_proxy="MSFT.NE"),
    "AMZN": WealthsimpleAsset("AMZN", "AMZN", "Amazon.com Inc.", "US_STOCK", "USD", "NASDAQ", False, True, alt_cad_proxy="AMZN.NE"),
    "GOOGL": WealthsimpleAsset("GOOGL", "GOOGL", "Alphabet Inc. (Class A)", "US_STOCK", "USD", "NASDAQ", False, True, alt_cad_proxy="GOOG.NE"),
    "META": WealthsimpleAsset("META", "META", "Meta Platforms Inc.", "US_STOCK", "USD", "NASDAQ", False, True, alt_cad_proxy="META.NE"),
    "TSLA": WealthsimpleAsset("TSLA", "TSLA", "Tesla Inc.", "US_STOCK", "USD", "NASDAQ", False, True, alt_cad_proxy="TSLA.NE"),
    "AMD": WealthsimpleAsset("AMD", "AMD", "Advanced Micro Devices Inc.", "US_STOCK", "USD", "NASDAQ", False, True, alt_cad_proxy="AMD.NE"),
    "PLTR": WealthsimpleAsset("PLTR", "PLTR", "Palantir Technologies", "US_STOCK", "USD", "NYSE", False, True, alt_cad_proxy="PLTR.NE"),
    "COIN": WealthsimpleAsset("COIN", "COIN", "Coinbase Global Inc.", "US_STOCK", "USD", "NASDAQ", False, True, alt_cad_proxy="COIN.NE"),
    "MSTR": WealthsimpleAsset("MSTR", "MSTR", "MicroStrategy Inc.", "US_STOCK", "USD", "NASDAQ", False, True),
    "AVGO": WealthsimpleAsset("AVGO", "AVGO", "Broadcom Inc.", "US_STOCK", "USD", "NASDAQ", False, True, alt_cad_proxy="AVGO.NE"),
    "NFLX": WealthsimpleAsset("NFLX", "NFLX", "Netflix Inc.", "US_STOCK", "USD", "NASDAQ", False, True, alt_cad_proxy="NFLX.NE"),
    "LLY": WealthsimpleAsset("LLY", "LLY", "Eli Lilly and Company", "US_STOCK", "USD", "NYSE", False, True, alt_cad_proxy="LLY.NE"),
    "JPM": WealthsimpleAsset("JPM", "JPM", "JPMorgan Chase & Co.", "US_STOCK", "USD", "NYSE", False, True, alt_cad_proxy="JPM.NE"),
    "V": WealthsimpleAsset("V", "V", "Visa Inc.", "US_STOCK", "USD", "NYSE", False, True, alt_cad_proxy="V.NE"),
    "MA": WealthsimpleAsset("MA", "MA", "Mastercard Inc.", "US_STOCK", "USD", "NYSE", False, True, alt_cad_proxy="MA.NE"),
    "COST": WealthsimpleAsset("COST", "COST", "Costco Wholesale Corp.", "US_STOCK", "USD", "NASDAQ", False, True, alt_cad_proxy="COST.NE"),
    "WMT": WealthsimpleAsset("WMT", "WMT", "Walmart Inc.", "US_STOCK", "USD", "NYSE", False, True, alt_cad_proxy="WMT.NE"),
    "CRM": WealthsimpleAsset("CRM", "CRM", "Salesforce Inc.", "US_STOCK", "USD", "NYSE", False, True, alt_cad_proxy="CRM.NE"),
    "UBER": WealthsimpleAsset("UBER", "UBER", "Uber Technologies Inc.", "US_STOCK", "USD", "NYSE", False, True, alt_cad_proxy="UBER.NE"),
    "HOOD": WealthsimpleAsset("HOOD", "HOOD", "Robinhood Markets Inc.", "US_STOCK", "USD", "NASDAQ", False, True),
    "SOFI": WealthsimpleAsset("SOFI", "SOFI", "SoFi Technologies Inc.", "US_STOCK", "USD", "NASDAQ", False, True),
    "ARM": WealthsimpleAsset("ARM", "ARM", "Arm Holdings plc", "US_STOCK", "USD", "NASDAQ", False, True),
    "SMCI": WealthsimpleAsset("SMCI", "SMCI", "Super Micro Computer Inc.", "US_STOCK", "USD", "NASDAQ", False, True),
    "MU": WealthsimpleAsset("MU", "MU", "Micron Technology Inc.", "US_STOCK", "USD", "NASDAQ", False, True),
    "PANW": WealthsimpleAsset("PANW", "PANW", "Palo Alto Networks", "US_STOCK", "USD", "NASDAQ", False, True),
    "CRWD": WealthsimpleAsset("CRWD", "CRWD", "CrowdStrike Holdings Inc.", "US_STOCK", "USD", "NASDAQ", False, True),
    "SNOW": WealthsimpleAsset("SNOW", "SNOW", "Snowflake Inc.", "US_STOCK", "USD", "NYSE", False, True),
    "TSM": WealthsimpleAsset("TSM", "TSM", "Taiwan Semiconductor ADR", "US_STOCK", "USD", "NYSE", False, True),
    "ASML": WealthsimpleAsset("ASML", "ASML", "ASML Holding NV", "US_STOCK", "USD", "NASDAQ", False, True),
    "RDDT": WealthsimpleAsset("RDDT", "RDDT", "Reddit Inc.", "US_STOCK", "USD", "NYSE", False, True),
    "RKLB": WealthsimpleAsset("RKLB", "RKLB", "Rocket Lab USA Inc.", "US_STOCK", "USD", "NASDAQ", False, True),
    "ASTS": WealthsimpleAsset("ASTS", "ASTS", "AST SpaceMobile Inc.", "US_STOCK", "USD", "NASDAQ", False, True),
    "IONQ": WealthsimpleAsset("IONQ", "IONQ", "IonQ Inc.", "US_STOCK", "USD", "NYSE", False, True),
    "SPY": WealthsimpleAsset("SPY", "SPY", "SPDR S&P 500 ETF Trust", "US_ETF", "USD", "NYSE", False, True, alt_cad_proxy="VFV.TO"),
    "QQQ": WealthsimpleAsset("QQQ", "QQQ", "Invesco QQQ Trust", "US_ETF", "USD", "NASDAQ", False, True, alt_cad_proxy="XQQ.TO"),
    "IWM": WealthsimpleAsset("IWM", "IWM", "iShares Russell 2000 ETF", "US_ETF", "USD", "NYSE", False, True),
    "SOXL": WealthsimpleAsset("SOXL", "SOXL", "Direxion Daily Semi Bull 3X", "US_ETF", "USD", "NYSE", False, True),
    "TQQQ": WealthsimpleAsset("TQQQ", "TQQQ", "ProShares UltraPro QQQ 3X", "US_ETF", "USD", "NASDAQ", False, True),
    "SMH": WealthsimpleAsset("SMH", "SMH", "VanEck Semiconductor ETF", "US_ETF", "USD", "NASDAQ", False, True),
    "GLD": WealthsimpleAsset("GLD", "GLD", "SPDR Gold Shares", "US_ETF", "USD", "NYSE", False, True, alt_cad_proxy="XGD.TO"),
    "TLT": WealthsimpleAsset("TLT", "TLT", "iShares 20+ Year Treasury Bond", "US_ETF", "USD", "NASDAQ", False, True),
    "IBIT": WealthsimpleAsset("IBIT", "IBIT", "iShares Bitcoin Trust ETF", "US_ETF", "USD", "NASDAQ", False, True, alt_cad_proxy="BTCC.TO"),

    # ─── WEALTHSIMPLE CRYPTO CAD PAIRS (~35+) ──────────────────────────────
    "BTC-CAD": WealthsimpleAsset("BTC-CAD", "BTC", "Bitcoin (CAD)", "CRYPTO", "CAD", "WS_CRYPTO", True, True, alt_cad_proxy="BTCC.TO"),
    "ETH-CAD": WealthsimpleAsset("ETH-CAD", "ETH", "Ethereum (CAD)", "CRYPTO", "CAD", "WS_CRYPTO", True, True, alt_cad_proxy="ETHX-B.TO"),
    "SOL-CAD": WealthsimpleAsset("SOL-CAD", "SOL", "Solana (CAD)", "CRYPTO", "CAD", "WS_CRYPTO", True, True, alt_cad_proxy="SOLQ.TO"),
    "XRP-CAD": WealthsimpleAsset("XRP-CAD", "XRP", "XRP (CAD)", "CRYPTO", "CAD", "WS_CRYPTO", True, True),
    "DOGE-CAD": WealthsimpleAsset("DOGE-CAD", "DOGE", "Dogecoin (CAD)", "CRYPTO", "CAD", "WS_CRYPTO", True, True),
    "ADA-CAD": WealthsimpleAsset("ADA-CAD", "ADA", "Cardano (CAD)", "CRYPTO", "CAD", "WS_CRYPTO", True, True),
    "AVAX-CAD": WealthsimpleAsset("AVAX-CAD", "AVAX", "Avalanche (CAD)", "CRYPTO", "CAD", "WS_CRYPTO", True, True),
    "LINK-CAD": WealthsimpleAsset("LINK-CAD", "LINK", "Chainlink (CAD)", "CRYPTO", "CAD", "WS_CRYPTO", True, True),
    "DOT-CAD": WealthsimpleAsset("DOT-CAD", "DOT", "Polkadot (CAD)", "CRYPTO", "CAD", "WS_CRYPTO", True, True),
    "SUI-CAD": WealthsimpleAsset("SUI-CAD", "SUI", "Sui (CAD)", "CRYPTO", "CAD", "WS_CRYPTO", True, True),
    "LTC-CAD": WealthsimpleAsset("LTC-CAD", "LTC", "Litecoin (CAD)", "CRYPTO", "CAD", "WS_CRYPTO", True, True),
    "BCH-CAD": WealthsimpleAsset("BCH-CAD", "BCH", "Bitcoin Cash (CAD)", "CRYPTO", "CAD", "WS_CRYPTO", True, True),
    "NEAR-CAD": WealthsimpleAsset("NEAR-CAD", "NEAR", "NEAR Protocol (CAD)", "CRYPTO", "CAD", "WS_CRYPTO", True, True),
    "UNI-CAD": WealthsimpleAsset("UNI-CAD", "UNI", "Uniswap (CAD)", "CRYPTO", "CAD", "WS_CRYPTO", True, True),
    "APT-CAD": WealthsimpleAsset("APT-CAD", "APT", "Aptos (CAD)", "CRYPTO", "CAD", "WS_CRYPTO", True, True),
    "PEPE-CAD": WealthsimpleAsset("PEPE-CAD", "PEPE", "Pepe (CAD)", "CRYPTO", "CAD", "WS_CRYPTO", True, True),
    "SHIB-CAD": WealthsimpleAsset("SHIB-CAD", "SHIB", "Shiba Inu (CAD)", "CRYPTO", "CAD", "WS_CRYPTO", True, True),
    "RENDER-CAD": WealthsimpleAsset("RENDER-CAD", "RENDER", "Render (CAD)", "CRYPTO", "CAD", "WS_CRYPTO", True, True),
    "FET-CAD": WealthsimpleAsset("FET-CAD", "FET", "Artificial Superintelligence Alliance", "CRYPTO", "CAD", "WS_CRYPTO", True, True),
    "ARB-CAD": WealthsimpleAsset("ARB-CAD", "ARB", "Arbitrum (CAD)", "CRYPTO", "CAD", "WS_CRYPTO", True, True),
    "OP-CAD": WealthsimpleAsset("OP-CAD", "OP", "Optimism (CAD)", "CRYPTO", "CAD", "WS_CRYPTO", True, True),
    "ATOM-CAD": WealthsimpleAsset("ATOM-CAD", "ATOM", "Cosmos (CAD)", "CRYPTO", "CAD", "WS_CRYPTO", True, True),
    "FIL-CAD": WealthsimpleAsset("FIL-CAD", "FIL", "Filecoin (CAD)", "CRYPTO", "CAD", "WS_CRYPTO", True, True),
    "INJ-CAD": WealthsimpleAsset("INJ-CAD", "INJ", "Injective (CAD)", "CRYPTO", "CAD", "WS_CRYPTO", True, True),
    "HBAR-CAD": WealthsimpleAsset("HBAR-CAD", "HBAR", "Hedera (CAD)", "CRYPTO", "CAD", "WS_CRYPTO", True, True),
    "XLM-CAD": WealthsimpleAsset("XLM-CAD", "XLM", "Stellar (CAD)", "CRYPTO", "CAD", "WS_CRYPTO", True, True),
    "ALGO-CAD": WealthsimpleAsset("ALGO-CAD", "ALGO", "Algorand (CAD)", "CRYPTO", "CAD", "WS_CRYPTO", True, True),
    "AAVE-CAD": WealthsimpleAsset("AAVE-CAD", "AAVE", "Aave (CAD)", "CRYPTO", "CAD", "WS_CRYPTO", True, True),
    "BONK-CAD": WealthsimpleAsset("BONK-CAD", "BONK", "Bonk (CAD)", "CRYPTO", "CAD", "WS_CRYPTO", True, True),
    "WIF-CAD": WealthsimpleAsset("WIF-CAD", "WIF", "dogwifhat (CAD)", "CRYPTO", "CAD", "WS_CRYPTO", True, True),
    "SEI-CAD": WealthsimpleAsset("SEI-CAD", "SEI", "Sei (CAD)", "CRYPTO", "CAD", "WS_CRYPTO", True, True),
    "TIA-CAD": WealthsimpleAsset("TIA-CAD", "TIA", "Celestia (CAD)", "CRYPTO", "CAD", "WS_CRYPTO", True, True),
    "JUP-CAD": WealthsimpleAsset("JUP-CAD", "JUP", "Jupiter (CAD)", "CRYPTO", "CAD", "WS_CRYPTO", True, True),
    "ONDO-CAD": WealthsimpleAsset("ONDO-CAD", "ONDO", "Ondo (CAD)", "CRYPTO", "CAD", "WS_CRYPTO", True, True),
}


class WealthsimpleCatalog:
    """Manages searching, caching, and resolving Wealthsimple assets."""

    _search_cache: Dict[str, tuple[float, List[WealthsimpleAsset]]] = {}

    @classmethod
    def get_asset(cls, symbol: str) -> Optional[WealthsimpleAsset]:
        """Lookup an asset from curated registry."""
        norm = symbol.strip().upper()
        # Direct hit
        if norm in CURATED_WS_ASSETS:
            return CURATED_WS_ASSETS[norm]
        # Check display symbol or USD version
        if norm + ".TO" in CURATED_WS_ASSETS:
            return CURATED_WS_ASSETS[norm + ".TO"]
        if norm + "-CAD" in CURATED_WS_ASSETS:
            return CURATED_WS_ASSETS[norm + "-CAD"]
        return None

    @classmethod
    def resolve_asset(cls, symbol: str) -> WealthsimpleAsset:
        """
        Resolve any ticker (curated or custom) into a valid WealthsimpleAsset instance.
        """
        hit = cls.get_asset(symbol)
        if hit is not None:
            return hit

        sym = symbol.strip().upper()
        if sym.endswith(".TO") or sym.endswith(".V"):
            cat = "TSX_STOCK"
            curr = "CAD"
            exch = "TSX" if sym.endswith(".TO") else "TSXV"
            is_crypto = False
        elif sym.endswith(".NE"):
            cat = "NEO_CDR"
            curr = "CAD"
            exch = "NEO"
            is_crypto = False
        elif "-CAD" in sym or "-USD" in sym:
            cat = "CRYPTO"
            curr = "CAD" if "-CAD" in sym else "USD"
            exch = "WS_CRYPTO"
            is_crypto = True
        else:
            cat = "US_STOCK"
            curr = "USD"
            exch = "NASDAQ"
            is_crypto = False

        display = sym.split(".")[0].replace("-CAD", "").replace("-USD", "")
        return WealthsimpleAsset(
            symbol=sym,
            ws_display_symbol=display,
            name=f"{sym} (Wealthsimple)",
            category=cat,
            currency=curr,
            exchange=exch,
            is_24_7=is_crypto,
            fractional_supported=True,
            alt_cad_proxy=None,
        )

    @classmethod
    def list_curated(cls, category: Optional[str] = None) -> List[WealthsimpleAsset]:
        """List curated assets optionally filtered by category."""
        if not category:
            return list(CURATED_WS_ASSETS.values())
        return [a for a in CURATED_WS_ASSETS.values() if a.category == category]

    @classmethod
    def search_assets(cls, query: str, limit: int = 25) -> List[WealthsimpleAsset]:
        """
        Unified search across Curated registry and live Yahoo Finance search API.
        """
        q = query.strip().upper()
        if not q:
            return list(CURATED_WS_ASSETS.values())[:limit]

        # 1. Curated Matches First
        curated_matches: List[WealthsimpleAsset] = []
        for asset in CURATED_WS_ASSETS.values():
            if (
                q == asset.symbol
                or q == asset.ws_display_symbol
                or asset.symbol.startswith(q)
                or asset.ws_display_symbol.startswith(q)
                or q in asset.name.upper()
            ):
                curated_matches.append(asset)
                if len(curated_matches) >= limit:
                    return curated_matches

        if len(curated_matches) >= limit or len(q) < 2:
            return curated_matches

        # 2. Check in-memory Yahoo Cache (TTL = 600s)
        now = time.time()
        if q in cls._search_cache:
            ts, cached_results = cls._search_cache[q]
            if now - ts < 600:
                merged = {a.symbol: a for a in (curated_matches + cached_results)}
                return list(merged.values())[:limit]

        # 3. Live Yahoo Finance Search Query
        live_results: List[WealthsimpleAsset] = []
        try:
            encoded = urllib.parse.quote(q)
            url = f"https://query2.finance.yahoo.com/v1/finance/search?q={encoded}&quotesCount=10"
            resp = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TradeSense/5.0"},
                timeout=2.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                quotes = data.get("quotes", [])
                for item in quotes:
                    raw_sym = item.get("symbol", "").upper()
                    raw_exch = item.get("exchange", "").upper()
                    short_name = item.get("shortname") or item.get("longname") or raw_sym
                    quote_type = item.get("quoteType", "").upper()

                    # Only map Wealthsimple-eligible venues
                    if raw_exch in ("TOR", "TSX") or raw_sym.endswith(".TO"):
                        live_results.append(WealthsimpleAsset(
                            symbol=raw_sym if raw_sym.endswith(".TO") else f"{raw_sym}.TO",
                            ws_display_symbol=raw_sym.split(".")[0],
                            name=short_name,
                            category="CANADIAN_ETF" if quote_type == "ETF" else "TSX_STOCK",
                            currency="CAD",
                            exchange="TSX",
                            is_24_7=False,
                            fractional_supported=True,
                        ))
                    elif raw_exch in ("NEO", "NEOE", "CNQ") or raw_sym.endswith(".NE"):
                        live_results.append(WealthsimpleAsset(
                            symbol=raw_sym if raw_sym.endswith(".NE") else f"{raw_sym}.NE",
                            ws_display_symbol=raw_sym.split(".")[0],
                            name=short_name,
                            category="NEO_CDR",
                            currency="CAD",
                            exchange="NEO",
                            is_24_7=False,
                            fractional_supported=True,
                        ))
                    elif raw_exch in ("NMS", "NYQ", "NGM", "PCX", "ASE", "NASDAQ", "NYSE"):
                        live_results.append(WealthsimpleAsset(
                            symbol=raw_sym,
                            ws_display_symbol=raw_sym,
                            name=short_name,
                            category="US_ETF" if quote_type == "ETF" else "US_STOCK",
                            currency="USD",
                            exchange="NASDAQ" if "N" in raw_exch else "NYSE",
                            is_24_7=False,
                            fractional_supported=True,
                        ))
                    elif quote_type == "CRYPTOCURRENCY" or raw_exch in ("CCC", "CRYPTO"):
                        cad_sym = raw_sym if raw_sym.endswith("-CAD") else f"{raw_sym.replace('-USD', '')}-CAD"
                        disp = cad_sym.replace("-CAD", "")
                        live_results.append(WealthsimpleAsset(
                            symbol=cad_sym,
                            ws_display_symbol=disp,
                            name=f"{short_name} (CAD)",
                            category="CRYPTO",
                            currency="CAD",
                            exchange="WS_CRYPTO",
                            is_24_7=True,
                            fractional_supported=True,
                        ))
                cls._search_cache[q] = (now, live_results)
        except Exception as exc:
            logger.debug("Live Yahoo search failed for %s: %s", q, exc)

        merged = {a.symbol: a for a in (curated_matches + live_results)}
        return list(merged.values())[:limit]


# ─────────────────────────────────────────────────────────────────────────────
# Accurate Wealthsimple Fee & Friction Engine
# ─────────────────────────────────────────────────────────────────────────────
class WealthsimpleFeeSchedule:
    """Computes exact round-trip fees, spreads, and friction on Wealthsimple."""

    @staticmethod
    def calculate_fee(
        asset: WealthsimpleAsset,
        notional_cad: float = 1000.0,
        ws_tier: str = "Core",             # "Core" | "Premium" | "Generation"
        usd_account_active: bool = False,
    ) -> FeeEstimate:
        """
        Compute one-way fee, slippage, round-trip total hurdle %, and notional CAD cost.
        """
        tier = ws_tier.capitalize()
        cat = asset.category

        if cat in ("TSX_STOCK", "CANADIAN_ETF"):
            fee_pct = 0.00
            slip_pct = 0.04
            round_trip = (fee_pct + slip_pct) * 2.0
            fee_cad = notional_cad * (round_trip / 100.0)
            return FeeEstimate(
                one_way_fee_pct=fee_pct,
                one_way_slippage_pct=slip_pct,
                round_trip_total_pct=round_trip,
                fee_cad_on_notional=fee_cad,
                badge_text="$0 Comm • ~0.08% Spread",
                optimization_tip=None,
            )

        if cat == "NEO_CDR":
            fee_pct = 0.00
            slip_pct = 0.08
            round_trip = (fee_pct + slip_pct) * 2.0
            fee_cad = notional_cad * (round_trip / 100.0)
            return FeeEstimate(
                one_way_fee_pct=fee_pct,
                one_way_slippage_pct=slip_pct,
                round_trip_total_pct=round_trip,
                fee_cad_on_notional=fee_cad,
                badge_text="$0 Comm • CAD Hedged (~0.16% Round-Trip)",
                optimization_tip=None,
            )

        if cat in ("US_STOCK", "US_ETF"):
            # Wealthsimple charges 1.5% FX conversion fee per side on Core without USD account
            if not usd_account_active and tier == "Core":
                fee_pct = 1.50
                slip_pct = 0.04
                round_trip = (fee_pct + slip_pct) * 2.0
                fee_cad = notional_cad * (round_trip / 100.0)
                tip = None
                if asset.alt_cad_proxy:
                    tip = f"Tip: Switch to {asset.alt_cad_proxy} (CDR) on Wealthsimple to save the 1.5% US FX fee!"
                return FeeEstimate(
                    one_way_fee_pct=fee_pct,
                    one_way_slippage_pct=slip_pct,
                    round_trip_total_pct=round_trip,
                    fee_cad_on_notional=fee_cad,
                    badge_text=f"Core US: 1.50% FX/side (~{round_trip:.2f}% Round-Trip)",
                    optimization_tip=tip,
                )
            else:
                # USD Account active or Premium/Generation ($10/mo waived or free)
                fee_pct = 0.00
                slip_pct = 0.04
                round_trip = (fee_pct + slip_pct) * 2.0
                fee_cad = notional_cad * (round_trip / 100.0)
                tier_label = "USD Account" if usd_account_active else f"{tier} Tier"
                return FeeEstimate(
                    one_way_fee_pct=fee_pct,
                    one_way_slippage_pct=slip_pct,
                    round_trip_total_pct=round_trip,
                    fee_cad_on_notional=fee_cad,
                    badge_text=f"$0 Comm ({tier_label}) • ~0.08% Spread",
                    optimization_tip=None,
                )

        if cat == "CRYPTO":
            # Wealthsimple Crypto fee tiers: Core = 2.0%, Premium = 1.0%, Generation = 0.5%
            fee_pct = 2.00 if tier == "Core" else (1.00 if tier == "Premium" else 0.50)
            slip_pct = 0.05
            round_trip = (fee_pct + slip_pct) * 2.0
            fee_cad = notional_cad * (round_trip / 100.0)
            tip = None
            if asset.alt_cad_proxy:
                tip = f"Tip: Trade {asset.alt_cad_proxy} spot ETF for $0 commission instead of {fee_pct}% crypto fee!"
            return FeeEstimate(
                one_way_fee_pct=fee_pct,
                one_way_slippage_pct=slip_pct,
                round_trip_total_pct=round_trip,
                fee_cad_on_notional=fee_cad,
                badge_text=f"{tier} Crypto: {fee_pct:.1f}% Fee/Side (~{round_trip:.2f}% RT)",
                optimization_tip=tip,
            )

        # Fallback default
        return FeeEstimate(
            one_way_fee_pct=0.10,
            one_way_slippage_pct=0.05,
            round_trip_total_pct=0.30,
            fee_cad_on_notional=notional_cad * 0.003,
            badge_text="Standard Friction",
            optimization_tip=None,
        )
