"""
OpenOptions v3 - Institutional Options Analytics Platform (NSE/BSE)
-------------------------------------------------------------------
Incremental upgrade of OpenOptions v2. The v2 Analyse screen (positions panel,
payoff/Greeks/P&L tabs, sliders, portfolio-analysis card) is preserved intact.

NEW MODULES (top-level tabs):
  Option Chain  - live multi-expiry chain w/ Greeks, OI, Max Pain, PCR
  Volatility    - HV, IV Rank/Percentile, smile, term structure, cone, surface
  Probability   - expected move, SD bands, P(ITM)/P(touch), Monte Carlo
  Backtest      - historical replay + expiry-cycle backtester w/ full stats
  Market        - VIX, PCR, Max Pain, regime, momentum (provider-abstracted)
  Journal       - SQLite trade journal with Excel/CSV export
  Risk tab      - stress heatmap, sizing, Kelly, SL/target, margin, gap risk

DATA PROVIDERS are abstracted (never hardcoded): jugaad-data -> yfinance ->
CSV, tried in order per capability. Historical option prices in the backtester
are BSM-MODELLED from historical spot + India-VIX-scaled IV (NSE does not
publish free historical chains); this is stated in the UI.

CORE QUANT LOGIC (BSM pricing/Greeks, strategy detection, expiry payoff,
POP, lot handling) is preserved verbatim from v2.
"""
from __future__ import annotations

import io
import json
import logging
import math
import sqlite3
import uuid
import warnings
from datetime import datetime, timedelta, date
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from scipy.stats import norm

# Optional live NSE data (jugaad_data). Chain access via NSELive class.
try:
    from jugaad_data.nse import NSELive
    nse_live = NSELive()
except ImportError:
    NSELive = None
    nse_live = None

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
log = logging.getLogger("openoptions")

# ==============================================================================
# 1. INDIAN MARKET CONFIG
# ==============================================================================
INDEX_YF_MAP: Dict[str, str] = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "NIFTYNXT50": "^NIFTYJR",
    "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
    "MIDCPNIFTY": "NIFTY_MID_SELECT.NS",
    "SENSEX": "^BSESN",
}

# NSE/SEBI revise lot sizes periodically - confirm against the latest circular.
LOT_SIZES: Dict[str, int] = {
    "NIFTY": 65, "BANKNIFTY": 30, "FINNIFTY": 60, "MIDCPNIFTY": 120,
    "NIFTYNXT50": 25, "SENSEX": 20, "RELIANCE": 500, "TCS": 175,
}
DEFAULT_STOCK_LOT_SIZE = 1
INDEX_SET = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"}
STRIKE_STEP = {"NIFTY": 50, "BANKNIFTY": 100, "FINNIFTY": 50,
               "MIDCPNIFTY": 25, "SENSEX": 100}

# Minimal sector map for portfolio sector exposure (extend as needed).
SECTOR_MAP: Dict[str, str] = {
    "NIFTY": "Index", "BANKNIFTY": "Index", "FINNIFTY": "Index",
    "MIDCPNIFTY": "Index", "SENSEX": "Index", "NIFTYNXT50": "Index",
    "RELIANCE": "Energy", "TCS": "IT", "INFY": "IT", "HDFCBANK": "Banks",
    "ICICIBANK": "Banks", "SBIN": "Banks", "ITC": "FMCG", "LT": "Infra",
    "TATAMOTORS": "Auto", "MARUTI": "Auto", "SUNPHARMA": "Pharma",
}

JOURNAL_DB = "openoptions_journal.db"

st.set_page_config(
    page_title="OpenOptions | Institutional Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==============================================================================
# 2. THEME  (v2 theme preserved; chain/heatmap utilities added)
# ==============================================================================
def apply_theme() -> None:
    """Inject the dark Sensibull-style theme. Kept from v2, extended for
    option-chain tables, nav tabs and status chips."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');
    :root {
        --bg:#0b0f17; --panel:#10151f; --panel2:#141b27; --border:#1e2734;
        --txt:#e6edf3; --muted:#7d8896; --green:#21ce99; --red:#ff5b6a;
        --blue:#4da3ff; --amber:#f7b731; --mono:'IBM Plex Mono',monospace;
    }
    html, body, .stApp { background: var(--bg) !important; color: var(--txt);
        font-family:'Inter',sans-serif; }
    .block-container { padding-top:1.1rem; padding-bottom:2rem; max-width:1550px; }
    #MainMenu, footer, header[data-testid="stHeader"] { visibility:hidden; height:0; }

    .oo-card { background:var(--panel); border:1px solid var(--border);
        border-radius:10px; padding:14px 16px; margin-bottom:12px; }
    .oo-title { font-size:1.5rem; font-weight:700; margin:0; }
    .oo-sub { color:var(--muted); font-size:.85rem; margin-top:2px; }
    .oo-ticker { font-family:var(--mono); font-size:1.02rem; font-weight:600; }
    .oo-chg-up { color:var(--green); font-size:.85rem; margin-left:6px; }
    .oo-chg-down { color:var(--red); font-size:.85rem; margin-left:6px; }

    .pos-name { font-weight:600; font-size:.95rem; }
    .pos-meta { color:var(--muted); font-size:.78rem; margin-top:2px; }
    .pos-num { font-family:var(--mono); font-size:.85rem; }
    .badge-s { background:#c0392b; color:#fff; border-radius:4px; padding:1px 7px;
        font-size:.7rem; font-weight:700; margin-right:6px; }
    .badge-b { background:#1f7aec; color:#fff; border-radius:4px; padding:1px 7px;
        font-size:.7rem; font-weight:700; margin-right:6px; }
    .badge-nrml { border:1px solid var(--border); color:var(--muted);
        border-radius:4px; padding:1px 6px; font-size:.68rem; margin-right:6px; }
    .grn { color:var(--green); } .rd { color:var(--red); } .mut { color:var(--muted); }

    .mstrip { display:flex; flex-wrap:wrap; border:1px solid var(--border);
        border-radius:10px; background:var(--panel2); overflow:hidden; margin-bottom:10px; }
    .mcell { flex:1 1 11%; min-width:110px; padding:10px 12px;
        border-right:1px solid var(--border); }
    .mcell:last-child { border-right:none; }
    .mlab { color:var(--muted); font-size:.64rem; letter-spacing:.08em;
        text-transform:uppercase; font-weight:600; }
    .mval { font-family:var(--mono); font-size:.95rem; font-weight:600; margin-top:3px; }

    .ai-h { color:#5aa7ff; font-weight:700; font-size:.8rem; letter-spacing:.06em; }
    .ai-sec { color:var(--amber); font-weight:700; font-size:.85rem; margin-top:10px; }
    .ai-body { font-size:.85rem; line-height:1.65; color:var(--txt); }
    .ai-body table { width:100%; border-collapse:collapse; margin:8px 0; }
    .ai-body th { color:#5aa7ff; text-align:left; font-size:.75rem;
        border-bottom:1px solid var(--border); padding:5px 8px; }
    .ai-body td { font-family:var(--mono); font-size:.8rem; padding:5px 8px;
        border-bottom:1px solid rgba(255,255,255,.04); }

    .pl-box { background:var(--panel2); border:1px solid var(--border);
        border-radius:8px; padding:14px; text-align:center; font-size:.95rem; }
    .pl-box .v { font-family:var(--mono); font-weight:700; font-size:1.15rem; }

    .chip { display:inline-block; background:var(--panel2); border:1px solid var(--border);
        border-radius:20px; padding:4px 12px; margin:0 6px 6px 0; font-size:.78rem; }
    .chip b { font-family:var(--mono); }

    .stTabs [data-baseweb="tab-list"] { gap:18px; border-bottom:1px solid var(--border); }
    .stTabs [data-baseweb="tab"] { color:var(--muted); font-weight:600; }
    .stTabs [aria-selected="true"] { color:var(--blue) !important;
        border-bottom:2px solid var(--blue); }
    .stButton>button { background:transparent; border:1px solid var(--border);
        color:var(--txt); border-radius:8px; font-weight:600; font-size:.85rem; }
    .stButton>button:hover { border-color:var(--blue); color:var(--blue); }
    div[data-testid="stSlider"] label { color:var(--muted) !important; font-size:.8rem; }
    div[data-testid="stExpander"] { background:var(--panel); border:1px solid var(--border);
        border-radius:10px; }
    div[data-testid="stDataFrame"] { border:1px solid var(--border); border-radius:10px; }
    </style>
    """, unsafe_allow_html=True)


# ==============================================================================
# 3. NUMBER FORMATTING (Indian grouping + compact)  -- from v2
# ==============================================================================
def fmt_inr(x, dec: int = 2, sign: bool = False) -> str:
    """Indian-style grouping: 1,36,978.00"""
    if x is None or (isinstance(x, float) and (math.isinf(x) or math.isnan(x))):
        return "—"
    neg = x < 0
    s = f"{abs(x):.{dec}f}"
    ip, fp = (s.split(".") + [""])[:2]
    if len(ip) > 3:
        head, tail = ip[:-3], ip[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        ip = ",".join(parts) + "," + tail
    out = ip + ("." + fp if fp else "")
    return ("-" if neg else ("+" if sign else "")) + out


def fmt_compact(x) -> str:
    """12.8K / 1.60L / 2.40Cr style."""
    if x is None or (isinstance(x, float) and (math.isinf(x) or math.isnan(x))):
        return "—"
    a, sgn = abs(x), ("-" if x < 0 else "")
    if a >= 1e7: return f"{sgn}{a/1e7:.2f}Cr"
    if a >= 1e5: return f"{sgn}{a/1e5:.2f}L"
    if a >= 1e3: return f"{sgn}{a/1e3:.1f}K"
    return f"{sgn}{a:.0f}"


def pnl_span(x, compact: bool = True, dec: int = 2) -> str:
    cls = "grn" if x >= 0 else "rd"
    txt = fmt_compact(x) if compact else "₹" + fmt_inr(x, dec)
    return f"<span class='{cls}'>{txt}</span>"


# ==============================================================================
# 4. QUANT PRICING ENGINE (Vectorized BSM) -- LOGIC PRESERVED VERBATIM FROM v2
# ==============================================================================
class BlackScholesEngine:
    """Vectorized Black-Scholes for real-time pricing and Greeks."""

    @staticmethod
    def _d1_d2(S, K, T, r, sigma):
        T = np.maximum(T, 1e-5)
        sigma = np.maximum(sigma, 1e-4)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2, T, sigma

    @classmethod
    def price(cls, S, K, T, r, sigma, opt_type):
        d1, d2, T, sigma = cls._d1_d2(S, K, T, r, sigma)
        if opt_type.upper() == 'C':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @classmethod
    def delta(cls, S, K, T, r, sigma, opt_type):
        d1, _, _, _ = cls._d1_d2(S, K, T, r, sigma)
        return norm.cdf(d1) if opt_type.upper() == 'C' else norm.cdf(d1) - 1

    @classmethod
    def gamma(cls, S, K, T, r, sigma):
        d1, _, T, sigma = cls._d1_d2(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    @classmethod
    def theta(cls, S, K, T, r, sigma, opt_type):
        d1, d2, T, sigma = cls._d1_d2(S, K, T, r, sigma)
        term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        if opt_type.upper() == 'C':
            return (term1 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        return (term1 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    @classmethod
    def vega(cls, S, K, T, r, sigma):
        d1, _, T, _ = cls._d1_d2(S, K, T, r, sigma)
        return (S * norm.pdf(d1) * np.sqrt(T)) / 100

    @classmethod
    def prob_itm(cls, S, K, T, r, sigma, opt_type):
        """Risk-neutral P(ITM at expiry): N(d2) for calls, N(-d2) for puts."""
        _, d2, _, _ = cls._d1_d2(S, K, T, r, sigma)
        return norm.cdf(d2) if opt_type.upper() == 'C' else norm.cdf(-d2)


# ==============================================================================
# 5. DATA PROVIDER ABSTRACTION
#    Sources are never hardcoded at call sites: each capability walks an
#    ordered provider chain and takes the first success. Add future providers
#    by appending a callable to the relevant chain.
# ==============================================================================
def _jugaad_chain(symbol: str) -> Optional[dict]:
    """Raw NSE option-chain payload via jugaad_data (None if unavailable)."""
    if nse_live is None:
        return None
    try:
        if symbol in INDEX_SET:
            return nse_live.index_option_chain(symbol)
        return nse_live.equities_option_chain(symbol)
    except Exception as e:
        log.warning("jugaad chain failed for %s: %s", symbol, e)
        return None


def _yf_symbol(symbol: str) -> str:
    return INDEX_YF_MAP.get(symbol, symbol if symbol.endswith(".NS") else f"{symbol}.NS")


# --- provider chains (ordered; extend for future data vendors) --------------
CHAIN_PROVIDERS: List[Callable[[str], Optional[dict]]] = [_jugaad_chain]


@st.cache_data(ttl=180, show_spinner=False)
def fetch_chain_raw(symbol: str) -> Optional[dict]:
    """Live option chain from the first working provider (cached 3 min)."""
    for provider in CHAIN_PROVIDERS:
        data = provider(symbol.upper().strip())
        if data:
            return data
    return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_data(ticker: str) -> Tuple[float, float, float]:
    """Spot, ATM IV, prev close. Chain: NSE live -> yfinance -> defaults.
    (v2 behaviour preserved.)"""
    if not ticker:
        return 100.0, 0.20, 100.0
    ticker = ticker.upper().strip()
    spot = iv = prev_close = None

    chain = fetch_chain_raw(ticker)
    if chain:
        try:
            spot = float(chain['records']['underlyingValue'])
            rows = [r_ for r_ in chain['records']['data']
                    if r_.get('strikePrice') is not None]   # a defaulted strike
            if rows:                                        # would fake distance 0
                nearest = min(rows, key=lambda r: abs(r['strikePrice'] - spot))
                ivs = [v['impliedVolatility'] for k in ('CE', 'PE') if k in nearest
                       for v in [nearest[k]] if v.get('impliedVolatility', 0)]
                iv = (sum(ivs) / len(ivs) / 100) if ivs else None
        except Exception:
            log.warning("chain spot/IV parse failed for %s", ticker, exc_info=True)

    try:
        hist = yf.Ticker(_yf_symbol(ticker)).history(period="5d")
        if not hist.empty:
            if spot is None:
                spot = float(hist['Close'].iloc[-1])
            if len(hist) >= 2:
                prev_close = float(hist['Close'].iloc[-2])
    except Exception as e:
        log.warning("yfinance spot failed: %s", e)

    if iv is None:
        try:
            vix = yf.Ticker("^INDIAVIX").history(period="1d")['Close'].iloc[-1]
            iv = float(vix) / 100
        except Exception:
            log.warning("VIX fetch failed for %s — defaulting IV to 15%%",
                        ticker, exc_info=True)
            iv = 0.15
    return (spot or 100.0), iv, (prev_close or spot or 100.0)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_spot_history(symbol: str, period: str = "2y") -> pd.DataFrame:
    """Daily OHLC history (yfinance provider)."""
    try:
        df = yf.Ticker(_yf_symbol(symbol)).history(period=period)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception as e:
        log.warning("history failed for %s: %s", symbol, e)
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_vix_history(period: str = "2y") -> pd.DataFrame:
    """India VIX daily history - IV proxy for rank/percentile and backtests."""
    try:
        df = yf.Ticker("^INDIAVIX").history(period=period)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()


def fetch_fii_dii() -> Optional[pd.DataFrame]:
    """FII/DII flows - no free stable provider wired yet. Returns None so the
    UI shows an 'add a provider' notice instead of fake numbers. Register a
    callable here when a source (broker API, paid feed) is available."""
    return None


def chain_expiries(chain: dict) -> List[str]:
    try:
        return list(chain['records']['expiryDates'])
    except Exception:
        return []


@st.cache_data(ttl=180, show_spinner=False)
def build_chain_dataframe(symbol: str, expiry: str, r: float = 0.065) -> Optional[pd.DataFrame]:
    """Processed option chain for one expiry with vectorized BSM Greeks,
    intrinsic/time value and bid/ask spread. One row per strike."""
    chain = fetch_chain_raw(symbol)
    if not chain:
        return None
    try:
        spot = float(chain['records']['underlyingValue'])
        rows = [r_ for r_ in chain['records']['data'] if r_.get('expiryDate') == expiry]
        if not rows:
            return None
        T = max((pd.to_datetime(expiry) - datetime.now()).days, 0) / 365.0
        recs = []
        for row in rows:
            k = float(row['strikePrice'])
            ce, pe = row.get('CE', {}) or {}, row.get('PE', {}) or {}
            recs.append({
                'Strike': k,
                'CE_LTP': ce.get('lastPrice', np.nan), 'PE_LTP': pe.get('lastPrice', np.nan),
                'CE_OI': ce.get('openInterest', 0), 'PE_OI': pe.get('openInterest', 0),
                'CE_ChgOI': ce.get('changeinOpenInterest', 0),
                'PE_ChgOI': pe.get('changeinOpenInterest', 0),
                'CE_Vol': ce.get('totalTradedVolume', 0), 'PE_Vol': pe.get('totalTradedVolume', 0),
                'CE_IV': ce.get('impliedVolatility', np.nan), 'PE_IV': pe.get('impliedVolatility', np.nan),
                'CE_Bid': ce.get('bidprice', np.nan), 'CE_Ask': ce.get('askPrice', np.nan),
                'PE_Bid': pe.get('bidprice', np.nan), 'PE_Ask': pe.get('askPrice', np.nan),
            })
        df = pd.DataFrame(recs).sort_values('Strike').reset_index(drop=True)

        # Vectorized Greeks (use quoted IV; fall back to ATM IV where missing)
        atm_iv = float(np.nanmedian(pd.concat([df['CE_IV'], df['PE_IV']])))
        if not math.isfinite(atm_iv) or atm_iv <= 0:   # nanmedian of all-NaN is
            atm_iv = 15.0                              # NaN, and bool(NaN) is True
        K = df['Strike'].values
        civ = np.nan_to_num(df['CE_IV'].values, nan=atm_iv) / 100
        piv = np.nan_to_num(df['PE_IV'].values, nan=atm_iv) / 100
        df['CE_Delta'] = np.round(BlackScholesEngine.delta(spot, K, T, r, civ, 'C'), 3)
        df['PE_Delta'] = np.round(BlackScholesEngine.delta(spot, K, T, r, piv, 'P'), 3)
        df['CE_Gamma'] = np.round(BlackScholesEngine.gamma(spot, K, T, r, civ), 5)
        df['PE_Gamma'] = np.round(BlackScholesEngine.gamma(spot, K, T, r, piv), 5)
        df['CE_Theta'] = np.round(BlackScholesEngine.theta(spot, K, T, r, civ, 'C'), 2)
        df['PE_Theta'] = np.round(BlackScholesEngine.theta(spot, K, T, r, piv, 'P'), 2)
        df['CE_Vega'] = np.round(BlackScholesEngine.vega(spot, K, T, r, civ), 2)
        df['PE_Vega'] = np.round(BlackScholesEngine.vega(spot, K, T, r, piv), 2)

        df['CE_Spread'] = np.round(df['CE_Ask'] - df['CE_Bid'], 2)
        df['PE_Spread'] = np.round(df['PE_Ask'] - df['PE_Bid'], 2)
        df['CE_Intr'] = np.round(np.maximum(spot - K, 0), 2)
        df['PE_Intr'] = np.round(np.maximum(K - spot, 0), 2)
        df['CE_TimeVal'] = np.round(df['CE_LTP'] - df['CE_Intr'], 2)
        df['PE_TimeVal'] = np.round(df['PE_LTP'] - df['PE_Intr'], 2)
        df.attrs['spot'] = spot
        df.attrs['expiry'] = expiry
        return df
    except Exception as e:
        log.warning("chain df failed: %s", e)
        return None


def chain_analytics(df: pd.DataFrame) -> dict:
    """PCR, Max Pain, ATM strike, highest OI / ChgOI strikes for a chain df."""
    spot = df.attrs.get('spot', float(df['Strike'].median()))
    atm = float(df.loc[(df['Strike'] - spot).abs().idxmin(), 'Strike'])
    tot_ce, tot_pe = float(df['CE_OI'].sum()), float(df['PE_OI'].sum())
    pcr = tot_pe / tot_ce if tot_ce else float('nan')

    # Max Pain: strike minimising total option-writer payout at expiry
    K = df['Strike'].values
    ce_oi, pe_oi = df['CE_OI'].values.astype(float), df['PE_OI'].values.astype(float)
    pain = [(np.maximum(k - K, 0) * ce_oi).sum() + (np.maximum(K - k, 0) * pe_oi).sum()
            for k in K]
    max_pain = float(K[int(np.argmin(pain))])

    return {
        'atm': atm, 'pcr': pcr, 'max_pain': max_pain,
        'tot_ce_oi': tot_ce, 'tot_pe_oi': tot_pe,
        'hi_ce_oi': float(df.loc[df['CE_OI'].idxmax(), 'Strike']),
        'hi_pe_oi': float(df.loc[df['PE_OI'].idxmax(), 'Strike']),
        'hi_ce_chg': float(df.loc[df['CE_ChgOI'].idxmax(), 'Strike']),
        'hi_pe_chg': float(df.loc[df['PE_ChgOI'].idxmax(), 'Strike']),
    }


@st.cache_data(ttl=300, show_spinner=False)
def fetch_oi_data(ticker: str):
    """OI by strike for nearest expiry (payoff-chart overlay). From v2."""
    chain = fetch_chain_raw(ticker)
    if not chain:
        return None
    try:
        expiry = chain['records']['expiryDates'][0]
        strikes, ce_oi, pe_oi = [], [], []
        for row in chain['records']['data']:
            if row.get('expiryDate') != expiry:
                continue
            strikes.append(row['strikePrice'])
            ce_oi.append(row.get('CE', {}).get('openInterest', 0) if 'CE' in row else 0)
            pe_oi.append(row.get('PE', {}).get('openInterest', 0) if 'PE' in row else 0)
        if not strikes:
            return None
        order = np.argsort(strikes)
        return (np.array(strikes)[order],
                np.array(ce_oi, dtype=float)[order],
                np.array(pe_oi, dtype=float)[order],
                float(np.sum(ce_oi)), float(np.sum(pe_oi)))
    except Exception:
        return None


# ==============================================================================
# 6. STRATEGY DETECTION -- PRESERVED VERBATIM -- plus template builder
# ==============================================================================
class StrategyEngine:
    @staticmethod
    def detect(df):
        if df.empty:
            return "No Position", "Neutral"
        legs = len(df)
        types = df['Type'].value_counts().to_dict()
        actions = df['Action'].value_counts().to_dict()
        c_count, p_count = types.get('Call', 0), types.get('Put', 0)
        l_count, s_count = actions.get('Buy', 0), actions.get('Sell', 0)
        strikes = sorted(df['Strike'].tolist())
        expiries = df['Expiry'].nunique()

        if legs == 1:
            row = df.iloc[0]
            if row['Action'] == 'Buy' and row['Type'] == 'Call': return "Long Call", "Bullish"
            if row['Action'] == 'Sell' and row['Type'] == 'Call': return "Short Call", "Bearish"
            if row['Action'] == 'Buy' and row['Type'] == 'Put': return "Long Put", "Bearish"
            if row['Action'] == 'Sell' and row['Type'] == 'Put': return "Short Put", "Bullish"
        if legs == 2 and expiries == 1:
            if c_count == 2 and l_count == 1 and s_count == 1:
                long_k = df[(df['Action'] == 'Buy') & (df['Type'] == 'Call')]['Strike'].values[0]
                short_k = df[(df['Action'] == 'Sell') & (df['Type'] == 'Call')]['Strike'].values[0]
                return ("Bull Call Spread", "Bullish") if long_k < short_k else ("Bear Call Spread", "Bearish")
            if p_count == 2 and l_count == 1 and s_count == 1:
                long_k = df[(df['Action'] == 'Buy') & (df['Type'] == 'Put')]['Strike'].values[0]
                short_k = df[(df['Action'] == 'Sell') & (df['Type'] == 'Put')]['Strike'].values[0]
                return ("Bear Put Spread", "Bearish") if long_k > short_k else ("Bull Put Spread", "Bullish")
            if c_count == 1 and p_count == 1:
                if l_count == 2: return ("Long Straddle" if strikes[0] == strikes[1] else "Long Strangle", "Volatile")
                if s_count == 2: return ("Short Straddle" if strikes[0] == strikes[1] else "Short Strangle", "Range-bound")
        if legs == 2 and expiries == 2 and c_count + p_count == 2:
            if strikes[0] == strikes[1]:
                return "Calendar Spread", "Range-bound"
            return "Diagonal Spread", "Directional"
        if legs == 4 and expiries == 1:
            if c_count == 2 and p_count == 2 and l_count == 2 and s_count == 2:
                short_strikes = df[df['Action'] == 'Sell']['Strike'].tolist()
                if short_strikes[0] == short_strikes[1]:
                    return "Iron Fly", "Range-bound"
                return "Iron Condor", "Range-bound"
        return "Custom Strategy", "Directional"


def strategy_templates(spot: float, symbol: str) -> Dict[str, List[dict]]:
    """Leg blueprints around ATM: (type, action, strike-offset-in-steps, qty,
    dte-offset-days). Basis of the strategy builder - every template is fully
    editable after insertion, and unlimited custom legs can be added."""
    s = STRIKE_STEP.get(symbol, max(round(spot * 0.01, -1), 5))
    atm = round(spot / s) * s
    L = lambda t, a, off, q=1, dteoff=0: {
        'Type': t, 'Action': a, 'Strike': atm + off * s, 'Qty': q, 'DTEoff': dteoff}
    return {
        "Long Call": [L('Call', 'Buy', 0)],
        "Long Put": [L('Put', 'Buy', 0)],
        "Short Call": [L('Call', 'Sell', 2)],
        "Short Put": [L('Put', 'Sell', -2)],
        "Bull Call Spread": [L('Call', 'Buy', 0), L('Call', 'Sell', 4)],
        "Bear Put Spread": [L('Put', 'Buy', 0), L('Put', 'Sell', -4)],
        "Bull Put Spread": [L('Put', 'Sell', -2), L('Put', 'Buy', -6)],
        "Bear Call Spread": [L('Call', 'Sell', 2), L('Call', 'Buy', 6)],
        "Long Straddle": [L('Call', 'Buy', 0), L('Put', 'Buy', 0)],
        "Short Straddle": [L('Call', 'Sell', 0), L('Put', 'Sell', 0)],
        "Long Strangle": [L('Call', 'Buy', 4), L('Put', 'Buy', -4)],
        "Short Strangle": [L('Call', 'Sell', 4), L('Put', 'Sell', -4)],
        "Iron Condor": [L('Put', 'Buy', -8), L('Put', 'Sell', -4),
                        L('Call', 'Sell', 4), L('Call', 'Buy', 8)],
        "Iron Butterfly": [L('Put', 'Buy', -6), L('Put', 'Sell', 0),
                           L('Call', 'Sell', 0), L('Call', 'Buy', 6)],
        "Calendar Spread": [L('Call', 'Sell', 0, dteoff=0), L('Call', 'Buy', 0, dteoff=28)],
        "Diagonal Spread": [L('Call', 'Sell', 2, dteoff=0), L('Call', 'Buy', 0, dteoff=28)],
        "Call Ratio Spread": [L('Call', 'Buy', 0), L('Call', 'Sell', 4, q=2)],
        "Put Ratio Spread": [L('Put', 'Buy', 0), L('Put', 'Sell', -4, q=2)],
        "Jade Lizard": [L('Put', 'Sell', -4), L('Call', 'Sell', 4), L('Call', 'Buy', 8)],
        "Broken Wing Butterfly": [L('Call', 'Buy', 0), L('Call', 'Sell', 4, q=2),
                                  L('Call', 'Buy', 10)],
        "Christmas Tree": [L('Call', 'Buy', 0), L('Call', 'Sell', 4, q=3),
                           L('Call', 'Buy', 6, q=2)],
    }


# ==============================================================================
# 7. RISK & PORTFOLIO ENGINE -- v2 math preserved; stress/exposure/prob added
# ==============================================================================
class PortfolioRiskEngine:
    def __init__(self, positions_df: pd.DataFrame, spot: float, r: float = 0.065):
        self.df = positions_df.copy()
        self.spot = spot
        self.r = r
        self._prep()

    def _prep(self) -> None:
        """Pre-compute per-leg static params (lot multiplier, DTE, direction)."""
        legs = []
        for _, row in self.df.iterrows():
            try:
                days = (pd.to_datetime(row['Expiry']) - datetime.now()).days
            except Exception:
                days = 30
            days = max(days, 0)
            lot_override = row.get('LotSize', None)
            if lot_override is not None and pd.notna(lot_override) and float(lot_override) > 0:
                mult = float(lot_override)
            else:
                mult = LOT_SIZES.get(str(row['Symbol']).upper(), DEFAULT_STOCK_LOT_SIZE)
            direction = 1 if row['Action'] == 'Buy' else -1
            legs.append({
                'row': row, 'K': float(row['Strike']), 'vol': float(row['IV']) / 100.0,
                'days': days, 'mult': mult, 'direction': direction,
                'actual_qty': float(row['Qty']) * mult * direction,
                'opt_type': 'C' if row['Type'] == 'Call' else 'P',
            })
        self.legs = legs
        self.dte = min((l['days'] for l in legs), default=0)

    def price_range(self, width: float = 0.11, n: int = 300) -> np.ndarray:
        return np.linspace(self.spot * (1 - width), self.spot * (1 + width), n)

    def payoff_expiry(self, s_range: np.ndarray) -> np.ndarray:
        total = np.zeros_like(s_range)
        for l in self.legs:
            K, aq, avg = l['K'], l['actual_qty'], float(l['row']['AvgPrice'])
            val = np.maximum(s_range - K, 0) if l['opt_type'] == 'C' else np.maximum(K - s_range, 0)
            total += (val - avg) * aq
        return total

    def payoff_at_days(self, s_range: np.ndarray, days_elapsed: float,
                       iv_shift: float = 0.0) -> np.ndarray:
        """Mark-to-model P&L at spot=s_range after days_elapsed. iv_shift is an
        absolute vol-point shift (e.g. 0.02 = +2 IV pts) for stress testing."""
        total = np.zeros_like(s_range, dtype=float)
        for l in self.legs:
            K, aq = l['K'], l['actual_qty']
            vol = max(l['vol'] + iv_shift, 1e-4)
            avg = float(l['row']['AvgPrice'])
            rem = max(l['days'] - days_elapsed, 0)
            if rem <= 0:
                px = np.maximum(s_range - K, 0) if l['opt_type'] == 'C' else np.maximum(K - s_range, 0)
            else:
                px = BlackScholesEngine.price(s_range, K, rem / 365.0, self.r, vol, l['opt_type'])
            total += (px - avg) * aq
        return total

    def pnl_point(self, target_price: float, days_elapsed: float, iv_shift: float = 0.0) -> float:
        return float(self.payoff_at_days(np.array([float(target_price)]), days_elapsed, iv_shift)[0])

    def get_metrics(self) -> dict:
        """Portfolio Greeks, P&L stats, POP, breakevens, margin, exposures.
        Per-leg BSM math identical to v2."""
        s_range = self.price_range(width=0.25, n=400)
        payoff_exp = self.payoff_expiry(s_range)

        net_delta = net_gamma = net_theta = net_vega = 0.0
        net_premium = capital_req = time_value = intrinsic_value = 0.0
        updated_rows = []
        S = self.spot

        for l in self.legs:
            row = l['row']
            K, vol, aq, mult = l['K'], l['vol'], l['actual_qty'], l['mult']
            qty = float(row['Qty'])
            T = max(l['days'] / 365.0, 0.001)
            ot = l['opt_type']

            d = BlackScholesEngine.delta(S, K, T, self.r, vol, ot) * aq
            g = BlackScholesEngine.gamma(S, K, T, self.r, vol) * aq
            t = BlackScholesEngine.theta(S, K, T, self.r, vol, ot) * aq
            v = BlackScholesEngine.vega(S, K, T, self.r, vol) * aq
            price = float(BlackScholesEngine.price(S, K, T, self.r, vol, ot))
            premium = price * aq

            net_delta += d; net_gamma += g; net_theta += t; net_vega += v
            net_premium -= premium

            intr = max(S - K, 0) if ot == 'C' else max(K - S, 0)
            intrinsic_value += intr * aq
            time_value += max(price - intr, 0) * aq

            if l['direction'] == -1:
                capital_req += S * 0.15 * qty * mult
            else:
                capital_req += abs(premium)

            booked = float(row.get('Booked', 0) or 0)
            unbooked = (price - float(row['AvgPrice'])) * aq
            updated_rows.append({
                **row, 'LTP': round(price, 2), 'Booked': round(booked, 2),
                'Unbooked': round(unbooked, 2), 'MTM': round(booked + unbooked, 2),
                'Delta': round(BlackScholesEngine.delta(S, K, T, self.r, vol, ot), 4),
                'Gamma': round(BlackScholesEngine.gamma(S, K, T, self.r, vol), 6),
                'Theta': round(BlackScholesEngine.theta(S, K, T, self.r, vol, ot), 4),
                'Vega': round(BlackScholesEngine.vega(S, K, T, self.r, vol), 4),
                'P_ITM': round(BlackScholesEngine.prob_itm(S, K, T, self.r, vol, ot) * 100, 1),
                'Exposure': round(abs(aq) * S, 0),
                'DTE': l['days'],
            })

        self.live = pd.DataFrame(updated_rows)

        max_profit = float('inf') if np.max(payoff_exp) > 1e7 else float(np.max(payoff_exp))
        max_loss = float('-inf') if np.min(payoff_exp) < -1e7 else float(np.min(payoff_exp))

        signs = np.sign(payoff_exp)
        sign_changes = ((np.roll(signs, 1) - signs) != 0).astype(int)
        sign_changes[0] = 0
        breakevens = s_range[sign_changes == 1]

        # POP: lognormal probability mass over the profit region at expiry.
        try:
            sigma = float(np.mean([l['vol'] for l in self.legs]))
            T = max(self.dte, 1) / 365.0
            mu = (self.r - 0.5 * sigma ** 2) * T
            sd = sigma * math.sqrt(T)
            z = (np.log(s_range / self.spot) - mu) / sd
            w = norm.pdf(z) / s_range
            w_sum = float(w.sum())
            if not math.isfinite(w_sum) or w_sum <= 0:
                # numpy 0/0 yields NaN silently (no exception) — force fallback
                raise ValueError("degenerate probability mass")
            pop = float(w[payoff_exp > 0].sum() / w_sum * 100)
            if not math.isfinite(pop):
                raise ValueError("non-finite POP")
            pop = min(max(pop, 0.1), 99.9)
        except Exception:
            if max_profit > 0 and max_loss < 0 and math.isfinite(max_loss):
                pop = min(max((abs(max_loss) / (abs(max_profit) + abs(max_loss))) * 100, 1), 99)
            else:
                pop = 100 if max_loss >= 0 else 0

        rr = None
        if math.isfinite(max_profit):
            if math.isfinite(max_loss) and max_loss < 0:
                rr = abs(max_profit / max_loss)
            elif capital_req > 0:
                rr = max_profit / capital_req

        return {
            'max_profit': max_profit, 'max_loss': max_loss,
            'breakevens': breakevens, 'pop': pop, 'capital': capital_req,
            'net_premium': net_premium, 'delta': net_delta, 'gamma': net_gamma,
            'theta': net_theta, 'vega': net_vega, 'time_value': time_value,
            'intrinsic_value': intrinsic_value, 'reward_risk': rr,
            'booked': float(self.live['Booked'].sum()),
            'unbooked': float(self.live['Unbooked'].sum()),
            'total_mtm': float(self.live['MTM'].sum()),
            'dte': self.dte,
        }

    # ---------- NEW: stress & exposure analytics ----------
    def stress_matrix(self, spot_moves=(-0.05, -0.03, -0.01, 0, 0.01, 0.03, 0.05),
                      iv_moves=(-0.05, -0.02, 0, 0.02, 0.05),
                      days_elapsed: float = 0) -> pd.DataFrame:
        """P&L grid across spot% x IV-point shifts (T+n model repricing)."""
        grid = np.zeros((len(iv_moves), len(spot_moves)))
        for i, ivm in enumerate(iv_moves):
            for j, sm in enumerate(spot_moves):
                grid[i, j] = self.pnl_point(self.spot * (1 + sm), days_elapsed, ivm)
        return pd.DataFrame(grid,
                            index=[f"IV {m*100:+.0f}pt" for m in iv_moves],
                            columns=[f"{m*100:+.0f}%" for m in spot_moves])

    def exposure_tables(self) -> Dict[str, pd.DataFrame]:
        """Greeks/notional grouped by expiry, strike and sector."""
        if not hasattr(self, 'live'):
            self.get_metrics()
        df = self.live.copy()
        df['Sector'] = df['Symbol'].map(lambda s: SECTOR_MAP.get(str(s).upper(), "Other"))
        agg = {'Delta': 'sum', 'Gamma': 'sum', 'Theta': 'sum', 'Vega': 'sum',
               'MTM': 'sum', 'Exposure': 'sum'}
        return {
            'By Expiry': df.groupby('Expiry').agg(agg).round(3),
            'By Strike': df.groupby('Strike').agg(agg).round(3),
            'By Sector': df.groupby('Sector').agg(agg).round(3),
        }

    def gap_analysis(self) -> pd.DataFrame:
        """Overnight gap scenarios: gap-down assumes an IV spike, gap-up an
        IV soften (typical index behaviour)."""
        rows = []
        for gap, ivm in [(-0.05, 0.06), (-0.03, 0.04), (-0.02, 0.02),
                         (0.02, -0.01), (0.03, -0.02), (0.05, -0.02)]:
            pnl = self.pnl_point(self.spot * (1 + gap), 0, ivm)
            rows.append({'Gap': f"{gap*100:+.0f}%", 'IV shift': f"{ivm*100:+.0f}pt",
                         'Spot': round(self.spot * (1 + gap), 0), 'P&L': round(pnl, 0)})
        return pd.DataFrame(rows)


# ==============================================================================
# 8. VOLATILITY ANALYTICS
# ==============================================================================
class VolAnalytics:
    """Historical/realized vol, IV rank & percentile (India-VIX based),
    vol cone, smile and term structure from the live chain."""

    @staticmethod
    def hv_series(close: pd.Series, window: int = 20) -> pd.Series:
        ret = np.log(close / close.shift(1))
        return ret.rolling(window).std() * math.sqrt(252) * 100

    @classmethod
    def realized_vol(cls, close: pd.Series, window: int = 20) -> float:
        s = cls.hv_series(close, window).dropna()
        return float(s.iloc[-1]) if len(s) else float('nan')

    @staticmethod
    def iv_rank_percentile(vix: pd.Series) -> Tuple[float, float, float]:
        """(current, rank 0-100, percentile 0-100) over the series window."""
        v = vix.dropna()
        if v.empty:
            return float('nan'), float('nan'), float('nan')
        cur, lo, hi = float(v.iloc[-1]), float(v.min()), float(v.max())
        rank = (cur - lo) / (hi - lo) * 100 if hi > lo else 50.0
        pct = float((v < cur).mean() * 100)
        return cur, rank, pct

    @classmethod
    def vol_cone(cls, close: pd.Series,
                 windows=(10, 20, 30, 60, 90)) -> pd.DataFrame:
        """Percentile envelope of realized vol across lookback windows."""
        rows = []
        for w in windows:
            s = cls.hv_series(close, w).dropna()
            if s.empty:
                continue
            rows.append({'Window': w, 'Min': s.min(), 'P25': s.quantile(.25),
                         'Median': s.median(), 'P75': s.quantile(.75),
                         'Max': s.max(), 'Current': s.iloc[-1]})
        return pd.DataFrame(rows)

    @staticmethod
    def smile(chain_df: pd.DataFrame) -> pd.DataFrame:
        return chain_df[['Strike', 'CE_IV', 'PE_IV']].dropna(how='all')

    @staticmethod
    def term_structure(symbol: str, r: float) -> pd.DataFrame:
        """ATM IV per expiry (first 6 expiries) from the live chain."""
        chain = fetch_chain_raw(symbol)
        if not chain:
            return pd.DataFrame()
        rows = []
        for exp in chain_expiries(chain)[:6]:
            df = build_chain_dataframe(symbol, exp, r)
            if df is None or df.empty:
                continue
            spot = df.attrs['spot']
            atm_row = df.iloc[(df['Strike'] - spot).abs().argmin()]
            ivs = [x for x in (atm_row['CE_IV'], atm_row['PE_IV']) if pd.notna(x)]
            if ivs:
                dte = max((pd.to_datetime(exp) - datetime.now()).days, 0)
                rows.append({'Expiry': exp, 'DTE': dte, 'ATM_IV': float(np.mean(ivs))})
        return pd.DataFrame(rows)


# ==============================================================================
# 9. PROBABILITY ANALYTICS (expected move, SD bands, Monte Carlo)
# ==============================================================================
class ProbAnalytics:
    @staticmethod
    def expected_move(spot: float, iv: float, dte: int) -> float:
        """1-sigma expected move to expiry: S * sigma * sqrt(T)."""
        return spot * iv * math.sqrt(max(dte, 1) / 365.0)

    @staticmethod
    def prob_touch(p_itm: float) -> float:
        """Standard approximation: P(touch) ~ 2 x P(ITM), capped at 100%."""
        return min(2 * p_itm, 100.0)

    @staticmethod
    def monte_carlo(engine: 'PortfolioRiskEngine', iv: float, n: int = 5000,
                    seed: int = 42) -> dict:
        """GBM terminal-price simulation to expiry; P&L distribution via the
        exact expiry payoff. Vectorized - no per-path Python loops."""
        rng = np.random.default_rng(seed)
        T = max(engine.dte, 1) / 365.0
        z = rng.standard_normal(n)
        st_prices = engine.spot * np.exp((engine.r - 0.5 * iv ** 2) * T
                                         + iv * math.sqrt(T) * z)
        pnl = engine.payoff_expiry(st_prices)
        return {
            'prices': st_prices, 'pnl': pnl,
            'pop_mc': float((pnl > 0).mean() * 100),
            'exp_pnl': float(pnl.mean()),
            'p5': float(np.percentile(pnl, 5)),
            'p95': float(np.percentile(pnl, 95)),
            'cvar5': float(pnl[pnl <= np.percentile(pnl, 5)].mean()),
        }


# ==============================================================================
# 10. HISTORICAL BACKTESTING ENGINE
#     Historical option premiums are BSM-MODELLED from historical spot with
#     leg IVs scaled by the India-VIX path (NSE publishes no free historical
#     chains). Stated in the UI; adequate for strategy-shape research, not for
#     tick-accurate fills.
# ==============================================================================
class BacktestEngine:
    def __init__(self, symbol: str, r: float = 0.065, cost_bps: float = 5.0,
                 store: Optional[object] = None):
        """cost_bps: round-trip friction per leg as bps of premium notional
        (brokerage + slippage + charges, user-tunable).
        store: optional BhavcopyStore — when supplied, REAL historical EOD
        option premiums are used wherever coverage exists; BSM-modelled
        prices fill only the gaps (per-trade data-source flag reported)."""
        self.symbol = symbol
        self.r = r
        self.cost_bps = cost_bps
        self.store = store
        self.spot_hist = fetch_spot_history(symbol, "5y")
        self.vix_hist = fetch_vix_history("5y")

    # ---------- shared helpers ----------
    def _iv_path(self, dates: pd.DatetimeIndex, base_iv: float) -> pd.Series:
        """Leg IV through time = base_iv scaled by VIX relative to its value
        on the first date (keeps each leg's smile offset)."""
        if self.vix_hist.empty:
            return pd.Series(base_iv, index=dates)
        vix = self.vix_hist['Close'].reindex(dates).ffill().bfill()
        v0 = float(vix.iloc[0]) if len(vix) else float('nan')
        if not math.isfinite(v0) or v0 <= 0 or vix.isna().all():
            # No overlapping VIX data for this window — flat IV, never NaN-poison
            return pd.Series(base_iv, index=dates)
        return (base_iv * vix.fillna(v0) / v0).clip(0.05, 1.5)

    def _leg_price(self, S, K, dte_days, iv, opt_type):
        if dte_days <= 0:
            return max(S - K, 0) if opt_type == 'C' else max(K - S, 0)
        return float(BlackScholesEngine.price(S, K, dte_days / 365.0, self.r, iv, opt_type))

    # ---------- real-data helpers (BhavcopyStore-backed) ----------
    def _pick_real_expiry(self, d0: pd.Timestamp, target_dte: int) -> Optional[str]:
        """Listed expiry (from stored bhavcopies) nearest to the target DTE."""
        if self.store is None:
            return None
        try:
            exps = self.store.expiries_on(str(d0.date()), self.symbol)
            if not exps:
                return None
            return min(exps, key=lambda e: abs((pd.to_datetime(e) - d0).days - target_dte))
        except Exception:
            return None

    def _dated_price(self, dt: pd.Timestamp, S: float, K: float, dte_days: int,
                     iv: float, opt_type: str,
                     real_expiry: Optional[str]) -> Tuple[float, bool]:
        """(price, is_real). Tries the historical store first; falls back to
        the BSM model so backtests never break on data gaps."""
        if self.store is not None and real_expiry:
            try:
                px = self.store.lookup(str(dt.date()), self.symbol, real_expiry,
                                       K, opt_type)
                if px is not None and px > 0:
                    return float(px), True
            except Exception:
                pass
        return self._leg_price(S, K, dte_days, iv, opt_type), False

    # ---------- Mode 1: Replay of the CURRENT position ----------
    def replay(self, positions: pd.DataFrame, start: date, end: date) -> Optional[pd.DataFrame]:
        """Daily modelled MTM & Greeks of the given legs between two dates,
        as if the position had been opened at each leg's AvgPrice on `start`.
        Supports forward (chronological) and backward scrub in the UI slider."""
        if self.spot_hist.empty:
            return None
        px = self.spot_hist.loc[str(start):str(end), 'Close']
        if px.empty:
            return None
        rows = []
        real_q = tot_q = 0
        for dt, S in px.items():
            mtm = delta = theta = vega = gamma = 0.0
            for _, leg in positions.iterrows():
                K = float(leg['Strike'])
                mult = float(leg.get('LotSize') or LOT_SIZES.get(str(leg['Symbol']).upper(), 1))
                aq = float(leg['Qty']) * mult * (1 if leg['Action'] == 'Buy' else -1)
                ot = 'C' if leg['Type'] == 'Call' else 'P'
                dte = (pd.to_datetime(leg['Expiry']) - dt).days
                iv_series = self._iv_path(px.index, float(leg['IV']) / 100)
                iv = float(iv_series.loc[dt])
                leg_exp = pd.to_datetime(leg['Expiry']).strftime("%Y-%m-%d")
                p, is_real = self._dated_price(dt, S, K, dte, iv, ot, leg_exp)
                real_q += int(is_real); tot_q += 1
                mtm += (p - float(leg['AvgPrice'])) * aq
                T = max(dte, 0) / 365.0
                if dte > 0:
                    delta += float(BlackScholesEngine.delta(S, K, T, self.r, iv, ot)) * aq
                    gamma += float(BlackScholesEngine.gamma(S, K, T, self.r, iv)) * aq
                    theta += float(BlackScholesEngine.theta(S, K, T, self.r, iv, ot)) * aq
                    vega += float(BlackScholesEngine.vega(S, K, T, self.r, iv)) * aq
            rows.append({'Date': dt, 'Spot': float(S), 'MTM': mtm, 'Delta': delta,
                         'Gamma': gamma, 'Theta': theta, 'Vega': vega})
        out = pd.DataFrame(rows).set_index('Date')
        out.attrs['real_pct'] = round(real_q / tot_q * 100, 1) if tot_q else 0.0
        return out

    # ---------- Mode 2: Expiry-cycle backtest of the strategy SHAPE ----------
    def cycle_backtest(self, positions: pd.DataFrame, lookback_days: int = 730,
                       hold_days: int = 30, sl_pct: float = 200.0,
                       tgt_pct: float = 50.0) -> Optional[dict]:
        """Re-enter the current strategy shape (strike offsets from ATM kept
        constant) every `hold_days`; exit at horizon, stop-loss or target.
        SL/target are % of entry credit (credit strategies) or debit.
        Returns trade log + equity curve + full performance stats."""
        if self.spot_hist.empty:
            return None
        px = self.spot_hist['Close'].iloc[-lookback_days:]
        if len(px) < hold_days + 5:
            return None
        step = STRIKE_STEP.get(self.symbol, 50)
        spot_now, _, _ = fetch_market_data(self.symbol)
        atm_now = round(spot_now / step) * step

        # Leg blueprint: PERCENTAGE offset from today's ATM (absolute point
        # offsets don't scale across years — a −1,500pt strike is far more OTM
        # at NIFTY 19,000 than at 25,000, which produced degenerate ~₹0
        # premiums in early history).
        blueprint = []
        for _, leg in positions.iterrows():
            mult = float(leg.get('LotSize') or LOT_SIZES.get(str(leg['Symbol']).upper(), 1))
            blueprint.append({
                'off_pct': (float(leg['Strike']) - atm_now) / max(spot_now, 1e-6),
                'ot': 'C' if leg['Type'] == 'Call' else 'P',
                'dirn': 1 if leg['Action'] == 'Buy' else -1,
                'aq': float(leg['Qty']) * mult * (1 if leg['Action'] == 'Buy' else -1),
                'iv0': float(leg['IV']) / 100,
            })

        iv_scale = self._iv_path(px.index, 1.0)   # VIX ratio path
        dates = px.index
        trades, equity, eq_dates = [], [], []
        cum = 0.0
        skipped = 0
        i = 0
        while i < len(dates) - 2:
            d0 = dates[i]
            S0 = float(px.iloc[i])
            real_exp = self._pick_real_expiry(d0, hold_days)
            legs = []
            entry_prem = 0.0   # signed: credit > 0
            t_real = t_tot = 0
            for b in blueprint:
                K = round(S0 * (1 + b['off_pct']) / step) * step
                iv = float(b['iv0'] * iv_scale.loc[d0])
                p0, is_real = self._dated_price(d0, S0, K, hold_days, iv,
                                                b['ot'], real_exp)
                t_real += int(is_real); t_tot += 1
                legs.append({**b, 'K': K, 'p0': p0})
                entry_prem += -p0 * b['aq']
            notional = sum(abs(l['p0'] * l['aq']) for l in legs)
            if notional < 250:
                # Degenerate entry: modelled premiums ~0 (strikes too far OTM
                # at this spot level / IV) — a ₹1 SL/target basis would then
                # trigger exits on noise. Skip and move to the next day.
                skipped += 1
                i += 1
                continue
            cost = notional * self.cost_bps / 1e4 * 2   # entry + exit
            # SL/target basis: entry credit/debit, floored at 25% of notional
            # so balanced (near-zero net premium) structures still get a
            # meaningful threshold instead of ₹1.
            basis = max(abs(entry_prem), 0.25 * notional, 1.0)

            exit_j, exit_pnl, reason = None, 0.0, "Horizon"
            j_end = min(i + hold_days, len(dates) - 1)
            for j in range(i + 1, j_end + 1):
                d, S = dates[j], float(px.iloc[j])
                dte = hold_days - (j - i)
                pnl = 0.0
                for l in legs:
                    iv = float(l['iv0'] * iv_scale.loc[d])
                    p, is_real = self._dated_price(d, S, l['K'], dte, iv,
                                                   l['ot'], real_exp)
                    t_real += int(is_real); t_tot += 1
                    pnl += (p - l['p0']) * l['aq']
                if pnl <= -basis * sl_pct / 100:
                    exit_j, exit_pnl, reason = j, pnl, "Stop-loss"
                    break
                if pnl >= basis * tgt_pct / 100:
                    exit_j, exit_pnl, reason = j, pnl, "Target"
                    break
                exit_j, exit_pnl = j, pnl
            cum += exit_pnl - cost
            data_src = ("Real" if t_tot and t_real == t_tot else
                        "Mixed" if t_real else "Model")
            trades.append({
                'Entry': d0.date(), 'Exit': dates[exit_j].date(),
                'Days': (dates[exit_j] - d0).days, 'EntrySpot': round(S0, 1),
                'ExitSpot': round(float(px.iloc[exit_j]), 1),
                'Premium': round(entry_prem, 0), 'Costs': round(cost, 0),
                'P&L': round(exit_pnl - cost, 0), 'Reason': reason,
                'Data': data_src,
            })
            equity.append(cum); eq_dates.append(dates[exit_j])
            i = exit_j + 1

        if not trades:
            return None
        tl = pd.DataFrame(trades)
        eq = pd.Series(equity, index=pd.DatetimeIndex(eq_dates), name='Equity')
        real_pct = round((tl['Data'] != 'Model').mean() * 100, 1)
        return {'trades': tl, 'equity': eq, 'stats': perf_stats(tl, eq),
                'real_pct': real_pct, 'skipped': skipped}


def perf_stats(trades: pd.DataFrame, equity: pd.Series) -> dict:
    """Institutional performance summary: win rate, PF, expectancy, drawdown,
    Sharpe/Sortino (per-trade returns annualized by trade frequency), CAGR."""
    pnl = trades['P&L'].astype(float)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    win_rate = len(wins) / len(pnl) * 100 if len(pnl) else 0
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float('inf')
    expectancy = pnl.mean() if len(pnl) else 0

    peak = equity.cummax()
    dd = equity - peak
    max_dd = float(dd.min()) if len(dd) else 0

    ret = pnl.values
    n_per_year = max(len(pnl) / max((equity.index[-1] - equity.index[0]).days / 365.25, 0.1), 1)
    sharpe = (ret.mean() / ret.std() * math.sqrt(n_per_year)) if ret.std() > 0 else 0
    downside = ret[ret < 0]
    sortino = (ret.mean() / downside.std() * math.sqrt(n_per_year)) if len(downside) > 1 and downside.std() > 0 else float('inf')

    yrs = max((equity.index[-1] - equity.index[0]).days / 365.25, 0.1)
    total = float(equity.iloc[-1])
    # CAGR on a notional base = |max drawdown| capital-at-risk proxy (avoids
    # needing an arbitrary starting capital); reported alongside absolute P&L.
    base = max(abs(max_dd) * 2, abs(total), 1)
    cagr = ((base + total) / base) ** (1 / yrs) - 1

    monthly = trades.copy()
    monthly['Month'] = pd.to_datetime(monthly['Exit']).dt.to_period('M').astype(str)
    monthly_ret = monthly.groupby('Month')['P&L'].sum()

    return {
        'Trades': len(pnl), 'Win Rate %': round(win_rate, 1),
        'Avg P&L': round(float(pnl.mean()), 0),
        'Avg Win': round(float(wins.mean()), 0) if len(wins) else 0,
        'Avg Loss': round(float(losses.mean()), 0) if len(losses) else 0,
        'Avg Hold (days)': round(float(trades['Days'].mean()), 1),
        'Profit Factor': round(float(pf), 2) if math.isfinite(pf) else float('inf'),
        'Expectancy': round(float(expectancy), 0),
        'Total P&L': round(total, 0), 'Max Drawdown': round(max_dd, 0),
        'Sharpe': round(float(sharpe), 2),
        'Sortino': round(float(sortino), 2) if math.isfinite(sortino) else float('inf'),
        'CAGR % (on risk base)': round(cagr * 100, 1),
        'monthly': monthly_ret,
    }


# ==============================================================================
# 11. TRADE JOURNAL (SQLite, auto-capture, Excel/CSV export)
# ==============================================================================
class TradeJournal:
    """Persistent journal. Saves the full position snapshot (legs JSON),
    portfolio Greeks, MTM, strategy name and notes. Screenshots aren't
    capturable server-side in Streamlit; the payoff-chart parameters are
    stored instead so the chart can be regenerated."""

    def __init__(self, path: str = JOURNAL_DB):
        self.path = path
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.path)

    def _init_db(self) -> None:
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS journal (
                id TEXT PRIMARY KEY, ts TEXT, symbol TEXT, strategy TEXT,
                action TEXT, spot REAL, mtm REAL, booked REAL, unbooked REAL,
                delta REAL, gamma REAL, theta REAL, vega REAL,
                pop REAL, max_profit REAL, max_loss REAL,
                legs_json TEXT, exit_reason TEXT, notes TEXT)""")

    def save(self, symbol: str, strategy: str, action: str, spot: float,
             metrics: dict, legs: List[dict], exit_reason: str = "",
             notes: str = "") -> str:
        rid = str(uuid.uuid4())[:8]
        with self._conn() as c:
            c.execute("INSERT INTO journal VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                rid, datetime.now().isoformat(timespec='seconds'), symbol,
                strategy, action, spot, metrics.get('total_mtm', 0),
                metrics.get('booked', 0), metrics.get('unbooked', 0),
                metrics.get('delta', 0), metrics.get('gamma', 0),
                metrics.get('theta', 0), metrics.get('vega', 0),
                metrics.get('pop', 0),
                metrics.get('max_profit', 0) if math.isfinite(metrics.get('max_profit', 0)) else None,
                metrics.get('max_loss', 0) if math.isfinite(metrics.get('max_loss', 0)) else None,
                json.dumps(legs, default=str), exit_reason, notes))
        log.info("journal saved %s %s", rid, strategy)
        return rid

    def load(self) -> pd.DataFrame:
        try:
            with self._conn() as c:
                return pd.read_sql("SELECT * FROM journal ORDER BY ts DESC", c)
        except Exception:
            return pd.DataFrame()

    def delete(self, rid: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM journal WHERE id=?", (rid,))

    @staticmethod
    def to_excel_bytes(df: pd.DataFrame) -> bytes:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as xw:
            df.to_excel(xw, index=False, sheet_name='Journal')
        return buf.getvalue()


# ==============================================================================
# 12. ANALYSE-SCREEN RENDERERS -- preserved from v2
# ==============================================================================
STRAT_INTENT = {
    "Short Strangle": ("Time decay (Theta)", "Range-bound / Neutral"),
    "Short Straddle": ("Time decay (Theta)", "Pinned / very Range-bound"),
    "Iron Condor": ("Time decay (Theta), defined risk", "Range-bound"),
    "Iron Fly": ("Time decay (Theta), defined risk", "Pinned near short strike"),
    "Iron Butterfly": ("Time decay (Theta), defined risk", "Pinned near short strike"),
    "Long Strangle": ("Volatility expansion (Vega/Gamma)", "Big move either side"),
    "Long Straddle": ("Volatility expansion (Vega/Gamma)", "Big move either side"),
    "Bull Call Spread": ("Directional move up, defined risk", "Moderately Bullish"),
    "Bear Put Spread": ("Directional move down, defined risk", "Moderately Bearish"),
    "Bull Put Spread": ("Time decay + direction", "Neutral to Bullish"),
    "Bear Call Spread": ("Time decay + direction", "Neutral to Bearish"),
    "Long Call": ("Directional move up (Delta)", "Bullish"),
    "Long Put": ("Directional move down (Delta)", "Bearish"),
    "Short Call": ("Time decay (Theta)", "Neutral to Bearish"),
    "Short Put": ("Time decay (Theta)", "Neutral to Bullish"),
    "Calendar Spread": ("Front-month decay vs back-month Vega", "Range-bound near strike"),
    "Diagonal Spread": ("Decay + direction", "Mildly directional"),
}


def instrument_name(row) -> str:
    try:
        exp = pd.to_datetime(row['Expiry']).strftime("%d %b").lstrip("0")
    except Exception:
        exp = str(row['Expiry'])
    ot = "CE" if row['Type'] == 'Call' else "PE"
    return f"{row['Symbol']} {exp} {fmt_inr(row['Strike'],0)} {ot}"


def metrics_strip(m: dict) -> None:
    mp = ("<span class='grn'>₹" + fmt_inr(m['max_profit']) + "</span>") \
        if math.isfinite(m['max_profit']) else "<span class='grn'>Unlimited</span>"
    ml = ("<span class='rd'>₹" + fmt_inr(abs(m['max_loss'])) + "</span>") \
        if math.isfinite(m['max_loss']) else "<span class='rd'>Unlimited</span>"
    rr = f"{m['reward_risk']:.2f} : 1" if m['reward_risk'] is not None else "—"
    tv_cls = 'grn' if m['time_value'] >= 0 else 'rd'
    bes = m['breakevens']
    be_txt = ",<br>".join(fmt_inr(b, 0) for b in bes[:2]) if len(bes) else "—"
    npv = m['net_premium']
    np_txt = (f"<span class='mut' style='font-size:.7rem'>{'Cr' if npv>=0 else 'Dr'}</span> "
              f"<span class='{'grn' if npv>=0 else 'rd'}'>₹{fmt_inr(abs(npv),0)}</span>")
    st.markdown(f"""
    <div class="mstrip">
      <div class="mcell"><div class="mlab">Max Profit</div><div class="mval">{mp}</div></div>
      <div class="mcell"><div class="mlab">Max Loss</div><div class="mval">{ml}</div></div>
      <div class="mcell"><div class="mlab">Reward : Risk</div><div class="mval">{rr}</div></div>
      <div class="mcell"><div class="mlab">POP</div><div class="mval">{m['pop']:.1f}%</div></div>
      <div class="mcell"><div class="mlab">Time Value</div>
           <div class="mval {tv_cls}">₹{fmt_inr(m['time_value'])}</div></div>
      <div class="mcell"><div class="mlab">Intrinsic Value</div>
           <div class="mval">₹{fmt_inr(m['intrinsic_value'])}</div></div>
      <div class="mcell"><div class="mlab">Breakeven</div><div class="mval">{be_txt}</div></div>
      <div class="mcell"><div class="mlab">Net Premium</div><div class="mval">{np_txt}</div></div>
    </div>""", unsafe_allow_html=True)


def render_payoff_chart(engine, metrics, days_elapsed, target_price, show_oi, oi_data):
    """Green/red expiry payoff + blue target-date curve + optional OI bars."""
    s_range = engine.price_range(width=0.11, n=300)
    exp = engine.payoff_expiry(s_range)
    tgt = engine.payoff_at_days(s_range, days_elapsed)
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if show_oi and oi_data is not None:
        strikes, ce, pe, _, _ = oi_data
        mask = (strikes >= s_range[0]) & (strikes <= s_range[-1])
        fig.add_trace(go.Bar(x=strikes[mask], y=ce[mask], name="Call OI",
                             marker_color="rgba(255,91,106,0.45)",
                             width=engine.spot * 0.004), secondary_y=True)
        fig.add_trace(go.Bar(x=strikes[mask], y=pe[mask], name="Put OI",
                             marker_color="rgba(33,206,153,0.45)",
                             width=engine.spot * 0.004), secondary_y=True)

    pos = np.where(exp >= 0, exp, np.nan)
    neg = np.where(exp < 0, exp, np.nan)
    fig.add_trace(go.Scatter(x=s_range, y=pos, mode='lines', name='On Expiry',
                             line=dict(color='#21ce99', width=2.4),
                             fill='tozeroy', fillcolor='rgba(33,206,153,0.12)'))
    fig.add_trace(go.Scatter(x=s_range, y=neg, mode='lines', showlegend=False,
                             line=dict(color='#ff5b6a', width=2.4),
                             fill='tozeroy', fillcolor='rgba(255,91,106,0.10)',
                             hoverinfo='skip'))
    lbl = "On Target Date" if days_elapsed > 0 else "On Target Date (T+0)"
    fig.add_trace(go.Scatter(x=s_range, y=tgt, mode='lines', name=lbl,
                             line=dict(color='#4da3ff', width=2.2)))

    fig.add_vline(x=engine.spot, line_dash="dash", line_color="#8b98a8",
                  annotation_text=f"Current price: {fmt_inr(engine.spot,0)}",
                  annotation_position="top", annotation_font_color="#4da3ff",
                  annotation_font_size=11)
    if target_price and abs(target_price - engine.spot) > 1e-9:
        fig.add_vline(x=target_price, line_dash="dot", line_color="#f7b731",
                      annotation_text=f"Target {fmt_inr(target_price,0)}",
                      annotation_position="bottom", annotation_font_size=10)
    for be in metrics['breakevens']:
        if s_range[0] <= be <= s_range[-1]:
            fig.add_vline(x=be, line_dash="dot", line_color="rgba(255,255,255,0.25)")
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.15)")
    fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", height=430,
                      margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1,
                                  xanchor="right", font=dict(size=11)),
                      barmode="overlay",
                      font=dict(family="IBM Plex Mono, monospace", size=11))
    fig.update_yaxes(tickformat="~s", secondary_y=False, gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(tickformat="~s", secondary_y=True, showgrid=False)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_greeks_tab(metrics, live_df):
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='oo-card'><div class='mlab'>Net Delta</div><div class='mval'>{metrics['delta']:.2f}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='oo-card'><div class='mlab'>Net Gamma</div><div class='mval'>{metrics['gamma']:.4f}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='oo-card'><div class='mlab'>Net Theta / day</div><div class='mval'>₹{fmt_inr(metrics['theta'])}</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='oo-card'><div class='mlab'>Net Vega</div><div class='mval'>₹{fmt_inr(metrics['vega'])}</div></div>", unsafe_allow_html=True)
    cols = ['Symbol', 'Strike', 'Type', 'Action', 'Qty', 'Delta', 'Gamma',
            'Theta', 'Vega', 'P_ITM', 'DTE']
    st.dataframe(live_df[cols], use_container_width=True, hide_index=True)
    fig = go.Figure()
    names = ["Delta", "Gamma×100", "Theta", "Vega"]
    vals = [metrics['delta'], metrics['gamma'] * 100, metrics['theta'], metrics['vega']]
    colors = ['#4da3ff', '#f7b731',
              '#ff5b6a' if metrics['theta'] < 0 else '#21ce99', '#21ce99']
    fig.add_trace(go.Bar(x=names, y=vals, marker_color=colors))
    fig.update_layout(template="plotly_dark", height=260,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_analysis_card(strat, outlook, live_df, metrics, symbol):
    driver, view = STRAT_INTENT.get(strat, ("Custom exposure", outlook))
    rows_html = ""
    for _, r in live_df.iterrows():
        side = str(r['Action']).upper()
        rows_html += (f"<tr><td>{r['Type']}</td><td>{fmt_inr(r['Strike'],0)}</td>"
                      f"<td>{side}</td>"
                      f"<td>{fmt_inr(float(r['Qty'])*(r.get('LotSize') or LOT_SIZES.get(symbol,1)),0)}</td>"
                      f"<td><span class='rd'>₹{fmt_inr(r['AvgPrice'])}</span></td>"
                      f"<td><span class='rd'>₹{fmt_inr(r['LTP'])}</span></td></tr>")
    bes = metrics['breakevens']
    be_txt = " and ".join(fmt_inr(b, 0) for b in bes[:2]) if len(bes) else "—"
    mp = "Unlimited" if not math.isfinite(metrics['max_profit']) else "₹" + fmt_inr(metrics['max_profit'])
    ml = "Unlimited" if not math.isfinite(metrics['max_loss']) else "₹" + fmt_inr(abs(metrics['max_loss']))
    theta_line = ("working in your favour — you earn roughly ₹" +
                  fmt_inr(abs(metrics['theta'])) + " per day") if metrics['theta'] > 0 else (
                  "working against you at ~₹" + fmt_inr(abs(metrics['theta'])) + " per day")
    st.markdown(f"""
    <div class="oo-card">
      <div class="ai-h">📊 {symbol} OPTIONS PORTFOLIO ANALYSIS</div>
      <div class="ai-sec">1. Strategy Name / Intent</div>
      <div class="ai-body">
        <b style="color:var(--amber)">{strat}</b><br>
        You are running a <b>{strat.lower()}</b> on <b>{symbol}</b>:
        <table><tr><th>Leg</th><th>Strike</th><th>Side</th><th>Qty</th>
        <th>Avg Price</th><th>Last Price</th></tr>{rows_html}</table>
        <b style="color:#5aa7ff">Intent &amp; Profit Profile</b>
        <ul style="margin-top:6px">
          <li><b>Primary profit driver:</b> <span style="color:var(--amber)">{driver}</span> — theta is currently {theta_line}.</li>
          <li><b>Market view:</b> <span style="color:var(--amber)">{view}</span> — you need {symbol} to behave accordingly into expiry.</li>
          <li><b>Breakevens:</b> {be_txt}. Beyond these levels the position loses money at expiry.</li>
          <li><b>Risk:</b> Max profit {mp}, max loss {ml}. Size positions and place stops accordingly.</li>
          <li><b>Vega:</b> You are <b>{'long' if metrics['vega']>0 else 'short'}</b> volatility — an IV {'drop hurts' if metrics['vega']>0 else 'spike hurts'} this position.</li>
        </ul>
        <span class="mut" style="font-size:.75rem">Rule-based analysis for education only — not investment advice.</span>
      </div>
    </div>""", unsafe_allow_html=True)


def position_row(row, live_row, key) -> bool:
    """Selectable position row: checkbox + badges + numbers (v2)."""
    c_chk, c_body = st.columns([0.06, 0.94])
    with c_chk:
        checked = st.checkbox("sel", value=st.session_state.selected.get(row['ID'], True),
                              key=f"chk_{key}", label_visibility="collapsed")
        st.session_state.selected[row['ID']] = checked
    badge = "<span class='badge-s'>S</span>" if row['Action'] == 'Sell' else "<span class='badge-b'>B</span>"
    booked = live_row['Booked'] if live_row is not None else 0.0
    unbooked = live_row['Unbooked'] if live_row is not None else 0.0
    pnl = booked + unbooked
    ltp = live_row['LTP'] if live_row is not None else float('nan')
    with c_body:
        st.markdown(f"""
        <div class="oo-card" style="padding:10px 14px; margin-bottom:8px;
             opacity:{1 if checked else .45};">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
              {badge}<span class='badge-nrml'>NRML</span>
              <span class="pos-name">{instrument_name(row)}</span>
              <div class="pos-meta">Lots <b>{int(row['Qty'])}</b> &nbsp;·&nbsp;
                Booked {pnl_span(booked)} &nbsp; Unbooked {pnl_span(unbooked)} &nbsp;
                P&amp;L {pnl_span(pnl)}</div>
            </div>
            <div style="text-align:right">
              <div class="pos-num mut">Avg <b style="color:var(--txt)">{fmt_inr(row['AvgPrice'])}</b></div>
              <div class="pos-num mut">LTP <b style="color:var(--txt)">{fmt_inr(ltp)}</b></div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
    return checked


# ==============================================================================
# 13. NEW MODULE RENDERERS
# ==============================================================================
def _dark_fig(fig, h=340):
    fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", height=h,
                      margin=dict(l=10, r=10, t=30, b=10),
                      font=dict(family="IBM Plex Mono, monospace", size=11))
    return fig


# --------------------------- OPTION CHAIN ------------------------------------
def render_chain_tab(symbol: str, r: float):
    chain = fetch_chain_raw(symbol)
    if not chain:
        st.warning("Live option chain unavailable — install `jugaad-data` and ensure "
                   "NSE connectivity, or add another chain provider in CHAIN_PROVIDERS.")
        return
    expiries = chain_expiries(chain)
    exp = st.selectbox("Expiry", expiries[:8], key="chain_exp")
    df = build_chain_dataframe(symbol, exp, r)
    if df is None or df.empty:
        st.warning("No chain rows for this expiry.")
        return
    a = chain_analytics(df)
    spot = df.attrs['spot']

    st.markdown(
        f"<span class='chip'>Spot <b>{fmt_inr(spot)}</b></span>"
        f"<span class='chip'>ATM <b>{fmt_inr(a['atm'],0)}</b></span>"
        f"<span class='chip'>PCR <b>{a['pcr']:.2f}</b></span>"
        f"<span class='chip'>Max Pain <b>{fmt_inr(a['max_pain'],0)}</b></span>"
        f"<span class='chip'>Hi Call OI <b class='rd'>{fmt_inr(a['hi_ce_oi'],0)}</b></span>"
        f"<span class='chip'>Hi Put OI <b class='grn'>{fmt_inr(a['hi_pe_oi'],0)}</b></span>"
        f"<span class='chip'>Hi ΔOI CE <b>{fmt_inr(a['hi_ce_chg'],0)}</b> / PE <b>{fmt_inr(a['hi_pe_chg'],0)}</b></span>",
        unsafe_allow_html=True)

    n_around = st.slider("Strikes around ATM", 5, 40, 15, key="chain_n")
    view = df[(df['Strike'] >= a['atm'] - n_around * (df['Strike'].diff().median() or 50)) &
              (df['Strike'] <= a['atm'] + n_around * (df['Strike'].diff().median() or 50))].copy()

    ce_cols = ['CE_OI', 'CE_ChgOI', 'CE_Vol', 'CE_IV', 'CE_Delta', 'CE_Gamma',
               'CE_Theta', 'CE_Vega', 'CE_Bid', 'CE_Ask', 'CE_Spread',
               'CE_Intr', 'CE_TimeVal', 'CE_LTP']
    pe_cols = ['PE_LTP', 'PE_Intr', 'PE_TimeVal', 'PE_Bid', 'PE_Ask', 'PE_Spread',
               'PE_Delta', 'PE_Gamma', 'PE_Theta', 'PE_Vega', 'PE_IV',
               'PE_Vol', 'PE_ChgOI', 'PE_OI']
    show = view[ce_cols + ['Strike'] + pe_cols]

    def style_row(row):
        css = [''] * len(row)
        k = row['Strike']
        if k == a['atm']:
            css = ['background-color: rgba(77,163,255,0.16)'] * len(row)
        # ITM shading: calls ITM below spot, puts ITM above spot
        for i, col in enumerate(show.columns):
            if col.startswith('CE') and k < spot:
                css[i] += ';background-color: rgba(33,206,153,0.06)'
            if col.startswith('PE') and k > spot:
                css[i] += ';background-color: rgba(33,206,153,0.06)'
        return css

    styler = (show.style.apply(style_row, axis=1)
              .format(precision=2, na_rep="—")
              .format({'Strike': lambda v: fmt_inr(v, 0)}))
    st.dataframe(styler, use_container_width=True, hide_index=True, height=520)
    st.caption("Blue row = ATM · green tint = ITM side · OTM untinted. "
               "Greeks computed from quoted IVs (vectorized BSM).")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=view['Strike'], y=view['CE_OI'], name='Call OI',
                         marker_color='rgba(255,91,106,0.7)'))
    fig.add_trace(go.Bar(x=view['Strike'], y=view['PE_OI'], name='Put OI',
                         marker_color='rgba(33,206,153,0.7)'))
    fig.add_vline(x=spot, line_dash='dash', line_color='#8b98a8',
                  annotation_text='Spot')
    fig.add_vline(x=a['max_pain'], line_dash='dot', line_color='#f7b731',
                  annotation_text='Max Pain')
    st.plotly_chart(_dark_fig(fig, 300), use_container_width=True,
                    config={'displayModeBar': False})


# --------------------------- VOLATILITY --------------------------------------
def render_vol_tab(symbol: str, r: float, current_iv: float):
    hist = fetch_spot_history(symbol, "2y")
    vix = fetch_vix_history("1y")
    if hist.empty:
        st.warning("No price history available for volatility analytics.")
        return

    hv20 = VolAnalytics.realized_vol(hist['Close'], 20)
    hv10 = VolAnalytics.realized_vol(hist['Close'], 10)
    cur_vix, iv_rank, iv_pct = VolAnalytics.iv_rank_percentile(
        vix['Close'] if not vix.empty else pd.Series(dtype=float))

    c = st.columns(5)
    cards = [("ATM IV", f"{current_iv*100:.1f}%"), ("HV (20d)", f"{hv20:.1f}%"),
             ("HV (10d)", f"{hv10:.1f}%"),
             ("IV Rank (VIX 1y)", f"{iv_rank:.0f}" if math.isfinite(iv_rank) else "—"),
             ("IV Percentile", f"{iv_pct:.0f}%" if math.isfinite(iv_pct) else "—")]
    for col, (lab, val) in zip(c, cards):
        col.markdown(f"<div class='oo-card'><div class='mlab'>{lab}</div>"
                     f"<div class='mval'>{val}</div></div>", unsafe_allow_html=True)

    # IV (VIX) vs realized vol
    fig = go.Figure()
    hv_s = VolAnalytics.hv_series(hist['Close'], 20).dropna()
    fig.add_trace(go.Scatter(x=hv_s.index, y=hv_s, name='HV 20d',
                             line=dict(color='#4da3ff', width=1.8)))
    if not vix.empty:
        fig.add_trace(go.Scatter(x=vix.index, y=vix['Close'], name='India VIX',
                                 line=dict(color='#f7b731', width=1.8)))
    fig.update_layout(title="Implied (India VIX) vs Realized Volatility")
    st.plotly_chart(_dark_fig(fig, 300), use_container_width=True,
                    config={'displayModeBar': False})

    col1, col2 = st.columns(2)
    with col1:  # Volatility cone
        cone = VolAnalytics.vol_cone(hist['Close'])
        if not cone.empty:
            fig = go.Figure()
            for name, color in [('Max', 'rgba(255,91,106,.5)'), ('P75', 'rgba(247,183,49,.6)'),
                                ('Median', '#8b98a8'), ('P25', 'rgba(33,206,153,.6)'),
                                ('Min', 'rgba(77,163,255,.5)')]:
                fig.add_trace(go.Scatter(x=cone['Window'], y=cone[name], name=name,
                                         line=dict(color=color, width=1.4)))
            fig.add_trace(go.Scatter(x=cone['Window'], y=cone['Current'], name='Current',
                                     mode='markers+lines',
                                     line=dict(color='#21ce99', width=2.4)))
            fig.update_layout(title="Volatility Cone (realized, by window)")
            st.plotly_chart(_dark_fig(fig, 320), use_container_width=True,
                            config={'displayModeBar': False})
    with col2:  # Term structure
        ts = VolAnalytics.term_structure(symbol, r)
        if not ts.empty:
            fig = go.Figure(go.Scatter(x=ts['DTE'], y=ts['ATM_IV'],
                                       mode='lines+markers',
                                       line=dict(color='#4da3ff', width=2.2)))
            fig.update_layout(title="IV Term Structure (ATM IV vs DTE)",
                              xaxis_title="DTE", yaxis_title="IV %")
            st.plotly_chart(_dark_fig(fig, 320), use_container_width=True,
                            config={'displayModeBar': False})
        else:
            st.info("Term structure needs the live chain (jugaad-data).")

    # Smile + skew + surface from the live chain
    chain = fetch_chain_raw(symbol)
    if chain:
        exps = chain_expiries(chain)[:4]
        exp = st.selectbox("Smile expiry", exps, key="smile_exp")
        cdf = build_chain_dataframe(symbol, exp, r)
        if cdf is not None:
            spot = cdf.attrs['spot']
            sm = VolAnalytics.smile(cdf)
            sm = sm[(sm['Strike'] > spot * 0.9) & (sm['Strike'] < spot * 1.1)]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sm['Strike'], y=sm['CE_IV'], name='Call IV',
                                     line=dict(color='#ff5b6a', width=1.8)))
            fig.add_trace(go.Scatter(x=sm['Strike'], y=sm['PE_IV'], name='Put IV',
                                     line=dict(color='#21ce99', width=1.8)))
            fig.add_vline(x=spot, line_dash='dash', line_color='#8b98a8')
            fig.update_layout(title=f"IV Smile — {exp} (put-side skew = downside fear)")
            st.plotly_chart(_dark_fig(fig, 300), use_container_width=True,
                            config={'displayModeBar': False})

        if len(exps) >= 2 and st.toggle("Show IV surface (3D)", value=False):
            zs, ys = [], []
            xs = None
            for e in exps:
                d = build_chain_dataframe(symbol, e, r)
                if d is None:
                    continue
                spot = d.attrs['spot']
                d = d[(d['Strike'] > spot * 0.92) & (d['Strike'] < spot * 1.08)]
                iv = d[['CE_IV', 'PE_IV']].mean(axis=1)
                if xs is None:
                    xs = d['Strike'].values
                zs.append(np.interp(xs, d['Strike'].values, iv.values))
                ys.append(max((pd.to_datetime(e) - datetime.now()).days, 0))
            if xs is not None and len(zs) >= 2:
                fig = go.Figure(go.Surface(x=xs, y=ys, z=np.array(zs),
                                           colorscale='Viridis'))
                fig.update_layout(title="IV Surface",
                                  scene=dict(xaxis_title='Strike', yaxis_title='DTE',
                                             zaxis_title='IV %'))
                st.plotly_chart(_dark_fig(fig, 480), use_container_width=True)


# --------------------------- PROBABILITY -------------------------------------
def render_prob_tab(engine: PortfolioRiskEngine, metrics: dict, iv: float):
    dte = max(int(metrics['dte']), 1)
    em1 = ProbAnalytics.expected_move(engine.spot, iv, dte)
    c = st.columns(5)
    for col, (lab, val) in zip(c, [
            ("POP (lognormal)", f"{metrics['pop']:.1f}%"),
            ("Expected Move", f"±₹{fmt_inr(em1,0)}"),
            ("1 SD Range", f"{fmt_inr(engine.spot-em1,0)} – {fmt_inr(engine.spot+em1,0)}"),
            ("2 SD Range", f"{fmt_inr(engine.spot-2*em1,0)} – {fmt_inr(engine.spot+2*em1,0)}"),
            ("DTE", f"{dte}d")]):
        col.markdown(f"<div class='oo-card'><div class='mlab'>{lab}</div>"
                     f"<div class='mval' style='font-size:.82rem'>{val}</div></div>",
                     unsafe_allow_html=True)

    # Per-leg P(ITM)/P(OTM)/P(touch)
    ldf = engine.live[['Symbol', 'Strike', 'Type', 'Action', 'P_ITM']].copy()
    ldf['P_OTM'] = (100 - ldf['P_ITM']).round(1)
    ldf['P_Touch'] = ldf['P_ITM'].apply(ProbAnalytics.prob_touch).round(1)
    st.dataframe(ldf, use_container_width=True, hide_index=True)

    if st.button("🎲 Run Monte Carlo (5,000 GBM paths)"):
        st.session_state['mc'] = ProbAnalytics.monte_carlo(engine, iv)
    mc = st.session_state.get('mc')
    if mc:
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='oo-card'><div class='mlab'>MC POP</div><div class='mval'>{mc['pop_mc']:.1f}%</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='oo-card'><div class='mlab'>Expected P&L</div><div class='mval'>{pnl_span(mc['exp_pnl'], False, 0)}</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='oo-card'><div class='mlab'>P5 / P95</div><div class='mval' style='font-size:.8rem'>{fmt_compact(mc['p5'])} / {fmt_compact(mc['p95'])}</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='oo-card'><div class='mlab'>CVaR (5%)</div><div class='mval rd'>{fmt_compact(mc['cvar5'])}</div></div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Histogram(x=mc['prices'], nbinsx=80,
                                         marker_color='rgba(77,163,255,.7)'))
            for k, ls in [(engine.spot - em1, 'dot'), (engine.spot + em1, 'dot'),
                          (engine.spot - 2 * em1, 'dash'), (engine.spot + 2 * em1, 'dash')]:
                fig.add_vline(x=k, line_dash=ls, line_color='#f7b731')
            fig.update_layout(title="Terminal price distribution (±1SD/±2SD)")
            st.plotly_chart(_dark_fig(fig, 300), use_container_width=True,
                            config={'displayModeBar': False})
        with col2:
            colors = np.where(np.sort(mc['pnl']) >= 0, 'rgba(33,206,153,.75)',
                              'rgba(255,91,106,.75)')
            fig = go.Figure(go.Histogram(x=mc['pnl'], nbinsx=80,
                                         marker_color='rgba(33,206,153,.7)'))
            fig.add_vline(x=0, line_color='#8b98a8')
            fig.update_layout(title="P&L distribution at expiry")
            st.plotly_chart(_dark_fig(fig, 300), use_container_width=True,
                            config={'displayModeBar': False})


# --------------------------- RISK MANAGEMENT ---------------------------------
def render_risk_tab(engine: PortfolioRiskEngine, metrics: dict):
    st.markdown("##### Stress Matrix — P&L across Spot % × IV shift (T+0 reprice)")
    sm = engine.stress_matrix()
    fig = go.Figure(go.Heatmap(
        z=sm.values, x=sm.columns, y=sm.index,
        colorscale=[[0, '#ff5b6a'], [0.5, '#10151f'], [1, '#21ce99']],
        zmid=0, text=np.vectorize(fmt_compact)(sm.values),
        texttemplate="%{text}", textfont=dict(size=10)))
    st.plotly_chart(_dark_fig(fig, 300), use_container_width=True,
                    config={'displayModeBar': False})

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Gap Risk (overnight scenarios)")
        st.dataframe(engine.gap_analysis(), use_container_width=True, hide_index=True)
    with col2:
        st.markdown("##### Position Sizing")
        cap = st.number_input("Trading capital (₹)", value=1_000_000, step=50_000)
        risk_pct = st.number_input("Risk per trade (%)", value=2.0, step=0.5)
        risk_amt = cap * risk_pct / 100
        ml = abs(metrics['max_loss']) if math.isfinite(metrics['max_loss']) else None
        if ml and ml > 0:
            st.markdown(f"Defined max loss/position: **₹{fmt_inr(ml,0)}** → "
                        f"suggested size: **{max(int(risk_amt // ml),0)}×** current position "
                        f"(risking ₹{fmt_inr(risk_amt,0)}).")
        else:
            st.markdown(f"Undefined max loss — size by margin: capital at risk "
                        f"₹{fmt_inr(risk_amt,0)} vs est. margin "
                        f"₹{fmt_inr(metrics['capital'],0)}. Use hard stop-losses.")
        # Kelly from POP + reward:risk
        p = metrics['pop'] / 100
        b = metrics['reward_risk'] or 0
        if b > 0:
            kelly = max(p - (1 - p) / b, 0)
            st.markdown(f"**Kelly fraction:** {kelly*100:.1f}% of capital "
                        f"(half-Kelly {kelly*50:.1f}% recommended).")
        util = metrics['capital'] / cap * 100 if cap else 0
        st.progress(min(util / 100, 1.0),
                    text=f"Margin utilization ≈ {util:.1f}% (est. ₹{fmt_inr(metrics['capital'],0)})")

    st.markdown("##### Stop-loss / Target suggestions (heuristics)")
    npv = metrics['net_premium']
    if npv > 0:   # net credit
        st.markdown(f"- Credit received **₹{fmt_inr(npv,0)}** → common rules: "
                    f"target **50%** of credit (₹{fmt_inr(npv*0.5,0)}), "
                    f"stop at **2×** credit loss (−₹{fmt_inr(npv*2,0)}).")
    else:
        st.markdown(f"- Debit paid **₹{fmt_inr(abs(npv),0)}** → common rules: "
                    f"target **100%** gain, stop at **50%** of debit "
                    f"(−₹{fmt_inr(abs(npv)*0.5,0)}).")
    st.caption("Heuristic rules, not advice — align with your own system's exits.")

    st.markdown("##### Exposure Breakdown")
    for name, tbl in engine.exposure_tables().items():
        with st.expander(name):
            st.dataframe(tbl, use_container_width=True)


# ==============================================================================
# 15. ADVANCED GREEKS (second/third order, closed-form BSM, q=0)
# ==============================================================================
class AdvancedGreeks:
    """Vanna, Charm, Vomma, Speed, Color, Zomma. Conventions:
    vanna per 1 vol-pt, charm/color per day, vomma per 1 vol-pt."""

    @staticmethod
    def _d(S, K, T, r, sigma):
        T = max(T, 1e-5); sigma = max(sigma, 1e-4)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return d1, d1 - sigma * math.sqrt(T), T, sigma

    @classmethod
    def vanna(cls, S, K, T, r, sigma):
        d1, d2, T, sigma = cls._d(S, K, T, r, sigma)
        return -norm.pdf(d1) * d2 / sigma / 100          # dDelta / d(1 vol-pt)

    @classmethod
    def charm(cls, S, K, T, r, sigma, opt_type='C'):
        d1, d2, T, sigma = cls._d(S, K, T, r, sigma)
        c = -norm.pdf(d1) * (2 * r * T - d2 * sigma * math.sqrt(T)) \
            / (2 * T * sigma * math.sqrt(T))
        return c / 365                                   # dDelta / day (call==put, q=0)

    @classmethod
    def vomma(cls, S, K, T, r, sigma):
        d1, d2, T, sigma = cls._d(S, K, T, r, sigma)
        vega = S * norm.pdf(d1) * math.sqrt(T)
        return vega * d1 * d2 / sigma / 1e4              # dVega(per pt) / d(1 vol-pt)

    @classmethod
    def speed(cls, S, K, T, r, sigma):
        d1, _, T, sigma = cls._d(S, K, T, r, sigma)
        gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
        return -gamma / S * (d1 / (sigma * math.sqrt(T)) + 1)   # dGamma/dSpot

    @classmethod
    def zomma(cls, S, K, T, r, sigma):
        d1, d2, T, sigma = cls._d(S, K, T, r, sigma)
        gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
        return gamma * (d1 * d2 - 1) / sigma / 100       # dGamma / d(1 vol-pt)

    @classmethod
    def color(cls, S, K, T, r, sigma):
        d1, d2, T, sigma = cls._d(S, K, T, r, sigma)
        term = 1 + (2 * r * T - d2 * sigma * math.sqrt(T)) / (sigma * math.sqrt(T)) * d1
        return -norm.pdf(d1) / (2 * S * T * sigma * math.sqrt(T)) * term / 365  # dGamma/day


def second_order_table(engine: 'PortfolioRiskEngine') -> pd.DataFrame:
    """Per-leg and portfolio second/third-order Greeks (position-scaled).
    Computed lazily - only when the Greeks tab is rendered."""
    rows = []
    S, r = engine.spot, engine.r
    for l in engine.legs:
        K, vol, aq = l['K'], l['vol'], l['actual_qty']
        T = max(l['days'] / 365.0, 0.001)
        rows.append({
            'Strike': K, 'Type': l['row']['Type'], 'Action': l['row']['Action'],
            'Vanna': AdvancedGreeks.vanna(S, K, T, r, vol) * aq,
            'Charm': AdvancedGreeks.charm(S, K, T, r, vol) * aq,
            'Vomma': AdvancedGreeks.vomma(S, K, T, r, vol) * aq,
            'Speed': AdvancedGreeks.speed(S, K, T, r, vol) * aq,
            'Color': AdvancedGreeks.color(S, K, T, r, vol) * aq,
            'Zomma': AdvancedGreeks.zomma(S, K, T, r, vol) * aq,
        })
    df = pd.DataFrame(rows)
    total = df.drop(columns=['Strike', 'Type', 'Action']).sum()
    total_row = {'Strike': '', 'Type': 'TOTAL', 'Action': '', **total.to_dict()}
    return pd.concat([df, pd.DataFrame([total_row])], ignore_index=True).round(5)


# ==============================================================================
# 16. GAMMA EXPOSURE (GEX) & DEALER POSITIONING
#     Standard convention (SpotGamma-style): dealers are assumed LONG calls
#     and SHORT puts vs customer flow -> call gamma positive, put gamma
#     negative. An assumption, not observed positioning - stated in the UI.
# ==============================================================================
def gex_profile(chain_df: pd.DataFrame, lot_size: int, r: float = 0.065) -> dict:
    """Per-strike dealer gamma exposure (₹ per 1% spot move), total GEX and
    the gamma-flip level (cumulative GEX zero crossing)."""
    spot = chain_df.attrs['spot']
    expiry = chain_df.attrs['expiry']
    T = max((pd.to_datetime(expiry) - datetime.now()).days, 1) / 365.0
    K = chain_df['Strike'].values
    atm_iv = float(np.nanmedian(pd.concat([chain_df['CE_IV'], chain_df['PE_IV']])))
    if not math.isfinite(atm_iv) or atm_iv <= 0:       # NaN is truthy — explicit guard
        atm_iv = 15.0
    civ = np.nan_to_num(chain_df['CE_IV'].values, nan=atm_iv) / 100
    piv = np.nan_to_num(chain_df['PE_IV'].values, nan=atm_iv) / 100

    g_c = BlackScholesEngine.gamma(spot, K, T, r, civ)
    g_p = BlackScholesEngine.gamma(spot, K, T, r, piv)
    # ₹ gamma exposure per 1% move: gamma * OI * lot * S^2 * 1%
    gex_c = g_c * chain_df['CE_OI'].values * lot_size * spot ** 2 * 0.01
    gex_p = -g_p * chain_df['PE_OI'].values * lot_size * spot ** 2 * 0.01
    net = gex_c + gex_p

    cum = np.cumsum(net)
    flip = None
    sign = np.sign(cum)
    for i in range(1, len(cum)):
        if sign[i] != sign[i - 1] and sign[i - 1] != 0:
            flip = float(K[i])
            break
    return {'strikes': K, 'gex_call': gex_c, 'gex_put': gex_p, 'net': net,
            'total': float(net.sum()), 'flip': flip, 'spot': spot}


def render_gex_section(symbol: str, r: float):
    chain = fetch_chain_raw(symbol)
    if not chain:
        return
    exps = chain_expiries(chain)
    if not exps:
        return
    st.markdown("---")
    st.markdown("##### 🎯 Gamma Exposure (GEX) & Dealer Positioning")
    exp = st.selectbox("GEX expiry", exps[:4], key="gex_exp")
    cdf = build_chain_dataframe(symbol, exp, r)
    if cdf is None:
        return
    lot = LOT_SIZES.get(symbol, 1)
    g = gex_profile(cdf, lot, r)
    regime = ("🟢 LONG gamma — dealers dampen moves (mean-reversion regime)"
              if g['total'] > 0 else
              "🔴 SHORT gamma — dealer hedging amplifies moves (trend/whipsaw regime)")
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='oo-card'><div class='mlab'>Total GEX / 1% move</div>"
                f"<div class='mval'>₹{fmt_compact(g['total'])}</div></div>",
                unsafe_allow_html=True)
    c2.markdown(f"<div class='oo-card'><div class='mlab'>Gamma Flip</div>"
                f"<div class='mval'>{fmt_inr(g['flip'],0) if g['flip'] else '—'}</div></div>",
                unsafe_allow_html=True)
    c3.markdown(f"<div class='oo-card'><div class='mlab'>Dealer Regime</div>"
                f"<div class='mval' style='font-size:.72rem'>{regime}</div></div>",
                unsafe_allow_html=True)

    mask = (g['strikes'] > g['spot'] * 0.94) & (g['strikes'] < g['spot'] * 1.06)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=g['strikes'][mask], y=g['gex_call'][mask], name='Call GEX',
                         marker_color='rgba(33,206,153,.7)'))
    fig.add_trace(go.Bar(x=g['strikes'][mask], y=g['gex_put'][mask], name='Put GEX',
                         marker_color='rgba(255,91,106,.7)'))
    fig.add_trace(go.Scatter(x=g['strikes'][mask], y=g['net'][mask], name='Net GEX',
                             line=dict(color='#4da3ff', width=2)))
    fig.add_vline(x=g['spot'], line_dash='dash', line_color='#8b98a8',
                  annotation_text='Spot')
    if g['flip']:
        fig.add_vline(x=g['flip'], line_dash='dot', line_color='#f7b731',
                      annotation_text='Flip')
    fig.update_layout(barmode='relative', title="Dealer GEX by strike (₹ per 1% move)")
    st.plotly_chart(_dark_fig(fig, 330), use_container_width=True,
                    config={'displayModeBar': False})
    st.caption("Assumes dealers long calls / short puts vs customer flow "
               "(standard GEX convention) — an inference from OI, not observed "
               "dealer books.")


# ==============================================================================
# 17. VOLATILITY FORECASTING (EWMA, GARCH(1,1) MLE, VIX regime detection)
# ==============================================================================
class VolForecaster:
    @staticmethod
    def ewma_vol(close: pd.Series, lam: float = 0.94) -> float:
        """RiskMetrics EWMA annualized vol (%) - lambda 0.94 daily standard."""
        ret = np.log(close / close.shift(1)).dropna().values
        if len(ret) < 30:
            return float('nan')
        var = ret[0] ** 2
        for x in ret[1:]:
            var = lam * var + (1 - lam) * x ** 2
        return math.sqrt(var * 252) * 100

    @staticmethod
    def garch11(close: pd.Series, horizon: int = 10) -> Optional[dict]:
        """GARCH(1,1) fitted by MLE (scipy). Returns params, current sigma and
        an n-day forecast path (annualized %). None if fit fails."""
        from scipy.optimize import minimize
        ret = np.log(close / close.shift(1)).dropna().values
        ret = ret - ret.mean()
        if len(ret) < 250:
            return None
        var0 = ret.var()

        def neg_ll(p):
            w, a, b = p
            if w <= 0 or a < 0 or b < 0 or a + b >= 0.999:
                return 1e9
            var = var0
            ll = 0.0
            for x in ret:
                ll += math.log(var) + x * x / var
                var = w + a * x * x + b * var
            return ll

        try:
            res = minimize(neg_ll, x0=[var0 * 0.05, 0.08, 0.90],
                           method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-10})
            w, a, b = res.x
            if not res.success and res.fun >= 1e9:
                return None
            var = var0
            for x in ret:
                var = w + a * x * x + b * var
            lt_var = w / (1 - a - b)
            path = []
            v = var
            for _ in range(horizon):
                path.append(math.sqrt(v * 252) * 100)
                v = w + (a + b) * v
            return {'omega': w, 'alpha': a, 'beta': b,
                    'persistence': a + b,
                    'sigma_now': math.sqrt(var * 252) * 100,
                    'sigma_lt': math.sqrt(lt_var * 252) * 100,
                    'forecast': path}
        except Exception as e:
            log.warning("GARCH fit failed: %s", e)
            return None

    @staticmethod
    def vix_regime(vix: pd.Series, smooth: int = 5) -> Optional[dict]:
        """3-state vol regime from VIX terciles with persistence smoothing.
        (Simple, transparent alternative to an HMM - states: Low/Normal/High.)"""
        v = vix.dropna()
        if len(v) < 120:
            return None
        q1, q2 = v.quantile(1 / 3), v.quantile(2 / 3)
        sm = pd.cut(v, [-np.inf, q1, q2, np.inf], labels=['Low', 'Normal', 'High'])
        # Smoothing: majority vote over the last `smooth` observations (rolling
        # mode on categoricals is unreliable in pandas, so it isn't used).
        recent = sm.iloc[-smooth:].astype(str)
        cur = recent.mode().iloc[0]
        # Consecutive raw days matching the smoothed state (>=1: the majority
        # state can differ from the very last raw observation)
        days_in = max(int((sm.astype(str)[::-1] == cur).cummin().sum()), 1)
        return {'state': cur, 'q_low': float(q1), 'q_high': float(q2),
                'current_vix': float(v.iloc[-1]), 'days_in_state': days_in,
                'series': sm.astype(str)}


def render_vol_forecast_section(symbol: str):
    st.markdown("---")
    st.markdown("##### 🔮 Volatility Forecasting")
    hist = fetch_spot_history(symbol, "3y")
    vix = fetch_vix_history("2y")
    if hist.empty:
        st.info("Needs price history.")
        return
    ewma = VolForecaster.ewma_vol(hist['Close'])
    reg = VolForecaster.vix_regime(vix['Close']) if not vix.empty else None

    if st.button("Fit GARCH(1,1) — MLE (takes a few seconds)"):
        with st.spinner("Fitting GARCH…"):
            st.session_state['garch'] = VolForecaster.garch11(hist['Close'], horizon=20)
    g = st.session_state.get('garch')

    c = st.columns(4)
    cards = [("EWMA vol (λ=0.94)", f"{ewma:.1f}%"),
             ("GARCH σ (now)", f"{g['sigma_now']:.1f}%" if g else "run fit →"),
             ("GARCH long-run σ", f"{g['sigma_lt']:.1f}%" if g else "—"),
             ("VIX Regime", f"{reg['state']} ({reg['days_in_state']}d)" if reg else "—")]
    for col, (lab, val) in zip(c, cards):
        col.markdown(f"<div class='oo-card'><div class='mlab'>{lab}</div>"
                     f"<div class='mval' style='font-size:.85rem'>{val}</div></div>",
                     unsafe_allow_html=True)
    if g:
        st.caption(f"GARCH params: ω={g['omega']:.2e}, α={g['alpha']:.3f}, "
                   f"β={g['beta']:.3f}, persistence={g['persistence']:.3f}")
        fig = go.Figure(go.Scatter(y=g['forecast'], mode='lines+markers',
                                   line=dict(color='#f7b731', width=2)))
        fig.add_hline(y=g['sigma_lt'], line_dash='dot', line_color='#8b98a8',
                      annotation_text='long-run')
        fig.update_layout(title="GARCH(1,1) 20-day vol forecast (annualized %)",
                          xaxis_title="days ahead")
        st.plotly_chart(_dark_fig(fig, 280), use_container_width=True,
                        config={'displayModeBar': False})
    if reg is not None:
        st.caption(f"Regime terciles: Low < {reg['q_low']:.1f} < Normal < "
                   f"{reg['q_high']:.1f} < High · current VIX {reg['current_vix']:.1f}. "
                   "Transparent tercile+persistence method (not a fitted HMM).")


# ==============================================================================
# 18. HISTORICAL OPTION CHAIN DATABASE (NSE F&O bhavcopy -> SQLite)
#     Real EOD option premiums. Downloader supports both bhavcopy formats:
#     legacy (INSTRUMENT/SYMBOL/EXPIRY_DT/...) via jugaad-data, and the new
#     UDiFF format (TckrSymb/XpryDt/StrkPric/...) via direct NSE archive URL.
#     CSV/zip files can also be imported manually (broker EOD dumps etc).
# ==============================================================================
class BhavcopyStore:
    DB = "openoptions_history.db"

    def __init__(self, path: str = None):
        self.path = path or self.DB
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS fo_eod (
                trade_date TEXT, symbol TEXT, expiry TEXT, strike REAL,
                opt_type TEXT, close REAL, settle REAL, oi REAL, volume REAL,
                PRIMARY KEY (trade_date, symbol, expiry, strike, opt_type))""")
            c.execute("CREATE INDEX IF NOT EXISTS idx_fo ON fo_eod(symbol, trade_date)")

    def _conn(self):
        return sqlite3.connect(self.path)

    # ------------------------- ingestion -------------------------
    @staticmethod
    def _normalize(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Map either bhavcopy format to the canonical schema."""
        cols = {c.strip().upper(): c for c in df.columns}
        if 'INSTRUMENT' in cols:                       # legacy format
            d = df[df[cols['INSTRUMENT']].astype(str).str.startswith('OPT')].copy()
            if d.empty:
                return None
            out = pd.DataFrame({
                'trade_date': pd.to_datetime(d[cols['TIMESTAMP']]).dt.strftime('%Y-%m-%d'),
                'symbol': d[cols['SYMBOL']].astype(str).str.upper(),
                'expiry': pd.to_datetime(d[cols['EXPIRY_DT']]).dt.strftime('%Y-%m-%d'),
                'strike': pd.to_numeric(d[cols['STRIKE_PR']], errors='coerce'),
                'opt_type': d[cols['OPTION_TYP']].astype(str).str[0].str.upper(),
                'close': pd.to_numeric(d[cols['CLOSE']], errors='coerce'),
                'settle': pd.to_numeric(d[cols.get('SETTLE_PR', cols['CLOSE'])], errors='coerce'),
                'oi': pd.to_numeric(d[cols.get('OPEN_INT', cols['CLOSE'])], errors='coerce'),
                'volume': pd.to_numeric(d[cols.get('CONTRACTS', cols['CLOSE'])], errors='coerce'),
            })
        elif 'TCKRSYMB' in cols:                       # new UDiFF format
            fin = cols.get('FININSTRMTP')
            d = df.copy()
            if fin:
                d = d[d[fin].astype(str).str.upper().str.contains('IDO|STO|OPT', na=False)]
            ot_col = cols.get('OPTNTP')
            if ot_col is None or d.empty:
                return None
            out = pd.DataFrame({
                'trade_date': pd.to_datetime(d[cols['TRADDT']]).dt.strftime('%Y-%m-%d'),
                'symbol': d[cols['TCKRSYMB']].astype(str).str.upper(),
                'expiry': pd.to_datetime(d[cols['XPRYDT']]).dt.strftime('%Y-%m-%d'),
                'strike': pd.to_numeric(d[cols['STRKPRIC']], errors='coerce'),
                'opt_type': d[ot_col].astype(str).str[0].str.upper(),
                'close': pd.to_numeric(d[cols['CLSPRIC']], errors='coerce'),
                'settle': pd.to_numeric(d[cols.get('STTLMPRIC', cols['CLSPRIC'])], errors='coerce'),
                'oi': pd.to_numeric(d[cols.get('OPNINTRST', cols['CLSPRIC'])], errors='coerce'),
                'volume': pd.to_numeric(d[cols.get('TTLTRADGVOL', cols['CLSPRIC'])], errors='coerce'),
            })
        else:
            return None
        out = out.dropna(subset=['strike', 'close'])
        out = out[out['opt_type'].isin(['C', 'P'])]
        return out

    def ingest_dataframe(self, df: pd.DataFrame) -> int:
        norm = self._normalize(df)
        if norm is None or norm.empty:
            return 0
        with self._conn() as c:
            norm.to_sql('_stage', c, if_exists='replace', index=False)
            c.execute("""INSERT OR REPLACE INTO fo_eod
                         SELECT trade_date,symbol,expiry,strike,opt_type,
                                close,settle,oi,volume FROM _stage""")
            c.execute("DROP TABLE _stage")
        return len(norm)

    def ingest_file(self, file_obj) -> int:
        """CSV or zip-of-CSV upload (both bhavcopy formats auto-detected)."""
        import zipfile
        try:
            name = getattr(file_obj, 'name', '')
            if str(name).endswith('.zip'):
                with zipfile.ZipFile(file_obj) as z:
                    inner = [n for n in z.namelist() if n.lower().endswith('.csv')]
                    total = 0
                    for n in inner:
                        total += self.ingest_dataframe(pd.read_csv(z.open(n)))
                    return total
            return self.ingest_dataframe(pd.read_csv(file_obj))
        except Exception as e:
            log.warning("ingest failed: %s", e)
            return 0

    def download_day(self, d: date) -> int:
        """Fetch one trading day's F&O bhavcopy. Tries jugaad-data (legacy),
        then the NSE UDiFF archive URL. Returns rows ingested (0 = no data,
        holiday, or blocked network)."""
        # Provider 1: jugaad-data
        try:
            from jugaad_data.nse import bhavcopy_fo_save
            import tempfile, os as _os
            with tempfile.TemporaryDirectory() as tmp:
                path = bhavcopy_fo_save(d, tmp)
                if path and _os.path.exists(path):
                    return self.ingest_dataframe(pd.read_csv(path))
        except Exception as e:
            log.info("jugaad bhavcopy %s: %s", d, e)
        # Provider 2: NSE UDiFF archive (new format, post Jul-2024)
        try:
            import requests, zipfile
            url = ("https://nsearchives.nseindia.com/content/fo/"
                   f"BhavCopy_NSE_FO_0_0_0_{d.strftime('%Y%m%d')}_F_0000.csv.zip")
            rsp = requests.get(url, timeout=20, headers={
                'User-Agent': 'Mozilla/5.0', 'Accept': '*/*'})
            if rsp.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(rsp.content)) as z:
                    inner = z.namelist()[0]
                    return self.ingest_dataframe(pd.read_csv(z.open(inner)))
        except Exception as e:
            log.info("UDiFF bhavcopy %s: %s", d, e)
        return 0

    def download_range(self, start: date, end: date,
                       progress_cb: Optional[Callable[[float, str], None]] = None) -> dict:
        days = pd.bdate_range(start, end)
        ok = fail = rows = 0
        for i, ts in enumerate(days):
            n = self.download_day(ts.date())
            rows += n
            ok += int(n > 0)
            fail += int(n == 0)
            if progress_cb:
                progress_cb((i + 1) / len(days), f"{ts.date()} — {n} rows")
        return {'days_ok': ok, 'days_empty': fail, 'rows': rows}

    # ------------------------- queries -------------------------
    def coverage(self, symbol: str) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute(
                "SELECT MIN(trade_date), MAX(trade_date), COUNT(*), "
                "COUNT(DISTINCT trade_date) FROM fo_eod WHERE symbol=?",
                (symbol.upper(),)).fetchone()
        if not row or not row[0]:
            return None
        return {'first': row[0], 'last': row[1], 'rows': row[2], 'days': row[3]}

    def expiries_on(self, trade_date: str, symbol: str) -> List[str]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT DISTINCT expiry FROM fo_eod WHERE trade_date=? AND symbol=? "
                "AND expiry >= ? ORDER BY expiry",
                (trade_date, symbol.upper(), trade_date)).fetchall()
        return [r[0] for r in rows]

    def lookup(self, trade_date: str, symbol: str, expiry: str,
               strike: float, opt_type: str) -> Optional[float]:
        """Real EOD close for an exact contract (None if not stored)."""
        with self._conn() as c:
            row = c.execute(
                "SELECT close, settle FROM fo_eod WHERE trade_date=? AND symbol=? "
                "AND expiry=? AND strike=? AND opt_type=?",
                (trade_date, symbol.upper(), expiry, float(strike),
                 opt_type[0].upper())).fetchone()
        if row is None:
            return None
        close, settle = row
        return close if close and close > 0 else (settle or None)

    def historical_chain(self, trade_date: str, symbol: str,
                         expiry: str) -> pd.DataFrame:
        """Full stored chain for a past date (real premiums + OI)."""
        with self._conn() as c:
            return pd.read_sql(
                "SELECT strike, opt_type, close, settle, oi, volume FROM fo_eod "
                "WHERE trade_date=? AND symbol=? AND expiry=? ORDER BY strike",
                c, params=(trade_date, symbol.upper(), expiry))


# ==============================================================================
# 19. ADJUSTMENT ENGINE - roll / repair / convert / hedge candidates with
#     full before-vs-after metric comparison (each candidate is re-priced
#     and re-analysed through the same PortfolioRiskEngine).
# ==============================================================================
class AdjustmentEngine:
    def __init__(self, positions_df: pd.DataFrame, spot: float, live_iv: float,
                 r: float = 0.065):
        self.df = positions_df.copy()
        self.spot = spot
        self.iv = live_iv
        self.r = r
        self.symbol = str(positions_df.iloc[0]['Symbol']).upper()
        self.step = STRIKE_STEP.get(self.symbol, 50)

    # ---------- leg utilities ----------
    def _bsm_px(self, strike: float, expiry: str, opt_type: str,
                iv_pct: float) -> float:
        dte = max((pd.to_datetime(expiry) - datetime.now()).days, 1)
        return round(float(BlackScholesEngine.price(
            self.spot, strike, dte / 365, self.r, iv_pct / 100,
            'C' if opt_type == 'Call' else 'P')), 2)

    def _reprice_leg(self, leg: dict, strike: float = None,
                     expiry: str = None) -> dict:
        """Clone a leg with new strike/expiry at live BSM price (new AvgPrice
        because rolling closes the old leg and opens the new one)."""
        new = dict(leg)
        new['ID'] = str(uuid.uuid4())
        if strike is not None:
            new['Strike'] = strike
        if expiry is not None:
            new['Expiry'] = expiry
        new['AvgPrice'] = self._bsm_px(new['Strike'], new['Expiry'],
                                       new['Type'], float(new['IV']))
        return new

    def _tested_leg(self) -> Optional[dict]:
        """The short leg spot is pressing against (closest short strike)."""
        shorts = [l for l in self.df.to_dict('records') if l['Action'] == 'Sell']
        if not shorts:
            return None
        return min(shorts, key=lambda l: abs(float(l['Strike']) - self.spot))

    # ---------- candidates ----------
    def candidates(self) -> List[dict]:
        """Each candidate: name, rationale, roll_cost (₹, +ve = debit paid to
        adjust) and the full replacement leg list."""
        base_legs = self.df.to_dict('records')
        out = []
        lot = float(self.df.iloc[0].get('LotSize') or LOT_SIZES.get(self.symbol, 1))

        tested = self._tested_leg()
        if tested:
            away = 2 * self.step * (1 if tested['Type'] == 'Call' else -1)
            rolled = self._reprice_leg(tested, strike=float(tested['Strike']) + away)
            close_px = self._bsm_px(tested['Strike'], tested['Expiry'],
                                    tested['Type'], float(tested['IV']))
            credit_new = rolled['AvgPrice'] * float(rolled['Qty']) * lot
            cost_close = close_px * float(tested['Qty']) * lot
            out.append({
                'name': f"Roll tested {tested['Type'].lower()} "
                        f"{fmt_inr(tested['Strike'],0)} → {fmt_inr(rolled['Strike'],0)}",
                'rationale': "Moves the pressured short strike further OTM, "
                             "re-centering the profit zone around spot.",
                'roll_cost': cost_close - credit_new,
                'legs': [l for l in base_legs if l['ID'] != tested['ID']] + [rolled],
            })

        # Roll out in time (+28d), same strikes
        new_exp = (pd.to_datetime(self.df['Expiry']).min()
                   + timedelta(days=28)).strftime("%Y-%m-%d")
        rolled_out, cost = [], 0.0
        for leg in base_legs:
            nl = self._reprice_leg(leg, expiry=new_exp)
            sgn = 1 if leg['Action'] == 'Buy' else -1
            close_px = self._bsm_px(leg['Strike'], leg['Expiry'], leg['Type'],
                                    float(leg['IV']))
            cost += sgn * (nl['AvgPrice'] - close_px) * float(leg['Qty']) * lot
            rolled_out.append(nl)
        out.append({
            'name': f"Roll out to {pd.to_datetime(new_exp).strftime('%d %b')}",
            'rationale': "Buys time and (for credit structures) collects fresh "
                         "premium; resets theta runway.",
            'roll_cost': cost, 'legs': rolled_out})

        # Convert to Iron Condor: buy wings beyond every naked short
        strat, _ = StrategyEngine.detect(self.df)
        if strat in ("Short Strangle", "Short Straddle"):
            wings, wing_cost = [], 0.0
            for leg in base_legs:
                if leg['Action'] != 'Sell':
                    continue
                off = 4 * self.step * (1 if leg['Type'] == 'Call' else -1)
                wing = self._reprice_leg(leg, strike=float(leg['Strike']) + off)
                wing['Action'] = 'Buy'
                wing_cost += wing['AvgPrice'] * float(wing['Qty']) * lot
                wings.append(wing)
            out.append({
                'name': "Convert to Iron Condor (buy wings)",
                'rationale': "Caps the unlimited tails; margin drops sharply "
                             "for a small debit.",
                'roll_cost': wing_cost, 'legs': base_legs + wings})

            # Convert to Iron Fly: shorts to ATM + wings
            atm = round(self.spot / self.step) * self.step
            fly, fly_cost = [], 0.0
            for leg in base_legs:
                if leg['Action'] == 'Sell':
                    close_px = self._bsm_px(leg['Strike'], leg['Expiry'],
                                            leg['Type'], float(leg['IV']))
                    nl = self._reprice_leg(leg, strike=atm)
                    fly_cost += (close_px - nl['AvgPrice']) * float(leg['Qty']) * lot
                    fly.append(nl)
                    woff = 6 * self.step * (1 if leg['Type'] == 'Call' else -1)
                    wing = self._reprice_leg(leg, strike=atm + woff)
                    wing['Action'] = 'Buy'
                    fly_cost += wing['AvgPrice'] * float(leg['Qty']) * lot
                    fly.append(wing)
                else:
                    fly.append(leg)
            out.append({
                'name': "Convert to Iron Butterfly (ATM)",
                'rationale': "Maximum theta collection pinned at spot with "
                             "defined wings — for a strong range-bound view.",
                'roll_cost': fly_cost, 'legs': fly})

        # Delta hedge with futures
        eng = PortfolioRiskEngine(self.df, self.spot, self.r)
        m = eng.get_metrics()
        if abs(m['delta']) > lot * 0.4:
            fut_lots = -m['delta'] / lot
            out.append({
                'name': f"Delta hedge: {'Buy' if fut_lots>0 else 'Sell'} "
                        f"{abs(fut_lots):.1f} futures lot(s)",
                'rationale': f"Neutralizes net delta of {m['delta']:.0f}; leaves "
                             "theta/vega exposure unchanged. (Futures leg is a "
                             "suggestion — not added to the option book.)",
                'roll_cost': 0.0, 'legs': None})

        out.append({'name': "Exit all legs",
                    'rationale': "Locks in current MTM; eliminates gamma/vega "
                                 "risk into events or expiry week.",
                    'roll_cost': -m['total_mtm'], 'legs': []})
        return out

    def compare(self, cand: dict) -> Optional[pd.DataFrame]:
        """Before/after metric table for a candidate with replacement legs."""
        if not cand.get('legs'):
            return None
        before = PortfolioRiskEngine(self.df, self.spot, self.r).get_metrics()
        after = PortfolioRiskEngine(pd.DataFrame(cand['legs']),
                                    self.spot, self.r).get_metrics()

        def row(m):
            return {
                'POP %': round(m['pop'], 1),
                'Max Profit': fmt_compact(m['max_profit']) if math.isfinite(m['max_profit']) else '∞',
                'Max Loss': fmt_compact(m['max_loss']) if math.isfinite(m['max_loss']) else '−∞',
                'Delta': round(m['delta'], 1), 'Theta/day': round(m['theta'], 0),
                'Vega': round(m['vega'], 0),
                'Margin est.': fmt_compact(m['capital']),
                'Breakevens': " / ".join(fmt_inr(b, 0) for b in m['breakevens'][:2]) or "—",
            }
        return pd.DataFrame([row(before), row(after)], index=['Before', 'After'])


# ==============================================================================
# 20. RECOMMENDATION ENGINE - deterministic, explainable rules (no black box).
#     Scores Hold / Book profit / Exit / Roll / Hedge / Add wings from the
#     live Greeks, POP, DTE, IV-rank and distance-to-strike. Every suggestion
#     carries its triggering reason.
# ==============================================================================
class RecommendationEngine:
    @staticmethod
    def analyse(engine: PortfolioRiskEngine, metrics: dict, strat: str,
                iv_rank: float) -> List[dict]:
        recs = []
        spot, dte = engine.spot, metrics['dte']
        mtm, npv = metrics['total_mtm'], metrics['net_premium']
        is_credit = npv > 0
        lot = engine.legs[0]['mult'] if engine.legs else 1

        # Profit capture (credit: % of premium collected)
        if is_credit and npv > 0:
            capture = mtm / npv * 100
            if capture >= 50:
                recs.append({'action': '✅ Book profit / Exit', 'score': 90,
                             'why': f"{capture:.0f}% of max credit captured — "
                                    "risk-adjusted edge decays past 50%."})
            elif capture <= -100:
                recs.append({'action': '🛑 Exit or repair', 'score': 85,
                             'why': f"Loss is {abs(capture):.0f}% of credit — "
                                    "beyond the common 1–2× credit stop zone."})

        # Gamma week
        if dte <= 7 and any(l['direction'] == -1 for l in engine.legs):
            recs.append({'action': '🔁 Roll out or exit shorts', 'score': 80,
                         'why': f"{dte} DTE — short gamma risk explodes in "
                                "expiry week; pin risk at short strikes."})

        # Tested strike proximity
        shorts = [l for l in engine.legs if l['direction'] == -1]
        if shorts:
            nearest = min(shorts, key=lambda l: abs(l['K'] - spot))
            dist_pct = abs(nearest['K'] - spot) / spot * 100
            if dist_pct < 1.0:
                recs.append({'action': '🔁 Roll tested side', 'score': 75,
                             'why': f"Spot within {dist_pct:.1f}% of short "
                                    f"{fmt_inr(nearest['K'],0)} strike."})

        # Delta drift
        if abs(metrics['delta']) * spot > 0.35 * max(metrics['capital'], 1):
            recs.append({'action': '⚖️ Delta hedge', 'score': 70,
                         'why': f"Net delta {metrics['delta']:.0f} — directional "
                                "exposure dominates the theta engine."})

        # POP degradation
        if metrics['pop'] < 40:
            recs.append({'action': '🛠 Repair / restructure', 'score': 65,
                         'why': f"POP fallen to {metrics['pop']:.0f}% — "
                                "probability no longer supports the position."})

        # Naked tails
        if not math.isfinite(metrics['max_loss']) and strat in (
                "Short Strangle", "Short Straddle"):
            recs.append({'action': '🦋 Add wings (→ Iron Condor)', 'score': 60,
                         'why': "Unlimited tail risk; wings cap gap risk and "
                                "cut margin sharply."})

        # Vol environment
        if math.isfinite(iv_rank):
            if iv_rank < 25 and metrics['vega'] < 0:
                recs.append({'action': '⚠️ Reduce short vega', 'score': 55,
                             'why': f"IV rank {iv_rank:.0f} — selling cheap vol; "
                                    "expansion risk outweighs decay edge."})
            elif iv_rank > 70 and metrics['vega'] < 0:
                recs.append({'action': '🟢 Hold (favourable vol)', 'score': 50,
                             'why': f"IV rank {iv_rank:.0f} — short vega is "
                                    "being paid well; mean-reversion tailwind."})

        if not recs:
            recs.append({'action': '🟢 Hold', 'score': 40,
                         'why': "No rule triggered — position within normal "
                                "risk parameters."})
        return sorted(recs, key=lambda r_: -r_['score'])


# ==============================================================================
# 21. PORTFOLIO OPTIMIZER - risk-budget sizing with sector caps and a
#     cross-underlying correlation check (multi-symbol books).
# ==============================================================================
def portfolio_optimizer(positions_df: pd.DataFrame, spot_map: Dict[str, float],
                        capital: float, risk_pct: float, sector_cap_pct: float,
                        r: float = 0.065) -> dict:
    """Groups the book by underlying, measures each group's capital-at-risk
    (max loss if defined, est. margin otherwise), and returns suggested size
    multipliers honouring per-position risk budgets and sector caps."""
    groups = []
    for sym, g in positions_df.groupby('Symbol'):
        eng = PortfolioRiskEngine(g, spot_map.get(sym, 100.0), r)
        m = eng.get_metrics()
        risk = abs(m['max_loss']) if math.isfinite(m['max_loss']) else m['capital']
        groups.append({'Symbol': sym,
                       'Sector': SECTOR_MAP.get(str(sym).upper(), 'Other'),
                       'Risk (₹)': risk, 'Margin (₹)': m['capital'],
                       'POP %': round(m['pop'], 1),
                       'Theta/day': round(m['theta'], 0)})
    gdf = pd.DataFrame(groups)
    if gdf.empty:
        return {}
    budget = capital * risk_pct / 100
    gdf['Risk Budget (₹)'] = budget
    gdf['Suggested ×'] = (budget / gdf['Risk (₹)'].replace(0, np.nan)).clip(upper=10).round(2)

    sector_risk = gdf.groupby('Sector')['Risk (₹)'].sum()
    sector_cap = capital * sector_cap_pct / 100
    breaches = sector_risk[sector_risk > sector_cap]

    corr = None
    syms = gdf['Symbol'].unique().tolist()
    if len(syms) > 1:
        rets = {}
        for s_ in syms:
            h = fetch_spot_history(s_, "1y")
            if not h.empty:
                rets[s_] = h['Close'].pct_change()
        if len(rets) > 1:
            corr = pd.DataFrame(rets).corr().round(2)

    return {'groups': gdf, 'sector_risk': sector_risk, 'sector_cap': sector_cap,
            'breaches': breaches, 'corr': corr,
            'total_risk': float(gdf['Risk (₹)'].sum()),
            'utilization_pct': float(gdf['Margin (₹)'].sum() / capital * 100)}


# ==============================================================================
# 22. BROKER INTEGRATION (Zerodha Kite / Dhan / Fyers / Angel One SmartAPI)
#     Read-only position import via official REST APIs. Adapters map broker
#     payloads to the app's leg schema. NOTE: written against each broker's
#     published API docs but NOT live-tested here (requires your API keys) -
#     treat the first sync as verification and report field mismatches.
# ==============================================================================
_MONTHS = {m: i + 1 for i, m in enumerate(
    ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
     'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'])}
# NSE monthly expiry weekday: 3=Thursday (historic). NSE has revised expiry
# days by circular before - change here if your contracts differ.
NSE_MONTHLY_EXPIRY_WEEKDAY = 3
_WEEKLY_MONTH = {**{str(i): i for i in range(1, 10)}, 'O': 10, 'N': 11, 'D': 12}


def _last_weekday(year: int, month: int, weekday: int) -> date:
    d = date(year, month, 28) + timedelta(days=4)
    d = d.replace(day=1) - timedelta(days=1)          # last day of month
    while d.weekday() != weekday:
        d -= timedelta(days=1)
    return d


def parse_nse_option_symbol(ts: str) -> Optional[dict]:
    """Parse NSE option trading symbols into (symbol, expiry, strike, type).
    Handles monthly (NIFTY24JUL22500CE) and weekly (NIFTY2470322500CE,
    month code 1-9/O/N/D) formats used by Zerodha/Fyers/Angel."""
    import re
    ts = ts.upper().strip()
    m = re.match(r'^([A-Z&\-]+?)(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)'
                 r'(\d+(?:\.\d+)?)(CE|PE)$', ts)
    if m:
        sym, yy, mon, strike, ot = m.groups()
        exp = _last_weekday(2000 + int(yy), _MONTHS[mon], NSE_MONTHLY_EXPIRY_WEEKDAY)
        return {'Symbol': sym, 'Expiry': exp.strftime('%Y-%m-%d'),
                'Strike': float(strike), 'Type': 'Call' if ot == 'CE' else 'Put'}
    m = re.match(r'^([A-Z&\-]+?)(\d{2})([1-9OND])(\d{2})(\d+(?:\.\d+)?)(CE|PE)$', ts)
    if m:
        sym, yy, mc, dd, strike, ot = m.groups()
        exp = date(2000 + int(yy), _WEEKLY_MONTH[mc], int(dd))
        return {'Symbol': sym, 'Expiry': exp.strftime('%Y-%m-%d'),
                'Strike': float(strike), 'Type': 'Call' if ot == 'CE' else 'Put'}
    return None


def _mk_leg(parsed: dict, qty_units: float, avg: float, iv_pct: float) -> dict:
    """Broker positions report NET UNITS; convert to lots via lot size."""
    lot = LOT_SIZES.get(parsed['Symbol'], DEFAULT_STOCK_LOT_SIZE)
    lots = max(round(abs(qty_units) / lot), 1)
    return {'ID': str(uuid.uuid4()), 'Symbol': parsed['Symbol'],
            'Expiry': parsed['Expiry'], 'Strike': parsed['Strike'],
            'Type': parsed['Type'],
            'Action': 'Buy' if qty_units > 0 else 'Sell',
            'Qty': lots, 'AvgPrice': round(float(avg), 2),
            'IV': round(iv_pct, 1), 'LotSize': lot, 'Booked': 0.0}


class BrokerAdapter:
    """Base adapter. fetch(creds) -> (legs, message)."""
    name = "base"
    fields: List[Tuple[str, bool]] = []      # (field_label, is_password)

    def fetch(self, creds: Dict[str, str], iv_pct: float) -> Tuple[List[dict], str]:
        raise NotImplementedError


class ZerodhaAdapter(BrokerAdapter):
    name = "Zerodha Kite"
    fields = [("API Key", False), ("Access Token", True)]

    def fetch(self, creds, iv_pct):
        import requests
        rsp = requests.get("https://api.kite.trade/portfolio/positions",
                           headers={"X-Kite-Version": "3",
                                    "Authorization": f"token {creds['API Key']}:"
                                                     f"{creds['Access Token']}"},
                           timeout=15)
        rsp.raise_for_status()
        legs = []
        for p in rsp.json().get('data', {}).get('net', []):
            if p.get('quantity', 0) == 0 or 'FO' not in str(p.get('exchange', '')) \
                    and p.get('exchange') != 'NFO':
                continue
            parsed = parse_nse_option_symbol(p.get('tradingsymbol', ''))
            if parsed:
                legs.append(_mk_leg(parsed, p['quantity'],
                                    p.get('average_price', 0), iv_pct))
        return legs, f"Kite: {len(legs)} option legs mapped"


class DhanAdapter(BrokerAdapter):
    name = "Dhan"
    fields = [("Access Token", True)]

    def fetch(self, creds, iv_pct):
        import requests
        rsp = requests.get("https://api.dhan.co/v2/positions",
                           headers={"access-token": creds['Access Token'],
                                    "Accept": "application/json"}, timeout=15)
        rsp.raise_for_status()
        legs = []
        for p in rsp.json():
            if p.get('netQty', 0) == 0 or 'OPT' not in str(p.get('drvOptionType', '')) \
                    and p.get('productType') is None:
                pass
            qty = p.get('netQty', 0)
            if qty == 0:
                continue
            ot = str(p.get('drvOptionType', '')).upper()
            if ot not in ('CALL', 'PUT', 'CE', 'PE'):
                continue
            exp = p.get('drvExpiryDate', '')[:10]
            legs.append(_mk_leg({'Symbol': str(p.get('tradingSymbol', p.get('customSymbol', ''))).split('-')[0].upper(),
                                 'Expiry': exp,
                                 'Strike': float(p.get('drvStrikePrice', 0)),
                                 'Type': 'Call' if ot.startswith('C') else 'Put'},
                                qty, p.get('costPrice', p.get('buyAvg', 0)), iv_pct))
        return legs, f"Dhan: {len(legs)} option legs mapped"


class FyersAdapter(BrokerAdapter):
    name = "Fyers"
    fields = [("App ID", False), ("Access Token", True)]

    def fetch(self, creds, iv_pct):
        import requests
        rsp = requests.get("https://api-t1.fyers.in/api/v3/positions",
                           headers={"Authorization": f"{creds['App ID']}:"
                                                     f"{creds['Access Token']}"},
                           timeout=15)
        rsp.raise_for_status()
        legs = []
        for p in rsp.json().get('netPositions', []):
            qty = p.get('netQty', 0)
            if qty == 0:
                continue
            ts = str(p.get('symbol', '')).replace('NSE:', '')
            parsed = parse_nse_option_symbol(ts)
            if parsed:
                legs.append(_mk_leg(parsed, qty, p.get('netAvg', 0), iv_pct))
        return legs, f"Fyers: {len(legs)} option legs mapped"


class AngelOneAdapter(BrokerAdapter):
    name = "Angel One SmartAPI"
    fields = [("API Key", False), ("JWT Token", True)]

    def fetch(self, creds, iv_pct):
        import requests
        rsp = requests.get(
            "https://apiconnect.angelone.in/rest/secure/angelbroking/order/v1/getPosition",
            headers={"Authorization": f"Bearer {creds['JWT Token']}",
                     "X-PrivateKey": creds['API Key'],
                     "Content-Type": "application/json",
                     "X-UserType": "USER", "X-SourceID": "WEB",
                     "X-ClientLocalIP": "127.0.0.1",
                     "X-ClientPublicIP": "127.0.0.1",
                     "X-MACAddress": "00:00:00:00:00:00"},
            timeout=15)
        rsp.raise_for_status()
        legs = []
        for p in (rsp.json().get('data') or []):
            qty = float(p.get('netqty', 0) or 0)
            if qty == 0:
                continue
            parsed = parse_nse_option_symbol(str(p.get('tradingsymbol', '')))
            if parsed:
                legs.append(_mk_leg(parsed, qty, float(p.get('avgnetprice', 0) or 0),
                                    iv_pct))
        return legs, f"Angel One: {len(legs)} option legs mapped"


BROKER_ADAPTERS: List[BrokerAdapter] = [
    ZerodhaAdapter(), DhanAdapter(), FyersAdapter(), AngelOneAdapter()]


def render_broker_sync(live_iv: float) -> None:
    """Sidebar broker-sync UI. Credentials are used in-memory for one request
    and never persisted to disk or the journal DB."""
    with st.expander("🔗 Broker Sync (read-only import)"):
        st.caption("Positions are imported read-only. Adapters follow each "
                   "broker's published API but are **untested without your "
                   "keys** — verify the first import against your terminal. "
                   "Credentials are not stored.")
        names = [a.name for a in BROKER_ADAPTERS]
        pick = st.selectbox("Broker", names)
        adapter = BROKER_ADAPTERS[names.index(pick)]
        creds = {}
        for label, secret in adapter.fields:
            creds[label] = st.text_input(label, type="password" if secret else "default",
                                         key=f"cred_{pick}_{label}")
        if st.button("Import positions", use_container_width=True):
            try:
                legs, msg = adapter.fetch(creds, live_iv * 100)
                if legs:
                    st.session_state.positions = st.session_state.positions + legs
                    st.success(msg)
                    st.rerun()
                else:
                    st.warning(f"{msg} — no open option positions found, or "
                               "symbol format unrecognised (check logs).")
            except Exception as e:
                st.error(f"{adapter.name} sync failed: {e}")


# ==============================================================================
# 23. ADVISOR TAB (recommendations + adjustment candidates + optimizer)
# ==============================================================================
def render_advisor_tab(engine: Optional[PortfolioRiskEngine], extras,
                       symbol: str, spot: float, live_iv: float, r: float):
    if engine is None:
        st.info("Add and select positions on the Analyse tab first.")
        return
    metrics, strat, df_sel = extras
    vix = fetch_vix_history("1y")
    _, iv_rank, _ = VolAnalytics.iv_rank_percentile(
        vix['Close'] if not vix.empty else pd.Series(dtype=float))

    st.markdown("##### 🧭 Position Recommendations (rule-based, explainable)")
    st.caption("Deterministic rules on Greeks / POP / DTE / IV-rank — every "
               "suggestion shows its trigger. Decision support, not advice.")
    recs = RecommendationEngine.analyse(engine, metrics, strat, iv_rank)
    for rec in recs:
        st.markdown(f"""<div class="oo-card" style="padding:10px 14px">
          <b>{rec['action']}</b>
          <span class="mut" style="float:right">confidence {rec['score']}/100</span>
          <div class="pos-meta" style="margin-top:4px">{rec['why']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("##### 🔧 Adjustment Candidates (before → after)")
    adj = AdjustmentEngine(df_sel, spot, live_iv, r)
    cands = adj.candidates()
    names = [c['name'] for c in cands]
    pick = st.selectbox("Candidate", names, key="adj_pick")
    cand = cands[names.index(pick)]
    cost_lbl = ("debit ₹" + fmt_inr(cand['roll_cost'], 0)) if cand['roll_cost'] > 0 \
        else ("credit ₹" + fmt_inr(abs(cand['roll_cost']), 0))
    st.markdown(f"*{cand['rationale']}*  \nAdjustment cash flow: **{cost_lbl}** "
                "(BSM-priced at live IV; actual fills will differ).")
    cmp_df = adj.compare(cand)
    if cmp_df is not None:
        st.dataframe(cmp_df, use_container_width=True)
        if st.button("Apply this adjustment to the position book"):
            st.session_state.positions = cand['legs']
            st.session_state.selected = {}
            st.success("Book replaced with adjusted legs — review on Analyse tab.")
            st.rerun()
    elif cand['legs'] is not None and len(cand['legs']) == 0:
        if st.button("Confirm exit — clear position book"):
            st.session_state.positions = []
            st.session_state.selected = {}
            st.rerun()

    st.markdown("##### 📐 Portfolio Optimizer (risk budgets & caps)")
    c1, c2, c3 = st.columns(3)
    cap = c1.number_input("Capital (₹)", value=1_000_000, step=100_000, key="opt_cap")
    rpct = c2.number_input("Risk per underlying (%)", value=2.0, step=0.5, key="opt_r")
    scap = c3.number_input("Sector cap (%)", value=25.0, step=5.0, key="opt_s")
    all_pos = pd.DataFrame(st.session_state.positions)
    if not all_pos.empty:
        spot_map = {symbol: spot}
        for s_ in all_pos['Symbol'].unique():
            if s_ not in spot_map:
                spot_map[s_], _, _ = fetch_market_data(s_)
        opt = portfolio_optimizer(all_pos, spot_map, cap, rpct, scap, r)
        if opt:
            st.dataframe(opt['groups'].style.format(
                {c_: (lambda v: fmt_compact(v)) for c_ in
                 ['Risk (₹)', 'Margin (₹)', 'Risk Budget (₹)']}),
                use_container_width=True, hide_index=True)
            st.progress(min(opt['utilization_pct'] / 100, 1.0),
                        text=f"Margin utilization {opt['utilization_pct']:.1f}% "
                             f"· total risk {fmt_compact(opt['total_risk'])}")
            if len(opt['breaches']):
                st.warning("Sector cap breached: " + ", ".join(
                    f"{s_} ({fmt_compact(v)})" for s_, v in opt['breaches'].items()))
            if opt['corr'] is not None:
                st.markdown("**Cross-underlying correlation (1y daily returns)** "
                            "— pairs > 0.8 concentrate risk:")
                st.dataframe(opt['corr'], use_container_width=True)


# --------------------------- BACKTEST ----------------------------------------
def render_backtest_tab(symbol: str, positions_df: pd.DataFrame, r: float):
    store = BhavcopyStore()
    cov = store.coverage(symbol)
    with st.expander("🗄️ Historical Chain Database (real NSE EOD premiums)", expanded=cov is None):
        if cov:
            st.markdown(f"Coverage for **{symbol}**: `{cov['first']}` → `{cov['last']}` "
                        f"· **{cov['days']}** trading days · {cov['rows']:,} contracts")
        else:
            st.markdown("No stored history yet for this symbol — download bhavcopies "
                        "or import files below.")
        c1, c2, c3 = st.columns([0.3, 0.3, 0.4])
        d_from = c1.date_input("From", date.today() - timedelta(days=30), key="bc_from")
        d_to = c2.date_input("To", date.today(), key="bc_to")
        if c3.button("⬇️ Download NSE F&O bhavcopies", use_container_width=True):
            bar = st.progress(0.0)
            res = store.download_range(d_from, d_to,
                                       lambda p, t: bar.progress(p, text=t))
            st.success(f"{res['rows']:,} contract rows stored "
                       f"({res['days_ok']} days ok, {res['days_empty']} empty/holiday/blocked).")
            st.rerun()
        up = st.file_uploader("Or import bhavcopy CSV/ZIP (legacy & UDiFF formats auto-detected)",
                              type=['csv', 'zip'], key="bc_up")
        if up is not None:
            n = store.ingest_file(up)
            st.success(f"Ingested {n:,} option rows.") if n else st.error(
                "Unrecognised format — expected NSE F&O bhavcopy columns.")
        st.caption("Backtests automatically use REAL premiums wherever this DB has "
                   "coverage; BSM-modelled prices fill only the gaps and every trade "
                   "is flagged Real / Mixed / Model.")

    if positions_df.empty:
        st.info("Add positions on the Analyse tab first.")
        return
    mode = st.radio("Mode", ["🔁 Replay current position", "📈 Expiry-cycle backtest"],
                    horizontal=True)
    bt = BacktestEngine(symbol, r,
                        cost_bps=st.number_input("Friction (bps of premium, round-trip/leg)",
                                                 value=5.0, step=1.0),
                        store=store)

    if mode.startswith("🔁"):
        c1, c2 = st.columns(2)
        start = c1.date_input("Start date", date.today() - timedelta(days=90))
        end = c2.date_input("End date", date.today())
        if st.button("Run Replay"):
            st.session_state['replay'] = bt.replay(positions_df, start, end)
        rp = st.session_state.get('replay')
        if rp is None or (isinstance(rp, pd.DataFrame) and rp.empty):
            return
        real_pct = rp.attrs.get('real_pct', 0.0)
        st.markdown(f"<span class='chip'>Data: <b>{real_pct:.0f}% real EOD quotes</b>"
                    f" · {100-real_pct:.0f}% BSM-modelled</span>", unsafe_allow_html=True)
        # Scrub slider: forward & backward through history
        idx = st.slider("Replay day", 0, len(rp) - 1, len(rp) - 1,
                        format="%d", help="Drag backward/forward to scrub through time")
        snap = rp.iloc[idx]
        c = st.columns(6)
        for col, (lab, val, money) in zip(c, [
                ("Date", str(rp.index[idx].date()), False),
                ("Spot", fmt_inr(snap['Spot'], 0), False),
                ("MTM", snap['MTM'], True), ("Delta", f"{snap['Delta']:.1f}", False),
                ("Theta/day", snap['Theta'], True), ("Vega", snap['Vega'], True)]):
            body = pnl_span(val, False, 0) if money else f"<span>{val}</span>"
            col.markdown(f"<div class='oo-card'><div class='mlab'>{lab}</div>"
                         f"<div class='mval' style='font-size:.85rem'>{body}</div></div>",
                         unsafe_allow_html=True)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
                            subplot_titles=("Modelled MTM", "Spot"))
        fig.add_trace(go.Scatter(x=rp.index, y=rp['MTM'], name='MTM',
                                 line=dict(color='#21ce99', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=rp.index, y=rp['Spot'], name='Spot',
                                 line=dict(color='#4da3ff', width=1.6)), row=2, col=1)
        fig.add_vline(x=rp.index[idx], line_dash='dash', line_color='#f7b731')
        st.plotly_chart(_dark_fig(fig, 460), use_container_width=True,
                        config={'displayModeBar': False})
        st.download_button("⬇️ Export replay CSV", rp.to_csv().encode(),
                           f"{symbol}_replay.csv", "text/csv")

    else:
        c1, c2, c3, c4 = st.columns(4)
        lookback = c1.selectbox("Lookback", [365, 730, 1095, 1825], index=1,
                                format_func=lambda d: f"{d//365}y")
        hold = c2.number_input("Hold days / cycle", 5, 60, 30)
        sl = c3.number_input("Stop-loss (% of premium)", 50, 500, 200, step=25)
        tgt = c4.number_input("Target (% of premium)", 10, 300, 50, step=10)
        if st.button("Run Cycle Backtest"):
            with st.spinner("Backtesting…"):
                st.session_state['cycle'] = bt.cycle_backtest(
                    positions_df, lookback, hold, sl, tgt)
        res = st.session_state.get('cycle')
        if not res:
            return
        stats, tl, eq = res['stats'], res['trades'], res['equity']
        st.markdown(f"<span class='chip'>Data: <b>{res.get('real_pct',0):.0f}% of trades "
                    f"used real EOD quotes</b> (rest BSM-modelled — see Data column)</span>",
                    unsafe_allow_html=True)
        if res.get('skipped', 0):
            st.warning(f"⚠️ {res['skipped']} degenerate entries skipped (modelled "
                       "premium < ₹250 — strikes too far OTM at that spot/IV level). "
                       "If many were skipped, this strike structure isn't tradeable "
                       "across the full lookback.")
        tiny = ((tl['Reason'] == 'Target') & (tl['P&L'].abs() < 50)).sum()
        if tiny > len(tl) * 0.2:
            st.error("🚩 Result validity check FAILED: a large share of 'Target' exits "
                     "have near-zero P&L — thresholds are not meaningful for this "
                     "structure. Do not trust the win rate.")

        cells = [(k, v) for k, v in stats.items() if k != 'monthly']
        html = "<div class='mstrip'>" + "".join(
            f"<div class='mcell'><div class='mlab'>{k}</div>"
            f"<div class='mval' style='font-size:.82rem'>"
            f"{fmt_compact(v) if isinstance(v,(int,float)) and abs(v)>=1000 else v}"
            f"</div></div>" for k, v in cells) + "</div>"
        st.markdown(html, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Scatter(x=eq.index, y=eq.values, name='Equity',
                                       line=dict(color='#21ce99', width=2.2),
                                       fill='tozeroy',
                                       fillcolor='rgba(33,206,153,.08)'))
            fig.update_layout(title="Equity Curve (cumulative ₹ P&L)")
            st.plotly_chart(_dark_fig(fig, 300), use_container_width=True,
                            config={'displayModeBar': False})
        with col2:
            dd = eq - eq.cummax()
            fig = go.Figure(go.Scatter(x=dd.index, y=dd.values, name='Drawdown',
                                       line=dict(color='#ff5b6a', width=2),
                                       fill='tozeroy',
                                       fillcolor='rgba(255,91,106,.12)'))
            fig.update_layout(title="Drawdown (₹)")
            st.plotly_chart(_dark_fig(fig, 300), use_container_width=True,
                            config={'displayModeBar': False})

        mon = stats['monthly']
        fig = go.Figure(go.Bar(x=mon.index, y=mon.values,
                               marker_color=['#21ce99' if v >= 0 else '#ff5b6a'
                                             for v in mon.values]))
        fig.update_layout(title="Monthly P&L")
        st.plotly_chart(_dark_fig(fig, 260), use_container_width=True,
                        config={'displayModeBar': False})

        st.markdown("##### Trade Log")
        st.dataframe(tl.style.map(
            lambda v: f"color: {'#21ce99' if v >= 0 else '#ff5b6a'}", subset=['P&L']),
            use_container_width=True, hide_index=True)
        st.download_button("⬇️ Export trade log (Excel)",
                           TradeJournal.to_excel_bytes(tl),
                           f"{symbol}_backtest.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# --------------------------- MARKET DASHBOARD --------------------------------
def render_market_tab(symbol: str, r: float):
    vix = fetch_vix_history("1y")
    hist = fetch_spot_history(symbol, "1y")
    chain = fetch_chain_raw(symbol)

    pcr = max_pain = None
    if chain:
        exp = chain_expiries(chain)[0] if chain_expiries(chain) else None
        if exp:
            cdf = build_chain_dataframe(symbol, exp, r)
            if cdf is not None:
                a = chain_analytics(cdf)
                pcr, max_pain = a['pcr'], a['max_pain']

    # Regime / trend / momentum from price history
    regime = trend_str = rsi_txt = "—"
    if not hist.empty and len(hist) > 200:
        close = hist['Close']
        sma50, sma200 = close.rolling(50).mean().iloc[-1], close.rolling(200).mean().iloc[-1]
        last = close.iloc[-1]
        regime = ("🟢 Bull" if last > sma50 > sma200 else
                  "🔴 Bear" if last < sma50 < sma200 else "🟡 Sideways")
        roc20 = (last / close.iloc[-21] - 1) * 100
        trend_str = f"{abs(roc20):.1f}% (20d ROC)"
        delta_p = close.diff()
        up = delta_p.clip(lower=0).rolling(14).mean()
        dn = (-delta_p.clip(upper=0)).rolling(14).mean()
        rs = up / dn.replace(0, np.nan)
        rsi = float((100 - 100 / (1 + rs)).iloc[-1])
        rsi_txt = f"{rsi:.0f}"

    cur_vix = float(vix['Close'].iloc[-1]) if not vix.empty else float('nan')
    c = st.columns(6)
    for col, (lab, val) in zip(c, [
            ("India VIX", f"{cur_vix:.2f}" if math.isfinite(cur_vix) else "—"),
            ("PCR (nearest exp)", f"{pcr:.2f}" if pcr else "—"),
            ("Max Pain", fmt_inr(max_pain, 0) if max_pain else "—"),
            ("Regime", regime), ("Trend Strength", trend_str),
            ("Momentum (RSI-14)", rsi_txt)]):
        col.markdown(f"<div class='oo-card'><div class='mlab'>{lab}</div>"
                     f"<div class='mval' style='font-size:.85rem'>{val}</div></div>",
                     unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if not vix.empty:
            fig = go.Figure(go.Scatter(x=vix.index, y=vix['Close'],
                                       line=dict(color='#f7b731', width=1.8)))
            fig.update_layout(title="India VIX — 1y")
            st.plotly_chart(_dark_fig(fig, 280), use_container_width=True,
                            config={'displayModeBar': False})
    with col2:
        if not hist.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Close',
                                     line=dict(color='#4da3ff', width=1.8)))
            if len(hist) > 200:
                fig.add_trace(go.Scatter(x=hist.index,
                                         y=hist['Close'].rolling(50).mean(),
                                         name='SMA50', line=dict(color='#21ce99', width=1.2)))
                fig.add_trace(go.Scatter(x=hist.index,
                                         y=hist['Close'].rolling(200).mean(),
                                         name='SMA200', line=dict(color='#ff5b6a', width=1.2)))
            fig.update_layout(title=f"{symbol} — trend")
            st.plotly_chart(_dark_fig(fig, 280), use_container_width=True,
                            config={'displayModeBar': False})

    fii = fetch_fii_dii()
    if fii is None:
        st.info("FII/DII flows & market breadth: no free stable provider is wired. "
                "Register a source (broker API / paid feed) in `fetch_fii_dii()` — "
                "the dashboard will pick it up automatically.")
    else:
        st.dataframe(fii, use_container_width=True, hide_index=True)


# --------------------------- JOURNAL ------------------------------------------
def render_journal_tab(journal: TradeJournal, symbol: str, strat: str,
                       spot: float, metrics: dict, legs: List[dict]):
    c1, c2 = st.columns([0.65, 0.35])
    with c1:
        notes = st.text_input("Notes", placeholder="Adjustment rationale, exit reason…")
        reason = st.selectbox("Exit reason (if closing)",
                              ["", "Target", "Stop-loss", "Expiry", "Adjustment", "Discretionary"])
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 Snapshot current position to journal", use_container_width=True):
            rid = journal.save(symbol, strat, "SNAPSHOT", spot, metrics, legs,
                               reason, notes)
            st.success(f"Saved entry {rid} — Greeks, MTM and legs captured.")

    jdf = journal.load()
    if jdf.empty:
        st.info("Journal is empty. Snapshots capture entry/exit state, Greeks, "
                "MTM, strategy and notes automatically.")
        return
    st.dataframe(jdf.drop(columns=['legs_json']), use_container_width=True,
                 hide_index=True, height=320)
    c1, c2, c3 = st.columns(3)
    c1.download_button("⬇️ Export Excel", TradeJournal.to_excel_bytes(
        jdf.drop(columns=['legs_json'])), "journal.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    c2.download_button("⬇️ Export CSV", jdf.to_csv(index=False).encode(),
                       "journal.csv", "text/csv")
    rid = c3.selectbox("Delete entry", [""] + jdf['id'].tolist())
    if rid and c3.button("Confirm delete"):
        journal.delete(rid)
        st.rerun()


# ==============================================================================
# 14. MAIN APPLICATION
# ==============================================================================
def render_analyse_screen(symbol: str, spot: float, chg_pct: float, live_iv: float,
                          risk_free_rate: float):
    """The full v2 Analyse screen (positions left / analysis right), extended
    with a strategy-template builder and a Risk tab."""
    positions = st.session_state.positions
    left, right = st.columns([0.42, 0.58], gap="medium")

    with left:
        chg_cls = "oo-chg-up" if chg_pct >= 0 else "oo-chg-down"
        st.markdown(f"""<div class="oo-card" style="padding:10px 14px">
          <span class="oo-ticker">{symbol} {fmt_inr(spot)}</span>
          <span class="{chg_cls}">{fmt_inr(chg_pct,2,sign=True)}%</span></div>""",
                    unsafe_allow_html=True)

        hdr1, hdr2 = st.columns([0.6, 0.4])
        hdr1.markdown(f"**{symbol} Positions**")
        if hdr2.button("Clear Positions", use_container_width=True):
            st.session_state.positions = []
            st.session_state.selected = {}
            st.rerun()

        # ----- strategy builder: template + manual leg -----
        with st.expander("➕ Add New Trade / Strategy Builder"):
            tmpl_names = ["— Manual leg —"] + list(strategy_templates(spot, symbol).keys())
            tmpl = st.selectbox("Strategy template", tmpl_names)
            base_exp = st.date_input("Expiry", datetime.now() + timedelta(days=30),
                                     key="bld_exp")
            default_lot = LOT_SIZES.get(symbol, DEFAULT_STOCK_LOT_SIZE)
            lots_t = st.number_input("Lots per leg", min_value=1, value=1, key="bld_lots")
            if tmpl != "— Manual leg —":
                if st.button(f"Insert {tmpl} legs (BSM-priced, editable after)"):
                    new_legs = []
                    for L in strategy_templates(spot, symbol)[tmpl]:
                        exp_d = base_exp + timedelta(days=L['DTEoff'])
                        dte = max((datetime.combine(exp_d, datetime.min.time())
                                   - datetime.now()).days, 1)
                        px = float(BlackScholesEngine.price(
                            spot, L['Strike'], dte / 365, risk_free_rate,
                            live_iv, 'C' if L['Type'] == 'Call' else 'P'))
                        new_legs.append({
                            'ID': str(uuid.uuid4()), 'Symbol': symbol,
                            'Expiry': exp_d.strftime("%Y-%m-%d"),
                            'Strike': L['Strike'], 'Type': L['Type'],
                            'Action': L['Action'], 'Qty': L['Qty'] * lots_t,
                            'AvgPrice': round(px, 2), 'IV': round(live_iv * 100, 1),
                            'LotSize': default_lot, 'Booked': 0.0})
                    # reassign (not mutate) so Streamlit state stays reactive
                    st.session_state.positions = st.session_state.positions + new_legs
                    st.rerun()
            else:
                c1, c2, c3 = st.columns(3)
                action = c1.selectbox("Action", ["Sell", "Buy"])
                opt_type = c2.selectbox("Type", ["Call", "Put"])
                strike = c3.number_input("Strike",
                                         value=float(math.ceil(spot / 50) * 50), step=50.0)
                c4, c5, c6 = st.columns(3)
                entry = c4.number_input("Entry ₹", value=50.0, step=0.5)
                ivp = c5.number_input("IV %", value=float(live_iv * 100), step=0.5)
                lot_size = c6.number_input("Lot Size", min_value=1, value=int(default_lot))
                if st.button("Add Leg", use_container_width=True):
                    st.session_state.positions = st.session_state.positions + [{
                        'ID': str(uuid.uuid4()), 'Symbol': symbol,
                        'Expiry': base_exp.strftime("%Y-%m-%d"), 'Strike': strike,
                        'Type': opt_type, 'Action': action, 'Qty': lots_t,
                        'AvgPrice': entry, 'IV': ivp, 'LotSize': lot_size,
                        'Booked': 0.0}]
                    st.rerun()

        if not positions:
            st.info("No positions. Add a trade, insert a strategy template, "
                    "or import a CSV from the sidebar.")
            return None, None

        df_all = pd.DataFrame(positions)
        engine_all = PortfolioRiskEngine(df_all, spot, r=risk_free_rate)
        m_all = engine_all.get_metrics()
        live_all = engine_all.live.set_index('ID')

        b, u, t = m_all['booked'], m_all['unbooked'], m_all['total_mtm']
        st.markdown(f"""<div class="oo-card" style="padding:10px 14px">
          <div style="display:flex; gap:28px">
            <div><div class="mlab">Booked</div><div class="mval">{pnl_span(b)}</div></div>
            <div><div class="mlab">Unbooked</div><div class="mval">{pnl_span(u)}</div></div>
            <div><div class="mlab">Total P&amp;L</div><div class="mval">{pnl_span(t)}</div></div>
          </div></div>""", unsafe_allow_html=True)

        n_sel = sum(1 for p in positions if st.session_state.selected.get(p['ID'], True))
        st.caption(f"☑ {n_sel} of {len(positions)} selected")
        for i, p in enumerate(positions):
            lr = live_all.loc[p['ID']] if p['ID'] in live_all.index else None
            position_row(p, lr, key=i)

        with st.expander("✏️ Edit legs (bulk editor)"):
            edit_cols = ['Symbol', 'Expiry', 'Strike', 'Type', 'Action', 'Qty',
                         'AvgPrice', 'IV', 'LotSize', 'Booked']
            edited = st.data_editor(df_all[edit_cols], num_rows="dynamic",
                                    use_container_width=True, key="leg_editor")
            if st.button("Apply edits"):
                new_positions = []
                for _, r_ in edited.iterrows():
                    d = r_.to_dict()
                    d['ID'] = str(uuid.uuid4())
                    d.setdefault('Booked', 0.0)
                    new_positions.append(d)
                st.session_state.positions = new_positions
                st.session_state.selected = {}
                st.rerun()

        with st.expander("🗑️ Remove a leg"):
            names = {p['ID']: instrument_name(p) for p in positions}
            rid = st.selectbox("Leg", list(names.keys()), format_func=lambda x: names[x])
            if st.button("Remove selected leg"):
                st.session_state.positions = [p for p in positions if p['ID'] != rid]
                st.session_state.selected.pop(rid, None)
                st.rerun()

    sel_positions = [p for p in positions if st.session_state.selected.get(p['ID'], True)]
    if not sel_positions:
        with right:
            st.info("Select at least one position to analyse.")
        return None, None

    df_sel = pd.DataFrame(sel_positions)
    engine = PortfolioRiskEngine(df_sel, spot, r=risk_free_rate)
    metrics = engine.get_metrics()
    strat, outlook = StrategyEngine.detect(df_sel)

    with right:
        tab_payoff, tab_greeks, tab_pl, tab_risk = st.tabs(
            ["Payoff Graph", "Greeks", "P&L Table", "Risk"])

        with tab_payoff:
            metrics_strip(metrics)
            oi_data = fetch_oi_data(symbol)
            top1, top2 = st.columns([0.75, 0.25])
            if oi_data is not None:
                strikes, ce, pe, tot_ce, tot_pe = oi_data
                atm = strikes[np.argmin(np.abs(strikes - spot))]
                top1.markdown(
                    f"<span class='mut' style='font-size:.78rem'>OI data at {fmt_inr(atm,0)}: "
                    f"<span class='rd'>Call OI {fmt_compact(tot_ce)}</span> "
                    f"<span class='grn'>Put OI {fmt_compact(tot_pe)}</span></span>",
                    unsafe_allow_html=True)
                show_oi = top2.toggle("Open Interest", value=True)
            else:
                top1.markdown("<span class='mut' style='font-size:.78rem'>OI overlay "
                              "unavailable (needs live NSE chain)</span>",
                              unsafe_allow_html=True)
                show_oi = False

            dte = max(int(metrics['dte']), 0)
            rng = engine.price_range()
            ph_chart = st.container()
            target_price = st.slider("Target Price", float(round(rng[0])),
                                     float(round(rng[-1])), float(round(spot)),
                                     step=1.0, format="%d")
            days_to_target = st.slider(f"Days to Target — {dte}d to expiry",
                                       0, max(dte, 1), 0, format="%dd")
            with ph_chart:
                render_payoff_chart(engine, metrics, days_to_target, target_price,
                                    show_oi, oi_data)
                proj = engine.pnl_point(target_price, days_to_target)
                st.markdown(f"<div style='text-align:center; font-size:.82rem' class='mut'>"
                            f"Projected P&amp;L at {fmt_inr(target_price,0)}: "
                            f"{pnl_span(proj, compact=False)}</div>", unsafe_allow_html=True)
            est = engine.pnl_point(target_price, days_to_target)
            st.markdown(f"""<div class="pl-box">Estimated P&amp;L at
              {fmt_inr(target_price,0)} in {days_to_target}d:
              <span class="v {'grn' if est>=0 else 'rd'}">₹{fmt_inr(est)}</span></div>""",
                        unsafe_allow_html=True)

        with tab_greeks:
            render_greeks_tab(metrics, engine.live)
            with st.expander("🧬 Second & third-order Greeks (Vanna · Charm · Vomma · Speed · Color · Zomma)"):
                st.dataframe(second_order_table(engine), use_container_width=True,
                             hide_index=True)
                st.caption("Position-scaled. Vanna/Vomma/Zomma per 1 vol-pt; "
                           "Charm/Color per day; Speed per ₹1 of spot.")

        with tab_pl:
            cols = ['Symbol', 'Expiry', 'Strike', 'Type', 'Action', 'Qty',
                    'AvgPrice', 'LTP', 'Booked', 'Unbooked', 'MTM']
            st.dataframe(engine.live[cols].style.map(
                lambda v: f"color: {'#21ce99' if v >= 0 else '#ff5b6a'}",
                subset=['Booked', 'Unbooked', 'MTM']),
                use_container_width=True, hide_index=True)
            st.markdown(f"**Combined MTM:** {pnl_span(metrics['total_mtm'], compact=False)}",
                        unsafe_allow_html=True)

        with tab_risk:
            render_risk_tab(engine, metrics)

    with left:
        render_analysis_card(strat, outlook, engine.live, metrics, symbol)

    return engine, (metrics, strat, df_sel)


def main() -> None:
    apply_theme()

    if 'positions' not in st.session_state:
        expiry = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        st.session_state.positions = [
            {'ID': str(uuid.uuid4()), 'Symbol': 'NIFTY', 'Expiry': expiry,
             'Strike': 22500, 'Type': 'Put', 'Action': 'Sell', 'Qty': 30,
             'AvgPrice': 37.0, 'IV': 14.0, 'LotSize': 65, 'Booked': 0.0},
            {'ID': str(uuid.uuid4()), 'Symbol': 'NIFTY', 'Expiry': expiry,
             'Strike': 25500, 'Type': 'Call', 'Action': 'Sell', 'Qty': 42,
             'AvgPrice': 23.75, 'IV': 12.5, 'LotSize': 65, 'Booked': 0.0},
        ]
    st.session_state.setdefault('selected', {})
    for p in st.session_state.positions:
        st.session_state.selected.setdefault(p['ID'], True)

    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        symbol = st.text_input("Underlying Symbol", "NIFTY").upper().strip()
        risk_free_rate = st.number_input("Risk-Free Rate (%)", value=6.5, step=0.25) / 100
        st.markdown("---")
        up = st.file_uploader("📥 Import positions (CSV/Excel)", type=['csv', 'xlsx'],
                              help="Columns: Symbol, Expiry, Strike, Type, Action, "
                                   "Qty, AvgPrice, IV, LotSize[, Booked]")
        if up is not None:
            try:
                imp = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
                new_legs = []
                for _, r_ in imp.iterrows():
                    d = r_.to_dict()
                    d.setdefault('ID', str(uuid.uuid4()))
                    d.setdefault('Booked', 0.0)
                    new_legs.append(d)
                st.session_state.positions = st.session_state.positions + new_legs
                st.success(f"Imported {len(imp)} legs")
                st.rerun()
            except Exception as e:
                st.error(f"Import error: {e}")
        st.caption("NSE lot sizes revised periodically — confirm on nseindia.com.")

    spot, live_iv, prev_close = fetch_market_data(symbol)
    chg_pct = (spot - prev_close) / prev_close * 100 if prev_close else 0.0

    with st.sidebar:
        render_broker_sync(live_iv)

    st.markdown(f"""<div style="margin-bottom:8px">
      <div class="oo-title">{symbol} Analysis</div>
      <div class="oo-sub">Analyse combined payoff, Greeks and P&amp;L — institutional
      options analytics for NSE/BSE</div></div>""", unsafe_allow_html=True)

    tabs = st.tabs(["📊 Analyse", "⛓️ Option Chain", "🌊 Volatility",
                    "🎲 Probability", "🧠 Advisor", "⏪ Backtest", "🌐 Market",
                    "📓 Journal"])

    with tabs[0]:
        result = render_analyse_screen(symbol, spot, chg_pct, live_iv, risk_free_rate)
        engine = result[0] if result else None
        extras = result[1] if result else None

    with tabs[1]:
        render_chain_tab(symbol, risk_free_rate)
        render_gex_section(symbol, risk_free_rate)

    with tabs[2]:
        render_vol_tab(symbol, risk_free_rate, live_iv)
        render_vol_forecast_section(symbol)

    with tabs[3]:
        if engine is not None:
            render_prob_tab(engine, extras[0], live_iv)
        else:
            st.info("Add and select positions on the Analyse tab first.")

    with tabs[4]:
        render_advisor_tab(engine, extras, symbol, spot, live_iv, risk_free_rate)

    with tabs[5]:
        df_pos = pd.DataFrame(st.session_state.positions)
        render_backtest_tab(symbol, df_pos, risk_free_rate)

    with tabs[6]:
        render_market_tab(symbol, risk_free_rate)

    with tabs[7]:
        journal = TradeJournal()
        if engine is not None:
            metrics, strat, df_sel = extras
            render_journal_tab(journal, symbol, strat, spot, metrics,
                               df_sel.to_dict('records'))
        else:
            render_journal_tab(journal, symbol, "No Position", spot, {}, [])


if __name__ == "__main__":
    main()
