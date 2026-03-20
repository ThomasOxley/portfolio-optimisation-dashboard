#!/usr/bin/env python
# coding: utf-8

# In[2]:


# ================================================================================
# Full-Depth Portfolio Optimisation Dashboard
# ================================================================================
# Author:      Thomas Oxley
# Institution: Bristol Business School, UWE — MSc Financial Technology
# Date:        2024–2025
# Version:     1.0

# Description:
#    An interactive, end-to-end portfolio construction and risk analytics dashboard
#   built for JupyterLab. Combines four expected return models (MPT, CAPM, APT,
#    ML ensemble) with a four-engine Monte Carlo simulation framework, a complete
#    drawdown analytics suite, and an interactive Efficient Frontier Picker.

# Key Features:
#    - Expected return models: MPT (LedoitWolf shrinkage), CAPM (Ridge regression),
#      APT (PCA factor model), ML (Ridge + TimeSeriesSplit CV), and Blend ensemble
#    - Portfolio optimisation: Sharpe maximisation or MV Utility, with long/short,
#      vol targeting, Sharpe-weighted bounds, and beta-neutral penalty
#    - Subset selection: exhaustive search + random search for optimal asset subsets
#    - Monte Carlo: Normal, Student-t (fat tails), Regime-switching Student-t,
#      and Historical Bootstrap simulation engines
#    - Drawdown suite: stress test, density heatmap, breach probability curves,
#     time-under-water distributions, and max drawdown histogram
#      apply any frontier point as the active portfolio
#    - Interactive Efficient Frontier: click-to-preview with full MC + drawdowns;
#    - Dividend handling: reinvest or income mode
#    - Benchmark overlays: S&P 500, FTSE 100, Nasdaq
#    - Full diagnostics: per-model summaries, CAPM betas, APT factor exposures,
#      ML R² scores, covariance/correlation matrices
#    - Export: full CSV export of all outputs

# Dependencies:
#     See requirements.txt

# Usage:
#    Run the single cell in JupyterLab. Configure inputs using the 8-section
#    Accordion UI, then click "Build Portfolio".

# Note:
#    This is an academic research tool. Outputs are for educational and
#    illustrative purposes and do not constitute financial advice.
# ================================================================================

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML, Javascript
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime
import itertools, math, json, warnings


warnings.filterwarnings("ignore")
np.random.seed(42)

# ============================================================
# Floating tooltip system (JupyterLab-safe)
# ============================================================
def _tt_payload(title: str, body_html: str) -> str:
    return json.dumps({"t": title, "b": body_html}).replace('"', "&quot;")

def qmark(title: str, body_html: str) -> str:
    return f'<span class="mpt-help" data-tt="{_tt_payload(title, body_html)}">?</span>'

def install_tooltips_jlab():
    css = r"""
    <style>
    .mpt-help{
      display:inline-flex; align-items:center; justify-content:center;
      margin-left:6px; width:18px; height:18px; border-radius:999px;
      border:1px solid rgba(0,0,0,0.35); color:rgba(0,0,0,0.75);
      font-size:12px; line-height:18px; cursor:pointer; user-select:none;
      vertical-align:middle;
    }
    .mpt-help:hover{ background:rgba(0,0,0,0.05); }
    #mptTTOverlay{
      position:fixed; z-index:2147483647;
      width:460px; max-width:min(460px, 92vw);
      background:rgba(20,20,20,0.96); color:#fff;
      padding:10px 12px; border-radius:12px;
      box-shadow:0 10px 30px rgba(0,0,0,0.25);
      font-size:12.8px; line-height:1.25rem;
      opacity:0; visibility:hidden;
      transition:opacity 0.12s ease-in;
      pointer-events:auto;
    }
    #mptTTOverlay.visible{ opacity:1; visibility:visible; }
    #mptTTOverlay .ttTitle{ font-weight:800; margin-bottom:6px; }
    #mptTTOverlay .ttBody{ color:rgba(255,255,255,0.92); }
    .mpt-note{ color:#555; font-size:12.5px; line-height:1.25rem; margin:6px 0 0 0; }
    .mpt-note b{ color:#333; }
    </style>
    """
    display(HTML(css))

    js = r"""
    (function(){
      let overlay = document.getElementById("mptTTOverlay");
      if(!overlay){
        overlay = document.createElement("div");
        overlay.id = "mptTTOverlay";
        overlay.innerHTML = '<div class="ttTitle"></div><div class="ttBody"></div>';
        document.body.appendChild(overlay);
      }
      const titleEl = overlay.querySelector(".ttTitle");
      const bodyEl = overlay.querySelector(".ttBody");
      function hide(){ overlay.classList.remove("visible"); }
      function showAt(clientX, clientY, title, body){
        titleEl.textContent = title || "Help";
        bodyEl.innerHTML = body || "";
        overlay.style.left = "0px";
        overlay.style.top  = "0px";
        overlay.classList.add("visible");
        const pad = 14;
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        const ow = overlay.offsetWidth || 460;
        const oh = overlay.offsetHeight || 160;
        let x = clientX + 14;
        let y = clientY + 14;
        if(x + ow + pad > vw) x = Math.max(pad, vw - ow - pad);
        if(y + oh + pad > vh) y = Math.max(pad, vh - oh - pad);
        overlay.style.left = x + "px";
        overlay.style.top  = y + "px";
      }
      function onDocClick(e){
        const t = e.target;
        if(t && t.classList && t.classList.contains("mpt-help")){
          e.preventDefault(); e.stopPropagation();
          const raw = t.getAttribute("data-tt") || "";
          let obj = null;
          try { obj = JSON.parse(raw); }
          catch(err){
            try { obj = JSON.parse(raw.replace(/&quot;/g,'"')); }
            catch(e2){ obj = {t:"Help", b: raw}; }
          }
          showAt(e.clientX, e.clientY, obj.t, obj.b);
          return;
        }
        if(t && (t.id === "mptTTOverlay" || (t.closest && t.closest("#mptTTOverlay")))) return;
        hide();
      }
      if(!window.__mpt_tt_bound){
        document.addEventListener("click", onDocClick, true);
        document.addEventListener("keydown", (ev)=>{ if(ev.key==="Escape") hide(); }, true);
        window.addEventListener("scroll", hide, {passive:true});
        window.addEventListener("resize", hide, {passive:true});
        window.__mpt_tt_bound = true;
      }
    })();
    """
    display(Javascript(js))

install_tooltips_jlab()

# ============================================================
# CONFIG
# ============================================================
TRADING_DAYS = 252
START_DATE_DEFAULT = "2018-01-01"
RISK_FREE_DEFAULT = 0.042
INFLATION_DEFAULT = 0.025

DEFAULT_ASSETS = ["AAPL","MSFT","NVDA","GOOGL","META","AMZN","JPM","AVGO","LLY","XOM",
                  "UNH","COST","TSLA","AMD","PEP","AZN.L","SHEL.L","HSBA.L","ULVR.L",
                  "BP.L","RIO.L","GSK.L","DGE.L","REL.L","NG.L","VOD.L","BATS.L","LSEG.L",
                  "BARC.L","AAL.L"]

BENCHMARKS = {"S&P 500": "^GSPC", "FTSE 100": "^FTSE", "Nasdaq": "^IXIC"}

# ============================================================
# Active override state (Frontier Picker -> Apply)
# ============================================================
_ACTIVE = {"enabled": False, "chosen_assets": None, "w_risky": None, "mu_model": None, "label": None}

# ============================================================
# Small helpers
# ============================================================
def _parse_tickers(text: str):
    tickers = [t.strip() for t in text.split(",") if t.strip()]
    out, seen = [], set()
    for t in tickers:
        if t not in seen:
            out.append(t); seen.add(t)
    return out

def is_crypto_ticker(t: str) -> bool:
    u = t.upper()
    return ("-USD" in u) or ("-GBP" in u) or u.endswith("USD") or u.endswith("GBP") or u in {"BTC","ETH"}

def _as_bizdays(df: pd.DataFrame):
    return df.sort_index().asfreq("B")

# ============================================================
# Robust prices + dividends download
# ============================================================
def download_prices_and_dividends(assets, start_date):
    raw = yf.download(
        assets, start=start_date,
        auto_adjust=False,
        actions=True,
        progress=False,
        group_by="column"
    )
    if raw is None or raw.empty:
        raise ValueError("yfinance returned empty data. Check tickers/date range.")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" in raw.columns.get_level_values(0):
            px = raw["Adj Close"]
        elif "Close" in raw.columns.get_level_values(0):
            px = raw["Close"]
        else:
            raise ValueError("yfinance data did not include 'Adj Close' or 'Close'.")
        div = raw["Dividends"] if ("Dividends" in raw.columns.get_level_values(0)) else None
    else:
        px = raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"]
        div = raw["Dividends"] if "Dividends" in raw.columns else None

    px = px.dropna(how="all")
    if div is None:
        div = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    else:
        div = div.reindex_like(px).fillna(0.0)

    if (div.sum().sum() == 0) and (len(px.columns) > 0):
        div_fix = pd.DataFrame(0.0, index=px.index, columns=px.columns)
        for t in px.columns:
            try:
                d = yf.Ticker(t).dividends
                if d is None or d.empty:
                    continue
                d = d[d.index >= pd.to_datetime(start_date)]
                d = d.reindex(px.index, fill_value=0.0)
                div_fix[t] = d.values
            except Exception:
                pass
        if div_fix.sum().sum() > 0:
            div = div_fix.reindex_like(px).fillna(0.0)

    return px, div

# ============================================================
# Cleaning + returns
# ============================================================
def _robust_returns_from_prices(prices: pd.DataFrame,
                                align_business_days: bool = True,
                                ffill_limit: int = 3,
                                min_obs_per_asset: int = 252,
                                min_day_coverage: float = 0.80,
                                min_asset_coverage: float = 0.80):
    px = prices.copy()
    failed = [c for c in px.columns if px[c].isna().all()]
    if failed: px = px.drop(columns=failed)

    px = px.dropna(axis=1, thresh=int(min_obs_per_asset))
    if px.shape[1] < 2:
        raise ValueError(f"Too few assets after cleaning prices (need ≥2). Remaining={px.shape[1]}.")

    if align_business_days: px = _as_bizdays(px)
    if ffill_limit and int(ffill_limit) > 0: px = px.ffill(limit=int(ffill_limit))

    r = px.pct_change()
    r = r.dropna(thresh=int(min_day_coverage * r.shape[1]))
    r = r.dropna(axis=1, thresh=int(min_asset_coverage * r.shape[0]))

    if r.shape[0] < 2: raise ValueError("Returns became empty after cleaning. Relax thresholds or extend date range.")
    if r.shape[1] < 2: raise ValueError("Too few assets after cleaning returns. Relax thresholds.")
    return r, px, failed

def autotune_cleaning_settings(prices_raw: pd.DataFrame, assets: list[str]):
    n_assets = len(assets)
    has_crypto = any(is_crypto_ticker(a) for a in assets)
    align_bd = True
    ffill = 3 if has_crypto else 2
    counts = prices_raw.notna().sum()
    med = int(np.nanmedian(counts.values)) if len(counts) else 0
    min_obs = int(np.clip(min(504, med), 126, 756))
    if n_assets >= 30:
        day_cov = 0.65; asset_cov = 0.65
    elif n_assets >= 15:
        day_cov = 0.75; asset_cov = 0.75
    else:
        day_cov = 0.85; asset_cov = 0.85
    return align_bd, ffill, min_obs, day_cov, asset_cov

def try_cleaning_with_fallback(prices_raw: pd.DataFrame, align_bd, ffill, min_obs, day_cov, asset_cov):
    ladders = [
        (align_bd, ffill, min_obs, day_cov, asset_cov),
        (align_bd, max(ffill, 1), max(126, min_obs - 126), min(day_cov, 0.70), min(asset_cov, 0.70)),
        (align_bd, max(ffill, 1), 126, 0.60, 0.60),
        (align_bd, 0, 126, 0.55, 0.55),
    ]
    last_err = None
    for p in ladders:
        try:
            r, px, failed = _robust_returns_from_prices(
                prices_raw, align_business_days=p[0], ffill_limit=p[1],
                min_obs_per_asset=p[2], min_day_coverage=p[3], min_asset_coverage=p[4]
            )
            return r, px, failed, p
        except Exception as e:
            last_err = e
    raise last_err

# ============================================================
# MPT core
# ============================================================
def estimate_mu_cov(r: pd.DataFrame, inflation: float):
    mu = (r.mean() - 0.5 * r.var()) * TRADING_DAYS - inflation
    cov = LedoitWolf().fit(r.values).covariance_ * TRADING_DAYS
    return mu.values, cov

def portfolio_perf(w: np.ndarray, mu: np.ndarray, cov: np.ndarray):
    ret = float(w @ mu)
    vol = float(np.sqrt(max(1e-18, w.T @ cov @ w)))
    return ret, vol

def risk_contribution(w: np.ndarray, cov: np.ndarray):
    port_var = float(max(1e-18, w.T @ cov @ w))
    port_vol = float(np.sqrt(port_var))
    mrc = (cov @ w) / port_vol
    rc = w * mrc
    s = rc.sum()
    rc_pct = rc / (s if abs(s) > 1e-12 else 1.0)
    return rc, rc_pct

def per_asset_sharpe(mu: np.ndarray, cov: np.ndarray, risk_free: float) -> np.ndarray:
    vols = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    s = (mu - risk_free) / vols
    return np.clip(s, -3.0, 6.0)

def build_max_weight_bounds(mu: np.ndarray, cov: np.ndarray, risk_free: float,
                            hard_cap: float,
                            long_short: bool,
                            use_sharpe_caps: bool,
                            sharpe_strength: float):
    n = len(mu)
    cap = float(np.clip(hard_cap, 0.02, 1.0))
    lo_cap = -cap if long_short else 0.0
    hi_cap = cap

    if (not use_sharpe_caps) or sharpe_strength <= 0:
        return [(lo_cap, hi_cap)] * n

    s = per_asset_sharpe(mu, cov, risk_free=risk_free)
    s_pos = np.maximum(s, 0.0)
    if np.all(s_pos == 0):
        return [(lo_cap, hi_cap)] * n

    s_norm = s_pos / (np.max(s_pos) if np.max(s_pos) > 0 else 1.0)
    base = cap * (1.0 - 0.5 * float(sharpe_strength))
    boost = cap * (1.0 + 0.25 * float(sharpe_strength))
    max_abs = np.clip(base + (boost - base) * s_norm, 0.01, cap)
    if long_short:
        return [(-float(m), float(m)) for m in max_abs]
    return [(0.0, float(m)) for m in max_abs]

def optimise(mu: np.ndarray, cov: np.ndarray, mode: str, risk_aversion: float, risk_free: float,
             bounds, long_short: bool, max_gross: float,
             beta_penalty: float = 0.0, beta_vec: np.ndarray | None = None):
    n = len(mu)
    w0 = np.ones(n) / n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if long_short:
        mg = float(max(1.0, max_gross))
        cons.append({"type": "ineq", "fun": lambda w, mg=mg: mg - np.sum(np.abs(w))})

    def obj(w):
        r, v = portfolio_perf(w, mu, cov)
        if mode == "Sharpe":
            core = -((r - risk_free) / v) if v > 0 else 1e9
        else:
            core = -(r - 0.5 * risk_aversion * v * v)
        if beta_penalty > 0 and (beta_vec is not None):
            b = float(np.dot(w, beta_vec))
            core += float(beta_penalty) * (b ** 2)
        return core

    res = minimize(obj, w0, bounds=bounds, constraints=cons, method="SLSQP",
                   options={"maxiter": 500, "ftol": 1e-10})
    return res.x if res.success else w0

# ============================================================
# Dividends
# ============================================================
def dividend_yield_by_asset(prices: pd.DataFrame, dividends: pd.DataFrame):
    prices = prices.copy()
    dividends = dividends.reindex_like(prices).fillna(0.0)
    div_ttm = dividends.rolling(window=252, min_periods=1).sum().iloc[-1]
    last_px = prices.iloc[-1]
    yld = (div_ttm / last_px).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return yld

def portfolio_dividend_yield(w, prices: pd.DataFrame, dividends: pd.DataFrame):
    if isinstance(w, pd.Series):
        w_series = w.copy()
    else:
        w_series = pd.Series(np.asarray(w, dtype=float), index=prices.columns[:len(np.asarray(w))])

    cols = [c for c in w_series.index if c in prices.columns and c in dividends.columns]
    if len(cols) == 0:
        return 0.0
    wv = w_series.loc[cols].values.astype(float)
    yld = dividend_yield_by_asset(prices[cols], dividends[cols]).values.astype(float)
    return float(np.nansum(wv * yld))

def contribution_steps_per_year(freq: str):
    return {"Monthly": 12, "Quarterly": 4, "Annual": 1}.get(freq, 12)

def rebalance_steps_per_year(freq: str):
    return {"None": 0, "Monthly": 12, "Quarterly": 4, "Annual": 1}.get(freq, 0)

# ============================================================
# CAPM + APT + ML expected returns
# ============================================================
def monthly_asset_returns_from_daily(returns: pd.DataFrame) -> pd.DataFrame:
    return (1.0 + returns).resample("M").prod() - 1.0

def capm_estimates(asset_monthly: pd.DataFrame, bench_monthly: pd.Series, rf_annual: float):
    rf_m = rf_annual / 12.0
    y = asset_monthly.sub(rf_m, axis=0).dropna(how="all")
    x = (bench_monthly - rf_m).reindex(y.index).dropna()
    y = y.reindex(x.index).dropna(how="any")
    if y.empty or x.empty:
        raise ValueError("Not enough overlapping monthly data for CAPM.")
    xm = x.values.reshape(-1, 1)
    betas, alphas = {}, {}
    for c in y.columns:
        yc = y[c].values
        reg = Ridge(alpha=1.0, fit_intercept=True).fit(xm, yc)
        betas[c] = float(reg.coef_[0])
        alphas[c] = float(reg.intercept_)
    beta = pd.Series(betas)
    alpha = pd.Series(alphas)
    mkt_prem_m = float(x.mean())
    capm_mu_m = rf_m + beta * mkt_prem_m + alpha
    capm_mu_ann = (1.0 + capm_mu_m).pow(12) - 1.0
    return beta, alpha, capm_mu_ann

def apt_statistical(asset_monthly: pd.DataFrame, n_factors: int, include_benchmark: bool, bench_monthly: pd.Series | None):
    X = asset_monthly.dropna(how="any")
    if X.shape[0] < 24:
        raise ValueError("APT needs more monthly history (try earlier start date).")

    if include_benchmark and bench_monthly is not None:
        bm = bench_monthly.reindex(X.index)
        if bm.notna().sum() > 0:
            X = X.copy()
            X["__BM__"] = bm

    Z = (X - X.mean()) / (X.std(ddof=0) + 1e-12)
    k = int(np.clip(n_factors, 1, min(10, Z.shape[1]-1)))
    pca = PCA(n_components=k).fit(Z.values)
    factors = pd.DataFrame(pca.transform(Z.values), index=Z.index, columns=[f"F{i+1}" for i in range(k)])
    premia = factors.mean(axis=0)
    loadings = pd.DataFrame(pca.components_.T, index=Z.columns, columns=factors.columns)
    z_mu = loadings @ premia
    mu_m = (z_mu * (X.std(ddof=0) + 1e-12) + X.mean())
    mu_m = mu_m.drop(index="__BM__", errors="ignore")
    mu_ann = (1.0 + mu_m).pow(12) - 1.0
    exposures = loadings.drop(index="__BM__", errors="ignore")
    return mu_ann, exposures, premia, factors

def ml_predict_next_month_mu(asset_monthly: pd.DataFrame, bench_monthly: pd.Series | None, n_lags: int = 6):
    X = asset_monthly.dropna(how="any")
    if X.shape[0] < (24 + n_lags + 1):
        raise ValueError("ML forecast needs more monthly history (try earlier start date).")
    bm = bench_monthly.reindex(X.index) if bench_monthly is not None else None

    mu_next = {}
    r2s = {}
    for c in X.columns:
        s = X[c].copy()
        df = pd.DataFrame({"y": s.shift(-1)})
        for k in range(1, n_lags+1):
            df[f"lag{k}"] = s.shift(k)
        if bm is not None:
            for k in range(1, min(3, n_lags)+1):
                df[f"bm_lag{k}"] = bm.shift(k)
        df = df.dropna()
        if df.shape[0] < 30:
            mu_next[c] = float(s.iloc[-1])
            r2s[c] = np.nan
            continue

        Xmat = df.drop(columns=["y"]).values
        y = df["y"].values
        tscv = TimeSeriesSplit(n_splits=4)
        scores = []
        for tr, te in tscv.split(Xmat):
            reg = Ridge(alpha=10.0).fit(Xmat[tr], y[tr])
            scores.append(reg.score(Xmat[te], y[te]))
        r2 = float(np.nanmean(scores)) if len(scores) else np.nan

        reg = Ridge(alpha=10.0).fit(Xmat, y)
        last_row = df.drop(columns=["y"]).iloc[[-1]].values
        pred = float(reg.predict(last_row)[0])

        mu_next[c] = pred
        r2s[c] = r2
    return pd.Series(mu_next), pd.Series(r2s)

# ============================================================
# Monte Carlo models (normal / t / regime_t / bootstrap)
# ============================================================
def sample_multivariate_t(rng, df: int, dim: int) -> np.ndarray:
    z = rng.standard_normal(dim)
    u = rng.chisquare(df)
    scale = np.sqrt(u / df) if u > 0 else 1.0
    return z / scale

def simulate_paths(mu_ann, cov_ann, w_risky, rf_weight, years, initial, contrib_amount, contrib_freq,
                   rebalance_freq, runs, risk_free, prices, dividends,
                   reinvest_divs=True, seed=42,
                   mc_model="t", t_df=6,
                   regime_high_vol_mult=2.2, p_stay_low=0.95, p_stay_high=0.85,
                   hist_monthly_asset_rets: pd.DataFrame | None = None):

    rng = np.random.default_rng(seed)
    steps = years * 12

    mu_m = np.asarray(mu_ann) / 12.0
    cov_m = np.asarray(cov_ann) / 12.0
    rf_m = float(risk_free) / 12.0

    w_series = pd.Series(np.asarray(w_risky, dtype=float), index=list(prices.columns))
    div_y = portfolio_dividend_yield(w_series, prices, dividends)
    div_m = div_y / 12.0

    contrib_per_year = contribution_steps_per_year(contrib_freq)
    contrib_interval = int(12 / contrib_per_year) if contrib_per_year else 1

    reb_per_year = rebalance_steps_per_year(rebalance_freq)
    reb_interval = int(12 / reb_per_year) if reb_per_year else 0

    try:
        L = np.linalg.cholesky(cov_m + 1e-12*np.eye(len(mu_m)))
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(np.diag(np.clip(np.diag(cov_m), 1e-12, None)))

    boot_vals = None
    if mc_model == "bootstrap":
        if hist_monthly_asset_rets is None or hist_monthly_asset_rets.empty:
            raise ValueError("Bootstrap selected but monthly asset returns are empty.")
        boot = hist_monthly_asset_rets.dropna(how="any")
        if boot.shape[0] < 24:
            raise ValueError("Bootstrap needs more monthly history. Try an earlier start date.")
        boot_vals = boot.values

    values = np.zeros((runs, steps), dtype=float)
    incomes = np.zeros((runs, steps), dtype=float)
    gross_rets = np.zeros((runs, steps), dtype=float)
    reinv_divs = np.zeros((runs, steps), dtype=float)

    for i in range(runs):
        income = 0.0
        reinv_cum = 0.0

        risky_bucket = float(initial) * (1 - rf_weight)
        rf_bucket = float(initial) * rf_weight

        state = 0

        for t in range(steps):
            total_before = risky_bucket + rf_bucket

            contrib_this = 0.0
            if t % contrib_interval == 0:
                contrib_this = float(contrib_amount)
                risky_bucket += contrib_this * (1 - rf_weight)
                rf_bucket += contrib_this * rf_weight

            if mc_model == "normal":
                z = rng.standard_normal(len(mu_m))
                asset_ret = mu_m + (L @ z)
            elif mc_model == "t":
                zt = sample_multivariate_t(rng, int(t_df), len(mu_m))
                asset_ret = mu_m + (L @ zt)
            elif mc_model == "regime_t":
                if state == 0:
                    state = 0 if (rng.random() < float(p_stay_low)) else 1
                else:
                    state = 1 if (rng.random() < float(p_stay_high)) else 0
                mult = 1.0 if state == 0 else float(regime_high_vol_mult)
                zt = sample_multivariate_t(rng, int(t_df), len(mu_m))
                asset_ret = mu_m + mult * (L @ zt)
            elif mc_model == "bootstrap":
                j = rng.integers(0, boot_vals.shape[0])
                asset_ret = boot_vals[j, :].astype(float)
            else:
                raise ValueError(f"Unknown mc_model={mc_model}")

            rp = float(np.dot(w_risky, asset_ret))

            risky_bucket *= (1.0 + rp)
            rf_bucket *= (1.0 + rf_m)

            div_cash = risky_bucket * div_m
            if reinvest_divs:
                risky_bucket += div_cash
                reinv_cum += div_cash
            else:
                income += div_cash

            if reb_interval and (t+1) % reb_interval == 0:
                total = risky_bucket + rf_bucket
                risky_bucket = total * (1 - rf_weight)
                rf_bucket = total * rf_weight

            total_after = risky_bucket + rf_bucket
            gross_rets[i, t] = (total_after - contrib_this) / total_before - 1.0 if total_before > 0 else 0.0
            values[i, t] = total_after
            incomes[i, t] = income
            reinv_divs[i, t] = reinv_cum

    return values, incomes, gross_rets, reinv_divs

# ============================================================
# Drawdowns helpers (full suite)
# ============================================================
def drawdown_from_path(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if v.size == 0 or np.all(np.isnan(v)):
        return v
    peak = np.maximum.accumulate(np.where(np.isnan(v), -np.inf, v))
    dd = (v / peak) - 1.0
    dd[np.isinf(dd)] = np.nan
    return dd

def drawdown_from_returns(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    idx = np.cumprod(1.0 + r)
    return drawdown_from_path(idx)

def drawdown_density_heatmap(dd_subset: np.ndarray, bins: int = 45, clip_lo: int = 1, clip_hi: int = 99):
    A = np.asarray(dd_subset, dtype=float)
    A = A[~np.isnan(A).all(axis=1)]
    if A.size == 0: return None, None, None
    flat = A[np.isfinite(A)]
    if flat.size == 0: return None, None, None
    lo = np.percentile(flat, clip_lo)
    hi = np.percentile(flat, clip_hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = np.nanmin(flat), np.nanmax(flat)
    edges = np.linspace(lo, hi, int(bins) + 1)
    steps = A.shape[1]
    z = np.zeros((int(bins), steps), dtype=float)
    for t in range(steps):
        col = A[:, t]
        col = col[np.isfinite(col)]
        if col.size == 0: continue
        hist, _ = np.histogram(np.clip(col, lo, hi), bins=edges)
        z[:, t] = hist
    denom = max(1, A.shape[0])
    z = z / denom
    y_centers = 0.5 * (edges[:-1] + edges[1:]) * 100.0
    x = np.arange(1, steps + 1)
    return z, y_centers, x

def breach_probabilities(dd_subset: np.ndarray, thresholds_pct: list[int]):
    A = np.asarray(dd_subset, dtype=float)
    probs = {}
    for thr in thresholds_pct:
        thr_dec = thr / 100.0
        probs[thr] = np.nanmean(A <= thr_dec, axis=0)
    return probs

def time_under_water_months_from_returns(gross_ret_path: np.ndarray):
    r = np.asarray(gross_ret_path, dtype=float)
    idx = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(idx)
    underwater = idx < peak - 1e-12
    durs = []
    cur = 0
    for u in underwater:
        if u: cur += 1
        else:
            if cur > 0:
                durs.append(cur); cur = 0
    if cur > 0: durs.append(cur)
    return durs

# ============================================================
# Rolling correlation helpers
# ============================================================
def rolling_corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    df = pd.concat([a, b], axis=1).dropna()
    if df.empty: return pd.Series(dtype=float)
    return df.iloc[:,0].rolling(window=window, min_periods=max(5, window//4)).corr(df.iloc[:,1])

def portfolio_daily_returns(returns: pd.DataFrame, w_risky: np.ndarray, rf_weight: float, risk_free_ann: float) -> pd.Series:
    r_risky = pd.Series(returns.values @ w_risky, index=returns.index, name="risky")
    rf_daily = (risk_free_ann / TRADING_DAYS)
    r_total = (1.0 - rf_weight) * r_risky + rf_weight * rf_daily
    r_total.name = "portfolio"
    return r_total

# ============================================================
# Sortino helpers (NEW)
# ============================================================
def downside_deviation_ann(r_daily: pd.Series, mar_annual: float = 0.0) -> float:
    if r_daily is None or len(r_daily) < 5:
        return np.nan
    mar_d = float(mar_annual) / TRADING_DAYS
    d = (r_daily - mar_d).values
    d = d[d < 0.0]
    if d.size < 2:
        return 0.0
    return float(np.sqrt(np.mean(d**2)) * np.sqrt(TRADING_DAYS))

def sortino_ratio_ann(r_daily: pd.Series, rf_annual: float) -> float:
    if r_daily is None or len(r_daily) < 5:
        return np.nan
    mu_ann = float(r_daily.mean() * TRADING_DAYS)
    dd = downside_deviation_ann(r_daily, mar_annual=rf_annual)
    if dd <= 0:
        return np.nan
    return float((mu_ann - float(rf_annual)) / dd)

# ============================================================
# Vol targeting helper
# ============================================================
def apply_vol_target(rf_weight: float, port_vol_risky: float, vol_target: float) -> float:
    vol_target = float(vol_target)
    if vol_target <= 0 or port_vol_risky <= 0: return rf_weight
    risky_frac = 1.0 - float(rf_weight)
    cur_total_vol = risky_frac * port_vol_risky
    if cur_total_vol <= 0: return rf_weight
    new_risky_frac = risky_frac * (vol_target / cur_total_vol)
    new_risky_frac = float(np.clip(new_risky_frac, 0.0, 1.0))
    return 1.0 - new_risky_frac

# ============================================================
# Benchmark helpers
# ============================================================
def benchmark_paths(bench_rets: pd.DataFrame, years, initial, contrib_amount, contrib_freq, label_map):
    steps = years * 12
    contrib_per_year = contribution_steps_per_year(contrib_freq)
    contrib_interval = int(12 / contrib_per_year) if contrib_per_year else 1
    out = {}
    for name, ticker in label_map.items():
        if ticker not in bench_rets.columns:
            out[name] = np.full(steps, np.nan); continue
        daily = bench_rets[ticker].dropna()
        if daily.empty:
            out[name] = np.full(steps, np.nan); continue
        monthly = (1 + daily).resample("M").prod() - 1
        if len(monthly) < steps:
            monthly = pd.concat([monthly] * ((steps // len(monthly)) + 1), ignore_index=True)
        v = float(initial)
        path = np.zeros(steps)
        for t in range(steps):
            if t % contrib_interval == 0:
                v += float(contrib_amount)
            v *= (1.0 + float(monthly.iloc[t]))
            path[t] = v
        out[name] = path
    return out

def benchmark_monthly_returns(bench_rets: pd.DataFrame, years: int, label_map: dict) -> dict:
    steps = years * 12
    out = {}
    for name, ticker in label_map.items():
        if ticker not in bench_rets.columns:
            out[name] = pd.Series(dtype=float); continue
        daily = bench_rets[ticker].dropna()
        if daily.empty:
            out[name] = pd.Series(dtype=float); continue
        m = (1 + daily).resample("M").prod() - 1
        if len(m) < steps:
            m = pd.concat([m] * ((steps // len(m)) + 1), ignore_index=True)
        out[name] = m.iloc[:steps].reset_index(drop=True)
    return out

# ============================================================
# Subset selection helpers
# ============================================================
def nCk(n, k):
    if k < 0 or k > n: return 0
    return math.comb(n, k)

def subset_search_best(assets, mu_s: pd.Series, cov_df: pd.DataFrame,
                       mode, A, rf, bounds_builder,
                       long_short, max_gross,
                       subset_size: int,
                       exhaustive_limit: int,
                       random_budget: int,
                       beta_penalty: float,
                       beta_vec_full: pd.Series | None,
                       seed=42):
    rng = np.random.default_rng(seed)
    N = len(assets)
    k = int(np.clip(subset_size, 2, N))
    combos = nCk(N, k)

    def score_of(sub):
        mu = mu_s.loc[sub].values
        cov = cov_df.loc[sub, sub].values
        beta_vec = beta_vec_full.loc[sub].values if beta_vec_full is not None else None
        bounds = bounds_builder(mu, cov)
        w = optimise(mu, cov, mode, A, rf, bounds, long_short, max_gross, beta_penalty=beta_penalty, beta_vec=beta_vec)
        r, v = portfolio_perf(w, mu, cov)
        s = ((r - rf) / v) if (mode == "Sharpe" and v > 0) else (r - 0.5 * A * v * v)
        return w, r, v, s

    if combos <= exhaustive_limit:
        best = None
        best_score = -1e18
        for sub in itertools.combinations(assets, k):
            sub = list(sub)
            w, r, v, s = score_of(sub)
            if s > best_score:
                best_score = s
                best = (sub, w, r, v, s, "Exhaustive", combos, combos)
        return best

    tried = set()
    best = None
    best_score = -1e18
    budget = int(max(50, random_budget))
    for _ in range(budget):
        sub = tuple(sorted(rng.choice(assets, size=k, replace=False)))
        if sub in tried: continue
        tried.add(sub)
        sub = list(sub)
        w, r, v, s = score_of(sub)
        if s > best_score:
            best_score = s
            best = (sub, w, r, v, s, "Random search", combos, len(tried))
    return best

# ============================================================
# Frontier solver for click-pick (NEW)
# ============================================================
def solve_min_var_for_target_return(mu: np.ndarray, cov: np.ndarray, target_ret: float,
                                   bounds, long_short: bool, max_gross: float,
                                   beta_penalty: float = 0.0, beta_vec: np.ndarray | None = None):
    n = len(mu)
    w0 = np.ones(n) / n
    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w, tr=target_ret: float(w @ mu) - float(tr)},
    ]
    if long_short:
        mg = float(max(1.0, max_gross))
        cons.append({"type": "ineq", "fun": lambda w, mg=mg: mg - np.sum(np.abs(w))})

    def obj(w):
        v = float(w.T @ cov @ w)
        if beta_penalty > 0 and (beta_vec is not None):
            b = float(np.dot(w, beta_vec))
            v += float(beta_penalty) * (b ** 2)
        return v

    res = minimize(obj, w0, bounds=bounds, constraints=cons, method="SLSQP",
                   options={"maxiter": 800, "ftol": 1e-10})
    return res.x if res.success else w0

def efficient_frontier_with_weights(mu: np.ndarray, cov: np.ndarray, points: int,
                                    bounds, long_short: bool, max_gross: float,
                                    beta_penalty: float = 0.0, beta_vec: np.ndarray | None = None):
    targets = np.linspace(float(np.nanmin(mu)), float(np.nanmax(mu)), int(points))
    vols, rets, ws = [], [], []
    for tr in targets:
        w = solve_min_var_for_target_return(mu, cov, tr, bounds, long_short, max_gross,
                                            beta_penalty=beta_penalty, beta_vec=beta_vec)
        r, v = portfolio_perf(w, mu, cov)
        if np.isfinite(r) and np.isfinite(v) and v > 0:
            vols.append(v); rets.append(r); ws.append(w)
    return np.asarray(vols), np.asarray(rets), ws

# ============================================================
# Efficient frontier (simple long-only view retained)
# ============================================================
def efficient_frontier(mu: np.ndarray, cov: np.ndarray, points: int = 60):
    n = len(mu)
    w0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    targets = np.linspace(float(mu.min()), float(mu.max()), points)
    vols, rets = [], []
    for t in targets:
        cons = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, tt=t: (w @ mu) - tt},
        )
        res = minimize(lambda w: float(w.T @ cov @ w), w0, bounds=bounds, constraints=cons, method="SLSQP")
        if res.success:
            vols.append(float(np.sqrt(max(1e-18, res.fun))))
            rets.append(float(t))
    return np.array(vols), np.array(rets)

# ============================================================
# Export
# ============================================================
def export_reports(summary_df, weights_df, holdings_df, setup_df, diagnostics):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"portfolio_export_{ts}"
    files = []
    paths = {
        "summary": f"{base}_summary.csv",
        "weights": f"{base}_weights.csv",
        "holdings": f"{base}_holdings.csv",
        "setup": f"{base}_portfolio_setup.csv",
        "mu": f"{base}_mu_selected.csv",
        "cov": f"{base}_cov_selected.csv",
        "corr": f"{base}_corr_selected.csv",
        "capm": f"{base}_capm.csv",
        "apt_mu": f"{base}_apt_mu.csv",
        "ml": f"{base}_ml.csv",
    }
    summary_df.to_csv(paths["summary"], index=False); files.append(paths["summary"])
    weights_df.to_csv(paths["weights"]); files.append(paths["weights"])
    holdings_df.to_csv(paths["holdings"], index=False); files.append(paths["holdings"])
    setup_df.to_csv(paths["setup"], index=False); files.append(paths["setup"])
    diagnostics["mu_selected"].to_csv(paths["mu"]); files.append(paths["mu"])
    diagnostics["cov"].to_csv(paths["cov"]); files.append(paths["cov"])
    diagnostics["corr"].to_csv(paths["corr"]); files.append(paths["corr"])
    diagnostics["capm"].to_csv(paths["capm"]); files.append(paths["capm"])
    diagnostics["apt_mu"].to_csv(paths["apt_mu"]); files.append(paths["apt_mu"])
    diagnostics["ml"].to_csv(paths["ml"]); files.append(paths["ml"])
    return files

# ============================================================
# UI widgets
# ============================================================
tickers_box = widgets.Textarea(value=",".join(DEFAULT_ASSETS), layout=widgets.Layout(width="100%", height="90px"))
start_date_box = widgets.Text(value=START_DATE_DEFAULT, layout=widgets.Layout(width="180px"))
risk_free_box = widgets.FloatText(value=RISK_FREE_DEFAULT, layout=widgets.Layout(width="160px"))
inflation_box = widgets.FloatText(value=INFLATION_DEFAULT, layout=widgets.Layout(width="160px"))

auto_clean_toggle = widgets.Checkbox(value=True)
align_bd_toggle = widgets.Checkbox(value=True)
ffill_limit_box = widgets.IntSlider(value=3, min=0, max=10, step=1)
min_obs_box = widgets.IntSlider(value=252, min=60, max=1000, step=21)
min_day_cov_box = widgets.FloatSlider(value=0.80, min=0.50, max=1.00, step=0.05)
min_asset_cov_box = widgets.FloatSlider(value=0.80, min=0.50, max=1.00, step=0.05)

def set_clean_controls_enabled(enabled: bool):
    align_bd_toggle.disabled = not enabled
    ffill_limit_box.disabled = not enabled
    min_obs_box.disabled = not enabled
    min_day_cov_box.disabled = not enabled
    min_asset_cov_box.disabled = not enabled
set_clean_controls_enabled(not auto_clean_toggle.value)
auto_clean_toggle.observe(lambda ch: set_clean_controls_enabled(not ch["new"]), names="value")

initial_input = widgets.IntText(value=10000, layout=widgets.Layout(width="180px"))
contrib_input = widgets.IntText(value=500, layout=widgets.Layout(width="180px"))
contrib_freq = widgets.Dropdown(options=["Monthly", "Quarterly", "Annual"], value="Monthly", layout=widgets.Layout(width="180px"))

years_slider = widgets.IntSlider(value=20, min=1, max=40)
risk_slider = widgets.FloatSlider(value=3.0, min=0.5, max=10.0, step=0.5)

strategy_select = widgets.ToggleButtons(options=["Sharpe","MV Utility"], value="Sharpe")
dividend_toggle = widgets.ToggleButtons(options=["Reinvest", "Income"], value="Reinvest")
rebalance_select = widgets.Dropdown(options=["None","Monthly","Quarterly","Annual"], value="Annual")

model_select = widgets.Dropdown(
    options=[
        ("Blend (MPT + CAPM + APT + ML)", "blend"),
        ("MPT (historical μ/Σ)", "mpt"),
        ("CAPM (beta vs benchmark)", "capm"),
        ("APT (PCA factor model)", "apt"),
    ],
    value="blend",
    layout=widgets.Layout(width="360px")
)
capm_bench = widgets.Dropdown(options=list(BENCHMARKS.keys()), value="S&P 500", layout=widgets.Layout(width="240px"))
apt_factors = widgets.IntSlider(value=4, min=1, max=8, step=1, layout=widgets.Layout(width="260px"))
apt_include_bench = widgets.Checkbox(value=True)
ml_blend = widgets.FloatSlider(value=0.35, min=0.0, max=1.0, step=0.05, readout_format=".2f", layout=widgets.Layout(width="320px"))

longshort_toggle = widgets.Checkbox(value=False)
max_weight_toggle = widgets.Checkbox(value=True)
max_weight_slider = widgets.FloatSlider(value=0.20, min=0.05, max=1.00, step=0.01, readout_format=".2f",
                                        layout=widgets.Layout(width="320px"))
max_gross_slider = widgets.FloatSlider(value=1.30, min=1.00, max=3.00, step=0.05, readout_format=".2f",
                                       layout=widgets.Layout(width="320px"))
beta_neutral_penalty = widgets.FloatSlider(value=0.0, min=0.0, max=25.0, step=0.5, readout_format=".1f",
                                          layout=widgets.Layout(width="320px"))

risk_controls_toggle = widgets.Checkbox(value=True)
vol_target_toggle = widgets.Checkbox(value=True)
vol_target_slider = widgets.FloatSlider(value=0.15, min=0.06, max=0.30, step=0.01, readout_format=".2f",
                                        layout=widgets.Layout(width="320px"))
sharpe_caps_toggle = widgets.Checkbox(value=False)
sharpe_caps_strength = widgets.FloatSlider(value=0.35, min=0.0, max=1.0, step=0.05, readout_format=".2f",
                                           layout=widgets.Layout(width="320px"))

mc_toggle = widgets.Checkbox(value=True)
full_mc_toggle = widgets.Checkbox(value=False)
mc_runs_slider = widgets.IntSlider(value=300, min=50, max=3000, step=50)
mc_model = widgets.Dropdown(
    options=[
        ("Normal (MVN)", "normal"),
        ("Student-t (fat tails)", "t"),
        ("Regime-switch Student-t", "regime_t"),
        ("Historical bootstrap (monthly)", "bootstrap")
    ],
    value="t",
    layout=widgets.Layout(width="320px")
)
t_df_slider = widgets.IntSlider(value=6, min=3, max=30, step=1, layout=widgets.Layout(width="320px"))
regime_high_vol_mult = widgets.FloatSlider(value=2.2, min=1.2, max=5.0, step=0.1, layout=widgets.Layout(width="320px"))
regime_p_stay_low = widgets.FloatSlider(value=0.95, min=0.50, max=0.995, step=0.005, layout=widgets.Layout(width="320px"))
regime_p_stay_high = widgets.FloatSlider(value=0.85, min=0.50, max=0.995, step=0.005, layout=widgets.Layout(width="320px"))

bench_toggle = widgets.Checkbox(value=True)
bench_select = widgets.SelectMultiple(
    options=list(BENCHMARKS.keys()),
    value=tuple(BENCHMARKS.keys()),
    layout=widgets.Layout(width="320px", height="80px")
)
ef_toggle = widgets.Checkbox(value=True)
diag_toggle = widgets.Checkbox(value=True)
corrhm_toggle = widgets.Checkbox(value=True)
rollcorr_toggle = widgets.Checkbox(value=True)

selection_toggle = widgets.Checkbox(value=True)
subset_size_slider = widgets.IntSlider(value=12, min=2, max=20, step=1, layout=widgets.Layout(width="300px"))
selection_budget = widgets.IntSlider(value=450, min=100, max=6000, step=50, layout=widgets.Layout(width="320px"))
exhaustive_limit = widgets.IntSlider(value=20000, min=1000, max=100000, step=1000, layout=widgets.Layout(width="320px"))

dd_toggle = widgets.Checkbox(value=True)
dd_method = widgets.Dropdown(
    options=[
        ("Return Index (recommended: excludes contributions)", "ret_index"),
        ("Portfolio Value (includes contributions)", "value_incl_contrib"),
        ("Portfolio Value (net of contributions)", "value_net_contrib"),
    ],
    value="ret_index",
    layout=widgets.Layout(width="460px")
)
dd_stress_mode = widgets.Dropdown(
    options=[
        ("Worst X% average (by Max Drawdown) — recommended", "worst_x_avg"),
        ("Worst X% median (by Max Drawdown)", "worst_x_med"),
        ("Worst single path (min Max Drawdown)", "worst_single"),
        ("Single representative path (nearest P10/P50/P90 by final)", "single_by_final"),
    ],
    value="worst_x_avg",
    layout=widgets.Layout(width="520px")
)
dd_worst_q = widgets.IntSlider(value=10, min=1, max=50, step=1, layout=widgets.Layout(width="360px"))
dd_mc_stat = widgets.Dropdown(options=[("P10", "p10"), ("Median", "p50"), ("P90", "p90")], value="p10",
                              layout=widgets.Layout(width="160px"))
dd_show_band = widgets.Checkbox(value=True)
dd_show_table = widgets.Checkbox(value=True)
dd_include_bench = widgets.Checkbox(value=True)

dd_density_toggle = widgets.Checkbox(value=True)
dd_density_bins = widgets.IntSlider(value=45, min=15, max=140, step=5, layout=widgets.Layout(width="320px"))
dd_density_quantiles = widgets.IntRangeSlider(value=[1, 99], min=0, max=100, step=1,
                                             layout=widgets.Layout(width="360px"), description="Clip %:")

dd_breach_toggle = widgets.Checkbox(value=True)
dd_breach_levels = widgets.SelectMultiple(
    options=[-10, -15, -20, -25, -30, -40, -50],
    value=(-20, -30),
    layout=widgets.Layout(width="240px", height="90px")
)
dd_tuw_toggle = widgets.Checkbox(value=True)
dd_tuw_max_months = widgets.IntSlider(value=240, min=24, max=600, step=12, layout=widgets.Layout(width="320px"))

dd_maxdd_hist_toggle = widgets.Checkbox(value=True)
dd_maxdd_bins = widgets.IntSlider(value=45, min=20, max=120, step=5, layout=widgets.Layout(width="320px"))

run_btn = widgets.Button(description="Build Portfolio", button_style="success")
reset_btn = widgets.Button(description="Reset", button_style="warning")
export_btn = widgets.Button(description="Export CSVs", button_style="info")

out = widgets.Output()
_last = {"summary": None, "weights": None, "holdings": None, "setup": None, "diagnostics": None}

# In-session cache
_CACHE = {"assets_key": None, "start_date": None, "prices": None, "dividends": None, "returns": None, "bench_rets": None, "failed": None}

# ============================================================
# MAIN UI container (Accordion)
# ============================================================
accordion = widgets.Accordion(children=[
    widgets.VBox([
        widgets.HTML("<b>Tickers & Inputs</b>"),
        tickers_box,
        widgets.HBox([widgets.HTML("Start date"), start_date_box,
                      widgets.HTML("Rf (ann)"), risk_free_box,
                      widgets.HTML("Inflation"), inflation_box], layout=widgets.Layout(gap="10px", flex_wrap="wrap")),
        widgets.HBox([widgets.HTML("Benchmarks"), bench_toggle,
                      widgets.HTML("Which"), bench_select], layout=widgets.Layout(gap="10px", flex_wrap="wrap"))
    ]),
    widgets.VBox([
        widgets.HTML("<b>Cleaning</b>"),
        widgets.HBox([widgets.HTML("Auto-tune"), auto_clean_toggle]),
        widgets.HBox([widgets.HTML("Align business days"), align_bd_toggle]),
        widgets.HBox([widgets.HTML("FFill limit"), ffill_limit_box]),
        widgets.HBox([widgets.HTML("Min obs"), min_obs_box]),
        widgets.HBox([widgets.HTML("Day coverage"), min_day_cov_box]),
        widgets.HBox([widgets.HTML("Asset coverage"), min_asset_cov_box]),
    ]),
    widgets.VBox([
        widgets.HTML("<b>Portfolio</b>"),
        widgets.HBox([widgets.HTML("Initial (£)"), initial_input,
                      widgets.HTML("Contribution (£)"), contrib_input,
                      widgets.HTML("Contrib freq"), contrib_freq], layout=widgets.Layout(gap="10px", flex_wrap="wrap")),
        widgets.HBox([widgets.HTML("Years"), years_slider,
                      widgets.HTML("Risk aversion A"), risk_slider], layout=widgets.Layout(gap="10px", flex_wrap="wrap")),
        widgets.HBox([widgets.HTML("Strategy"), strategy_select,
                      widgets.HTML("Model"), model_select], layout=widgets.Layout(gap="10px", flex_wrap="wrap")),
        widgets.HBox([widgets.HTML("CAPM benchmark"), capm_bench,
                      widgets.HTML("APT factors"), apt_factors,
                      widgets.HTML("APT include benchmark"), apt_include_bench],
                     layout=widgets.Layout(gap="10px", flex_wrap="wrap")),
        widgets.HBox([widgets.HTML("ML blend"), ml_blend,
                      widgets.HTML("Rebalance"), rebalance_select,
                      widgets.HTML("Dividends"), dividend_toggle],
                     layout=widgets.Layout(gap="10px", flex_wrap="wrap")),
    ]),
    widgets.VBox([
        widgets.HTML("<b>Constraints</b>"),
        widgets.HBox([widgets.HTML("Enable long/short"), longshort_toggle]),
        widgets.HBox([widgets.HTML("Max |w| cap"), max_weight_toggle,
                      widgets.HTML("Cap"), max_weight_slider], layout=widgets.Layout(gap="10px", flex_wrap="wrap")),
        widgets.HBox([widgets.HTML("Max gross"), max_gross_slider,
                      widgets.HTML("Beta-neutral penalty"), beta_neutral_penalty],
                     layout=widgets.Layout(gap="10px", flex_wrap="wrap")),
    ]),
    widgets.VBox([
        widgets.HTML("<b>Selection</b>"),
        widgets.HBox([widgets.HTML("Enable subset selection"), selection_toggle]),
        widgets.HBox([widgets.HTML("Subset size"), subset_size_slider,
                      widgets.HTML("Exhaustive limit"), exhaustive_limit,
                      widgets.HTML("Random budget"), selection_budget],
                     layout=widgets.Layout(gap="10px", flex_wrap="wrap")),
    ]),
    widgets.VBox([
        widgets.HTML("<b>Risk Controls</b>"),
        widgets.HBox([widgets.HTML("Enable"), risk_controls_toggle]),
        widgets.HBox([widgets.HTML("Vol target"), vol_target_toggle,
                      widgets.HTML("Target"), vol_target_slider],
                     layout=widgets.Layout(gap="10px", flex_wrap="wrap")),
        widgets.HBox([widgets.HTML("Sharpe caps"), sharpe_caps_toggle,
                      widgets.HTML("Strength"), sharpe_caps_strength],
                     layout=widgets.Layout(gap="10px", flex_wrap="wrap")),
    ]),
    widgets.VBox([
        widgets.HTML("<b>Analytics</b>"),
        widgets.HBox([widgets.HTML("Run MC"), mc_toggle,
                      widgets.HTML("Spaghetti"), full_mc_toggle],
                     layout=widgets.Layout(gap="10px", flex_wrap="wrap")),
        widgets.HBox([widgets.HTML("MC runs"), mc_runs_slider]),
        widgets.HBox([widgets.HTML("MC model"), mc_model,
                      widgets.HTML("t df"), t_df_slider],
                     layout=widgets.Layout(gap="10px", flex_wrap="wrap")),
        widgets.HBox([widgets.HTML("Regime hi-vol ×"), regime_high_vol_mult,
                      widgets.HTML("P(stay low)"), regime_p_stay_low,
                      widgets.HTML("P(stay high)"), regime_p_stay_high],
                     layout=widgets.Layout(gap="10px", flex_wrap="wrap")),
        widgets.HBox([widgets.HTML("Eff frontier"), ef_toggle,
                      widgets.HTML("Diagnostics"), diag_toggle,
                      widgets.HTML("Corr heatmap"), corrhm_toggle,
                      widgets.HTML("Rolling corr"), rollcorr_toggle],
                     layout=widgets.Layout(gap="10px", flex_wrap="wrap")),
    ]),
    widgets.VBox([
        widgets.HTML("<b>Drawdowns</b>"),
        widgets.HBox([widgets.HTML("Enable"), dd_toggle]),
        widgets.HBox([widgets.HTML("Method"), dd_method]),
        widgets.HBox([widgets.HTML("Stress mode"), dd_stress_mode]),
        widgets.HBox([widgets.HTML("Worst X%"), dd_worst_q,
                      widgets.HTML("Single path uses"), dd_mc_stat,
                      widgets.HTML("Band"), dd_show_band,
                      widgets.HTML("Max-DD table"), dd_show_table,
                      widgets.HTML("Include benchmarks"), dd_include_bench],
                     layout=widgets.Layout(gap="10px", flex_wrap="wrap")),
        widgets.HBox([widgets.HTML("Density heatmap"), dd_density_toggle,
                      widgets.HTML("Bins"), dd_density_bins,
                      widgets.HTML("Clip"), dd_density_quantiles],
                     layout=widgets.Layout(gap="10px", flex_wrap="wrap")),
        widgets.HBox([widgets.HTML("Breach probs"), dd_breach_toggle,
                      widgets.HTML("Thresholds"), dd_breach_levels],
                     layout=widgets.Layout(gap="10px", flex_wrap="wrap")),
        widgets.HBox([widgets.HTML("Time-under-water"), dd_tuw_toggle,
                      widgets.HTML("Max months"), dd_tuw_max_months,
                      widgets.HTML("MaxDD hist"), dd_maxdd_hist_toggle,
                      widgets.HTML("Bins"), dd_maxdd_bins],
                     layout=widgets.Layout(gap="10px", flex_wrap="wrap")),
    ]),
])
for i, t in enumerate(["1) Data","2) Cleaning","3) Portfolio","4) Constraints","5) Selection","6) Risk Controls","7) Analytics","8) Drawdowns"]):
    accordion.set_title(i, t)
accordion.selected_index = 0

# ============================================================
# Main callback
# ============================================================
def run_model(_=None):
    with out:
        clear_output()

        assets = _parse_tickers(tickers_box.value)
        if len(assets) < 2:
            raise ValueError("Please provide at least 2 tickers.")

        start_date = start_date_box.value.strip()
        risk_free = float(risk_free_box.value)
        inflation = float(inflation_box.value)

        # ---------- Load data ----------
        if auto_clean_toggle.value:
            prices_raw, dividends_raw = download_prices_and_dividends(assets, start_date)

            align_bd, ffill, min_obs, day_cov, asset_cov = autotune_cleaning_settings(prices_raw, assets)
            align_bd_toggle.value = bool(align_bd)
            ffill_limit_box.value = int(ffill)
            min_obs_box.value = int(min_obs)
            min_day_cov_box.value = float(day_cov)
            min_asset_cov_box.value = float(asset_cov)

            returns, prices_clean, failed, chosen = try_cleaning_with_fallback(
                prices_raw, align_bd_toggle.value, ffill_limit_box.value,
                min_obs_box.value, min_day_cov_box.value, min_asset_cov_box.value
            )
            dividends = dividends_raw.reindex_like(prices_clean).fillna(0.0)

            bench_raw = yf.download(list(BENCHMARKS.values()), start=start_date, auto_adjust=True, progress=False)
            bench_prices = bench_raw["Close"].dropna(how="all") if isinstance(bench_raw.columns, pd.MultiIndex) else bench_raw.dropna(how="all")
            bench_rets = bench_prices.pct_change().dropna(how="all")

            prices = prices_clean
        else:
            prices, dividends = download_prices_and_dividends(assets, start_date)
            returns, prices, failed = _robust_returns_from_prices(
                prices,
                align_business_days=align_bd_toggle.value,
                ffill_limit=ffill_limit_box.value,
                min_obs_per_asset=min_obs_box.value,
                min_day_coverage=min_day_cov_box.value,
                min_asset_coverage=min_asset_cov_box.value,
            )
            bench_raw = yf.download(list(BENCHMARKS.values()), start=start_date, auto_adjust=True, progress=False)
            bench_prices = bench_raw["Close"].dropna(how="all") if isinstance(bench_raw.columns, pd.MultiIndex) else bench_raw.dropna(how="all")
            bench_rets = bench_prices.pct_change().dropna(how="all")

        _CACHE.update({
            "assets_key": tuple(assets),
            "start_date": start_date,
            "prices": prices,
            "dividends": dividends,
            "returns": returns,
            "bench_rets": bench_rets,
            "failed": failed
        })

        assets_used = list(returns.columns)
        if len(assets_used) < 2:
            raise ValueError("Too few assets after cleaning. Relax thresholds or extend date range.")

        # ---------- MPT μ/Σ ----------
        mu_mpt, cov = estimate_mu_cov(returns, inflation=inflation)
        mu_mpt_s = pd.Series(mu_mpt, index=assets_used, name="μ_MPT (ann)")
        cov_df = pd.DataFrame(cov, index=assets_used, columns=assets_used)

        # ---------- Monthly series ----------
        asset_monthly = monthly_asset_returns_from_daily(returns[assets_used]).dropna(how="any")
        capm_ticker = BENCHMARKS[capm_bench.value]
        if capm_ticker not in bench_rets.columns:
            raise ValueError(f"Benchmark {capm_bench.value} not available in downloaded benchmark returns.")
        bench_monthly = (1.0 + bench_rets[capm_ticker].dropna()).resample("M").prod() - 1.0
        bench_monthly = bench_monthly.reindex(asset_monthly.index).dropna()

        # CAPM
        try:
            capm_beta, capm_alpha, capm_mu_ann = capm_estimates(asset_monthly, bench_monthly, rf_annual=risk_free)
            capm_mu_ann = capm_mu_ann.reindex(assets_used).fillna(mu_mpt_s)
            capm_beta = capm_beta.reindex(assets_used).fillna(0.0)
            capm_alpha = capm_alpha.reindex(assets_used).fillna(0.0)
        except Exception:
            capm_beta = pd.Series(0.0, index=assets_used)
            capm_alpha = pd.Series(0.0, index=assets_used)
            capm_mu_ann = mu_mpt_s.copy()

        # APT
        try:
            apt_mu_ann, apt_exposures, apt_premia, apt_factors_ts = apt_statistical(
                asset_monthly[assets_used],
                n_factors=int(apt_factors.value),
                include_benchmark=bool(apt_include_bench.value),
                bench_monthly=bench_monthly
            )
            apt_mu_ann = apt_mu_ann.reindex(assets_used).fillna(mu_mpt_s)
        except Exception:
            apt_mu_ann = mu_mpt_s.copy()
            apt_exposures = pd.DataFrame(index=assets_used)
            apt_premia = pd.Series(dtype=float)
            apt_factors_ts = pd.DataFrame(index=asset_monthly.index)

        # ML
        try:
            ml_mu_next_m, ml_r2 = ml_predict_next_month_mu(asset_monthly[assets_used], bench_monthly, n_lags=6)
        except Exception:
            ml_mu_next_m = pd.Series(0.0, index=assets_used)
            ml_r2 = pd.Series(np.nan, index=assets_used)
        ml_mu_ann = (1.0 + ml_mu_next_m).pow(12) - 1.0

        # ---------- Select μ based on model ----------
        if model_select.value == "mpt":
            mu_sel_s = mu_mpt_s.copy()
        elif model_select.value == "capm":
            mu_sel_s = capm_mu_ann.copy()
        elif model_select.value == "apt":
            mu_sel_s = apt_mu_ann.copy()
        else:
            base = (mu_mpt_s + capm_mu_ann + apt_mu_ann) / 3.0
            mu_sel_s = (1.0 - float(ml_blend.value)) * base + float(ml_blend.value) * ml_mu_ann.reindex(assets_used).fillna(0.0)
        mu_sel_s = mu_sel_s.replace([np.inf, -np.inf], np.nan).fillna(mu_mpt_s)

        # Keep a canonical "blend" μ for diagnostics comparisons
        base_blend = (mu_mpt_s + capm_mu_ann + apt_mu_ann) / 3.0
        mu_blend_s = (1.0 - float(ml_blend.value)) * base_blend + float(ml_blend.value) * ml_mu_ann.reindex(assets_used).fillna(0.0)

        # ---------- Constraints / bounds ----------
        long_short = bool(longshort_toggle.value)
        use_bounds = bool(max_weight_toggle.value)
        max_abs = float(max_weight_slider.value)
        max_gross = float(max_gross_slider.value)
        beta_pen = float(beta_neutral_penalty.value)
        beta_vec_full = capm_beta.copy()

        def bounds_builder(mu_arr, cov_arr):
            if not use_bounds:
                if long_short: return [(-1.0, 1.0)] * len(mu_arr)
                return [(0.0, 1.0)] * len(mu_arr)
            return build_max_weight_bounds(
                mu=np.asarray(mu_arr),
                cov=np.asarray(cov_arr),
                risk_free=risk_free,
                hard_cap=max_abs,
                long_short=long_short,
                use_sharpe_caps=bool(sharpe_caps_toggle.value) if risk_controls_toggle.value else False,
                sharpe_strength=float(sharpe_caps_strength.value) if risk_controls_toggle.value else 0.0
            )

        # ---------- Subset Selection ----------
        selection_report = None
        if selection_toggle.value:
            k = int(np.clip(subset_size_slider.value, 2, len(assets_used)))
            best = subset_search_best(
                assets=assets_used,
                mu_s=mu_sel_s,
                cov_df=cov_df,
                mode=strategy_select.value,
                A=float(risk_slider.value),
                rf=risk_free,
                bounds_builder=lambda mu_arr, cov_arr: bounds_builder(mu_arr, cov_arr),
                long_short=long_short,
                max_gross=max_gross,
                subset_size=k,
                exhaustive_limit=int(exhaustive_limit.value),
                random_budget=int(selection_budget.value),
                beta_penalty=beta_pen if long_short else 0.0,
                beta_vec_full=beta_vec_full if long_short else None,
                seed=42
            )
            chosen_assets, w_risky, port_r, port_v, best_score, method_sel, combos, tried = best
            selection_report = {
                "Method": method_sel,
                "N": len(assets_used),
                "Subset size": k,
                "Combos (theoretical)": combos,
                "Combos tried": tried,
                "Best objective": best_score,
                "Best ret": port_r,
                "Best vol": port_v
            }
        else:
            chosen_assets = assets_used
            mu_arr = mu_sel_s.loc[chosen_assets].values
            cov_arr = cov_df.loc[chosen_assets, chosen_assets].values
            bnds = bounds_builder(mu_arr, cov_arr)
            w_risky = optimise(mu_arr, cov_arr, strategy_select.value, float(risk_slider.value), risk_free,
                               bnds, long_short, max_gross,
                               beta_penalty=(beta_pen if long_short else 0.0),
                               beta_vec=(beta_vec_full.loc[chosen_assets].values if long_short else None))
            port_r, port_v = portfolio_perf(w_risky, mu_arr, cov_arr)

        chosen_assets = list(chosen_assets)

        # ---------- Apply ACTIVE override (Frontier Picker) ----------
        active_label = None
        if _ACTIVE["enabled"] and _ACTIVE["chosen_assets"] is not None and _ACTIVE["w_risky"] is not None:
            # only apply override if the asset universe matches current chosen_assets (same tickers after cleaning/selection)
            if list(_ACTIVE["chosen_assets"]) == list(chosen_assets):
                w_risky = np.asarray(_ACTIVE["w_risky"], dtype=float)
                if _ACTIVE["mu_model"] is not None:
                    # override μ for simulation/perf if the user applied a specific model from Frontier Picker
                    mu_sel_s = pd.Series(np.asarray(_ACTIVE["mu_model"], dtype=float), index=chosen_assets).reindex(chosen_assets).fillna(mu_sel_s.loc[chosen_assets]).rename("μ_selected (ann)")
                active_label = _ACTIVE["label"]

        # Recompute selected μ/cov after override (aligned to chosen)
        mu_sel = mu_sel_s.loc[chosen_assets].values
        cov_sel = cov_df.loc[chosen_assets, chosen_assets].values
        beta_vec = beta_vec_full.loc[chosen_assets].values
        port_r, port_v = portfolio_perf(w_risky, mu_sel, cov_sel)

        # ---------- Risky vs risk-free split ----------
        var_risky = port_v**2
        risky_weight = float(np.clip((port_r - risk_free) / (float(risk_slider.value) * (var_risky if var_risky > 0 else 1e-9)), 0.0, 1.0))
        rf_weight = 1.0 - risky_weight

        if risk_controls_toggle.value and vol_target_toggle.value:
            rf_weight = apply_vol_target(rf_weight=rf_weight, port_vol_risky=port_v, vol_target=float(vol_target_slider.value))
            risky_weight = 1.0 - rf_weight

        reinvest = (dividend_toggle.value == "Reinvest")
        w_series = pd.Series(w_risky, index=chosen_assets)
        dy = portfolio_dividend_yield(w_series, prices[chosen_assets], dividends[chosen_assets])

        years = int(years_slider.value)
        steps = years * 12

        contrib_per_year = contribution_steps_per_year(contrib_freq.value)
        total_contribs = (years * contrib_per_year) * float(contrib_input.value)
        base_invested = float(initial_input.value) + float(total_contribs)

        # ---------- Simulate ----------
        sims = incomes = gross_rets = reinv_divs = None
        if mc_toggle.value:
            hist_monthly = monthly_asset_returns_from_daily(returns[chosen_assets]).dropna(how="any")
            sims, incomes, gross_rets, reinv_divs = simulate_paths(
                mu_ann=mu_sel, cov_ann=cov_sel,
                w_risky=w_risky, rf_weight=rf_weight,
                years=years,
                initial=float(initial_input.value),
                contrib_amount=float(contrib_input.value),
                contrib_freq=contrib_freq.value,
                rebalance_freq=rebalance_select.value,
                runs=int(mc_runs_slider.value),
                risk_free=risk_free,
                prices=prices[chosen_assets],
                dividends=dividends[chosen_assets],
                reinvest_divs=reinvest,
                seed=42,
                mc_model=mc_model.value,
                t_df=int(t_df_slider.value),
                regime_high_vol_mult=float(regime_high_vol_mult.value),
                p_stay_low=float(regime_p_stay_low.value),
                p_stay_high=float(regime_p_stay_high.value),
                hist_monthly_asset_rets=hist_monthly.reindex(columns=chosen_assets)
            )
            p10 = np.percentile(sims, 10, axis=0)
            p50 = np.percentile(sims, 50, axis=0)
            p90 = np.percentile(sims, 90, axis=0)
            final_val = float(p50[-1])
            final_income = float(np.percentile(incomes, 50, axis=0)[-1])
            reinv_div_final = float(np.percentile(reinv_divs, 50, axis=0)[-1]) if reinvest else 0.0
        else:
            # keep a minimal deterministic fallback
            monthly_r = port_r / 12.0
            rf_m = risk_free / 12.0
            dy_m = dy / 12.0
            contrib_interval = int(12 / contrib_per_year) if contrib_per_year else 1
            reb_per_year = rebalance_steps_per_year(rebalance_select.value)
            reb_interval = int(12 / reb_per_year) if reb_per_year else 0

            risky_bucket = float(initial_input.value) * (1 - rf_weight)
            rf_bucket = float(initial_input.value) * rf_weight
            income = 0.0
            reinv_cum = 0.0

            path = np.zeros(steps)
            income_path = np.zeros(steps)
            gross_r = np.zeros(steps, dtype=float)
            reinv_path = np.zeros(steps, dtype=float)

            for t in range(steps):
                total_before = risky_bucket + rf_bucket
                contrib_this = 0.0
                if t % contrib_interval == 0:
                    contrib_this = float(contrib_input.value)
                    risky_bucket += contrib_this * (1 - rf_weight)
                    rf_bucket += contrib_this * rf_weight

                risky_bucket *= (1 + monthly_r)
                rf_bucket *= (1 + rf_m)

                div_cash = risky_bucket * dy_m
                if reinvest:
                    risky_bucket += div_cash
                    reinv_cum += div_cash
                else:
                    income += div_cash

                if reb_interval and (t+1) % reb_interval == 0:
                    total = risky_bucket + rf_bucket
                    risky_bucket = total * (1 - rf_weight)
                    rf_bucket = total * rf_weight

                total_after = risky_bucket + rf_bucket
                gross_r[t] = (total_after - contrib_this) / total_before - 1.0 if total_before > 0 else 0.0

                path[t] = total_after
                income_path[t] = income
                reinv_path[t] = reinv_cum

            sims = path[None, :]
            incomes = income_path[None, :]
            gross_rets = gross_r[None, :]
            reinv_divs = reinv_path[None, :]

            final_val = float(path[-1])
            final_income = float(income_path[-1])
            reinv_div_final = float(reinv_cum) if reinvest else 0.0
            p10 = p50 = p90 = None

        total_invested = base_invested + (reinv_div_final if reinvest else 0.0)
        profit = final_val - total_invested
        growth_pct = (final_val / total_invested - 1.0) if total_invested > 0 else np.nan

        # ---------- Sortino (NEW) ----------
        r_port_daily = portfolio_daily_returns(returns[chosen_assets], w_risky=w_risky, rf_weight=rf_weight, risk_free_ann=risk_free)
        sortino_total = sortino_ratio_ann(r_port_daily, rf_annual=risk_free)

        # ---------- Tables: weights + holdings ----------
        rc, rc_pct = risk_contribution(w_risky, cov_sel)
        weights_df = pd.DataFrame({
            "Weight": w_risky,
            "Weight %": w_risky * 100.0,
            "Asset Vol %": np.sqrt(np.clip(np.diag(cov_sel), 1e-18, None)) * 100.0,
            "Risk Contrib %": rc_pct * 100.0,
            "CAPM Beta": beta_vec
        }, index=chosen_assets).sort_values("Weight %", ascending=False)

        rf_row = pd.DataFrame({
            "Weight": [rf_weight],
            "Weight %": [rf_weight * 100.0],
            "Asset Vol %": [0.0],
            "Risk Contrib %": [0.0],
            "CAPM Beta": [0.0]
        }, index=["RISK-FREE (T-bills proxy)"])
        weights_with_rf = pd.concat([weights_df, rf_row], axis=0)

        risky_capital = total_invested * (1 - rf_weight)
        rf_capital = total_invested * rf_weight
        holdings_vals = weights_df.loc[chosen_assets, "Weight"].values * risky_capital
        holdings_df = pd.DataFrame({
            "Asset": chosen_assets,
            "Target Weight %": weights_df.loc[chosen_assets, "Weight %"].values,
            "£ Value (target, invested)": holdings_vals
        }).sort_values("£ Value (target, invested)", ascending=False)

        holdings_df = pd.concat([
            holdings_df,
            pd.DataFrame([{
                "Asset": "RISK-FREE (T-bills proxy)",
                "Target Weight %": rf_weight * 100.0,
                "£ Value (target, invested)": rf_capital
            }])
        ], ignore_index=True)

        initial_total = float(initial_input.value)
        initial_risky_cash = initial_total * (1 - rf_weight)
        initial_rf_cash = initial_total * rf_weight
        setup_vals = weights_df.loc[chosen_assets, "Weight"].values * initial_risky_cash
        setup_df = pd.DataFrame({
            "Asset": chosen_assets,
            "Target Weight %": weights_df.loc[chosen_assets, "Weight %"].values,
            "£ To Invest (Initial Only)": setup_vals
        }).sort_values("£ To Invest (Initial Only)", ascending=False)

        setup_df = pd.concat([
            setup_df,
            pd.DataFrame([{
                "Asset": "RISK-FREE (T-bills proxy)",
                "Target Weight %": rf_weight * 100.0,
                "£ To Invest (Initial Only)": initial_rf_cash
            }])
        ], ignore_index=True)

        # ---------- Diagnostics packs ----------
        mu_selected_df = mu_sel_s.loc[chosen_assets].to_frame("μ_selected (ann)")
        diagnostics = {
            "mu_selected": mu_selected_df,
            "cov": pd.DataFrame(cov_sel, index=chosen_assets, columns=chosen_assets),
            "corr": returns[chosen_assets].corr(),
            "capm": pd.DataFrame({
                "Beta": capm_beta.reindex(chosen_assets),
                "Alpha (monthly)": capm_alpha.reindex(chosen_assets),
                "μ_CAPM (ann)": capm_mu_ann.reindex(chosen_assets)
            }).sort_values("μ_CAPM (ann)", ascending=False),
            "apt_mu": apt_mu_ann.reindex(chosen_assets).to_frame("μ_APT (ann)"),
            "ml": pd.DataFrame({
                "ML μ_next_month": ml_mu_next_m.reindex(chosen_assets),
                "ML μ_annualised": ml_mu_ann.reindex(chosen_assets),
                "CV R²": ml_r2.reindex(chosen_assets)
            }).sort_values("ML μ_annualised", ascending=False),
            "apt_exposures": apt_exposures.reindex(chosen_assets) if isinstance(apt_exposures, pd.DataFrame) else pd.DataFrame(index=chosen_assets),
            "apt_premia": pd.DataFrame({"Premia": apt_premia}) if isinstance(apt_premia, (pd.Series, pd.DataFrame)) else pd.DataFrame()
        }

        # ---------- Model comparison portfolios (Diagnostics) + Sortino (NEW) ----------
        def solve_for_mu(mu_series: pd.Series):
            mu_vec = mu_series.reindex(chosen_assets).fillna(mu_mpt_s.reindex(chosen_assets)).values
            bnds = bounds_builder(mu_vec, cov_sel)
            w = optimise(mu_vec, cov_sel, strategy_select.value, float(risk_slider.value), risk_free,
                         bnds, long_short, max_gross,
                         beta_penalty=(beta_pen if long_short else 0.0),
                         beta_vec=(beta_vec if long_short else None))
            r, v = portfolio_perf(w, mu_vec, cov_sel)
            # Sortino for risky sleeve only (no rf sleeve) for apples-to-apples between models:
            r_risky_daily = pd.Series(returns[chosen_assets].values @ w, index=returns.index)
            sortino_risky = sortino_ratio_ann(r_risky_daily, rf_annual=risk_free)
            return w, r, v, sortino_risky

        w_mpt, r_mpt, v_mpt, so_mpt = solve_for_mu(mu_mpt_s)
        w_capm, r_capm, v_capm, so_capm = solve_for_mu(capm_mu_ann)
        w_apt, r_apt, v_apt, so_apt = solve_for_mu(apt_mu_ann)
        w_blend, r_blend, v_blend, so_blend = solve_for_mu(mu_blend_s)

        def summary_tbl(name, w, r, v, sortino_val):
            beta_p = float(np.dot(w, beta_vec)) if beta_vec is not None else np.nan
            return pd.DataFrame({
                "Metric": ["Model", "Expected Return (ann)", "Volatility (ann)", "Sharpe", "Sortino (risky sleeve)", "MV Utility", "Gross", "Net", "Portfolio Beta"],
                "Value": [
                    name,
                    f"{r*100:.2f}%",
                    f"{v*100:.2f}%",
                    f"{((r-risk_free)/v if v>0 else np.nan):.2f}",
                    f"{sortino_val:.2f}" if np.isfinite(sortino_val) else "(n/a)",
                    f"{(r - 0.5*float(risk_slider.value)*v*v):.4f}",
                    f"{np.sum(np.abs(w)):.2f}",
                    f"{np.sum(w):.2f}",
                    f"{beta_p:.2f}"
                ]
            })

        mpt_summary_df = summary_tbl("MPT", w_mpt, r_mpt, v_mpt, so_mpt)
        capm_summary_df = summary_tbl("CAPM", w_capm, r_capm, v_capm, so_capm)
        apt_summary_df = summary_tbl("APT", w_apt, r_apt, v_apt, so_apt)
        blend_summary_df = summary_tbl("Blend", w_blend, r_blend, v_blend, so_blend)

        # ---------- Main summary (+ Sortino NEW) ----------
        mv_utility = port_r - 0.5 * float(risk_slider.value) * (port_v ** 2)
        summary_df_plain = pd.DataFrame({
            "Metric": [
                "Active override",
                "Tickers (requested)", "Tickers (used)", "Failed tickers dropped",
                "Return window used",
                "Model", "Strategy",
                "Subset selection",
                "Annualised Return (risky sleeve)", "Volatility (risky sleeve)", "Sharpe (risky sleeve)",
                "Sortino (total portfolio, incl rf sleeve)",
                "MV Utility (risky sleeve)",
                "Risky Weight", "Risk-Free Weight",
                "Dividend Yield (risky sleeve)",
                "Dividends reinvested (median)", "Cash income (median, if Income mode)",
                "Initial", "Total Contributions", "Base Invested (Init+Contrib)",
                "Total Invested (incl reinvested divs)", "Final Portfolio Value (median)",
                "Growth % (vs total invested)", "Profit (vs total invested)"
            ],
            "Value": [
                active_label if active_label else "(none)",
                str(len(assets)), str(len(assets_used)), ", ".join(failed) if failed else "(none)",
                f"{returns.index.min().date()} → {returns.index.max().date()}",
                model_select.label, strategy_select.value,
                (selection_report["Method"] if selection_report else "(disabled)"),
                f"{port_r*100:.2f}%", f"{port_v*100:.2f}%",
                f"{((port_r - risk_free)/port_v if port_v>0 else np.nan):.2f}",
                f"{sortino_total:.2f}" if np.isfinite(sortino_total) else "(n/a)",
                f"{mv_utility:.4f}",
                f"{(1-rf_weight)*100:.1f}%", f"{rf_weight*100:.1f}%",
                f"{dy*100:.2f}%",
                f"£{reinv_div_final:,.0f}" if reinvest else "(not reinvesting)",
                f"£{final_income:,.0f}" if not reinvest else "(reinvest mode)",
                f"£{float(initial_input.value):,.0f}",
                f"£{float(total_contribs):,.0f}",
                f"£{base_invested:,.0f}",
                f"£{total_invested:,.0f}",
                f"£{final_val:,.0f}",
                f"{growth_pct*100:.2f}%" if np.isfinite(growth_pct) else "(n/a)",
                f"£{profit:,.0f}",
            ]
        })

        _last["summary"] = summary_df_plain.copy()
        _last["weights"] = weights_with_rf.copy()
        _last["holdings"] = holdings_df.copy()
        _last["setup"] = setup_df.copy()
        _last["diagnostics"] = diagnostics

        # ============================================================
        # BUILD MAIN TABS
        # ============================================================
        tabs = widgets.Tab()

        # --- Summary ---
        summary_out = widgets.Output()
        with summary_out:
            display(HTML("<h3>Summary</h3>"))
            display(summary_df_plain)
            if selection_report:
                display(HTML("<h4>Selection Report</h4>"))
                display(pd.DataFrame([selection_report]))

        # --- Setup ---
        setup_out = widgets.Output()
        with setup_out:
            display(HTML("<h3>Portfolio Set-up</h3>"))
            df = setup_df.copy()
            df["Target Weight %"] = df["Target Weight %"].map(lambda x: f"{float(x):.2f}%")
            df["£ To Invest (Initial Only)"] = df["£ To Invest (Initial Only)"].map(lambda x: f"£{float(x):,.0f}")
            display(df)

        # --- Weights & Holdings ---
        weights_out = widgets.Output()
        with weights_out:
            display(HTML("<h3>Weights & Holdings</h3>"))
            dfw = weights_with_rf.copy()
            dfw["Weight %"] = dfw["Weight %"].map(lambda x: f"{float(x):.2f}%")
            dfw["Asset Vol %"] = dfw["Asset Vol %"].map(lambda x: f"{float(x):.2f}%")
            dfw["Risk Contrib %"] = dfw["Risk Contrib %"].map(lambda x: f"{float(x):.2f}%")
            dfw["CAPM Beta"] = dfw["CAPM Beta"].map(lambda x: f"{float(x):.2f}")
            display(dfw)

            display(HTML("<div style='height:10px'></div><h4>Holdings (target, total invested)</h4>"))
            dfh = holdings_df.copy()
            dfh["Target Weight %"] = dfh["Target Weight %"].map(lambda x: f"{float(x):.2f}%")
            dfh["£ Value (target, invested)"] = dfh["£ Value (target, invested)"].map(lambda x: f"£{float(x):,.0f}")
            display(dfh)

        # --- Growth ---
        growth_out = widgets.Output()
        with growth_out:
            fig = go.Figure()
            x = np.arange(1, steps+1)
            if mc_toggle.value and sims is not None and sims.shape[0] > 1:
                fig.add_trace(go.Scatter(x=x, y=np.percentile(sims, 50, axis=0), name="Portfolio (median)", mode="lines"))
                fig.add_trace(go.Scatter(x=x, y=np.percentile(sims, 10, axis=0), name="10th percentile", mode="lines",
                                         line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=x, y=np.percentile(sims, 90, axis=0), name="90th percentile", mode="lines",
                                         fill="tonexty", opacity=0.25))
            else:
                fig.add_trace(go.Scatter(x=x, y=sims[0], name="Portfolio", mode="lines"))

            if bench_toggle.value:
                chosen_bench = list(bench_select.value)
                bp = benchmark_paths(bench_rets, years, float(initial_input.value), float(contrib_input.value), contrib_freq.value,
                                     {k: BENCHMARKS[k] for k in chosen_bench})
                for name, path_ in bp.items():
                    fig.add_trace(go.Scatter(x=x, y=path_, name=name, mode="lines", line=dict(dash="dot")))

            fig.update_layout(title="Portfolio Growth (optional Monte Carlo band & benchmarks)",
                              xaxis_title="Months", yaxis_title="Value (£)",
                              hovermode="x unified", legend_title="Series")
            fig.show()

        # --- Monte Carlo ---
        mc_out = widgets.Output()
        with mc_out:
            if not mc_toggle.value:
                display(HTML("<i>Monte Carlo is disabled.</i>"))
            else:
                finals = sims[:, -1]
                fig2 = go.Figure()
                fig2.add_trace(go.Histogram(x=finals, nbinsx=45, name="Final values"))
                fig2.update_layout(title="Distribution of Final Portfolio Values (Monte Carlo)",
                                   xaxis_title="£ Value", yaxis_title="Count")
                fig2.show()

                if full_mc_toggle.value:
                    max_lines = 250
                    idx = np.linspace(0, sims.shape[0]-1, min(max_lines, sims.shape[0])).astype(int)
                    fig3 = go.Figure()
                    for i2 in idx:
                        fig3.add_trace(go.Scatter(y=sims[i2], mode="lines", line=dict(width=1),
                                                  opacity=0.12, showlegend=False))
                    fig3.update_layout(title="Monte Carlo Spaghetti (downsampled)",
                                       xaxis_title="Months", yaxis_title="£ Value")
                    fig3.show()
                else:
                    display(HTML("<i>Enable ‘Spaghetti (MC)’ in Analytics to show individual simulation paths.</i>"))

        # --- Annual Growth (kept compact) ---
        annual_out = widgets.Output()
        with annual_out:
            if mc_toggle.value and sims is not None and sims.shape[0] > 1:
                v = np.percentile(sims, 50, axis=0)
            else:
                v = sims[0].astype(float)
            years_n = int(np.ceil(steps/12))
            year_end_idx = [min((k+1)*12, steps)-1 for k in range(years_n)]
            year_vals = v[year_end_idx]
            year_growth = []
            for k in range(years_n):
                start_idx = 0 if k == 0 else year_end_idx[k-1]
                start_val = v[start_idx]
                end_val = v[year_end_idx[k]]
                g = (end_val / start_val - 1.0) if start_val > 0 else np.nan
                year_growth.append(g)
            year_labels = [f"Y{k+1}" for k in range(years_n)]
            figy = go.Figure()
            figy.add_trace(go.Bar(x=year_labels, y=np.array(year_growth)*100, name="Year growth %"))
            figy.update_layout(title="Year-by-Year Growth (%)",
                               xaxis_title="Year", yaxis_title="Growth (%)",
                               hovermode="x unified")
            figy.show()

        # --- Drawdowns (full suite) ---
        dd_out = widgets.Output()
        with dd_out:
            if not dd_toggle.value:
                display(HTML("<i>Drawdowns are disabled.</i>"))
            else:
                dd_tabs = widgets.Tab()
                dd_chart_out = widgets.Output()
                dd_heat_out = widgets.Output()
                dd_breach_out = widgets.Output()
                dd_tuw_out = widgets.Output()
                dd_hist_out = widgets.Output()

                method = dd_method.value
                x = np.arange(1, steps+1)
                dd_rows = []

                def dd_band_from_ddmat(A):
                    lo = np.nanpercentile(A, 10, axis=0)
                    hi = np.nanpercentile(A, 90, axis=0)
                    return lo, hi

                if method == "ret_index":
                    R = gross_rets
                    dd_mat = np.apply_along_axis(drawdown_from_returns, 1, R)
                    max_dd = np.nanmin(dd_mat, axis=1)
                elif method == "value_incl_contrib":
                    dd_mat = np.apply_along_axis(drawdown_from_path, 1, sims)
                    max_dd = np.nanmin(dd_mat, axis=1)
                else:
                    contrib_per_year2 = contribution_steps_per_year(contrib_freq.value)
                    contrib_interval2 = int(12 / contrib_per_year2) if contrib_per_year2 else 1
                    cum_contrib = np.zeros(steps, dtype=float)
                    c = 0.0
                    for t in range(steps):
                        if t % contrib_interval2 == 0:
                            c += float(contrib_input.value)
                        cum_contrib[t] = c
                    netV = sims - cum_contrib[None, :]
                    dd_mat = np.apply_along_axis(drawdown_from_path, 1, netV)
                    max_dd = np.nanmin(dd_mat, axis=1)

                q = int(dd_worst_q.value)
                k = max(1, int(np.floor(dd_mat.shape[0] * q / 100)))
                worst_idx = np.argsort(max_dd)[:k]

                mode_dd = dd_stress_mode.value
                dd_port = None
                band_lo = band_hi = None
                band_name = None

                if mode_dd in {"worst_x_avg", "worst_x_med", "worst_single"}:
                    if mode_dd == "worst_x_avg":
                        dd_port = np.nanmean(dd_mat[worst_idx, :], axis=0)
                    elif mode_dd == "worst_x_med":
                        dd_port = np.nanmedian(dd_mat[worst_idx, :], axis=0)
                    else:
                        dd_port = dd_mat[worst_idx[0], :]

                    if dd_show_band.value and dd_mat.shape[0] > 10:
                        band_lo, band_hi = dd_band_from_ddmat(dd_mat[worst_idx, :])
                        band_name = f"Worst {q}% band (P10–P90)"
                else:
                    finals = sims[:, -1] if method != "ret_index" else np.cumprod(1.0 + R, axis=1)[:, -1]
                    p = 10 if dd_mc_stat.value == "p10" else 50 if dd_mc_stat.value == "p50" else 90
                    tgt = np.percentile(finals, p)
                    j = int(np.argmin(np.abs(finals - tgt)))
                    dd_port = dd_mat[j, :]

                    if dd_show_band.value and dd_mat.shape[0] > 10:
                        band_lo, band_hi = dd_band_from_ddmat(dd_mat)
                        band_name = "All runs band (P10–P90)"

                bench_overlays = []
                if dd_include_bench.value and bench_toggle.value:
                    bm = benchmark_monthly_returns(bench_rets, years, BENCHMARKS)
                    for name in list(bench_select.value):
                        mr = bm.get(name, pd.Series(dtype=float))
                        if mr is None or mr.empty:
                            continue
                        dd_b = drawdown_from_returns(mr.values[:steps])
                        bench_overlays.append((name, dd_b))

                with dd_chart_out:
                    figdd = go.Figure()
                    figdd.add_trace(go.Scatter(x=x, y=dd_port*100, name="Portfolio (stress line)", mode="lines"))
                    dd_rows.append({"Series": "Portfolio (stress line)", "Max Drawdown": f"{float(np.nanmin(dd_port))*100:.2f}%"})

                    if dd_show_band.value and band_lo is not None and band_hi is not None:
                        figdd.add_trace(go.Scatter(x=x, y=band_lo*100, mode="lines", line=dict(width=0), showlegend=False))
                        figdd.add_trace(go.Scatter(x=x, y=band_hi*100, mode="lines", fill="tonexty", opacity=0.18,
                                                   name=band_name or "Band"))

                    for (nm, dd_b) in bench_overlays:
                        figdd.add_trace(go.Scatter(x=x, y=dd_b*100, name=nm, mode="lines", line=dict(dash="dot")))
                        dd_rows.append({"Series": nm, "Max Drawdown": f"{float(np.nanmin(dd_b))*100:.2f}%"})

                    figdd.update_layout(title="Drawdown Stress Test (Peak-to-Trough %)",
                                        xaxis_title="Months", yaxis_title="Drawdown (%)",
                                        hovermode="x unified", legend_title="Series")
                    figdd.show()

                    if dd_show_table.value and dd_rows:
                        display(HTML("<h4>Max Drawdown Summary</h4>"))
                        display(pd.DataFrame(dd_rows))

                with dd_heat_out:
                    if not dd_density_toggle.value:
                        display(HTML("<i>Density heatmap is disabled.</i>"))
                    else:
                        z, y_centers, xh = drawdown_density_heatmap(
                            dd_mat[worst_idx, :],
                            bins=int(dd_density_bins.value),
                            clip_lo=int(dd_density_quantiles.value[0]),
                            clip_hi=int(dd_density_quantiles.value[1]),
                        )
                        if z is None:
                            display(HTML("<i>Not enough data to build density heatmap.</i>"))
                        else:
                            figH = go.Figure()
                            figH.add_trace(go.Heatmap(
                                z=z, x=xh, y=y_centers,
                                hovertemplate="Month=%{x}<br>Drawdown=%{y:.1f}%<br>Share=%{z:.2%}<extra></extra>",
                                colorbar=dict(title="Share")
                            ))
                            figH.add_trace(go.Scatter(x=x, y=dd_port*100, mode="lines", name="Stress line"))
                            figH.update_layout(title=f"Drawdown Density Heatmap (Worst {q}% subset)",
                                               xaxis_title="Months", yaxis_title="Drawdown (%)",
                                               hovermode="x unified")
                            figH.show()

                with dd_breach_out:
                    if not dd_breach_toggle.value:
                        display(HTML("<i>Breach probabilities are disabled.</i>"))
                    else:
                        levels = sorted(list(dd_breach_levels.value))
                        probs = breach_probabilities(dd_mat[worst_idx, :], thresholds_pct=levels)
                        figB = go.Figure()
                        for thr in levels:
                            figB.add_trace(go.Scatter(x=x, y=probs[thr], mode="lines", name=f"P(DD ≤ {thr}%)"))
                        figB.update_layout(title=f"Breach Probabilities Over Time (Worst {q}% subset)",
                                           xaxis_title="Months", yaxis_title="Probability",
                                           hovermode="x unified", yaxis=dict(range=[0, 1]))
                        figB.show()

                with dd_tuw_out:
                    if not dd_tuw_toggle.value:
                        display(HTML("<i>Time-under-water is disabled.</i>"))
                    else:
                        all_durs = []
                        if method == "ret_index":
                            for j in worst_idx:
                                all_durs.extend(time_under_water_months_from_returns(R[j, :]))
                        else:
                            for j in worst_idx:
                                vpath = sims[j, :]
                                rr = np.diff(np.r_[1.0, vpath]) / np.r_[1.0, vpath[:-1]]
                                all_durs.extend(time_under_water_months_from_returns(rr))
                        if len(all_durs) == 0:
                            display(HTML("<i>No under-water episodes detected.</i>"))
                        else:
                            cap = int(dd_tuw_max_months.value)
                            clipped = np.clip(np.array(all_durs, dtype=int), 0, cap)
                            figT = go.Figure()
                            figT.add_trace(go.Histogram(x=clipped, nbinsx=min(60, max(10, cap//2))))
                            figT.update_layout(title=f"Time Under Water Distribution (Worst {q}% subset)",
                                               xaxis_title="Months below prior peak (episode length)",
                                               yaxis_title="Count")
                            figT.show()

                with dd_hist_out:
                    if not dd_maxdd_hist_toggle.value:
                        display(HTML("<i>Max drawdown histogram is disabled.</i>"))
                    else:
                        figM = go.Figure()
                        figM.add_trace(go.Histogram(x=(max_dd[worst_idx] * 100.0), nbinsx=int(dd_maxdd_bins.value),
                                                    name=f"Worst {q}% subset"))
                        figM.add_trace(go.Histogram(x=(max_dd * 100.0), nbinsx=int(dd_maxdd_bins.value),
                                                    name="All runs", opacity=0.5))
                        figM.update_layout(barmode="overlay",
                                           title="Max Drawdown Distribution (subset vs all)",
                                           xaxis_title="Max drawdown (%)", yaxis_title="Count")
                        figM.show()

                dd_tabs.children = [dd_chart_out, dd_heat_out, dd_breach_out, dd_tuw_out, dd_hist_out]
                for i2, t2 in enumerate(["Chart", "Density Heatmap", "Breach Prob", "Time Under Water", "Max-DD Hist"]):
                    dd_tabs.set_title(i2, t2)
                display(dd_tabs)

        # --- Efficient Frontier (INTERACTIVE + Submenu) ---
        ef_out = widgets.Output()
        with ef_out:
            if not ef_toggle.value:
                display(HTML("<i>Efficient frontier is disabled.</i>"))
            else:
                picker_out = widgets.Output()
                preview_out = widgets.Output()

                # Frontier for current selected μ (mu_sel) and cov_sel
                bnds_frontier = bounds_builder(mu_sel, cov_sel)
                vols_f, rets_f, w_list = efficient_frontier_with_weights(
                    mu=mu_sel, cov=cov_sel, points=80,
                    bounds=bnds_frontier, long_short=long_short, max_gross=max_gross,
                    beta_penalty=(beta_pen if long_short else 0.0),
                    beta_vec=(beta_vec if long_short else None)
                )

                fig_fw = go.FigureWidget()
                fig_fw.add_trace(go.Scatter(
                    x=vols_f*100, y=rets_f*100,
                    mode="lines+markers", marker=dict(size=6),
                    name="Frontier (click a point)",
                    hovertemplate="Vol=%{x:.2f}%<br>Ret=%{y:.2f}%<extra></extra>",
                ))
                fig_fw.add_trace(go.Scatter(
                    x=[port_v*100], y=[port_r*100],
                    mode="markers", marker=dict(size=12, symbol="x"),
                    name="Selected / Active",
                ))
                fig_fw.update_layout(
                    title="Interactive Efficient Frontier — Click a point to open Frontier Picker",
                    xaxis_title="Volatility (%)",
                    yaxis_title="Expected Return (%)",
                    hovermode="closest",
                )

                display(fig_fw)
                display(HTML("<div class='mpt-note'><b>Frontier Picker:</b> click a point → choose model → preview FULL Monte Carlo + FULL Drawdowns. "
                             "Then optionally replace the active portfolio.</div>"))
                display(picker_out)
                display(preview_out)

                def build_pack_for_preview(mu_model_vec, w_preview, label):
                    # recompute split
                    r_, v_ = portfolio_perf(w_preview, mu_model_vec, cov_sel)
                    var_ = v_**2
                    risky_w_ = float(np.clip((r_ - risk_free) / (float(risk_slider.value) * (var_ if var_ > 0 else 1e-9)), 0.0, 1.0))
                    rf_w_ = 1.0 - risky_w_
                    if risk_controls_toggle.value and vol_target_toggle.value:
                        rf_w_ = apply_vol_target(rf_weight=rf_w_, port_vol_risky=v_, vol_target=float(vol_target_slider.value))
                        risky_w_ = 1.0 - rf_w_

                    r_daily_ = portfolio_daily_returns(returns[chosen_assets], w_risky=w_preview, rf_weight=rf_w_, risk_free_ann=risk_free)
                    sortino_ = sortino_ratio_ann(r_daily_, rf_annual=risk_free)

                    dy_ = portfolio_dividend_yield(pd.Series(w_preview, index=chosen_assets), prices[chosen_assets], dividends[chosen_assets])

                    hist_monthly_ = monthly_asset_returns_from_daily(returns[chosen_assets]).dropna(how="any")
                    sims_, incomes_, gross_rets_, reinv_divs_ = simulate_paths(
                        mu_ann=mu_model_vec, cov_ann=cov_sel,
                        w_risky=w_preview, rf_weight=rf_w_,
                        years=years,
                        initial=float(initial_input.value),
                        contrib_amount=float(contrib_input.value),
                        contrib_freq=contrib_freq.value,
                        rebalance_freq=rebalance_select.value,
                        runs=int(mc_runs_slider.value),
                        risk_free=risk_free,
                        prices=prices[chosen_assets],
                        dividends=dividends[chosen_assets],
                        reinvest_divs=(dividend_toggle.value == "Reinvest"),
                        seed=42,
                        mc_model=mc_model.value,
                        t_df=int(t_df_slider.value),
                        regime_high_vol_mult=float(regime_high_vol_mult.value),
                        p_stay_low=float(regime_p_stay_low.value),
                        p_stay_high=float(regime_p_stay_high.value),
                        hist_monthly_asset_rets=hist_monthly_.reindex(columns=chosen_assets)
                    )

                    reinvest_ = (dividend_toggle.value == "Reinvest")
                    reinv_div_final_ = float(np.percentile(reinv_divs_, 50, axis=0)[-1]) if reinvest_ else 0.0
                    final_val_ = float(np.percentile(sims_, 50, axis=0)[-1])
                    total_invested_ = base_invested + (reinv_div_final_ if reinvest_ else 0.0)
                    profit_ = final_val_ - total_invested_
                    growth_pct_ = (final_val_ / total_invested_ - 1.0) if total_invested_ > 0 else np.nan

                    summary_ = pd.DataFrame({
                        "Metric": ["Preview portfolio", "Return (ann)", "Vol (ann)", "Sharpe (risky sleeve)", "Sortino (total portfolio)", "Risky %", "RF %", "Dividend yield (risky)", "Final (median)", "Profit vs invested"],
                        "Value": [
                            label,
                            f"{r_*100:.2f}%",
                            f"{v_*100:.2f}%",
                            f"{((r_-risk_free)/v_ if v_>0 else np.nan):.2f}",
                            f"{sortino_:.2f}" if np.isfinite(sortino_) else "(n/a)",
                            f"{(1-rf_w_)*100:.1f}%",
                            f"{rf_w_*100:.1f}%",
                            f"{dy_*100:.2f}%",
                            f"£{final_val_:,.0f}",
                            f"£{profit_:,.0f}",
                        ]
                    })

                    setup_vals_ = (pd.Series(w_preview, index=chosen_assets) * (float(initial_input.value)*(1-rf_w_))).values
                    setup_df_ = pd.DataFrame({"Asset": chosen_assets, "Target Weight %": np.asarray(w_preview)*100, "£ To Invest (Initial Only)": setup_vals_}).sort_values("£ To Invest (Initial Only)", ascending=False)
                    setup_df_ = pd.concat([setup_df_, pd.DataFrame([{"Asset":"RISK-FREE (T-bills proxy)","Target Weight %":rf_w_*100,"£ To Invest (Initial Only)": float(initial_input.value)*rf_w_}])], ignore_index=True)

                    return {
                        "label": label, "summary": summary_, "setup": setup_df_,
                        "sims": sims_, "incomes": incomes_, "gross_rets": gross_rets_, "reinv_divs": reinv_divs_,
                        "rf_weight": rf_w_, "r": r_, "v": v_, "w": w_preview, "mu": mu_model_vec
                    }

                def render_full_preview_tabs(pack):
                    tabsP = widgets.Tab()
                    o1,o2,o3,o4 = widgets.Output(), widgets.Output(), widgets.Output(), widgets.Output()

                    with o1:
                        display(HTML("<h4>Preview: Summary</h4>"))
                        display(pack["summary"])

                    with o2:
                        display(HTML("<h4>Preview: Portfolio Set-up</h4>"))
                        df = pack["setup"].copy()
                        df["Target Weight %"] = df["Target Weight %"].map(lambda x: f"{float(x):.2f}%")
                        df["£ To Invest (Initial Only)"] = df["£ To Invest (Initial Only)"].map(lambda x: f"£{float(x):,.0f}")
                        display(df)

                    with o3:
                        display(HTML("<h4>Preview: Full Monte Carlo</h4>"))
                        sims_ = pack["sims"]
                        x_ = np.arange(1, steps+1)
                        p10_ = np.percentile(sims_, 10, axis=0)
                        p50_ = np.percentile(sims_, 50, axis=0)
                        p90_ = np.percentile(sims_, 90, axis=0)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=x_, y=p50_, name="Median", mode="lines"))
                        fig.add_trace(go.Scatter(x=x_, y=p10_, name="P10", mode="lines", line=dict(width=0), showlegend=False))
                        fig.add_trace(go.Scatter(x=x_, y=p90_, name="P90", mode="lines", fill="tonexty", opacity=0.25))
                        fig.update_layout(title="Growth band (Preview)", xaxis_title="Months", yaxis_title="Value (£)", hovermode="x unified")
                        fig.show()

                        finals_ = sims_[:, -1]
                        fig2 = go.Figure()
                        fig2.add_trace(go.Histogram(x=finals_, nbinsx=45))
                        fig2.update_layout(title="Final Value Distribution (Preview)", xaxis_title="£ Value", yaxis_title="Count")
                        fig2.show()

                        if full_mc_toggle.value:
                            max_lines = 250
                            idx = np.linspace(0, sims_.shape[0]-1, min(max_lines, sims_.shape[0])).astype(int)
                            fig3 = go.Figure()
                            for i2 in idx:
                                fig3.add_trace(go.Scatter(y=sims_[i2], mode="lines", line=dict(width=1), opacity=0.12, showlegend=False))
                            fig3.update_layout(title="MC Spaghetti (Preview)", xaxis_title="Months", yaxis_title="£ Value")
                            fig3.show()

                    with o4:
                        display(HTML("<h4>Preview: Full Drawdowns Suite</h4>"))
                        sims_ = pack["sims"]
                        gross_ = pack["gross_rets"]

                        dd_tabsP = widgets.Tab()
                        dd_chart_outP = widgets.Output()
                        dd_heat_outP  = widgets.Output()
                        dd_breach_outP= widgets.Output()
                        dd_tuw_outP   = widgets.Output()
                        dd_hist_outP  = widgets.Output()

                        methodP = dd_method.value
                        xP = np.arange(1, steps+1)
                        dd_rowsP = []

                        def dd_band_from_ddmatP(A):
                            lo = np.nanpercentile(A, 10, axis=0)
                            hi = np.nanpercentile(A, 90, axis=0)
                            return lo, hi

                        if methodP == "ret_index":
                            RP = gross_
                            dd_matP = np.apply_along_axis(drawdown_from_returns, 1, RP)
                            max_ddP = np.nanmin(dd_matP, axis=1)
                        elif methodP == "value_incl_contrib":
                            dd_matP = np.apply_along_axis(drawdown_from_path, 1, sims_)
                            max_ddP = np.nanmin(dd_matP, axis=1)
                        else:
                            contrib_per_year2 = contribution_steps_per_year(contrib_freq.value)
                            contrib_interval2 = int(12 / contrib_per_year2) if contrib_per_year2 else 1
                            cum_contribP = np.zeros(steps, dtype=float)
                            cP = 0.0
                            for t in range(steps):
                                if t % contrib_interval2 == 0:
                                    cP += float(contrib_input.value)
                                cum_contribP[t] = cP
                            netVP = sims_ - cum_contribP[None, :]
                            dd_matP = np.apply_along_axis(drawdown_from_path, 1, netVP)
                            max_ddP = np.nanmin(dd_matP, axis=1)

                        qP = int(dd_worst_q.value)
                        kP = max(1, int(np.floor(dd_matP.shape[0] * qP / 100)))
                        worst_idxP = np.argsort(max_ddP)[:kP]

                        modeP = dd_stress_mode.value
                        dd_portP = None
                        band_loP = band_hiP = None
                        band_nameP = None

                        if modeP in {"worst_x_avg", "worst_x_med", "worst_single"}:
                            if modeP == "worst_x_avg":
                                dd_portP = np.nanmean(dd_matP[worst_idxP, :], axis=0)
                            elif modeP == "worst_x_med":
                                dd_portP = np.nanmedian(dd_matP[worst_idxP, :], axis=0)
                            else:
                                dd_portP = dd_matP[worst_idxP[0], :]
                            if dd_show_band.value and dd_matP.shape[0] > 10:
                                band_loP, band_hiP = dd_band_from_ddmatP(dd_matP[worst_idxP, :])
                                band_nameP = f"Worst {qP}% band (P10–P90)"
                        else:
                            finalsP = sims_[:, -1] if methodP != "ret_index" else np.cumprod(1.0 + RP, axis=1)[:, -1]
                            pP = 10 if dd_mc_stat.value == "p10" else 50 if dd_mc_stat.value == "p50" else 90
                            tgtP = np.percentile(finalsP, pP)
                            jP = int(np.argmin(np.abs(finalsP - tgtP)))
                            dd_portP = dd_matP[jP, :]
                            if dd_show_band.value and dd_matP.shape[0] > 10:
                                band_loP, band_hiP = dd_band_from_ddmatP(dd_matP)
                                band_nameP = "All runs band (P10–P90)"

                        bench_overlaysP = []
                        if dd_include_bench.value and bench_toggle.value:
                            bmP = benchmark_monthly_returns(bench_rets, years, BENCHMARKS)
                            for name in list(bench_select.value):
                                mr = bmP.get(name, pd.Series(dtype=float))
                                if mr is None or mr.empty:
                                    continue
                                dd_b = drawdown_from_returns(mr.values[:steps])
                                bench_overlaysP.append((name, dd_b))

                        with dd_chart_outP:
                            figdd = go.Figure()
                            figdd.add_trace(go.Scatter(x=xP, y=dd_portP*100, name="Portfolio (stress line)", mode="lines"))
                            dd_rowsP.append({"Series": "Portfolio (stress line)", "Max Drawdown": f"{float(np.nanmin(dd_portP))*100:.2f}%"})
                            if dd_show_band.value and band_loP is not None and band_hiP is not None:
                                figdd.add_trace(go.Scatter(x=xP, y=band_loP*100, mode="lines", line=dict(width=0), showlegend=False))
                                figdd.add_trace(go.Scatter(x=xP, y=band_hiP*100, mode="lines", fill="tonexty", opacity=0.18,
                                                           name=band_nameP or "Band"))
                            for (nm, dd_b) in bench_overlaysP:
                                figdd.add_trace(go.Scatter(x=xP, y=dd_b*100, name=nm, mode="lines", line=dict(dash="dot")))
                                dd_rowsP.append({"Series": nm, "Max Drawdown": f"{float(np.nanmin(dd_b))*100:.2f}%"})
                            figdd.update_layout(title="Drawdown Stress Test (Preview)",
                                                xaxis_title="Months", yaxis_title="Drawdown (%)",
                                                hovermode="x unified", legend_title="Series")
                            figdd.show()
                            if dd_show_table.value and dd_rowsP:
                                display(HTML("<h5>Max Drawdown Summary</h5>"))
                                display(pd.DataFrame(dd_rowsP))

                        with dd_heat_outP:
                            if not dd_density_toggle.value:
                                display(HTML("<i>Density heatmap is disabled.</i>"))
                            else:
                                z, y_centers, xh = drawdown_density_heatmap(
                                    dd_matP[worst_idxP, :],
                                    bins=int(dd_density_bins.value),
                                    clip_lo=int(dd_density_quantiles.value[0]),
                                    clip_hi=int(dd_density_quantiles.value[1]),
                                )
                                if z is None:
                                    display(HTML("<i>Not enough data.</i>"))
                                else:
                                    figH = go.Figure()
                                    figH.add_trace(go.Heatmap(z=z, x=xh, y=y_centers,
                                                              hovertemplate="Month=%{x}<br>DD=%{y:.1f}%<br>Share=%{z:.2%}<extra></extra>",
                                                              colorbar=dict(title="Share")))
                                    figH.add_trace(go.Scatter(x=xP, y=dd_portP*100, mode="lines", name="Stress line"))
                                    figH.update_layout(title=f"Drawdown Density Heatmap (Preview, Worst {qP}% subset)",
                                                       xaxis_title="Months", yaxis_title="Drawdown (%)",
                                                       hovermode="x unified")
                                    figH.show()

                        with dd_breach_outP:
                            if not dd_breach_toggle.value:
                                display(HTML("<i>Breach probabilities are disabled.</i>"))
                            else:
                                levels = sorted(list(dd_breach_levels.value))
                                probs = breach_probabilities(dd_matP[worst_idxP, :], thresholds_pct=levels)
                                figB = go.Figure()
                                for thr in levels:
                                    figB.add_trace(go.Scatter(x=xP, y=probs[thr], mode="lines", name=f"P(DD ≤ {thr}%)"))
                                figB.update_layout(title=f"Breach Probabilities (Preview, Worst {qP}% subset)",
                                                   xaxis_title="Months", yaxis_title="Probability",
                                                   hovermode="x unified", yaxis=dict(range=[0, 1]))
                                figB.show()

                        with dd_tuw_outP:
                            if not dd_tuw_toggle.value:
                                display(HTML("<i>Time-under-water is disabled.</i>"))
                            else:
                                all_durs = []
                                if methodP == "ret_index":
                                    for j in worst_idxP:
                                        all_durs.extend(time_under_water_months_from_returns(RP[j, :]))
                                else:
                                    for j in worst_idxP:
                                        vpath = sims_[j, :]
                                        rr = np.diff(np.r_[1.0, vpath]) / np.r_[1.0, vpath[:-1]]
                                        all_durs.extend(time_under_water_months_from_returns(rr))
                                if len(all_durs) == 0:
                                    display(HTML("<i>No under-water episodes detected.</i>"))
                                else:
                                    cap = int(dd_tuw_max_months.value)
                                    clipped = np.clip(np.array(all_durs, dtype=int), 0, cap)
                                    figT = go.Figure()
                                    figT.add_trace(go.Histogram(x=clipped, nbinsx=min(60, max(10, cap//2))))
                                    figT.update_layout(title=f"Time Under Water (Preview, Worst {qP}% subset)",
                                                       xaxis_title="Months below prior peak (episode length)",
                                                       yaxis_title="Count")
                                    figT.show()

                        with dd_hist_outP:
                            if not dd_maxdd_hist_toggle.value:
                                display(HTML("<i>Max drawdown histogram is disabled.</i>"))
                            else:
                                figM = go.Figure()
                                figM.add_trace(go.Histogram(x=(max_ddP[worst_idxP] * 100.0), nbinsx=int(dd_maxdd_bins.value),
                                                            name=f"Worst {qP}% subset"))
                                figM.add_trace(go.Histogram(x=(max_ddP * 100.0), nbinsx=int(dd_maxdd_bins.value),
                                                            name="All runs", opacity=0.5))
                                figM.update_layout(barmode="overlay",
                                                   title="Max Drawdown Distribution (Preview)",
                                                   xaxis_title="Max drawdown (%)", yaxis_title="Count")
                                figM.show()

                        dd_tabsP.children = [dd_chart_outP, dd_heat_outP, dd_breach_outP, dd_tuw_outP, dd_hist_outP]
                        for i2, t2 in enumerate(["Chart", "Density Heatmap", "Breach Prob", "Time Under Water", "Max-DD Hist"]):
                            dd_tabsP.set_title(i2, t2)
                        display(dd_tabsP)

                    tabsP.children = [o1, o2, o3, o4]
                    for i2, t2 in enumerate(["Summary", "Set-up", "Monte Carlo", "Drawdowns"]):
                        tabsP.set_title(i2, t2)
                    return tabsP

                def compute_candidates_at_target(target_ret_ann: float):
                    # Build per-model μ vectors aligned to chosen_assets
                    mu_models = {
                        "MPT": mu_mpt_s.reindex(chosen_assets).values,
                        "CAPM": capm_mu_ann.reindex(chosen_assets).values,
                        "APT": apt_mu_ann.reindex(chosen_assets).values,
                        "Blend": mu_blend_s.reindex(chosen_assets).values
                    }
                    outW = {}
                    for name, mu_vec in mu_models.items():
                        bnds = bounds_builder(mu_vec, cov_sel)
                        w = solve_min_var_for_target_return(
                            mu_vec, cov_sel, float(target_ret_ann),
                            bounds=bnds, long_short=long_short, max_gross=max_gross,
                            beta_penalty=(beta_pen if long_short else 0.0),
                            beta_vec=(beta_vec if long_short else None)
                        )
                        outW[name] = (w, mu_vec)
                    return outW

                def on_frontier_click(trace, points, selector):
                    if not points.point_inds:
                        return
                    i = int(points.point_inds[0])
                    target_ret = float(rets_f[i])
                    vol_guess = float(vols_f[i])

                    with fig_fw.batch_update():
                        fig_fw.data[1].x = [vol_guess*100]
                        fig_fw.data[1].y = [target_ret*100]
                        fig_fw.data[1].name = f"Selected (Vol {vol_guess*100:.2f}%, Ret {target_ret*100:.2f}%)"

                    cand = compute_candidates_at_target(target_ret)
                    model_dd = widgets.Dropdown(options=list(cand.keys()), value="Blend", description="Model:")
                    preview_btn = widgets.Button(description="Preview", button_style="info")
                    apply_btn = widgets.Button(description="Replace main portfolio", button_style="success")
                    clear_btn = widgets.Button(description="Clear override", button_style="warning")

                    def do_preview(_=None):
                        with preview_out:
                            clear_output(wait=True)
                            w_prev, mu_prev = cand[model_dd.value]
                            label = f"{model_dd.value} @ target {target_ret*100:.2f}% (frontier vol≈{vol_guess*100:.2f}%)"
                            pack = build_pack_for_preview(mu_prev, w_prev, label)
                            display(render_full_preview_tabs(pack))

                    def do_apply(_=None):
                        w_prev, mu_prev = cand[model_dd.value]
                        _ACTIVE["enabled"] = True
                        _ACTIVE["chosen_assets"] = list(chosen_assets)
                        _ACTIVE["w_risky"] = np.asarray(w_prev, dtype=float)
                        _ACTIVE["mu_model"] = np.asarray(mu_prev, dtype=float)
                        _ACTIVE["label"] = f"ACTIVE: {model_dd.value} @ target {target_ret*100:.2f}% (frontier vol≈{vol_guess*100:.2f}%)"
                        # Rebuild dashboard immediately
                        run_model(None)

                    def do_clear(_=None):
                        _ACTIVE["enabled"] = False
                        _ACTIVE["chosen_assets"] = None
                        _ACTIVE["w_risky"] = None
                        _ACTIVE["mu_model"] = None
                        _ACTIVE["label"] = None
                        run_model(None)

                    preview_btn.on_click(do_preview)
                    apply_btn.on_click(do_apply)
                    clear_btn.on_click(do_clear)

                    with picker_out:
                        clear_output(wait=True)
                        display(HTML(
                            f"<h4>Frontier Picker</h4>"
                            f"<div class='mpt-note'>Clicked: <b>{target_ret*100:.2f}%</b> return, "
                            f"frontier vol≈<b>{vol_guess*100:.2f}%</b>. Choose model, preview, or replace main portfolio.</div>"
                        ))
                        display(widgets.HBox([model_dd, preview_btn, apply_btn, clear_btn],
                                             layout=widgets.Layout(gap="10px", flex_wrap="wrap")))
                        do_preview()

                fig_fw.data[0].on_click(on_frontier_click)

        # --- Correlation heatmap ---
        corrhm_out = widgets.Output()
        with corrhm_out:
            if not corrhm_toggle.value:
                display(HTML("<i>Correlation heatmap is disabled.</i>"))
            else:
                corr = returns[chosen_assets].corr()
                figc = go.Figure(data=go.Heatmap(
                    z=corr.values,
                    x=corr.columns.tolist(),
                    y=corr.index.tolist(),
                    zmin=-1, zmax=1,
                    hovertemplate="X=%{x}<br>Y=%{y}<br>Corr=%{z:.2f}<extra></extra>"
                ))
                figc.update_layout(title="Correlation Heatmap (Daily Returns, Post-Cleaning)",
                                   xaxis_title="Asset", yaxis_title="Asset")
                figc.show()

        # --- Rolling corr ---
        rollcorr_out = widgets.Output()
        with rollcorr_out:
            if not rollcorr_toggle.value:
                display(HTML("<i>Rolling correlation is disabled.</i>"))
            else:
                bench_dropdown = widgets.Dropdown(
                    options=list(BENCHMARKS.keys()),
                    value=list(BENCHMARKS.keys())[0],
                    description="Benchmark:",
                    layout=widgets.Layout(width="340px")
                )
                window_dropdown = widgets.Dropdown(
                    options=[("3 months (~63d)", 63), ("6 months (~126d)", 126), ("12 months (~252d)", 252), ("24 months (~504d)", 504)],
                    value=252,
                    description="Window:",
                    layout=widgets.Layout(width="340px")
                )
                rc_fig_out = widgets.Output()
                r_port = portfolio_daily_returns(returns[chosen_assets], w_risky=w_risky, rf_weight=rf_weight, risk_free_ann=risk_free)

                def render_rollcorr(_=None):
                    with rc_fig_out:
                        clear_output(wait=True)
                        name = bench_dropdown.value
                        ticker = BENCHMARKS[name]
                        if ticker not in bench_rets.columns:
                            display(HTML(f"<i>No benchmark data for {name} ({ticker}).</i>"))
                            return
                        r_bench = bench_rets[ticker].dropna()
                        s = rolling_corr(r_port, r_bench, window=int(window_dropdown.value))
                        if s.empty:
                            display(HTML("<i>Not enough overlapping data to compute rolling correlation.</i>"))
                            return
                        figr = go.Figure()
                        figr.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=f"ρ(Portfolio, {name})"))
                        figr.update_layout(title="Rolling Correlation (Daily Returns)",
                                           xaxis_title="Date", yaxis_title="Correlation",
                                           hovermode="x unified",
                                           yaxis=dict(range=[-1, 1]))
                        figr.show()

                bench_dropdown.observe(render_rollcorr, names="value")
                window_dropdown.observe(render_rollcorr, names="value")
                display(widgets.HBox([bench_dropdown, window_dropdown], layout=widgets.Layout(gap="12px", flex_wrap="wrap")))
                display(rc_fig_out)
                render_rollcorr()

        # --- Diagnostics (sub-tabs) ---
        diag_out = widgets.Output()
        with diag_out:
            if not diag_toggle.value:
                display(HTML("<i>Diagnostics are disabled.</i>"))
            else:
                display(HTML("<h3>Diagnostics</h3>"))
                diag_tabs = widgets.Tab()
                out_mpt = widgets.Output()
                out_capm = widgets.Output()
                out_apt = widgets.Output()
                out_ml = widgets.Output()
                out_inputs = widgets.Output()

                with out_mpt:
                    display(HTML("<h4>MPT Portfolio Results</h4>"))
                    display(mpt_summary_df)
                    display(pd.DataFrame({"Weight %": w_mpt*100}, index=chosen_assets).sort_values("Weight %", ascending=False))

                with out_capm:
                    display(HTML("<h4>CAPM Portfolio Results</h4>"))
                    display(capm_summary_df)
                    display(HTML("<h5>CAPM per-asset estimates</h5>"))
                    display(diagnostics["capm"])

                with out_apt:
                    display(HTML("<h4>APT Portfolio Results</h4>"))
                    display(apt_summary_df)
                    display(HTML("<h5>APT μ per asset</h5>"))
                    display(diagnostics["apt_mu"])
                    if isinstance(diagnostics["apt_exposures"], pd.DataFrame) and not diagnostics["apt_exposures"].empty:
                        display(HTML("<h5>APT factor exposures</h5>"))
                        display(diagnostics["apt_exposures"])
                    if isinstance(diagnostics["apt_premia"], pd.DataFrame) and not diagnostics["apt_premia"].empty:
                        display(HTML("<h5>APT factor premia</h5>"))
                        display(diagnostics["apt_premia"])

                with out_ml:
                    display(HTML("<h4>ML Forecast Diagnostics</h4>"))
                    display(diagnostics["ml"])

                with out_inputs:
                    display(HTML("<h4>Inputs</h4>"))
                    display(HTML("<h5>Selected μ (annual)</h5>"))
                    display(diagnostics["mu_selected"].sort_values("μ_selected (ann)", ascending=False).style.format("{:.4f}"))
                    display(HTML("<h5>Covariance Σ (annual)</h5>"))
                    display(diagnostics["cov"])
                    display(HTML("<h5>Correlation (daily)</h5>"))
                    display(diagnostics["corr"])
                    display(HTML("<h5>Model comparison (MPT vs CAPM vs APT vs Blend)</h5>"))
                    display(blend_summary_df)

                diag_tabs.children = [out_mpt, out_capm, out_apt, out_ml, out_inputs]
                for i2, t2 in enumerate(["MPT", "CAPM", "APT", "ML", "Inputs"]):
                    diag_tabs.set_title(i2, t2)
                display(diag_tabs)

        # --- Assemble tabs ---
        tabs.children = [
            summary_out, setup_out, weights_out, growth_out, mc_out, annual_out,
            dd_out, ef_out, corrhm_out, rollcorr_out, diag_out
        ]
        titles = [
            "Summary","Portfolio Set-up","Weights & Holdings","Growth","Monte Carlo","Annual Growth",
            "Drawdowns","Efficient Frontier","Correlation Heatmap","Rolling Corr","Diagnostics"
        ]
        for i2, t2 in enumerate(titles):
            tabs.set_title(i2, t2)

        display(tabs)

def reset_model(_=None):
    tickers_box.value = ",".join(DEFAULT_ASSETS)
    start_date_box.value = START_DATE_DEFAULT
    risk_free_box.value = RISK_FREE_DEFAULT
    inflation_box.value = INFLATION_DEFAULT

    auto_clean_toggle.value = True
    align_bd_toggle.value = True
    ffill_limit_box.value = 3
    min_obs_box.value = 252
    min_day_cov_box.value = 0.80
    min_asset_cov_box.value = 0.80

    initial_input.value = 10000
    contrib_input.value = 500
    contrib_freq.value = "Monthly"
    years_slider.value = 20
    risk_slider.value = 3.0
    strategy_select.value = "Sharpe"
    dividend_toggle.value = "Reinvest"
    rebalance_select.value = "Annual"

    model_select.value = "blend"
    capm_bench.value = "S&P 500"
    apt_factors.value = 4
    apt_include_bench.value = True
    ml_blend.value = 0.35

    longshort_toggle.value = False
    max_weight_toggle.value = True
    max_weight_slider.value = 0.20
    max_gross_slider.value = 1.30
    beta_neutral_penalty.value = 0.0

    selection_toggle.value = True
    subset_size_slider.value = 12
    selection_budget.value = 450
    exhaustive_limit.value = 20000

    risk_controls_toggle.value = True
    vol_target_toggle.value = True
    vol_target_slider.value = 0.15
    sharpe_caps_toggle.value = False
    sharpe_caps_strength.value = 0.35

    mc_toggle.value = True
    full_mc_toggle.value = False
    mc_runs_slider.value = 300
    mc_model.value = "t"
    t_df_slider.value = 6
    regime_high_vol_mult.value = 2.2
    regime_p_stay_low.value = 0.95
    regime_p_stay_high.value = 0.85

    bench_toggle.value = True
    bench_select.value = tuple(BENCHMARKS.keys())
    ef_toggle.value = True
    diag_toggle.value = True
    corrhm_toggle.value = True
    rollcorr_toggle.value = True

    dd_toggle.value = True
    dd_method.value = "ret_index"
    dd_stress_mode.value = "worst_x_avg"
    dd_worst_q.value = 10
    dd_mc_stat.value = "p10"
    dd_show_band.value = True
    dd_show_table.value = True
    dd_include_bench.value = True
    dd_density_toggle.value = True
    dd_density_bins.value = 45
    dd_density_quantiles.value = (1, 99)
    dd_breach_toggle.value = True
    dd_breach_levels.value = (-20, -30)
    dd_tuw_toggle.value = True
    dd_tuw_max_months.value = 240
    dd_maxdd_hist_toggle.value = True
    dd_maxdd_bins.value = 45

    # clear override too
    _ACTIVE["enabled"] = False
    _ACTIVE["chosen_assets"] = None
    _ACTIVE["w_risky"] = None
    _ACTIVE["mu_model"] = None
    _ACTIVE["label"] = None

    out.clear_output()

def export_model(_=None):
    with out:
        if _last["summary"] is None:
            print("Nothing to export yet — run the model first.")
            return
        files = export_reports(_last["summary"], _last["weights"], _last["holdings"], _last["setup"], _last["diagnostics"])
        print("Exported:")
        for f in files:
            display(HTML(f"<a href='{f}' target='_blank'>{f}</a>"))

run_btn.on_click(run_model)
reset_btn.on_click(reset_model)
export_btn.on_click(export_model)

header = widgets.HTML(
    "<h2 style='margin:0.2em 0'>Full-Depth Portfolio Dashboard (MPT + CAPM + APT + ML + Selection + Long/Short)</h2>"
    "<div style='color:#555;margin:0 0 0.8em 0'>Efficient Frontier now has a <b>Frontier Picker</b>: click points to preview full MC + full drawdowns; apply to replace active portfolio.</div>"
)

display(widgets.VBox([
    header,
    accordion,
    widgets.HBox([run_btn, reset_btn, export_btn], layout=widgets.Layout(gap="10px"))
]), out)


# In[ ]:




