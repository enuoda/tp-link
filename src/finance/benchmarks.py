#!/usr/bin/env python3

"""
Utilities for cointegration benchmark files

Provides functions for loading, validating, and using benchmark data
for spread trading. Includes automatic symbol conversion between
canonical format (used in benchmarks) and exchange-specific formats.

Sam Dawley
08/2025
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import (
    EXCHANGE_NAME,
    canonical_to_exchange,
    exchange_to_canonical,
)


REQUIRED_TOP_KEYS = {"version", "computed_at_utc", "universe", "lookback_days", "cointegration_groups"}


# ==================================================
# PRIVATE UTILITIES
# ==================================================


def _read_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def _latest_path(root: Path) -> Optional[Path]:
    p = root / "benchmarks_latest.json"
    return p if p.exists() else None


def _convert_symbol_to_exchange(symbol: str) -> str:
    """
    Convert a symbol from any format to current exchange format.
    
    Handles:
        - Canonical format: "BTC" -> "BTC/USD:USD" (Kraken)
        - Old Alpaca format: "BTC/USD" -> "BTC/USD:USD" (Kraken)
        - Other exchange format: "BTC/USDT:USDT" -> "BTC/USD:USD" (Kraken)
    """
    canonical = exchange_to_canonical(symbol)
    return canonical_to_exchange(canonical)


def _convert_weights_to_exchange(weights: Dict[str, float]) -> Dict[str, float]:
    """Convert benchmark weights from canonical to exchange format."""
    return {_convert_symbol_to_exchange(sym): w for sym, w in weights.items()}


def _convert_assets_to_exchange(assets: List[str]) -> List[str]:
    """Convert asset list from any format to exchange format."""
    return [_convert_symbol_to_exchange(a) for a in assets]


# ==================================================
# PUBLIC UTILITIES
# ==================================================


def entry_exit_signal(
    group: dict, 
    price_map: Dict[str, float],
    entry_zscore: float = 2.0,
    exit_zscore: float = 0.5,
) -> Dict[str, float | str]:
    """
    Generate entry/exit signal based on current prices and z-score thresholds.
    
    Args:
        group: Cointegration group from benchmark file
        price_map: Current prices (can be in any symbol format)
        entry_zscore: Z-score threshold for entry signals (default: 2.0)
        exit_zscore: Z-score threshold for exit signals (default: 0.5)
        
    Returns:
        Dict with 'zscore' and 'signal' ('HOLD', 'SELL_SPREAD', 'BUY_SPREAD', 'EXIT')
    """
    z = zscore_from_prices(group, price_map)
    signal = "HOLD"
    if z >= entry_zscore:
        signal = "SELL_SPREAD"  # short the positive-spread direction
    elif z <= -entry_zscore:
        signal = "BUY_SPREAD"   # long the spread
    elif abs(z) <= exit_zscore:
        signal = "EXIT"
    return {"zscore": z, "signal": signal}


def get_group(payload: dict, group_id: str) -> Optional[dict]:
    """Get a specific cointegration group by ID."""
    for g in payload.get("cointegration_groups", []):
        if g.get("id") == group_id:
            return g
    return None


def get_weights(group: dict, convert_to_exchange: bool = True) -> Dict[str, float]:
    """
    Get weights from a cointegration group.
    
    Args:
        group: Cointegration group dict
        convert_to_exchange: If True, convert symbols to current exchange format
        
    Returns:
        Dict mapping symbol to weight
    """
    vectors = group.get("vectors") or []
    if not vectors:
        return {}
    
    weights = dict(vectors[0].get("weights", {}))
    
    if convert_to_exchange:
        weights = _convert_weights_to_exchange(weights)
    
    return weights


def get_assets(group: dict, convert_to_exchange: bool = True) -> List[str]:
    """
    Get assets from a cointegration group.
    
    Args:
        group: Cointegration group dict
        convert_to_exchange: If True, convert symbols to current exchange format
        
    Returns:
        List of asset symbols
    """
    assets = group.get("assets", [])
    
    if convert_to_exchange:
        assets = _convert_assets_to_exchange(assets)
    
    return assets


def get_spread_params(group: dict) -> Tuple[float, float]:
    """Get spread mean and std from a cointegration group."""
    vectors = group.get("vectors") or []
    if not vectors:
        return np.nan, np.nan
    v0 = vectors[0]
    return float(v0.get("spread_mean", np.nan)), float(v0.get("spread_std", np.nan))


def load_benchmarks(path: Optional[str | Path] = None) -> dict:
    """
    Load a benchmark JSON snapshot (defaults to data/benchmarks/benchmarks_latest.json).
    Validate minimally that required fields exist.
    
    Note: The benchmark file may contain symbols in various formats (canonical,
    old Alpaca format, etc.). Use get_weights() and get_assets() with
    convert_to_exchange=True to get symbols in the current exchange format.
    """
    if path is None:
        root = Path("data/benchmarks")
        p = _latest_path(root)
        if p is None:
            raise FileNotFoundError("No benchmarks_latest.json found under data/benchmarks")
    else:
        p = Path(path)
    payload = _read_json(p)
    missing = REQUIRED_TOP_KEYS - set(payload.keys())
    if missing:
        raise ValueError(f"Benchmark JSON missing required keys: {missing}")
    return payload


def is_stale(payload: dict, max_age_days: int = 7) -> bool:
    """Check if benchmark data is older than max_age_days."""
    ts = datetime.fromisoformat(payload["computed_at_utc"]).replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - ts
    return age.days > max_age_days


def zscore_from_prices(group: dict, price_map: Dict[str, float]) -> float:
    """
    Calculate z-score from current prices using benchmark parameters.
    
    Automatically handles symbol format conversion. The price_map can use
    any symbol format (exchange-specific, canonical, or old Alpaca format).
    
    Args:
        group: Cointegration group with weights and spread parameters
        price_map: Current prices keyed by symbol (any format)
        
    Returns:
        Z-score of current spread, or NaN if calculation fails
    """
    # Get weights in exchange format (what the price_map likely uses)
    weights_exchange = get_weights(group, convert_to_exchange=True)
    
    # Also get weights with canonical symbols for fallback lookup
    weights_canonical = get_weights(group, convert_to_exchange=False)
    
    mean_, std_ = get_spread_params(group)
    
    # Normalize price_map keys to canonical for consistent lookup
    price_map_canonical = {}
    for sym, price in price_map.items():
        canonical = exchange_to_canonical(sym)
        price_map_canonical[canonical] = price
    
    # Also keep exchange-format keys
    price_map_exchange = {}
    for sym, price in price_map.items():
        exchange_sym = _convert_symbol_to_exchange(sym)
        price_map_exchange[exchange_sym] = price
    
    spread = 0.0
    
    # Try to match using exchange format first
    for sym, w in weights_exchange.items():
        canonical = exchange_to_canonical(sym)
        
        # Try to find price using various key formats
        price = None
        
        # Try exact exchange format match
        if sym in price_map:
            price = price_map[sym]
        # Try canonical format
        elif canonical in price_map_canonical:
            price = price_map_canonical[canonical]
        # Try exchange-converted format
        elif sym in price_map_exchange:
            price = price_map_exchange[sym]
        
        if price is None:
            return np.nan
        
        spread += float(w) * float(price)
    
    if std_ and std_ > 0:
        return (spread - mean_) / std_
    
    return np.nan


def get_universe(payload: dict, convert_to_exchange: bool = True) -> List[str]:
    """
    Get the universe of symbols from benchmark data.
    
    Args:
        payload: Loaded benchmark data
        convert_to_exchange: If True, convert to current exchange format
        
    Returns:
        List of symbols
    """
    universe = payload.get("universe", [])
    
    if convert_to_exchange:
        universe = _convert_assets_to_exchange(universe)
    
    return universe


def get_all_cointegration_assets(payload: dict, convert_to_exchange: bool = True) -> List[str]:
    """
    Get all unique assets involved in cointegration groups.
    
    Args:
        payload: Loaded benchmark data
        convert_to_exchange: If True, convert to current exchange format
        
    Returns:
        List of unique asset symbols used in cointegration groups
    """
    all_assets = set()
    
    for group in payload.get("cointegration_groups", []):
        assets = group.get("assets", [])
        all_assets.update(assets)
    
    all_assets_list = list(all_assets)
    
    if convert_to_exchange:
        all_assets_list = _convert_assets_to_exchange(all_assets_list)
    
    return all_assets_list
