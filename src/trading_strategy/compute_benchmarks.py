#!/usr/bin/env python3
"""
Weekly benchmark computation for pairwise cointegration (Engle–Granger) with JSON persistence.

Uses existing utilities in this codebase for data retrieval and tests:
- finance.crypto.fetch_crypto_data_for_cointegration
- trading_strategy.cointegration_utils.run_pairwise_cointegration (Engle–Granger)

Outputs JSON snapshots under data/benchmarks/ containing cointegration_groups
with weights, spread stats, and ranking strictly by p-value.

CLI:
  python -m trading_strategy.compute_benchmarks --symbols BTC/USD ETH/USD SOL/USD \
      --days 30 --time-scale hour --max-groups 10
"""

# stdlib
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# numerics
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

# custom
from src.trading_strategy.cointegration_utils import (
    run_pairwise_cointegration,
)
from src.finance.crypto import fetch_crypto_data_for_cointegration
from src.finance import TIME_FRAMES

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class GroupVector:
    weights: Dict[str, float]
    spread_mean: float
    spread_std: float
    half_life: float
    test_stats: Dict[str, float]
    p_value: Optional[float]


@dataclass
class GroupRecord:
    id: str
    assets: List[str]
    method: str
    rank: int
    vectors: List[GroupVector]
    selection_score: float
    zscore_thresholds: Dict[str, float]
    notes: str = ""


def _slug(symbols: Sequence[str]) -> str:
    return "-".join(s.replace("/", "").upper() for s in symbols)


def _base_symbol(symbol: str) -> str:
    return symbol.split("/")[0].upper() if "/" in symbol else symbol.upper()


def _half_life(spread: np.ndarray) -> float:
    # Simple AR(1) half-life estimate; guard against invalid cases
    x = np.asarray(spread, dtype=float)
    if x.size < 10 or not np.isfinite(x).all():
        return float("nan")
    x_lag = x[:-1]
    y = x[1:]
    # Add constant
    X = np.column_stack([np.ones_like(x_lag), x_lag])
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        phi = float(beta[1])
        if phi <= 0 or phi >= 1:
            return float("inf")
        return -math.log(2) / math.log(phi)
    except Exception:
        return float("nan")


def _normalize_time_scale(time_scale: str) -> str:
    """Normalize time scale string to a valid TIME_FRAMES key."""
    key = time_scale.strip().lower()
    if key in {"min", "minute", "minutes", "m"}:
        return "min"
    if key in {"hour", "hours", "h"}:
        return "hour"
    if key in {"day", "days", "d"}:
        return "day"
    if key in TIME_FRAMES:
        return key
    logger.warning(f"Unknown time_scale '{time_scale}', defaulting to 'hour'")
    return "hour"


def _align_arrays(symbols: List[str], days_back: int, time_scale_key: str) -> Tuple[np.ndarray, List[str]]:
    """
    Fetch and align price arrays for given symbols.
    
    Args:
        symbols: List of crypto symbols
        days_back: Lookback period in days
        time_scale_key: Time scale string (e.g., "hour", "day", "min")
        
    Returns:
        Tuple of (stacked price array, list of symbols)
    """
    # Convert string key to TimeFrame object (FIX: was passing string directly)
    normalized_key = _normalize_time_scale(time_scale_key)
    frequency = TIME_FRAMES.get(normalized_key, TIME_FRAMES["hour"])
    
    logger.debug(f"Fetching data for {symbols} with frequency={frequency}")
    
    try:
        price_arrays, sorted_symbols, _timestamps = fetch_crypto_data_for_cointegration(
            symbols=symbols, days_back=days_back, frequency=frequency
        )
    except Exception as e:
        logger.error(f"Failed to fetch data for {symbols}: {e}")
        raise
    
    arrays = [np.asarray(a, dtype=float) for a in price_arrays]
    
    # Log data quality info
    for i, sym in enumerate(sorted_symbols):
        valid_count = np.sum(~np.isnan(arrays[i]) & np.isfinite(arrays[i]))
        logger.debug(f"  {sym}: {len(arrays[i])} points, {valid_count} valid")
    
    return np.column_stack(arrays), list(sorted_symbols)


def _pairs_payload(
    symbols: List[str], days_back: int, time_scale: str, max_groups: int, p_threshold: float = 0.05
) -> List[GroupRecord]:
    """
    Run pairwise cointegration and build GroupRecord list.
    
    Args:
        symbols: List of crypto symbols
        days_back: Lookback period in days
        time_scale: Time scale string
        max_groups: Maximum groups to return
        p_threshold: P-value threshold for cointegration
        
    Returns:
        List of GroupRecord objects for cointegrated pairs
    """
    # Calculate expected number of pairs
    n_symbols = len(symbols)
    n_pairs = n_symbols * (n_symbols - 1) // 2
    
    logger.info(f"🔍 Running pairwise cointegration analysis...")
    logger.info(f"   Symbols: {n_symbols}, Possible pairs: {n_pairs}")
    logger.info(f"   Lookback: {days_back} days, Time scale: {time_scale}")
    logger.info(f"   P-value threshold: {p_threshold}")
    
    # Reuse existing pairwise runner to identify candidate pairs + rough stats
    mapping = run_pairwise_cointegration(
        tickers=symbols, time_scale=time_scale, days_back=days_back, p_threshold=p_threshold
    )
    
    logger.info(f"   Found {len(mapping)} candidate pairs passing initial threshold")

    # Build base->full mapping (assume unique base, e.g., BTC/USD only)
    base_to_full: Dict[str, str] = {}
    for s in symbols:
        base_to_full[_base_symbol(s)] = s

    # For selection scoring, recompute p-values and spread stats on aligned arrays
    groups: List[GroupRecord] = []
    pairs_processed = 0
    pairs_skipped_data = 0
    pairs_skipped_error = 0
    
    for (a_base, b_base), (hedge_ratio, std_spread) in mapping.items():
        a = base_to_full.get(a_base, a_base)
        b = base_to_full.get(b_base, b_base)
        pairs_processed += 1
        
        try:
            Y, ordered = _align_arrays([a, b], days_back, time_scale)
            x = Y[:, 0]
            y = Y[:, 1]
            mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
            x = x[mask]
            y = y[mask]
            
            if x.size < 10:
                logger.debug(f"   {a}/{b}: Skipped - only {x.size} valid data points")
                pairs_skipped_data += 1
                continue
                
            t_stat, p_val, _ = coint(x, y)
            spread = y - hedge_ratio * x
            hl = _half_life(spread)
            
            logger.debug(f"   {a}/{b}: t={t_stat:.3f}, p={p_val:.4f}, half_life={hl:.1f}")
            
            vect = GroupVector(
                weights={a: float(-hedge_ratio), b: 1.0},
                spread_mean=float(np.nanmean(spread)),
                spread_std=float(np.nanstd(spread, ddof=1)) if spread.size > 1 else 0.0,
                half_life=float(hl),
                test_stats={"t_stat": float(t_stat)},
                p_value=float(p_val),
            )
            # Selection: prefer low p, low half-life, low std
            score = (1.0 - min(1.0, float(p_val))) + (1.0 / (1.0 + float(hl) if math.isfinite(hl) else 1e6)) + (
                1.0 / (1.0 + float(vect.spread_std))
            )
            rec = GroupRecord(
                id=_slug([a, b]),
                assets=[a, b],
                method="EngleGranger",
                rank=1,
                vectors=[vect],
                selection_score=float(score),
                zscore_thresholds={"entry": 2.0, "exit": 0.5},
            )
            groups.append(rec)
            
        except Exception as e:
            logger.warning(f"   {a}/{b}: Error during analysis - {e}")
            pairs_skipped_error += 1
            continue

    logger.info(f"   Processed: {pairs_processed}, Groups created: {len(groups)}")
    logger.info(f"   Skipped (insufficient data): {pairs_skipped_data}, Skipped (errors): {pairs_skipped_error}")

    # Rank and keep top
    groups.sort(key=lambda g: g.selection_score, reverse=True)
    return groups[:max_groups]


# No higher-order (Johansen) computation in pairwise-only mode


def compute_benchmarks(
    symbols: List[str],
    days_back: int = 30,
    time_scale: str = "hour",
    max_groups: int = 10,
    p_threshold: float = 0.05,
) -> Dict:
    """
    Compute cointegration benchmarks for pairs trading.
    
    Args:
        symbols: List of crypto symbols (e.g., ["BTC/USD", "ETH/USD"])
        days_back: Lookback period in days for historical data
        time_scale: Time scale for bars ("min", "hour", "day")
        max_groups: Maximum number of cointegration groups to keep
        p_threshold: P-value threshold for cointegration test (default 0.05).
                     Higher values (e.g., 0.10, 0.15) are less strict and will
                     find more pairs; lower values are more strict.
    
    Returns:
        Dict containing benchmark payload with cointegration groups
    """
    now = datetime.now(timezone.utc)
    payload = {
        "version": "1.1",
        "computed_at_utc": now.isoformat(),
        "universe": symbols,
        "lookback_days": int(days_back),
        "p_threshold": float(p_threshold),
        "cointegration_groups": [],
        "metrics": {},
    }

    # Pairs only; rank strictly by ascending p-value
    pair_groups = _pairs_payload(symbols, days_back, time_scale, max_groups * 5, p_threshold)
    # Sort by p-value from the first vector
    pair_groups.sort(
        key=lambda g: (g.vectors[0].p_value if g.vectors and g.vectors[0].p_value is not None else 1.0)
    )
    top = pair_groups[:max_groups]
    payload["cointegration_groups"] = [
        {
            "id": g.id,
            "assets": g.assets,
            "method": g.method,
            "rank": g.rank,
            "vectors": [asdict(v) for v in g.vectors],
            "selection_score": g.selection_score,
            "zscore_thresholds": g.zscore_thresholds,
            "notes": g.notes,
        }
        for g in top
    ]

    return payload


def save_benchmarks(payload: Dict, out_dir: Path | str = "data/benchmarks") -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ts_name = datetime.utcnow().strftime("benchmarks_%G-%V.json")
    ts_file = out_path / ts_name
    # Ensure JSON-serializable types
    with ts_file.open("w") as f:
        json.dump(payload, f, indent=2)
    latest = out_path / "benchmarks_latest.json"
    with latest.open("w") as f:
        json.dump(payload, f, indent=2)
    return ts_file


def _parse_args(argv: Optional[Sequence[str]] = None):
    import argparse

    p = argparse.ArgumentParser(description="Compute weekly cointegration benchmarks")
    p.add_argument("--symbols", nargs="+", required=True, help="List of symbols, e.g., BTC/USD ETH/USD")
    p.add_argument("--days", type=int, default=30, help="Lookback days")
    p.add_argument("--time-scale", type=str, default="hour", help="min|hour|day")
    p.add_argument("--max-groups", type=int, default=10, help="Max groups to keep")
    p.add_argument("--p-threshold", type=float, default=0.05, 
                   help="P-value threshold for cointegration (default: 0.05). Higher values find more pairs.")
    p.add_argument("--out", type=str, default="data/benchmarks", help="Output directory")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    payload = compute_benchmarks(
        symbols=args.symbols,
        days_back=args.days,
        time_scale=args.time_scale,
        max_groups=args.max_groups,
        p_threshold=args.p_threshold,
    )
    save_benchmarks(payload, args.out)
    print(f"Saved benchmarks for {len(payload['cointegration_groups'])} groups to {args.out}")
    print(f"Used p-value threshold: {args.p_threshold}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


