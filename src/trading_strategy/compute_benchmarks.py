#!/usr/bin/env python3
"""
Weekly benchmark computation for pairwise cointegration (Engle–Granger) with JSON persistence.

Uses existing utilities in this codebase for data retrieval and tests:
- finance.crypto.fetch_crypto_data_for_cointegration
- trading_strategy.cointegration_utils.run_pairwise_cointegration (Engle–Granger)

Outputs JSON snapshots under data/benchmarks/ containing cointegration_groups
with weights, spread stats, and ranking strictly by p-value.

**Symbol Format**: Benchmarks are stored using CANONICAL format (just base asset,
e.g., "BTC", "ETH") for portability across exchanges. When loading benchmarks
for trading, symbols are automatically converted to the current exchange format.

CLI:
  python -m trading_strategy.compute_benchmarks --symbols BTC ETH SOL \
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
# import pandas as pd
from statsmodels.tsa.stattools import coint

# custom
from src.trading_strategy.cointegration_utils import (
    run_pairwise_cointegration,
)
from src.finance.crypto import fetch_crypto_data_for_cointegration
from src.finance import (
    TIME_FRAMES,
    CRYPTO_TICKERS,
    EXCHANGE_NAME,
    EXCHANGE_CONFIG,
    DEMO_MODE,
    exchange_to_canonical,
    canonical_to_exchange,
    fetch_available_symbols,
)

# Configure logging
logger = logging.getLogger(__name__)


# ==================================================
# HEDGE RATIO FILTERING
# ==================================================

# Hedge ratios outside this range produce leg notionals that are impractical to trade.
# E.g., hedge_ratio=0.01 means one leg would be 1% of the other, likely below exchange minimums.
MIN_HEDGE_RATIO = 0.05  # Skip pairs where one leg would be <5% of the other
MAX_HEDGE_RATIO = 20.0  # Skip pairs requiring extreme leverage in one leg (1/0.05 = 20)


# ==================================================
# DATA CLASSES
# ==================================================


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
    notes: str = ""


# ==================================================
# PRIVATE UTILITIES
# ==================================================


def _half_life(spread: np.ndarray) -> float:
    """ Simple AR(1) half-life estimate; guard against invalid cases """
    x = np.asarray(spread, dtype=float)
    if x.size < 10 or not np.isfinite(x).all():
        return np.nan

    x_lag = x[:-1]
    y = x[1:]
    x_arr = np.column_stack([np.ones_like(x_lag), x_lag])

    try:
        beta = np.linalg.lstsq(x_arr, y, rcond=None)[0]
        phi = float(beta[1])
        if phi <= 0 or phi >= 1:
            return np.inf
        return -math.log(2) / math.log(phi)

    except Exception:
        return np.nan


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

    # ----- convert string key to TimeFrame object -----
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
    
    # ----- logging -----
    for i, sym in enumerate(sorted_symbols):
        valid_count = np.sum(~np.isnan(arrays[i]) & np.isfinite(arrays[i]))
        logger.debug(f"  {sym}: {len(arrays[i])} points, {valid_count} valid")
    
    return np.column_stack(arrays), list(sorted_symbols)


def _to_canonical(symbol: str) -> str:
    """Convert any symbol format to canonical (base asset only)."""
    return exchange_to_canonical(symbol)


def _to_exchange(canonical: str) -> str:
    """Convert canonical symbol to current exchange format."""
    return canonical_to_exchange(canonical)


def _pairs_payload(
    symbols: List[str], days_back: int, time_scale: str, max_groups: int, p_threshold: float = 0.05
) -> List[GroupRecord]:
    """
    Run pairwise cointegration and build GroupRecord list.
    
    Args:
        symbols: List of crypto symbols (exchange-specific format)
        days_back: Lookback period in days
        time_scale: Time scale string
        max_groups: Maximum groups to return
        p_threshold: P-value threshold for cointegration
        
    Returns:
        List of GroupRecord objects for cointegrated pairs (using canonical symbols)
    """
    # Helper to create a unique ID from canonical symbols
    def _slug(canonical_symbols: Sequence[str]) -> str:
        return "-".join(s.upper() for s in canonical_symbols)
    
    # Helper to get canonical (base) from any symbol format
    def _base_symbol(s: str) -> str:
        return _to_canonical(s)

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
    
    logger.info(f"\tFound {len(mapping)} candidate pairs passing initial threshold")

    # Build base->full mapping (assume unique base, e.g., BTC/USD only)
    base_to_full: Dict[str, str] = {}
    for s in symbols:
        base_to_full[_base_symbol(s)] = s

    # For selection scoring, recompute p-values and spread stats on aligned arrays
    groups: List[GroupRecord] = []
    pairs_processed = 0
    pairs_skipped_data = 0
    pairs_skipped_error = 0
    pairs_skipped_extreme_hedge = 0
    pairs_skipped_inverse_corr = 0
    
    for (a_base, b_base), (hedge_ratio, std_spread) in mapping.items():
        a = base_to_full.get(a_base, a_base)
        b = base_to_full.get(b_base, b_base)
        pairs_processed += 1
        
        # Filter extreme hedge ratios that would produce untradeable leg sizes
        abs_hedge = abs(hedge_ratio)
        if abs_hedge < MIN_HEDGE_RATIO or abs_hedge > MAX_HEDGE_RATIO:
            logger.debug(
                f"\t{a}/{b}: Skipped - hedge ratio {hedge_ratio:.4f} outside tradeable range "
                f"[{MIN_HEDGE_RATIO}, {MAX_HEDGE_RATIO}]"
            )
            pairs_skipped_extreme_hedge += 1
            continue
        
        # Filter pairs with negative hedge ratio (inverse correlation)
        # These produce same-sign weights, which aren't suitable for market-neutral spread trading
        # A proper pairs trade requires one long and one short position
        if hedge_ratio < 0:
            logger.debug(
                f"\t{a}/{b}: Skipped - negative hedge ratio {hedge_ratio:.4f} "
                f"(inverse correlation not suitable for pairs trading)"
            )
            pairs_skipped_inverse_corr += 1
            continue
        
        try:
            Y, ordered = _align_arrays([a, b], days_back, time_scale)
            x = Y[:, 0]
            y = Y[:, 1]
            mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
            x = x[mask]
            y = y[mask]
            
            if x.size < 10:
                logger.debug(f"\t{a}/{b}: Skipped - only {x.size} valid data points")
                pairs_skipped_data += 1
                continue
                
            t_stat, p_val, _ = coint(x, y)
            spread = y - hedge_ratio * x
            hl = _half_life(spread)
            
            logger.debug(f"\t{a}/{b}: t={t_stat:.3f}, p={p_val:.4f}, half_life={hl:.1f}")
            
            # Convert to canonical symbols for storage (portable across exchanges)
            a_canonical = _to_canonical(a)
            b_canonical = _to_canonical(b)
            
            vect = GroupVector(
                weights={a_canonical: float(-hedge_ratio), b_canonical: 1.0},
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
                id=_slug([a_canonical, b_canonical]),
                assets=[a_canonical, b_canonical],  # Store canonical symbols
                method="EngleGranger",
                rank=1,
                vectors=[vect],
                selection_score=float(score),
            )
            groups.append(rec)
            
        except Exception as e:
            logger.warning(f"\t{a}/{b}: Error during analysis - {e}")
            pairs_skipped_error += 1
            continue

    logger.info(f"\tProcessed: {pairs_processed}, Groups created: {len(groups)}")
    logger.info(f"\tSkipped (insufficient data): {pairs_skipped_data}, Skipped (errors): {pairs_skipped_error}")
    logger.info(f"\tSkipped (extreme hedge ratio): {pairs_skipped_extreme_hedge}")
    logger.info(f"\tSkipped (inverse correlation): {pairs_skipped_inverse_corr}")

    groups.sort(key=lambda g: g.selection_score, reverse=True)
    return groups[:max_groups]


# ==================================================
# BENCHMARKING
# ==================================================


def compute_benchmarks(
    symbols: List[str] = None,
    days_back: int = 30,
    time_scale: str = "hour",
    max_groups: int = 10,
    p_threshold: float = 0.05,
    use_dynamic_symbols: bool = True,
) -> Dict:
    """
    Compute cointegration benchmarks for pairs trading.
    
    Symbols can be provided in any format (canonical, exchange-specific, etc.)
    and will be automatically converted to exchange format for data fetching,
    then stored in canonical format for portability.
    
    Args:
        symbols: List of crypto symbols (optional). Can be:
            - Canonical: ["BTC", "ETH", "SOL"]
            - Exchange-specific: ["BTC/USD:USD", "ETH/USDT:USDT"]
            - Old Alpaca format: ["BTC/USD", "ETH/USD"]
            If None and use_dynamic_symbols=True, will fetch from exchange.
        days_back: Lookback period in days for historical data
        time_scale: Time scale for bars ("min", "hour", "day")
        max_groups: Maximum number of cointegration groups to keep
        p_threshold: P-value threshold for cointegration test (default 0.05).
                     Higher values (e.g., 0.10, 0.15) are less strict and will
                     find more pairs; lower values are more strict.
        use_dynamic_symbols: If True and symbols is None, fetch available
                            symbols from the exchange dynamically.
    
    Returns:
        Dict containing benchmark payload with cointegration groups.
        All symbols in the output are in canonical format (base asset only).
    """
    now = datetime.now(timezone.utc)
    
    # Get symbols - either provided, dynamic, or static fallback
    if symbols is None:
        if use_dynamic_symbols:
            logger.info(f"📡 Fetching available symbols from {EXCHANGE_CONFIG['name']}...")
            try:
                symbols = fetch_available_symbols()
                logger.info(f"   Found {len(symbols)} perpetual contracts")
            except Exception as e:
                logger.warning(f"⚠️ Could not fetch symbols dynamically: {e}")
                symbols = CRYPTO_TICKERS
                logger.info(f"   Using static symbol list ({len(symbols)} symbols)")
        else:
            symbols = CRYPTO_TICKERS
            logger.info(f"   Using static symbol list ({len(symbols)} symbols)")
    
    # Convert input symbols to exchange format for data fetching
    exchange_symbols = [_to_exchange(_to_canonical(s)) for s in symbols]
    
    # Store universe in canonical format for portability
    canonical_universe = [_to_canonical(s) for s in symbols]
    
    logger.info(f"📊 Computing benchmarks on {EXCHANGE_CONFIG['name']}")
    logger.info(f"   Mode: {'TESTNET' if DEMO_MODE else 'PRODUCTION'}")
    logger.info(f"   Input symbols: {len(symbols)}")
    logger.info(f"   Exchange symbols: {exchange_symbols[:5]}...")
    logger.info(f"   Canonical symbols: {canonical_universe[:5]}...")
    
    payload = {
        "version": "1.3",  # Updated version for dynamic symbol support
        "computed_at_utc": now.isoformat(),
        "computed_on_exchange": EXCHANGE_NAME,  # Track which exchange was used
        "computed_on_testnet": DEMO_MODE,  # Track testnet vs production
        "universe": canonical_universe,  # Store in canonical format
        "lookback_days": int(days_back),
        "p_threshold": float(p_threshold),
        "cointegration_groups": [],
        "metrics": {},
    }

    # ----- pairs only; rank strictly by ascending p-value -----
    # Use exchange-specific symbols for data fetching
    pair_groups = _pairs_payload(exchange_symbols, days_back, time_scale, 5 * max_groups, p_threshold)
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
            "notes": g.notes,
        }
        for g in top
    ]

    return payload


def save_benchmarks(payload: Dict, out_dir: Path | str = "data/benchmarks") -> Path:
    """
    Save benchmark payload to JSON files.
    
    Creates two files:
        - benchmarks_{YEAR}-{WEEK}.json (timestamped)
        - benchmarks_latest.json (always points to latest)
    
    Args:
        payload: Benchmark data dict (should use canonical symbols)
        out_dir: Output directory
        
    Returns:
        Path to the timestamped benchmark file
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ts_name = datetime.utcnow().strftime("benchmarks_%G-%V.json")
    ts_file = out_path / ts_name

    with ts_file.open("w") as f:
        json.dump(payload, f, indent=2)
    latest = out_path / "benchmarks_latest.json"
    with latest.open("w") as f:
        json.dump(payload, f, indent=2)
    
    logger.info(f"💾 Saved benchmarks to {ts_file}")
    logger.info(f"   Version: {payload.get('version', 'unknown')}")
    logger.info(f"   Universe: {len(payload.get('universe', []))} symbols (canonical format)")
    logger.info(f"   Groups: {len(payload.get('cointegration_groups', []))}")
    
    return ts_file


# ==================================================
# PRIMARY SUBROUTINE
# ==================================================


def _parse_args(argv: Optional[Sequence[str]] = None):
    import argparse

    p = argparse.ArgumentParser(description="Compute weekly cointegration benchmarks")
    p.add_argument("--symbols", nargs="+", required=False, 
                   help="List of symbols, e.g., BTC ETH SOL. If not provided, fetches from exchange.")
    p.add_argument("--days", type=float, default=30, help="Lookback days")
    p.add_argument("--time-scale", type=str, default="hour", help="min|hour|day")
    p.add_argument("--max-groups", type=int, default=10, help="Max groups to keep")
    p.add_argument("--p-threshold", type=float, default=0.05, 
                   help="P-value threshold for cointegration (default: 0.05). Higher values find more pairs.")
    p.add_argument("--out", type=str, default="data/benchmarks", help="Output directory")
    p.add_argument("--no-dynamic", action="store_true",
                   help="Don't fetch symbols dynamically, use static list")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    
    # Determine symbols to use
    symbols = args.symbols if args.symbols else None
    use_dynamic = not args.no_dynamic
    
    payload = compute_benchmarks(
        symbols=symbols,
        days_back=args.days,
        time_scale=args.time_scale,
        max_groups=args.max_groups,
        p_threshold=args.p_threshold,
        use_dynamic_symbols=use_dynamic,
    )
    save_benchmarks(payload, args.out)
    
    print(f"Saved benchmarks for {len(payload['cointegration_groups'])} groups to {args.out}")
    print(f"Exchange: {payload.get('computed_on_exchange', 'unknown')}")
    print(f"Testnet: {payload.get('computed_on_testnet', 'unknown')}")
    print(f"P-value threshold: {args.p_threshold}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


