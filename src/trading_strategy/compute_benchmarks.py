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
# from itertools import combinations
import math
# import os
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


def _align_arrays(symbols: List[str], days_back: int, time_scale_key) -> Tuple[np.ndarray, List[str]]:
    # Use existing data fetcher; returns arrays aligned to a common timestamp index
    price_arrays, sorted_symbols, _timestamps = fetch_crypto_data_for_cointegration(
        symbols=symbols, days_back=days_back, frequency=time_scale_key
    )
    arrays = [np.asarray(a, dtype=float) for a in price_arrays]
    return np.column_stack(arrays), list(sorted_symbols)


def _pairs_payload(
    symbols: List[str], days_back: int, time_scale: str, max_groups: int
) -> List[GroupRecord]:
    # Reuse existing pairwise runner to identify candidate pairs + rough stats
    mapping = run_pairwise_cointegration(
        tickers=symbols, time_scale=time_scale, days_back=days_back, p_threshold=0.05
    )

    # Build base->full mapping (assume unique base, e.g., BTC/USD only)
    base_to_full: Dict[str, str] = {}
    for s in symbols:
        base_to_full[_base_symbol(s)] = s

    # For selection scoring, recompute p-values and spread stats on aligned arrays
    groups: List[GroupRecord] = []
    for (a_base, b_base), (hedge_ratio, std_spread) in mapping.items():
        a = base_to_full.get(a_base, a_base)
        b = base_to_full.get(b_base, b_base)
        try:
            Y, ordered = _align_arrays([a, b], days_back, time_scale)
            x = Y[:, 0]
            y = Y[:, 1]
            mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
            x = x[mask]
            y = y[mask]
            if x.size < 10:
                continue
            t_stat, p_val, _ = coint(x, y)
            spread = y - hedge_ratio * x
            hl = _half_life(spread)
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
        except Exception:
            continue

    # Rank and keep top
    groups.sort(key=lambda g: g.selection_score, reverse=True)
    return groups[:max_groups]


# No higher-order (Johansen) computation in pairwise-only mode


def compute_benchmarks(
    symbols: List[str],
    days_back: int = 30,
    time_scale: str = "hour",
    max_groups: int = 10,
) -> Dict:
    now = datetime.now(timezone.utc)
    payload = {
        "version": "1.1",
        "computed_at_utc": now.isoformat(),
        "universe": symbols,
        "lookback_days": int(days_back),
        "cointegration_groups": [],
        "metrics": {},
    }

    # Pairs only; rank strictly by ascending p-value
    pair_groups = _pairs_payload(symbols, days_back, time_scale, max_groups * 5)
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
    p.add_argument("--out", type=str, default="data/benchmarks", help="Output directory")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    payload = compute_benchmarks(
        symbols=args.symbols,
        days_back=args.days,
        time_scale=args.time_scale,
        max_groups=args.max_groups,
    )
    save_benchmarks(payload, args.out)
    print(f"Saved benchmarks for {len(payload['cointegration_groups'])} groups to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


