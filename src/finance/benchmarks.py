from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REQUIRED_TOP_KEYS = {"version", "computed_at_utc", "universe", "lookback_days", "cointegration_groups"}


def _read_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def _latest_path(root: Path) -> Optional[Path]:
    p = root / "benchmarks_latest.json"
    return p if p.exists() else None


def load_benchmarks(path: Optional[str | Path] = None) -> dict:
    """
    Load a benchmark JSON snapshot (defaults to data/benchmarks/benchmarks_latest.json).
    Validate minimally that required fields exist.
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
    ts = datetime.fromisoformat(payload["computed_at_utc"]).replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - ts
    return age.days > max_age_days


def list_groups(payload: dict) -> List[dict]:
    return list(payload.get("cointegration_groups", []))


def get_group(payload: dict, group_id: str) -> Optional[dict]:
    for g in payload.get("cointegration_groups", []):
        if g.get("id") == group_id:
            return g
    return None


def get_weights(group: dict) -> Dict[str, float]:
    vectors = group.get("vectors") or []
    if not vectors:
        return {}
    return dict(vectors[0].get("weights", {}))


def get_spread_params(group: dict) -> Tuple[float, float]:
    vectors = group.get("vectors") or []
    if not vectors:
        return float("nan"), float("nan")
    v0 = vectors[0]
    return float(v0.get("spread_mean", float("nan"))), float(v0.get("spread_std", float("nan")))


def zscore_from_prices(group: dict, price_map: Dict[str, float]) -> float:
    weights = get_weights(group)
    mean_, std_ = get_spread_params(group)
    spread = 0.0
    for sym, w in weights.items():
        if sym not in price_map:
            return float("nan")
        spread += float(w) * float(price_map[sym])
    if std_ and std_ > 0:
        return (spread - mean_) / std_
    return float("nan")


def entry_exit_signal(group: dict, price_map: Dict[str, float]) -> Dict[str, float | str]:
    z = zscore_from_prices(group, price_map)
    thr = group.get("zscore_thresholds", {"entry": 2.0, "exit": 0.5})
    entry, exit_ = float(thr.get("entry", 2.0)), float(thr.get("exit", 0.5))
    signal = "HOLD"
    if z >= entry:
        signal = "SELL_SPREAD"  # short the positive-spread direction
    elif z <= -entry:
        signal = "BUY_SPREAD"   # long the spread
    elif abs(z) <= exit_:
        signal = "EXIT"
    return {"zscore": z, "signal": signal}


