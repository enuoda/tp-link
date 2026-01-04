#!/usr/bin/env python3
"""
Spread Signal Engine

Real-time spread trading signal generation based on cointegration benchmarks.
Computes z-scores from live prices and emits entry/exit signals.

Sam Dawley
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from zoneinfo import ZoneInfo

from .benchmarks import (
    load_benchmarks,
    get_weights,
    get_assets,
    get_spread_params,
    is_stale,
)
from .rolling_buffer import RollingCointegrationBuffer
from . import get_staleness_for_pair, get_liquidity_tier


class SignalType(Enum):
    """Types of spread trading signals"""
    HOLD = "HOLD"
    BUY_SPREAD = "BUY_SPREAD"    # Go long the spread (buy positive weight, sell negative)
    SELL_SPREAD = "SELL_SPREAD"  # Go short the spread (sell positive weight, buy negative)
    EXIT = "EXIT"                 # Close existing spread position
    EMERGENCY_EXIT = "EMERGENCY_EXIT"  # Force close due to data staleness issues


@dataclass
class SpreadSignal:
    """A trading signal for a cointegrated spread"""
    group_id: str
    assets: List[str]
    signal_type: SignalType
    zscore: float
    confidence: float  # 0.0 to 1.0, based on z-score magnitude vs threshold
    weights: Dict[str, float]  # Asset -> weight for constructing the spread
    spread_mean: float
    spread_std: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(ZoneInfo("UTC")))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization"""
        return {
            "group_id": self.group_id,
            "assets": self.assets,
            "signal_type": self.signal_type.value,
            "zscore": self.zscore,
            "confidence": self.confidence,
            "weights": self.weights,
            "spread_mean": self.spread_mean,
            "spread_std": self.spread_std,
            "timestamp": self.timestamp.isoformat(),
        }


class SpreadSignalEngine:
    """
    Real-time spread signal engine using pre-computed cointegration benchmarks.
    
    Features:
    - Loads benchmark data (hedge ratios, spread stats)
    - Computes z-scores from live prices
    - Generates BUY_SPREAD / SELL_SPREAD / EXIT signals
    - Supports periodic recalibration from rolling buffer data
    - Tracks active signals and position state
    """
    
    def __init__(
        self,
        benchmarks: Dict = None,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        max_groups: int = 10,
        entry_max_staleness_secs: float = 30.0,
        exit_max_staleness_secs: float = 300.0,
        emergency_exit_staleness_secs: float = 900.0,
        available_symbols: List[str] = None,
    ):
        """
        Initialize the spread signal engine.
        
        Args:
            benchmarks: Pre-loaded benchmark dict, or None to load from default path
            entry_zscore: Z-score threshold for entry signals (default: 2.0)
            exit_zscore: Z-score threshold for exit signals (default: 0.5)
            max_groups: Maximum number of groups to monitor
            entry_max_staleness_secs: Global fallback for entry staleness (default: 30s).
                Per-pair thresholds from STALENESS_PROFILES take precedence.
            exit_max_staleness_secs: Global fallback for exit staleness (default: 5 min).
            available_symbols: List of symbols available on the exchange. If provided,
                groups with unavailable symbols will be filtered out.
                Per-pair thresholds from STALENESS_PROFILES take precedence.
            emergency_exit_staleness_secs: Global fallback for emergency staleness (default: 15 min).
                Per-pair thresholds from STALENESS_PROFILES take precedence.
        
        Note:
            Signal generation uses per-pair staleness thresholds based on asset liquidity
            tiers (high/medium/low). The global thresholds serve as fallbacks for unknown
            symbols. See src/finance/__init__.py STALENESS_PROFILES for tier definitions.
        """
        if benchmarks is None:
            try:
                benchmarks = load_benchmarks()
            except FileNotFoundError:
                print("⚠️ No benchmarks file found. Engine will need benchmarks to generate signals.")
                benchmarks = {"cointegration_groups": []}
        
        self.benchmarks = benchmarks
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.max_groups = max_groups
        
        # ----- staleness thresholds -----
        self.entry_max_staleness_secs = entry_max_staleness_secs
        self.exit_max_staleness_secs = exit_max_staleness_secs
        self.emergency_exit_staleness_secs = emergency_exit_staleness_secs
        
        # ----- extract and filter groups from benchmarks -----
        all_groups = list(benchmarks.get("cointegration_groups", []))
        
        # Filter groups to only include those where ALL assets are available
        if available_symbols is not None:
            available_set = set(available_symbols)
            filtered_groups = []
            skipped_groups = []
            
            for group in all_groups:
                group_assets = get_assets(group, convert_to_exchange=True)
                missing_assets = [a for a in group_assets if a not in available_set]
                
                if not missing_assets:
                    filtered_groups.append(group)
                else:
                    skipped_groups.append((group.get("id", "unknown"), missing_assets))
            
            if skipped_groups:
                print(f"⚠️ Filtered out {len(skipped_groups)} groups due to unavailable symbols:")
                for group_id, missing in skipped_groups[:5]:  # Show first 5
                    print(f"   - {group_id}: missing {missing}")
                if len(skipped_groups) > 5:
                    print(f"   ... and {len(skipped_groups) - 5} more")
            
            self.groups = filtered_groups[:max_groups]
        else:
            self.groups = all_groups[:max_groups]

        self._latest_prices: Dict[str, float] = {}
        self._price_staleness: Dict[str, float] = {} # symbol -> age in seconds
        self._current_signals: Dict[str, SpreadSignal] = {}
        self._active_positions: Dict[str, str] = {}  # group_id -> "LONG" or "SHORT"
        
        # ----- rolling price history for recalibration (symbol -> list of prices) -----
        self._price_history: Dict[str, List[float]] = {sym: [] for sym in self.get_required_symbols()}
        self._price_history_maxlen = 500
        self._last_recalibration: Optional[datetime] = None
        self._recalibration_count = 0
        
        # ----- check benchmark staleness and report status -----
        if benchmarks.get("cointegration_groups"):
            if is_stale(benchmarks):
                print("⚠️ Benchmark data is stale (>7 days old). Consider recomputing.")
            
            if self.groups:
                print(f"✅ Using {len(self.groups)} cointegration groups (of {len(all_groups)} total)")
            else:
                print("⚠️ No usable cointegration groups after filtering for available symbols")
    

    def get_required_symbols(self) -> List[str]:
        """
        Get all symbols needed for streaming from cointegration groups.
        
        Use this to determine which symbols to subscribe to via WebSocket.
        Only returns symbols that are part of loaded cointegration pairs.
        
        Returns:
            list: Unique symbols across all groups (e.g., ['BTC/USD', 'ETH/USD'])
            
        Example:
            >>> engine = SpreadSignalEngine(benchmarks)
            >>> symbols = engine.get_required_symbols()
            >>> trader.start_real_time_streaming(symbols)
        """
        symbols = set()
        for group in self.groups:
            symbols.update(get_assets(group))  # Converts to exchange format
        return list(symbols)
    
    def update_prices(
        self, 
        price_map: Dict[str, float], 
        staleness_map: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Update engine with latest prices before calling get_signals().
        
        Call this each trading cycle with fresh prices from streaming.
        Staleness info is used to block entries with stale data and
        trigger emergency exits when data is critically old.
        
        Args:
            price_map: {symbol: price} from get_latest_price()
            staleness_map: {symbol: age_secs} from get_connection_health()
            
        Returns:
            None
            
        Example:
            >>> engine.update_prices(
            ...     {'BTC/USD': 95000, 'ETH/USD': 3400},
            ...     {'BTC/USD': 2.5, 'ETH/USD': 1.8}
            ... )
            >>> signals = engine.get_signals()
        """
        self._latest_prices.update(price_map)
        if staleness_map is not None:
            self._price_staleness = staleness_map.copy()
        else:
            self._price_staleness = {sym: 0.0 for sym in price_map}
    

    def set_position(self, group_id: str, position_type: Optional[str]) -> None:
        """
        Set the current position state for a group.
        
        Args:
            group_id: The cointegration group ID
            position_type: "LONG", "SHORT", or None (no position)
        """
        if position_type is None:
            self._active_positions.pop(group_id, None)
        else:
            self._active_positions[group_id] = position_type
    

    def has_position(self, group_id: str) -> bool:
        """Check if there's an active position for a group"""
        return group_id in self._active_positions
    

    def get_position(self, group_id: str) -> Optional[str]:
        """Get the current position type for a group"""
        return self._active_positions.get(group_id)
    

    def _compute_zscore(self, group: Dict) -> Tuple[float, Dict[str, float], float, float]:
        """
        Compute z-score for a group from current prices.
        
        Returns:
            (zscore, weights, spread_mean, spread_std)
        """
        weights = get_weights(group)
        mean_, std_ = get_spread_params(group)
        
        # ----- compute spread value from current prices -----
        spread = 0.0
        for symbol, weight in weights.items():
            if symbol not in self._latest_prices:
                return np.nan, weights, mean_, std_

            spread += weight * self._latest_prices[symbol]
        
        # ----- compute z-score -----
        if std_ and std_ > 0:
            zscore = (spread - mean_) / std_
        else:
            zscore = np.nan
        
        return zscore, weights, mean_, std_
    

    def _generate_signal_for_group(self, group: Dict) -> Optional[SpreadSignal]:
        """
        Generate a signal for a single group based on current prices.
        
        Staleness-aware logic with per-pair thresholds:
        - Uses liquidity-based staleness thresholds for each asset pair
        - ENTRY: Requires all prices within pair-specific entry staleness
        - EXIT: Allows prices up to pair-specific exit staleness
        - EMERGENCY_EXIT: Triggered when any price exceeds pair-specific emergency threshold
                          and we have an open position
        
        Args:
            group: Cointegration group dictionary
            
        Returns:
            SpreadSignal or None if no signal
        """
        group_id = group.get("id", "unknown")
        assets = get_assets(group)  # Converts to exchange format
        
        # ----- per-pair staleness thresholds based on asset liquidity -----
        pair_staleness = get_staleness_for_pair(assets)
        entry_staleness_threshold = pair_staleness["entry"]
        exit_staleness_threshold = pair_staleness["exit"]
        emergency_staleness_threshold = pair_staleness["emergency"]
        
        has_pos = self.has_position(group_id)
        pos_type = self.get_position(group_id)
        
        # ----- calculate staleness for each asset -----
        asset_staleness = {}
        for asset in assets:
            if asset in self._price_staleness:
                asset_staleness[asset] = self._price_staleness[asset]
            elif asset in self._latest_prices:
                asset_staleness[asset] = 0.0
            else:
                asset_staleness[asset] = np.inf
        
        max_staleness = max(asset_staleness.values()) if asset_staleness else np.inf
        
        # ----- emergency exit check -----
        if has_pos and max_staleness >= emergency_staleness_threshold:
            stale_assets = [a for a, s in asset_staleness.items() if s >= emergency_staleness_threshold]
            tiers = [get_liquidity_tier(a) for a in assets]
            print(f"🚨 EMERGENCY EXIT triggered for {group_id} (tiers: {tiers}): "
                  f"critically stale assets {stale_assets} (threshold: {emergency_staleness_threshold}s)")

            return SpreadSignal(
                group_id=group_id,
                assets=assets,
                signal_type=SignalType.EMERGENCY_EXIT,
                zscore=np.nan,
                confidence=1.0,
                weights={},
                spread_mean=0.0,
                spread_std=0.0,
            )
        
        # ----- check for missing data -----
        missing = [a for a in assets if a not in self._latest_prices]
        if missing:
            return None
        
        zscore, weights, spread_mean, spread_std = self._compute_zscore(group)
        if np.isnan(zscore):
            return None
        
        thresholds = group.get("zscore_thresholds", {})
        entry_thr = float(thresholds.get("entry", self.entry_zscore))
        exit_thr = float(thresholds.get("exit", self.exit_zscore))
        
        confidence = 0.0
        signal_type = SignalType.HOLD
        
        if has_pos:
            
            # ----- exit logic -----
            if max_staleness > exit_staleness_threshold:
                stale_assets = [a for a, s in asset_staleness.items() if s > exit_staleness_threshold]
                return None
            
            if abs(zscore) <= exit_thr:
                signal_type = SignalType.EXIT
                confidence = 1.0 - abs(zscore) / exit_thr

            elif pos_type == "LONG" and zscore >= entry_thr:
                # Wrong direction - should exit
                signal_type = SignalType.EXIT
                confidence = 0.8

            elif pos_type == "SHORT" and zscore <= -entry_thr:
                # Wrong direction - should exit
                signal_type = SignalType.EXIT
                confidence = 0.8

        else:

            # ----- entry logic -----
            if max_staleness > entry_staleness_threshold:
                return None
            
            # if spread is too high or too low, expect reversion
            if zscore >= entry_thr:
                signal_type = SignalType.SELL_SPREAD
                confidence = min(1.0, abs(zscore) / (entry_thr * 2))

            elif zscore <= -entry_thr:
                signal_type = SignalType.BUY_SPREAD
                confidence = min(1.0, abs(zscore) / (entry_thr * 2))
        
        if signal_type == SignalType.HOLD:
            return None
        
        return SpreadSignal(
            group_id=group_id,
            assets=assets,
            signal_type=signal_type,
            zscore=zscore,
            confidence=confidence,
            weights=weights,
            spread_mean=spread_mean,
            spread_std=spread_std,
        )
    

    def get_signals(self) -> List[SpreadSignal]:
        """
        Generate trading signals for all monitored cointegration groups.
        
        Call after update_prices() each cycle. Returns actionable signals
        (BUY_SPREAD, SELL_SPREAD, EXIT, EMERGENCY_EXIT) based on z-scores
        and staleness thresholds. Uses per-symbol liquidity tiers.
        
        Returns:
            list[SpreadSignal]: Signals for groups needing action. Empty if
                all groups are in HOLD state or have missing/stale data.
            
        Example:
            >>> engine.update_prices(price_map, staleness_map)
            >>> for signal in engine.get_signals():
            ...     if signal.signal_type == SignalType.BUY_SPREAD:
            ...         execute_spread_buy(signal)
            ...     elif signal.signal_type == SignalType.EMERGENCY_EXIT:
            ...         force_close_position(signal.group_id)
        """
        signals = []
        
        for group in self.groups:
            signal = self._generate_signal_for_group(group)

            if signal is not None:
                signals.append(signal)
                self._current_signals[signal.group_id] = signal
        
        return signals
    

    def get_all_zscores(self) -> Dict[str, float]:
        """
        Get current z-scores for all groups.
        
        Returns:
            Dictionary mapping group_id to z-score
        """
        zscores = {}
        for group in self.groups:
            group_id = group.get("id", "unknown")
            zscore, _, _, _ = self._compute_zscore(group)
            zscores[group_id] = zscore
        return zscores
    

    def get_group_status(self, group_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed status for a specific group.
        
        Args:
            group_id: The cointegration group ID
            
        Returns:
            Dictionary with group status or None if not found
        """
        for group in self.groups:
            if group.get("id") == group_id:
                zscore, weights, mean_, std_ = self._compute_zscore(group)
                return {
                    "group_id": group_id,
                    "assets": get_assets(group),  # Converts to exchange format
                    "zscore": zscore,
                    "weights": weights,
                    "spread_mean": mean_,
                    "spread_std": std_,
                    "has_position": self.has_position(group_id),
                    "position_type": self.get_position(group_id),
                    "current_prices": {a: self._latest_prices.get(a) for a in get_assets(group)},
                }
        return None
    
    def recalibrate(
        self,
        buffer: RollingCointegrationBuffer,
        group_id: str = None,
    ) -> bool:
        """
        Re-estimate spread statistics from rolling buffer data.
        
        This updates the spread mean and std for more accurate z-score calculation
        based on recent market conditions.
        
        Args:
            buffer: RollingCointegrationBuffer with recent price data
            group_id: Specific group to recalibrate, or None for all
            
        Returns:
            True if recalibration was successful
        """
        try:
            prices, symbols, timestamps = buffer.get_aligned_arrays()
            
            if len(prices) < 2 or prices.shape[1] < 50:
                print("⚠️ Insufficient data for recalibration")
                return False
            
            # ----- build symbol to index mapping -----
            sym_to_idx = {s: i for i, s in enumerate(symbols)}
            
            groups_to_update = [g for g in self.groups if group_id is None or g.get("id") == group_id]
            
            for group in groups_to_update:
                assets = get_assets(group)  # Converts to exchange format
                weights = get_weights(group)  # Converts to exchange format
                
                # ----- check if we have data for all assets -----
                if not all(a in sym_to_idx for a in assets):
                    continue
                
                # ----- compute spread series -----
                spread = np.zeros(prices.shape[1])
                for asset, weight in weights.items():
                    if asset in sym_to_idx:
                        spread += weight * prices[sym_to_idx[asset]]
                
                # ----- update spread params in the group's first vector -----
                if group.get("vectors"):
                    new_mean = float(np.nanmean(spread))
                    new_std = float(np.nanstd(spread, ddof=1))
                    
                    group["vectors"][0]["spread_mean"] = new_mean
                    group["vectors"][0]["spread_std"] = new_std
                    
                    print(f"📊 Recalibrated {group.get('id')}: mean={new_mean:.4f}, std={new_std:.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during recalibration: {e}")
            return False
    

    def record_prices(self, price_map: Dict[str, float]) -> None:
        """
        Record current prices into the rolling history for recalibration.
        
        Call this each cycle to build up price history that can be used
        for rolling recalibration of spread mean/std.
        
        Args:
            price_map: Current prices {symbol: price}
            
        Example:
            >>> engine.record_prices({'BTC/USD': 95000, 'ETH/USD': 3400})
        """
        for symbol, price in price_map.items():
            if symbol in self._price_history:
                self._price_history[symbol].append(price)
                if len(self._price_history[symbol]) > self._price_history_maxlen:
                    self._price_history[symbol] = self._price_history[symbol][-self._price_history_maxlen:]
    

    def recalibrate_from_history(self, min_observations: int = 50) -> Dict[str, Dict]:
        """
        Recalibrate spread parameters using accumulated price history.
        
        Updates the spread mean and standard deviation for each cointegration
        group based on recent price observations. This helps z-scores stay
        relevant as market conditions drift from the original benchmark.
        
        Args:
            min_observations: Minimum price points needed per symbol (default: 50)
            
        Returns:
            Dict mapping group_id to recalibration results:
            {
                'group_id': {
                    'success': bool,
                    'old_mean': float,
                    'new_mean': float,
                    'old_std': float,
                    'new_std': float,
                    'observations': int,
                }
            }
            
        Example:
            >>> results = engine.recalibrate_from_history(min_observations=100)
            >>> for gid, r in results.items():
            ...     if r['success']:
            ...         print(f"{gid}: mean {r['old_mean']:.2f} -> {r['new_mean']:.2f}")
        """
        results = {}
        
        for group in self.groups:
            group_id = group.get("id", "unknown")
            assets = get_assets(group)  # Converts to exchange format
            
            # ----- check if we have enough data for all assets -----
            obs_counts = [len(self._price_history.get(a, [])) for a in assets]
            min_obs = min(obs_counts) if obs_counts else 0
            
            if min_obs < min_observations:
                results[group_id] = {
                    "success": False,
                    "reason": f'Insufficient data ({min_obs}/{min_observations} observations)',
                    "observations": min_obs,
                }
                continue
            
            try:
                weights = get_weights(group)
                old_mean, old_std = get_spread_params(group)
                
                # ----- compute spread series from price history -----
                spread_values = []
                for i in range(min_obs):
                    spread = 0.0
                    idx = -(min_obs - i)
                    for asset in assets:
                        if asset in weights and asset in self._price_history:
                            price = self._price_history[asset][idx]
                            spread += weights[asset] * price
                    spread_values.append(spread)
                
                new_mean = float(np.mean(spread_values))
                new_std = float(np.std(spread_values, ddof=1))
                
                if new_std < 1e-10:
                    results[group_id] = {
                        "success": False,
                        "reason": "Zero standard deviation",
                        "observations": min_obs,
                    }
                    continue
                
                # ----- update the group's spread params -----
                if group.get("vectors"):
                    group["vectors"][0]["spread_mean"] = new_mean
                    group["vectors"][0]["spread_std"] = new_std
                
                results[group_id] = {
                    "success": True,
                    "old_mean": old_mean,
                    "new_mean": new_mean,
                    "old_std": old_std,
                    "new_std": new_std,
                    "observations": min_obs,
                    "mean_shift": new_mean - old_mean,
                    "std_change_pct": ((new_std - old_std) / old_std * 100) if old_std else 0,
                }
                
            except Exception as e:
                results[group_id] = {
                    "success": False,
                    "reason": str(e),
                    "observations": min_obs,
                }
        
        self._last_recalibration = datetime.now(ZoneInfo("UTC"))
        self._recalibration_count += 1
        
        return results
    

    def get_recalibration_status(self) -> Dict[str, Any]:
        """
        Get status of the recalibration system.
        
        Returns:
            Dict with recalibration metrics
        """
        obs_counts = {sym: len(hist) for sym, hist in self._price_history.items()}
        return {
            "last_recalibration": self._last_recalibration.isoformat() if self._last_recalibration else None,
            "recalibration_count": self._recalibration_count,
            "price_history_lengths": obs_counts,
            "min_observations": min(obs_counts.values()) if obs_counts else 0,
            "max_history_length": self._price_history_maxlen,
        }
    

    def clear_price_history(self) -> None:
        """Clear the accumulated price history."""
        for sym in self._price_history:
            self._price_history[sym] = []
    

    def get_summary(self) -> str:
        """
        Get a formatted summary of current engine state.
        
        Returns:
            Multi-line string summary
        """
        lines = [
            "=" * 60,
            "SPREAD SIGNAL ENGINE STATUS",
            "=" * 60,
            f"Groups monitored: {len(self.groups)}",
            f"Symbols tracked: {len(self._latest_prices)}",
            f"Active positions: {len(self._active_positions)}",
            "",
            "Z-SCORES:",
        ]
        
        for group in self.groups:
            group_id = group.get("id", "?")
            zscore, _, _, _ = self._compute_zscore(group)
            pos = self._active_positions.get(group_id, "-")
            
            if np.isnan(zscore):
                z_str = "N/A (missing data)"
            else:
                z_str = f"{zscore:+.3f}"
            
            lines.append(f"  {group_id}: z={z_str}, pos={pos}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


def create_engine_from_benchmarks(
    benchmark_path: str = None,
    entry_zscore: float = 2.0,
    exit_zscore: float = 0.5,
    max_groups: int = 10,
    entry_max_staleness_secs: float = 30.0,
    exit_max_staleness_secs: float = 300.0,
    emergency_exit_staleness_secs: float = 900.0,
) -> SpreadSignalEngine:
    """
    Factory function to create a SpreadSignalEngine from benchmark file.
    
    Args:
        benchmark_path: Path to benchmarks JSON, or None for default
        entry_zscore: Entry threshold
        exit_zscore: Exit threshold
        max_groups: Max groups to monitor
        entry_max_staleness_secs: Max price age (secs) for new entries
        exit_max_staleness_secs: Max price age (secs) for exits
        emergency_exit_staleness_secs: Force exit after this staleness
        
    Returns:
        Configured SpreadSignalEngine
    """
    benchmarks = load_benchmarks(benchmark_path)
    return SpreadSignalEngine(
        benchmarks=benchmarks,
        entry_zscore=entry_zscore,
        exit_zscore=exit_zscore,
        max_groups=max_groups,
        entry_max_staleness_secs=entry_max_staleness_secs,
        exit_max_staleness_secs=exit_max_staleness_secs,
        emergency_exit_staleness_secs=emergency_exit_staleness_secs,
    )

