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
    list_groups,
    get_weights,
    get_spread_params,
    zscore_from_prices,
    entry_exit_signal,
    is_stale,
)
from .rolling_buffer import RollingCointegrationBuffer


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
    ):
        """
        Initialize the spread signal engine.
        
        Args:
            benchmarks: Pre-loaded benchmark dict, or None to load from default path
            entry_zscore: Z-score threshold for entry signals (default: 2.0)
            exit_zscore: Z-score threshold for exit signals (default: 0.5)
            max_groups: Maximum number of groups to monitor
            entry_max_staleness_secs: Max price age (secs) for new entries (default: 30s)
            exit_max_staleness_secs: Max price age (secs) for exits (default: 5 min)
            emergency_exit_staleness_secs: Force exit after this staleness (default: 15 min)
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
        
        # Staleness thresholds
        self.entry_max_staleness_secs = entry_max_staleness_secs
        self.exit_max_staleness_secs = exit_max_staleness_secs
        self.emergency_exit_staleness_secs = emergency_exit_staleness_secs
        
        # Extract groups from benchmarks
        self.groups = list_groups(benchmarks)[:max_groups]
        
        # Track latest prices
        self._latest_prices: Dict[str, float] = {}
        
        # Track price staleness (symbol -> age in seconds)
        self._price_staleness: Dict[str, float] = {}
        
        # Track current signals per group
        self._current_signals: Dict[str, SpreadSignal] = {}
        
        # Track which groups have active positions (managed externally)
        self._active_positions: Dict[str, str] = {}  # group_id -> "LONG" or "SHORT"
        
        # Check benchmark staleness
        if benchmarks.get("cointegration_groups"):
            if is_stale(benchmarks):
                print("⚠️ Benchmark data is stale (>7 days old). Consider recomputing.")
            else:
                print(f"✅ Loaded {len(self.groups)} cointegration groups from benchmarks")
    
    def get_required_symbols(self) -> List[str]:
        """
        Get list of all symbols required by the monitored groups.
        
        Returns:
            List of unique symbols across all groups
        """
        symbols = set()
        for group in self.groups:
            symbols.update(group.get("assets", []))
        return list(symbols)
    
    def update_prices(
        self, 
        price_map: Dict[str, float], 
        staleness_map: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Update the engine with latest prices and staleness info.
        
        Args:
            price_map: Dictionary mapping symbol to latest price
            staleness_map: Optional dict mapping symbol -> age in seconds (None = fresh)
        """
        self._latest_prices.update(price_map)
        if staleness_map is not None:
            self._price_staleness = staleness_map.copy()
        else:
            # If no staleness info provided, assume all prices are fresh
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
        
        # Compute spread value from current prices
        spread = 0.0
        for symbol, weight in weights.items():
            if symbol not in self._latest_prices:
                return float("nan"), weights, mean_, std_
            spread += weight * self._latest_prices[symbol]
        
        # Compute z-score
        if std_ and std_ > 0:
            zscore = (spread - mean_) / std_
        else:
            zscore = float("nan")
        
        return zscore, weights, mean_, std_
    
    def _generate_signal_for_group(self, group: Dict) -> Optional[SpreadSignal]:
        """
        Generate a signal for a single group based on current prices.
        
        Staleness-aware logic:
        - ENTRY: Requires all prices within entry_max_staleness_secs
        - EXIT: Allows prices up to exit_max_staleness_secs
        - EMERGENCY_EXIT: Triggered when any price exceeds emergency_exit_staleness_secs
                          and we have an open position
        
        Args:
            group: Cointegration group dictionary
            
        Returns:
            SpreadSignal or None if no signal
        """
        group_id = group.get("id", "unknown")
        assets = group.get("assets", [])
        
        # Check position state first
        has_pos = self.has_position(group_id)
        pos_type = self.get_position(group_id)
        
        # Calculate staleness for each asset
        asset_staleness = {}
        for asset in assets:
            if asset in self._price_staleness:
                asset_staleness[asset] = self._price_staleness[asset]
            elif asset in self._latest_prices:
                asset_staleness[asset] = 0.0  # Assume fresh if no staleness data
            else:
                asset_staleness[asset] = float('inf')  # Missing = infinitely stale
        
        max_staleness = max(asset_staleness.values()) if asset_staleness else float('inf')
        
        # ===== EMERGENCY EXIT CHECK =====
        # If we have a position and data is critically stale, force exit
        if has_pos and max_staleness >= self.emergency_exit_staleness_secs:
            stale_assets = [a for a, s in asset_staleness.items() 
                           if s >= self.emergency_exit_staleness_secs]
            print(f"🚨 EMERGENCY EXIT triggered for {group_id}: "
                  f"critically stale assets {stale_assets}")
            return SpreadSignal(
                group_id=group_id,
                assets=assets,
                signal_type=SignalType.EMERGENCY_EXIT,
                zscore=float('nan'),
                confidence=1.0,
                weights={},
                spread_mean=0.0,
                spread_std=0.0,
            )
        
        # ===== CHECK FOR MISSING DATA =====
        missing = [a for a in assets if a not in self._latest_prices]
        if missing:
            return None
        
        # Compute z-score
        zscore, weights, spread_mean, spread_std = self._compute_zscore(group)
        
        if np.isnan(zscore):
            return None
        
        # Get thresholds (group-specific or engine defaults)
        thresholds = group.get("zscore_thresholds", {})
        entry_thr = float(thresholds.get("entry", self.entry_zscore))
        exit_thr = float(thresholds.get("exit", self.exit_zscore))
        
        signal_type = SignalType.HOLD
        confidence = 0.0
        
        if has_pos:
            # ===== EXIT LOGIC (more tolerant of staleness) =====
            if max_staleness > self.exit_max_staleness_secs:
                # Data too stale even for exit - skip this cycle but log warning
                stale_assets = [a for a, s in asset_staleness.items() 
                               if s > self.exit_max_staleness_secs]
                # Don't spam logs, just return None
                return None
            
            # Check for exit conditions
            if abs(zscore) <= exit_thr:
                signal_type = SignalType.EXIT
                confidence = 1.0 - abs(zscore) / exit_thr  # Higher confidence as z approaches 0
            elif pos_type == "LONG" and zscore >= entry_thr:
                # Wrong direction - should exit
                signal_type = SignalType.EXIT
                confidence = 0.8
            elif pos_type == "SHORT" and zscore <= -entry_thr:
                # Wrong direction - should exit
                signal_type = SignalType.EXIT
                confidence = 0.8
            # Otherwise hold
        else:
            # ===== ENTRY LOGIC (strict staleness requirement) =====
            if max_staleness > self.entry_max_staleness_secs:
                # Data too stale for entry - don't trade
                return None
            
            # Check for entry conditions
            if zscore >= entry_thr:
                signal_type = SignalType.SELL_SPREAD  # Spread is too high, expect reversion
                confidence = min(1.0, abs(zscore) / (entry_thr * 2))
            elif zscore <= -entry_thr:
                signal_type = SignalType.BUY_SPREAD  # Spread is too low, expect reversion
                confidence = min(1.0, abs(zscore) / (entry_thr * 2))
        
        # Only return signal if it's actionable
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
        Generate signals for all monitored groups.
        
        Returns:
            List of SpreadSignal objects for groups with actionable signals
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
                    "assets": group.get("assets", []),
                    "zscore": zscore,
                    "weights": weights,
                    "spread_mean": mean_,
                    "spread_std": std_,
                    "has_position": self.has_position(group_id),
                    "position_type": self.get_position(group_id),
                    "current_prices": {a: self._latest_prices.get(a) for a in group.get("assets", [])},
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
            
            # Build symbol to index mapping
            sym_to_idx = {s: i for i, s in enumerate(symbols)}
            
            groups_to_update = [g for g in self.groups if group_id is None or g.get("id") == group_id]
            
            for group in groups_to_update:
                assets = group.get("assets", [])
                weights = get_weights(group)
                
                # Check if we have data for all assets
                if not all(a in sym_to_idx for a in assets):
                    continue
                
                # Compute spread series
                spread = np.zeros(prices.shape[1])
                for asset, weight in weights.items():
                    if asset in sym_to_idx:
                        spread += weight * prices[sym_to_idx[asset]]
                
                # Update spread params in the group's first vector
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

