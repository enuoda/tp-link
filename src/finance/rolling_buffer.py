#!/usr/bin/env python3
"""
Rolling Cointegration Buffer

Accumulates streamed bar data into rolling numpy arrays for cointegration analysis.
Maintains aligned timestamps across multiple symbols with thread-safe access.

Sam Dawley
"""

from __future__ import annotations

import threading
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo


class RollingCointegrationBuffer:
    """
    Thread-safe rolling buffer for accumulating streamed bar data.
    
    Designed to work with CryptoTrader's streaming callbacks to build
    aligned time series for real-time cointegration analysis.
    
    Features:
    - Accumulates OHLCV bar data for multiple symbols
    - Maintains timestamp alignment across symbols
    - Auto-prunes old data beyond lookback window
    - Provides aligned numpy arrays for cointegration tests
    - Thread-safe for concurrent streaming and analysis
    """
    
    def __init__(
        self,
        symbols: List[str],
        lookback_bars: int = 500,
        max_staleness_mins: int = 5,
    ):
        """
        Initialize the rolling buffer.
        
        Args:
            symbols: List of crypto symbols to track (e.g., ['BTC/USD', 'ETH/USD'])
            lookback_bars: Maximum number of bars to keep per symbol
            max_staleness_mins: Maximum age of data before considering symbol stale
        """
        self.symbols = list(symbols)
        self.lookback_bars = lookback_bars
        self.max_staleness_mins = max_staleness_mins
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Data storage: symbol -> list of bar dicts with 'timestamp', 'close', etc.
        self._bars: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Track last update time per symbol
        self._last_update: Dict[str, datetime] = {}
        
        # Common timestamp index (union of all bar timestamps)
        self._all_timestamps: set = set()
    
    def on_bar(self, bar_data) -> None:
        """
        Callback handler for incoming bar data from streaming.
        
        Designed to be registered with CCXTFuturesTrader streaming callbacks.
        
        Args:
            bar_data: Bar data dict from CCXT streaming (has symbol, timestamp, open, high, low, close, volume)
        """
        try:
            symbol = bar_data.symbol
            
            # Only track symbols we're interested in
            if symbol not in self.symbols:
                return
            
            with self._lock:
                # Create bar dict
                bar_dict = {
                    "timestamp": bar_data.timestamp,
                    "open": float(bar_data.open),
                    "high": float(bar_data.high),
                    "low": float(bar_data.low),
                    "close": float(bar_data.close),
                    "volume": float(bar_data.volume),
                }
                
                # Append to symbol's bar list
                self._bars[symbol].append(bar_dict)
                
                # Add timestamp to global set
                self._all_timestamps.add(bar_data.timestamp)
                
                # Update last update time
                self._last_update[symbol] = datetime.now(ZoneInfo("UTC"))
                
                # Prune old data if exceeding lookback
                if len(self._bars[symbol]) > self.lookback_bars * 2:
                    self._prune_old_data()
                    
        except Exception as e:
            print(f"Error in RollingCointegrationBuffer.on_bar: {e}")
    
    def add_bar_manually(
        self,
        symbol: str,
        timestamp: datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> None:
        """
        Add a bar manually (useful for testing or historical backfill).
        
        Args:
            symbol: Crypto symbol
            timestamp: Bar timestamp
            open_, high, low, close: OHLC prices
            volume: Trading volume
        """
        if symbol not in self.symbols:
            return
        
        with self._lock:
            bar_dict = {
                "timestamp": timestamp,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
            self._bars[symbol].append(bar_dict)
            self._all_timestamps.add(timestamp)
            self._last_update[symbol] = datetime.now(ZoneInfo("UTC"))
    
    def _prune_old_data(self) -> None:
        """
        Remove data older than lookback window.
        Must be called with lock held.
        """
        # Sort all timestamps and find cutoff
        sorted_ts = sorted(self._all_timestamps)
        if len(sorted_ts) <= self.lookback_bars:
            return
        
        cutoff_ts = sorted_ts[-self.lookback_bars]
        
        # Prune each symbol's bars
        for symbol in self.symbols:
            self._bars[symbol] = [
                b for b in self._bars[symbol]
                if b["timestamp"] >= cutoff_ts
            ]
        
        # Prune timestamp set
        self._all_timestamps = {ts for ts in self._all_timestamps if ts >= cutoff_ts}
    
    def get_aligned_arrays(
        self,
        symbols: List[str] = None,
        price_col: str = "close",
    ) -> Tuple[np.ndarray, List[str], List[datetime]]:
        """
        Get time-aligned price arrays for cointegration analysis.
        
        Returns arrays aligned to common timestamps where all requested
        symbols have data. Missing values are interpolated.
        
        Args:
            symbols: List of symbols to include (default: all tracked symbols)
            price_col: Which price to use ('close', 'open', 'high', 'low', 'vwap')
        
        Returns:
            prices: np.ndarray of shape (n_symbols, n_timestamps)
            symbols_out: List of symbols in same order as prices array
            timestamps: List of aligned timestamps
        """
        if symbols is None:
            symbols = self.symbols
        
        with self._lock:
            # Build DataFrames per symbol
            dfs = {}
            for symbol in symbols:
                if symbol not in self._bars or not self._bars[symbol]:
                    continue
                
                bars = self._bars[symbol]
                df = pd.DataFrame(bars)
                df = df.set_index("timestamp").sort_index()
                
                # Remove duplicate timestamps (keep last)
                df = df[~df.index.duplicated(keep="last")]
                dfs[symbol] = df
            
            if not dfs:
                return np.array([]), [], []
            
            # Find common timestamps across all symbols
            common_idx = None
            for symbol, df in dfs.items():
                if common_idx is None:
                    common_idx = set(df.index)
                else:
                    common_idx = common_idx.intersection(df.index)
            
            if not common_idx:
                # No common timestamps - try to use union with interpolation
                all_idx = pd.DatetimeIndex(sorted(self._all_timestamps))
                
                prices = []
                symbols_out = []
                for symbol in symbols:
                    if symbol not in dfs:
                        continue
                    df = dfs[symbol]
                    # Reindex and interpolate
                    reindexed = df.reindex(all_idx)
                    reindexed[price_col] = reindexed[price_col].interpolate(method="linear")
                    reindexed = reindexed.dropna(subset=[price_col])
                    
                    if not reindexed.empty:
                        prices.append(reindexed[price_col].values)
                        symbols_out.append(symbol)
                
                if not prices:
                    return np.array([]), [], []
                
                # Trim to common length
                min_len = min(len(p) for p in prices)
                prices = [p[-min_len:] for p in prices]
                timestamps = list(reindexed.index[-min_len:])
                
                return np.array(prices), symbols_out, timestamps
            
            # Use common timestamps
            common_idx_sorted = sorted(common_idx)
            
            prices = []
            symbols_out = []
            for symbol in symbols:
                if symbol not in dfs:
                    continue
                df = dfs[symbol]
                prices.append(df.loc[common_idx_sorted, price_col].values)
                symbols_out.append(symbol)
            
            return np.array(prices), symbols_out, common_idx_sorted
    
    def has_sufficient_data(self, min_bars: int = 100) -> bool:
        """
        Check if buffer has enough data for cointegration analysis.
        
        Args:
            min_bars: Minimum number of aligned bars required
        
        Returns:
            True if sufficient data is available
        """
        with self._lock:
            prices, symbols, _ = self.get_aligned_arrays()
            if len(prices) < 2:
                return False
            if prices.shape[1] < min_bars:
                return False
            return True
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current buffer state.
        
        Returns:
            Dictionary with buffer statistics
        """
        with self._lock:
            now = datetime.now(ZoneInfo("UTC"))
            stats = {
                "symbols_tracked": len(self.symbols),
                "total_timestamps": len(self._all_timestamps),
                "symbol_stats": {},
            }
            
            for symbol in self.symbols:
                bars = self._bars.get(symbol, [])
                last_update = self._last_update.get(symbol)
                
                if bars:
                    earliest = min(b["timestamp"] for b in bars)
                    latest = max(b["timestamp"] for b in bars)
                else:
                    earliest = None
                    latest = None
                
                staleness = None
                if last_update:
                    staleness = (now - last_update).total_seconds()
                
                stats["symbol_stats"][symbol] = {
                    "bar_count": len(bars),
                    "earliest_timestamp": earliest,
                    "latest_timestamp": latest,
                    "last_update_age_secs": staleness,
                    "is_stale": staleness is None or staleness > (self.max_staleness_mins * 60),
                }
            
            return stats
    
    def clear(self) -> None:
        """Clear all buffered data."""
        with self._lock:
            self._bars.clear()
            self._all_timestamps.clear()
            self._last_update.clear()
    
    def get_latest_prices(self) -> Dict[str, float]:
        """
        Get the latest close price for each symbol.
        
        Returns:
            Dictionary mapping symbol to latest close price
        """
        with self._lock:
            prices = {}
            for symbol in self.symbols:
                bars = self._bars.get(symbol, [])
                if bars:
                    # Get most recent bar
                    latest_bar = max(bars, key=lambda b: b["timestamp"])
                    prices[symbol] = latest_bar["close"]
            return prices
    
    def to_dataframe(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get all buffered data for a symbol as a DataFrame.
        
        Args:
            symbol: Crypto symbol
        
        Returns:
            DataFrame with OHLCV data indexed by timestamp, or None if no data
        """
        with self._lock:
            bars = self._bars.get(symbol, [])
            if not bars:
                return None
            
            df = pd.DataFrame(bars)
            df = df.set_index("timestamp").sort_index()
            return df


# def create_buffer_from_trader(
#     trader,
#     symbols: List[str],
#     lookback_bars: int = 500,
# ) -> RollingCointegrationBuffer:
#     """
#     Factory function to create a buffer and wire it to a CryptoTrader.
    
#     Args:
#         trader: CryptoTrader instance
#         symbols: Symbols to track
#         lookback_bars: Buffer size
    
#     Returns:
#         Configured RollingCointegrationBuffer
#     """
#     buffer = RollingCointegrationBuffer(
#         symbols=symbols,
#         lookback_bars=lookback_bars,
#     )
    
#     # Register the buffer's callback with the trader
#     trader.register_bar_callback(buffer.on_bar)
    
#     return buffer

