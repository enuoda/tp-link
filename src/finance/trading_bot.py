#!/usr/bin/env python3
"""
Live Trading Bot with Real-Time Data Streaming

This module provides live trading functionality using:
1. Real-time data streaming via CCXT (exchange-agnostic)
2. Live trading based on streaming market data
3. Risk management and position monitoring
4. Performance tracking and logging

The exchange is configurable via EXCHANGE_NAME environment variable.

Usage:
    python main.py --mode trade-indefinite

Requirements:
    - Valid exchange API credentials in environment variables
    - ccxt library installed
    - Demo/testnet account for paper trading (recommended for testing)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import time
import traceback
from typing import Dict, List, Optional

from . import CRYPTO_TICKERS, EXCHANGE_CONFIG, MAX_SPREAD_NOTIONAL, MIN_SPREAD_NOTIONAL, MAX_SPREAD_POSITIONS
from .ccxt_trader import CCXTFuturesTrader
from .benchmarks import (
    load_benchmarks,
    is_stale,
    # get_weights,
    # get_assets,
    # get_all_cointegration_assets,
)
from .spread_engine import SpreadSignalEngine, SpreadSignal, SignalType
from .rolling_buffer import RollingCointegrationBuffer

# global constants
MOMENTUM_BUY_THRESHOLD = 0.02
MOMENTUM_SELL_THRESHOLD = -0.02
MOMENTUM_MIN_DATA_POINTS = 10
MIN_LEG_NOTIONAL = 10.0  # minimum $ per spread leg to ensure order meets exchange minimums

# ==================================================
# LOGGING
# ==================================================


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("live_trading.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ==================================================
# DATA CLASSES
# ==================================================


@dataclass
class TradingSignal:
    """Data class for trading signals"""
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 to 1.0
    price: float
    reason: str
    timestamp: datetime


@dataclass
class Position:
    """Data class for position tracking"""
    symbol: str
    side: str  # 'LONG', 'SHORT'
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    entry_time: datetime


@dataclass
class SpreadPosition:
    """Data class for spread position tracking (pairs trading)"""
    
    group_id: str
    assets: List[str]
    side: str  # 'LONG' (bought the spread) or 'SHORT' (sold the spread)
    weights: Dict[str, float]  # Asset -> weight
    quantities: Dict[str, float]  # Asset -> quantity held
    entry_prices: Dict[str, float]  # Asset -> entry price
    current_prices: Dict[str, float] = field(default_factory=dict)
    notional: float = 0.0  # Total notional value of the spread
    entry_zscore: float = 0.0
    current_zscore: float = 0.0
    unrealized_pnl: float = 0.0
    entry_time: datetime = field(default_factory=datetime.now)
    
    def update_pnl(self, current_prices: Dict[str, float], current_zscore: float = None) -> None:
        """Update current prices and calculate unrealized P&L"""
        self.current_prices = current_prices

        if current_zscore is not None:
            self.current_zscore = current_zscore
        
        pnl = 0.0
        for asset in self.assets:
            qty = self.quantities.get(asset, 0)
            entry = self.entry_prices.get(asset, 0)
            current = current_prices.get(asset, entry)
            pnl += qty * (current - entry)
        
        self.unrealized_pnl = pnl


# ==================================================
# LIVE TRADING BOTS
# ==================================================


class TradingPartner:
    """
    Main trading interface that combines crypto trading and market data

    This class provides a unified interface for:
    - Real-time crypto trading with streaming data via Binance Futures
    - Market data analysis and visualization
    - Portfolio management and risk control
    - Spread trading based on cointegration analysis
    - Simplified trading operations for main.py integration
    """


    def __init__(
        self,
        paper: bool = True,
        spread_notional: float = MAX_SPREAD_NOTIONAL,
        min_spread_notional: float = MIN_SPREAD_NOTIONAL,
        max_spread_positions: int = MAX_SPREAD_POSITIONS,
        buying_power_buffer: float = 0.9,
    ) -> None:
        """
        Initialize the trading partner

        Args:
            paper: Whether to use paper trading/testnet (recommended for testing)
            spread_notional: Target USD per spread leg
            min_spread_notional: Minimum USD per leg (avoid dust orders)
            max_spread_positions: Maximum number of spread positions
            buying_power_buffer: Use 90% of buying power (10% buffer for fees)
        """
        self.crypto_trader = CCXTFuturesTrader(testnet=paper)
        self.paper = paper
        self.symbols: List[str] = []
        
        # ----- load benchmarks if available -----
        self.benchmarks: Optional[Dict] = None
        try:
            self.benchmarks = load_benchmarks()
            if is_stale(self.benchmarks):
                logger.warning("⚠️ Benchmark data is stale (>7 days). Consider recomputing.")
        except FileNotFoundError:
            logger.warning("⚠️ No benchmarks file found. Spread trading disabled.")
        
        self.spread_engine: Optional[SpreadSignalEngine] = None
        self.rolling_buffer: Optional[RollingCointegrationBuffer] = None
        self._spread_positions: Dict[str, SpreadPosition] = {}
        
        # ----- trading parameters for spread trades -----
        self.spread_notional = spread_notional
        self.min_spread_notional = min_spread_notional
        self.max_spread_positions = max_spread_positions
        self.buying_power_buffer = buying_power_buffer
        
        mode_str = "TESTNET" if paper else "PRODUCTION"
        logger.info(f"Initialized TradingPartner ({mode_str} mode)")


    def _calculate_spread_notional(self, num_legs: int) -> Optional[float]:
        """
        Calculate the actual notional per leg based on available buying power.
        
        Scales down from target notional if insufficient buying power.
        Returns None if buying power is below minimum threshold.
        
        Args:
            num_legs: Number of assets in the spread
            
        Returns:
            float: Actual notional per leg, or None if insufficient funds
            
        Example:
            >>> notional = self._calculate_spread_notional(2)
            >>> if notional is None:
            ...     print("Insufficient funds")
        """
        buying_power = self._get_available_buying_power()
        total_notional_needed = self.spread_notional * num_legs
        
        if buying_power >= total_notional_needed:
            return self.spread_notional
        
        usable_power = buying_power * self.buying_power_buffer
        actual_notional = usable_power / num_legs
        
        if actual_notional < self.min_spread_notional:
            logger.warning(
                f"⚠️ Insufficient buying power: ${buying_power:.2f} available, "
                f"need at least ${self.min_spread_notional * num_legs:.2f} "
                f"(min ${self.min_spread_notional:.2f}/leg × {num_legs} legs)"
            )
            return None
        
        logger.info(
            f"💰 Scaling spread notional: ${self.spread_notional:.2f} -> ${actual_notional:.2f} per leg "
            f"(buying power: ${buying_power:.2f})"
        )

        return actual_notional


    def _get_available_buying_power(self) -> float:
        """
        Get available buying power from the trading account.
        
        Refreshes account data and returns the current buying power.
        Uses the exchange's configured quote currency (USD for Kraken, USDT for Binance).
        
        Returns:
            float: Available buying power, or 0.0 on error
            
        Example:
            >>> bp = self._get_available_buying_power()
            >>> print(f"Available: ${bp:.2f}")
        """
        try:
            # Let the trader use its exchange-configured quote currency
            return self.crypto_trader.get_available_balance()

        except Exception as e:
            logger.warning(f"Could not get buying power: {e}")
            return 0.0


    def get_account(self):
        """Get and display account information"""
        self.crypto_trader.print_account_summary()


    def reconcile_positions(self) -> Dict[str, Dict]:
        """
        Compare local spread positions with actual exchange positions.
        
        This helps diagnose discrepancies between what your program thinks
        is open vs what the exchange actually shows. Essential for debugging
        order execution issues.
        
        Returns:
            Dict mapping symbol to reconciliation status with 'local_qty',
            'exchange_qty', 'match', and 'discrepancy' fields
        
        Example:
            >>> reconciliation = trader.reconcile_positions()
            >>> for symbol, status in reconciliation.items():
            ...     if not status['match']:
            ...         print(f"Mismatch: {symbol} local={status['local_qty']} exchange={status['exchange_qty']}")
        """
        # Aggregate local positions by symbol
        local_by_symbol: Dict[str, float] = {}
        local_spreads_by_symbol: Dict[str, List[str]] = {}
        
        for group_id, pos in self._spread_positions.items():
            for asset, qty in pos.quantities.items():
                local_by_symbol[asset] = local_by_symbol.get(asset, 0) + qty
                if asset not in local_spreads_by_symbol:
                    local_spreads_by_symbol[asset] = []
                local_spreads_by_symbol[asset].append(group_id)
        
        # Fetch exchange positions
        try:
            exchange_positions = {p.symbol: p for p in self.crypto_trader.get_all_positions()}
        except Exception as e:
            logger.error(f"Failed to fetch exchange positions for reconciliation: {e}")
            return {}
        
        # Build reconciliation report
        all_symbols = set(local_by_symbol.keys()) | set(exchange_positions.keys())
        reconciliation: Dict[str, Dict] = {}
        
        logger.info("\n🔍 POSITION RECONCILIATION:")
        logger.info("-" * 70)
        
        has_mismatch = False
        for symbol in sorted(all_symbols):
            local_qty = local_by_symbol.get(symbol, 0)
            
            exchange_pos = exchange_positions.get(symbol)
            if exchange_pos:
                exchange_qty = exchange_pos.contracts
                if exchange_pos.side == 'short':
                    exchange_qty = -exchange_qty
            else:
                exchange_qty = 0.0
            
            discrepancy = local_qty - exchange_qty
            match = abs(discrepancy) < 0.0001
            
            reconciliation[symbol] = {
                'local_qty': local_qty,
                'exchange_qty': exchange_qty,
                'match': match,
                'discrepancy': discrepancy,
                'spreads': local_spreads_by_symbol.get(symbol, []),
            }
            
            if match:
                status_icon = "✅"
            else:
                status_icon = "❌"
                has_mismatch = True
            
            local_side = "LONG" if local_qty > 0 else ("SHORT" if local_qty < 0 else "FLAT")
            exchange_side = "LONG" if exchange_qty > 0 else ("SHORT" if exchange_qty < 0 else "FLAT")
            
            logger.info(
                f"  {status_icon} {symbol}: "
                f"Local={local_side} {abs(local_qty):.6f} | "
                f"Exchange={exchange_side} {abs(exchange_qty):.6f} | "
                f"Diff={discrepancy:+.6f}"
            )
            
            if not match and local_spreads_by_symbol.get(symbol):
                logger.info(f"      └─ Involved in spreads: {local_spreads_by_symbol[symbol]}")
        
        # Check for exchange positions not tracked locally
        untracked_on_exchange = set(exchange_positions.keys()) - set(local_by_symbol.keys())
        if untracked_on_exchange:
            logger.warning(f"  ⚠️ Exchange has positions not tracked locally: {list(untracked_on_exchange)}")
        
        if has_mismatch:
            logger.warning("  ⚠️ POSITION MISMATCHES DETECTED - consider closing all and restarting")
        else:
            logger.info("  ✅ All positions reconciled successfully")
        
        logger.info("-" * 70)
        
        return reconciliation


    def monitor_data_only(self, symbols: List[str] = None, duration_minutes: int = 10):
        """
        Monitor real-time streaming data without executing trades.
        
        Useful for testing WebSocket connectivity and verifying price
        data is flowing correctly before enabling live trading.
        
        Args:
            symbols: Crypto symbols to monitor (default: CRYPTO_TICKERS)
            duration_minutes: How long to monitor (default: 10)
            
        Returns:
            bool: True if monitoring completed successfully
            
        Example:
            >>> trader = TradingPartner(paper=True)
            >>> trader.monitor_data_only(
            ...     symbols=['BTC/USDT:USDT', 'ETH/USDT:USDT'],
            ...     duration_minutes=5
            ... )
            # Prints prices every 5 seconds for 5 minutes
        """
        if symbols is None:
            symbols = CRYPTO_TICKERS

        logger.info(f"📡 Starting data monitoring for {duration_minutes} minutes")

        try:
            self.crypto_trader.start_real_time_streaming(symbols, ["tickers"])
            time.sleep(3)

            if not self.crypto_trader.is_streaming_active():
                logger.error("❌ Failed to start streaming")
                return False

            end_time = datetime.now() + timedelta(minutes=duration_minutes)

            while datetime.now() < end_time:
                print(f"\n🕐 {datetime.now().strftime('%H:%M:%S')}")

                for symbol in symbols:
                    price = self.crypto_trader.get_latest_price(symbol)
                    if price:
                        print(f"{symbol}: ${price:.2f}")
                    else:
                        print(f"{symbol}: No data")

                time.sleep(5)  # Update every 5 seconds

            logger.info("✅ Data monitoring completed")
            return True

        except KeyboardInterrupt:
            logger.info("🛑 Monitoring stopped by user")
        except Exception as e:
            logger.error(f"❌ Error in monitoring: {e}")
        finally:
            self.crypto_trader.stop_real_time_streaming()

        return False


    # ==================================================
    # UPDATING POSITIONS
    # ==================================================


    def _close_all_spread_positions(self, reason: str = "Session ended") -> None:
        """Close all open spread positions."""
        if not self._spread_positions:
            return
        
        logger.info(f"🔚 Closing all spread positions: {reason}")
        
        # ----- get current prices -----
        price_map = {}
        for symbol in self.symbols:
            p = self.crypto_trader.get_latest_price(symbol)
            if p is not None:
                price_map[symbol] = float(p)
        
        # ----- close each position -----
        for group_id in list(self._spread_positions.keys()):
            self._exit_spread(group_id, price_map, reason)


    def _enter_spread_long(self, signal: SpreadSignal, price_map: Dict[str, float]) -> bool:
        """
        Enter a long spread position (buy the spread).
        
        For a spread with weights {A: -beta, B: 1.0}:
        - BUY_SPREAD means we expect the spread to increase
        - We go long the asset with positive weight and short the asset with negative weight
        
        Uses Binance Futures for native long/short positions.
        
        Args:
            signal: SpreadSignal with entry information
            price_map: Current prices for all assets
            
        Returns:
            True if position was opened successfully
        """
        group_id = signal.group_id
        
        if group_id in self._spread_positions:
            logger.warning(f"Already have position in {group_id}")
            return False
        
        try:
            assets = signal.assets
            weights = signal.weights
            
            # Check buying power and calculate actual notional per leg
            actual_notional = self._calculate_spread_notional(len(assets))
            if actual_notional is None:
                logger.warning(f"⚠️ Skipping LONG spread {group_id}: insufficient buying power")
                return False
            
            # Calculate quantities and execute orders for each leg
            # Scale notional by absolute weight to maintain hedge ratio
            orders = []
            quantities: Dict[str, float] = {}
            entry_prices: Dict[str, float] = {}
            leg_notionals: Dict[str, float] = {}
            opened_assets: List[str] = []  # Track successfully opened legs for rollback
            
            for asset in assets:
                if asset not in price_map:
                    logger.error(f"Missing price for {asset}")
                    # Rollback: close any already-opened positions
                    for opened_asset in opened_assets:
                        logger.info(f"🔄 Rolling back: closing {opened_asset}")
                        self.crypto_trader.close_position(opened_asset)
                    return False

                price = price_map[asset]
                weight = weights.get(asset, 0)
                
                # positive weight = long, negative weight = short
                is_long = weight > 0
                
                # Scale notional by absolute weight to maintain hedge ratio
                leg_notional = actual_notional * abs(weight)
                
                # Check minimum leg notional before attempting order
                if leg_notional < MIN_LEG_NOTIONAL:
                    logger.warning(
                        f"⚠️ Skipping spread {group_id}: leg {asset} notional ${leg_notional:.2f} "
                        f"below minimum ${MIN_LEG_NOTIONAL:.2f} (weight: {weight:.4f})"
                    )
                    # Rollback: close any already-opened positions
                    for opened_asset in opened_assets:
                        logger.info(f"🔄 Rolling back: closing {opened_asset}")
                        self.crypto_trader.close_position(opened_asset)
                    return False
                
                leg_notionals[asset] = leg_notional

                # ----- execute futures orders -----
                # Store actual filled quantity and price from order response
                if is_long:
                    order = self.crypto_trader.open_long(
                        symbol=asset,
                        notional=leg_notional
                    )
                    if order is None:
                        logger.error(f"❌ LONG order failed for {asset}, aborting spread entry")
                        # Rollback: close any already-opened positions using their tracked quantities
                        for opened_asset in opened_assets:
                            rollback_qty = quantities.get(opened_asset, 0)
                            rollback_side = 'long' if rollback_qty > 0 else 'short'
                            logger.info(f"🔄 Rolling back: closing {opened_asset} ({rollback_side} {abs(rollback_qty):.6f})")
                            self.crypto_trader.reduce_position(opened_asset, abs(rollback_qty), rollback_side)
                        return False
                    # Use actual filled quantity and price from order
                    quantities[asset] = order.amount
                    # Use fill price if available, otherwise fall back to streaming price
                    entry_prices[asset] = order.price if order.price else price
                    logger.info(f"(LONG) 🟢 LONG {asset}: qty={order.amount:.6f} @ ${entry_prices[asset]:.2f} (${leg_notional:.2f}, weight: {weight:.3f})")
                else:
                    order = self.crypto_trader.open_short(
                        symbol=asset,
                        notional=leg_notional
                    )
                    if order is None:
                        logger.error(f"❌ SHORT order failed for {asset}, aborting spread entry")
                        # Rollback: close any already-opened positions using their tracked quantities
                        for opened_asset in opened_assets:
                            rollback_qty = quantities.get(opened_asset, 0)
                            rollback_side = 'long' if rollback_qty > 0 else 'short'
                            logger.info(f"🔄 Rolling back: closing {opened_asset} ({rollback_side} {abs(rollback_qty):.6f})")
                            self.crypto_trader.reduce_position(opened_asset, abs(rollback_qty), rollback_side)
                        return False
                    # Use actual filled quantity and price from order (negative qty for short)
                    quantities[asset] = -order.amount
                    # Use fill price if available, otherwise fall back to streaming price
                    entry_prices[asset] = order.price if order.price else price
                    logger.info(f"(LONG) 🔴 SHORT {asset}: qty={order.amount:.6f} @ ${entry_prices[asset]:.2f} (${leg_notional:.2f}, weight: {weight:.3f})")
                
                orders.append(order)
                opened_assets.append(asset)  # Track this leg as successfully opened
            
            # Calculate total notional from weighted leg notionals
            total_notional = sum(leg_notionals.values())
            
            position = SpreadPosition(
                group_id=group_id,
                assets=assets,
                side="LONG",
                weights=weights,
                quantities=quantities,
                entry_prices=entry_prices,
                current_prices=price_map.copy(),
                notional=round(total_notional, 2),
                entry_zscore=signal.zscore,
                current_zscore=signal.zscore,
                entry_time=datetime.now(),
            )
            
            self._spread_positions[group_id] = position
            
            # ----- update spread engine position state -----
            if self.spread_engine:
                self.spread_engine.set_position(group_id, "LONG")
            
            logger.info(f"✅ Opened LONG spread position: {group_id} (z={signal.zscore:.2f})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error entering long spread {group_id}: {e}")
            return False


    def _enter_spread_short(self, signal: SpreadSignal, price_map: Dict[str, float]) -> bool:
        """
        Enter a short spread position (sell the spread).
        
        For a spread with weights {A: -beta, B: 1.0}:
        - SELL_SPREAD means we expect the spread to decrease
        - We short the asset with positive weight and go long the asset with negative weight
        
        Uses Binance Futures for native long/short positions.
        
        Args:
            signal: SpreadSignal with entry information
            price_map: Current prices for all assets
            
        Returns:
            True if position was opened successfully
        """
        group_id = signal.group_id
        
        if group_id in self._spread_positions:
            logger.warning(f"Already have position in {group_id}")
            return False
        
        try:
            assets = signal.assets
            weights = signal.weights
            
            # Check buying power and calculate actual notional per leg
            actual_notional = self._calculate_spread_notional(len(assets))
            if actual_notional is None:
                logger.warning(f"⚠️ Skipping SHORT spread {group_id}: insufficient buying power")
                return False
            
            # Calculate quantities for each leg (opposite of long)
            # Scale notional by absolute weight to maintain hedge ratio
            orders = []
            quantities: Dict[str, float] = {}
            entry_prices: Dict[str, float] = {}
            leg_notionals: Dict[str, float] = {}
            opened_assets: List[str] = []  # Track successfully opened legs for rollback
            
            for asset in assets:
                if asset not in price_map:
                    logger.error(f"Missing price for {asset}")
                    # Rollback: close any already-opened positions
                    for opened_asset in opened_assets:
                        logger.info(f"🔄 Rolling back: closing {opened_asset}")
                        self.crypto_trader.close_position(opened_asset)
                    return False
                
                price = price_map[asset]
                weight = weights.get(asset, 0)
                
                # Opposite of long: positive weight = short, negative weight = long
                is_short = weight > 0
                
                # Scale notional by absolute weight to maintain hedge ratio
                leg_notional = actual_notional * abs(weight)
                
                # Check minimum leg notional before attempting order
                if leg_notional < MIN_LEG_NOTIONAL:
                    logger.warning(
                        f"⚠️ Skipping spread {group_id}: leg {asset} notional ${leg_notional:.2f} "
                        f"below minimum ${MIN_LEG_NOTIONAL:.2f} (weight: {weight:.4f})"
                    )
                    # Rollback: close any already-opened positions
                    for opened_asset in opened_assets:
                        logger.info(f"🔄 Rolling back: closing {opened_asset}")
                        self.crypto_trader.close_position(opened_asset)
                    return False
                
                leg_notionals[asset] = leg_notional
            
                # ----- execute futures orders -----
                # Store actual filled quantity and price from order response
                if is_short:
                    order = self.crypto_trader.open_short(
                        symbol=asset,
                        notional=leg_notional
                    )
                    if order is None:
                        logger.error(f"❌ SHORT order failed for {asset}, aborting spread entry")
                        # Rollback: close any already-opened positions using their tracked quantities
                        for opened_asset in opened_assets:
                            rollback_qty = quantities.get(opened_asset, 0)
                            rollback_side = 'long' if rollback_qty > 0 else 'short'
                            logger.info(f"🔄 Rolling back: closing {opened_asset} ({rollback_side} {abs(rollback_qty):.6f})")
                            self.crypto_trader.reduce_position(opened_asset, abs(rollback_qty), rollback_side)
                        return False
                    # Use actual filled quantity and price from order (negative qty for short)
                    quantities[asset] = -order.amount
                    # Use fill price if available, otherwise fall back to streaming price
                    entry_prices[asset] = order.price if order.price else price
                    logger.info(f"(SHORT) 🔴 SHORT {asset}: qty={order.amount:.6f} @ ${entry_prices[asset]:.2f} (${leg_notional:.2f}, weight: {weight:.3f})")
                else:
                    order = self.crypto_trader.open_long(
                        symbol=asset,
                        notional=leg_notional
                    )
                    if order is None:
                        logger.error(f"❌ LONG order failed for {asset}, aborting spread entry")
                        # Rollback: close any already-opened positions using their tracked quantities
                        for opened_asset in opened_assets:
                            rollback_qty = quantities.get(opened_asset, 0)
                            rollback_side = 'long' if rollback_qty > 0 else 'short'
                            logger.info(f"🔄 Rolling back: closing {opened_asset} ({rollback_side} {abs(rollback_qty):.6f})")
                            self.crypto_trader.reduce_position(opened_asset, abs(rollback_qty), rollback_side)
                        return False
                    # Use actual filled quantity and price from order
                    quantities[asset] = order.amount
                    # Use fill price if available, otherwise fall back to streaming price
                    entry_prices[asset] = order.price if order.price else price
                    logger.info(f"(SHORT) 🟢 LONG {asset}: qty={order.amount:.6f} @ ${entry_prices[asset]:.2f} (${leg_notional:.2f}, weight: {weight:.3f})")
                
                orders.append(order)
                opened_assets.append(asset)  # Track this leg as successfully opened
            
            # Calculate total notional from weighted leg notionals
            total_notional = sum(leg_notionals.values())
            
            position = SpreadPosition(
                group_id=group_id,
                assets=assets,
                side="SHORT",
                weights=weights,
                quantities=quantities,
                entry_prices=entry_prices,
                current_prices=price_map.copy(),
                notional=round(total_notional, 2),
                entry_zscore=signal.zscore,
                current_zscore=signal.zscore,
                entry_time=datetime.now(),
            )
            
            self._spread_positions[group_id] = position
            
            # ----- update spread engine position state -----
            if self.spread_engine:
                self.spread_engine.set_position(group_id, "SHORT")
            
            logger.info(f"✅ Opened SHORT spread position: {group_id} (z={signal.zscore:.2f})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error entering short spread {group_id}: {e}")
            return False


    def _exit_spread(self, group_id: str, price_map: Dict[str, float], reason: str = "") -> bool:
        """
        Exit an existing spread position.
        
        Closes only the specific quantities associated with this spread position,
        leaving any quantities from other spread positions intact. Uses the
        reduce_position method to close exact amounts rather than closing
        the entire position for each symbol.
        
        Args:
            group_id: The spread group ID
            price_map: Current prices for unwinding
            reason: Reason for exit (for logging)
            
        Returns:
            True if position was closed successfully
        """
        if group_id not in self._spread_positions:
            logger.warning(f"No position found for {group_id}")
            return False
        
        try:
            position = self._spread_positions[group_id]
            
            # Close each leg using the EXACT quantity from when we opened
            # This preserves quantities from other spread positions that share these symbols
            for asset in position.assets:
                qty = position.quantities.get(asset, 0)
                if qty == 0:
                    logger.warning(f"⚠️ Zero quantity for {asset} in {group_id}")
                    continue
                
                # Determine side based on sign: positive = long, negative = short
                side = 'long' if qty > 0 else 'short'
                
                order = self.crypto_trader.reduce_position(
                    symbol=asset,
                    qty=abs(qty),
                    side=side,
                )
                if order:
                    logger.info(f"🔚 Closed {asset}: {side} {abs(qty):.6f}")
                else:
                    logger.warning(f"⚠️ Failed to close {asset} for {group_id}")
            
            position.update_pnl(price_map)
            logger.info(f"🔚 Closed spread {group_id}: P&L=${position.unrealized_pnl:.2f} | Reason: {reason}")
            del self._spread_positions[group_id]
            
            # ----- update spread engine position state -----
            if self.spread_engine:
                self.spread_engine.set_position(group_id, None)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error closing spread {group_id}: {e}")
            return False


    # ==================================================
    # STREAMING MARKET DATA
    # ==================================================


    def _monitor_market_data(self, symbols: List[str], show_debug: bool = False) -> None:
        symbols_with_data = 0
        symbols_waiting = 0
        
        for symbol in symbols:
            price = self.crypto_trader.get_latest_price(symbol)
            ohlc = self.crypto_trader.get_real_time_ohlc(symbol)

            if price and ohlc:
                print(
                    f"{symbol:>20}: ${price:>10.2f} | "
                    f"OHLC: ${ohlc['open']:>8.2f}/${ohlc['high']:>8.2f}/"
                    f"${ohlc['low']:>8.2f}/${ohlc['close']:>8.2f}"
                )
                symbols_with_data += 1

            elif price:
                print(f"{symbol:>20}: ${price:>10.2f}")
                symbols_with_data += 1

            else:
                print(f"{symbol:>20}: ⏳ Waiting for data...")
                symbols_waiting += 1
        
        print("-" * 60)
        print(f"📈 {symbols_with_data}/{len(symbols)} symbols have data, {symbols_waiting} waiting")
        
        # ----- show detailed debug info if requested or if no data is coming through -----
        if show_debug or (symbols_waiting == len(symbols) and len(symbols) > 0):
            print(self.crypto_trader.get_streaming_debug_info())


    def _execute_trading_logic(
        self,
        symbols: List[str],
        staleness_threshold: float = 30.0,
    ) -> None:
        """Execute trading logic based on real-time data and spread signals"""
        price_map = {}
        staleness_map = {}
        
        # Get connection health info for staleness tracking
        health = self.crypto_trader.get_connection_health(staleness_threshold)
        symbol_health = health.get("symbols", {})
        
        for sym in symbols:
            p = self.crypto_trader.get_latest_price(sym)

            if p is not None:
                price_map[sym] = float(p)
                staleness = symbol_health.get(sym, {}).get("last_data_age_secs")
                staleness_map[sym] = 0.0 if staleness is None else staleness
        
        # Check for open positions with stale data and warn + logging
        stale_symbols = [s for s, age in staleness_map.items() if age > staleness_threshold]
        if stale_symbols:
            stale_info = {s: f"{staleness_map[s]:.0f}s" for s in stale_symbols}
            logger.warning(f"⚠️ Stale data detected: {stale_info}")
        
        for group_id, pos in self._spread_positions.items():
            stale_assets = [a for a in pos.assets if staleness_map.get(a, 0) > staleness_threshold]
            if stale_assets:
                ages = {a: f"{staleness_map.get(a, 0):.0f}s" for a in stale_assets}
                logger.warning(f"⚠️ Position {group_id} has stale assets: {ages}")
        
        # ----- use spread engine if available -----
        if self.spread_engine is not None:
            self.spread_engine.update_prices(price_map, staleness_map)
            
            for signal in self.spread_engine.get_signals():
                # Handle EMERGENCY_EXIT specially - force close position
                if signal.signal_type == SignalType.EMERGENCY_EXIT:
                    logger.warning(f"🚨 EMERGENCY EXIT: {signal.group_id} - Data critically stale!")
                    if signal.group_id in self._spread_positions:
                        self._exit_spread(signal.group_id, price_map, "EMERGENCY: Data staleness")
                    continue
                
                logger.info(f"📊 Spread {signal.group_id}: z={signal.zscore:.2f} -> {signal.signal_type.value} (conf={signal.confidence:.2f})")
                
                # ===== EXECUTE TRADES =====
                if signal.signal_type == SignalType.BUY_SPREAD:
                    if len(self._spread_positions) < self.max_spread_positions:
                        self._enter_spread_long(signal, price_map)

                    else:
                        logger.info(f"⚠️ Max spread positions ({self.max_spread_positions}) reached")
                        
                elif signal.signal_type == SignalType.SELL_SPREAD:
                    if len(self._spread_positions) < self.max_spread_positions:
                        self._enter_spread_short(signal, price_map)
                        
                    else:
                        logger.info(f"⚠️ Max spread positions ({self.max_spread_positions}) reached")
                        
                elif signal.signal_type == SignalType.EXIT:
                    if signal.group_id in self._spread_positions:
                        self._exit_spread(signal.group_id, price_map, "Signal EXIT")
            
            # ----- log z-score summary for monitored groups -----
            zscores = self.spread_engine.get_all_zscores()
            zscore_strs = [f"{gid}: {z:.2f}" for gid, z in zscores.items() if not (z != z)]  # skip NaN
            if zscore_strs:
                logger.info(f"📈 Z-scores: {', '.join(zscore_strs)}")

        else:
            # ----- fallback: simple per-symbol momentum signals (no execution) -----
            for symbol in symbols:
                price_history = self.crypto_trader.get_price_history(symbol, limit=20)

                if len(price_history) >= MOMENTUM_MIN_DATA_POINTS:
                    prices = [p['price'] for p in price_history]
                    price_change = (prices[-1] - prices[0]) / prices[0]

                    if price_change > MOMENTUM_BUY_THRESHOLD:
                        logger.info(f"📈 Momentum BUY signal: {symbol} up {price_change:.2%}")
                    elif price_change < MOMENTUM_SELL_THRESHOLD:
                        logger.info(f"📉 Momentum SELL signal: {symbol} down {price_change:.2%}")


    def _update_spread_positions(self) -> None:
        """Update current prices and P&L for all spread positions."""
        if not self._spread_positions:
            return
        
        # ----- get current prices -----
        price_map = {}
        for symbol in self.symbols:
            p = self.crypto_trader.get_latest_price(symbol)
            if p is not None:
                price_map[symbol] = float(p)
        
        # ----- get current z-scores -----
        zscores = {}
        if self.spread_engine:
            zscores = self.spread_engine.get_all_zscores()
        
        # ----- update each position -----
        for group_id, position in self._spread_positions.items():
            current_zscore = zscores.get(group_id)
            position.update_pnl(price_map, current_zscore)


    # ==================================================
    # LIVE TRADING LOOP
    # ==================================================


    def start_streaming_bot(
        self, 
        symbols: List[str] = None,
        lookback_bars: int = 500,
        duration_minutes: int = 30,
        max_stream_symbols: int = 10,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        cycle_interval: int = 30,
    ):
        """
        Start the live trading bot for a fixed duration.
        
        Connects to Binance Futures WebSocket, streams prices for cointegrated pairs,
        and executes spread trades based on z-score signals. Runs for the
        specified duration, then closes positions and stops.
        
        Args:
            symbols: Fallback symbols if no benchmarks exist
            duration_minutes: How long to run before stopping (default: 30)
            max_stream_symbols: Max WebSocket subscriptions (default: 10)
            entry_zscore: Z-score threshold to enter (default: 2.0)
            exit_zscore: Z-score threshold to exit (default: 0.5)
            
        Returns:
            None (runs until duration expires)
            
        Example:
            >>> trader = TradingPartner(paper=True)
            >>> trader.start_streaming_bot(
            ...     duration_minutes=60,
            ...     entry_zscore=2.5
            ... )
            # Runs for 60 minutes, then stops
        """
        if symbols is None:
            symbols = CRYPTO_TICKERS

        logger.info(f"🚀 Starting streaming trading bot for {duration_minutes} minutes")
        logger.info(f"Max streaming symbols: {max_stream_symbols}")
        logger.info(f"Z-score thresholds: entry={entry_zscore}, exit={exit_zscore}")

        try:
            # ===== DYNAMIC SYMBOL DISCOVERY =====
            # Get actually available symbols from the exchange (handles testnet vs production)
            # Note: CCXTFuturesTrader.__init__ already sets the runtime symbol map
            available_perpetuals = self.crypto_trader.get_tradeable_symbols()
            logger.info(f"📊 Found {len(available_perpetuals)} perpetual contracts on {EXCHANGE_CONFIG['name']}")
            
            if len(available_perpetuals) <= 20:
                # On testnet with limited symbols, log them all
                logger.info(f"📋 Available symbols: {available_perpetuals}")
            
            all_symbols = []
            
            # Initialize spread engine if benchmarks available
            if self.benchmarks is not None:
                # Pass available perpetuals to filter benchmark groups
                # Symbol conversion now uses runtime map set by CCXTFuturesTrader
                self.spread_engine = SpreadSignalEngine(
                    benchmarks=self.benchmarks,
                    entry_zscore=entry_zscore,
                    exit_zscore=exit_zscore,
                    max_groups=max_stream_symbols,
                    available_symbols=available_perpetuals if available_perpetuals else None,
                )
                
                # Get ONLY symbols from cointegration pairs (not full universe)
                # These are now correctly converted via runtime symbol map
                all_symbols = self.spread_engine.get_required_symbols()
                
                if self.spread_engine.groups:
                    logger.info(f"📊 Spread engine initialized with {len(self.spread_engine.groups)} groups")
                    logger.info(f"📊 Entry z={entry_zscore}, Exit z={exit_zscore}")
                    logger.info(f"📊 Required symbols from cointegration pairs: {all_symbols}")
                else:
                    logger.warning(f"⚠️ No cointegration groups available after filtering!")
                    logger.warning(f"   Benchmarks may have been computed on a different exchange/mode.")
                    logger.warning(f"   Consider recomputing benchmarks with: python main.py --mode benchmark")
            
            # Fallback if no spread engine or no symbols
            if not all_symbols:
                # Use dynamically fetched symbols if available, otherwise fall back to provided symbols
                if available_perpetuals:
                    all_symbols = available_perpetuals[:max_stream_symbols]
                    logger.warning(f"⚠️ No cointegration pairs - using available perpetuals: {all_symbols}")
                else:
                    all_symbols = symbols[:max_stream_symbols]
                    logger.warning(f"⚠️ No cointegration pairs - using fallback symbols: {all_symbols}")
            
            # Apply safety limit
            if len(all_symbols) > max_stream_symbols:
                logger.warning(f"⚠️ Limiting streams from {len(all_symbols)} to {max_stream_symbols} symbols")
                all_symbols = all_symbols[:max_stream_symbols]
            
            self.symbols = all_symbols
            logger.info(f"📡 Will stream {len(all_symbols)} symbols: {all_symbols}")
            
            # Initialize rolling buffer for streaming symbols
            self.rolling_buffer = RollingCointegrationBuffer(
                symbols=all_symbols,
                lookback_bars=lookback_bars,
            )
            
            # Start real-time streaming via CCXT/Binance
            self.crypto_trader.start_real_time_streaming(
                all_symbols, ["tickers"]
            )
            
            # Wait for streaming to initialize with data
            logger.info("⏳ Waiting for streaming data...")
            if not self.crypto_trader.wait_for_data(all_symbols, timeout_secs=30):
                logger.warning("⚠️ Timeout waiting for data, continuing anyway...")
            
            if not self.crypto_trader.is_streaming_active():
                logger.error("❌ Failed to start streaming")
                return False

            logger.info("✅ Real-time streaming started successfully")

            start_time = datetime.now()
            end_time = start_time + timedelta(minutes=duration_minutes)
            cycle_count = 0

            while datetime.now() < end_time:
                cycle_count += 1
                logger.info(f"\n🔄 Trading cycle #{cycle_count}")

                self._monitor_market_data(all_symbols, show_debug=(cycle_count <= 3))
                self._execute_trading_logic(all_symbols)
                self._update_spread_positions()
                
                if self._spread_positions:
                    self._log_spread_positions()

                time.sleep(cycle_interval)

            logger.info("✅ Trading session completed successfully")
            return True

        except KeyboardInterrupt:
            logger.info("🛑 Trading session stopped by user")

        except Exception as e:
            logger.error(f"❌ Error in trading session: {e}")
            traceback.print_exc()

        finally:
            # Close all spread positions before stopping
            self._close_all_spread_positions("Session ended")
            self.crypto_trader.stop_real_time_streaming()

        return False


    # ==================================================
    # LOGGING
    # ==================================================


    def _log_spread_positions(self) -> None:
        """Log summary of open spread positions with accurate per-spread P&L."""
        if not self._spread_positions:
            return
        
        logger.info("\n📋 OPEN SPREAD POSITIONS:")
        logger.info("-" * 60)
        
        # Fetch actual positions from exchange
        try:
            exchange_positions = {p.symbol: p for p in self.crypto_trader.get_all_positions()}
        except Exception as e:
            logger.warning(f"Could not fetch exchange positions: {e}")
            exchange_positions = {}
        
        # Calculate total local quantity per symbol to properly attribute exchange P&L
        # This avoids double-counting when multiple spreads share a symbol
        total_local_qty_by_symbol: Dict[str, float] = {}
        for pos in self._spread_positions.values():
            for asset, qty in pos.quantities.items():
                total_local_qty_by_symbol[asset] = total_local_qty_by_symbol.get(asset, 0) + qty
        
        total_pnl = 0.0
        for group_id, pos in self._spread_positions.items():
            z_str = f"{pos.current_zscore:.2f}" if pos.current_zscore == pos.current_zscore else "N/A"
            
            # Calculate P&L for this spread using local calculation (most accurate per-spread)
            # This avoids the double-counting issue with exchange P&L when symbols are shared
            spread_pnl = 0.0
            for asset in pos.assets:
                qty = pos.quantities.get(asset, 0)
                entry = pos.entry_prices.get(asset, 0)
                current = pos.current_prices.get(asset, entry)
                leg_pnl = qty * (current - entry)
                spread_pnl += leg_pnl
            
            total_pnl += spread_pnl
            
            logger.info(
                f"  {group_id}: {pos.side} | Entry z={pos.entry_zscore:.2f} | "
                f"Current z={z_str} | P&L=${spread_pnl:+.2f}"
            )
            
            # Log individual legs with details
            for asset in pos.assets:
                qty = pos.quantities.get(asset, 0)
                side = "LONG" if qty > 0 else "SHORT"
                entry = pos.entry_prices.get(asset, 0)
                current = pos.current_prices.get(asset, entry)
                leg_pnl = qty * (current - entry)
                
                # Show exchange position reconciliation info
                if asset in exchange_positions:
                    ep = exchange_positions[asset]
                    local_total = total_local_qty_by_symbol.get(asset, 0)
                    exchange_qty = ep.contracts if ep.side == 'long' else -ep.contracts
                    match_indicator = "✓" if abs(local_total - exchange_qty) < 0.0001 else "⚠️"
                    logger.info(
                        f"    └─ {asset}: {side} {abs(qty):.6f} @ ${entry:.2f} → ${current:.2f} "
                        f"(P&L: ${leg_pnl:+.2f}) {match_indicator}"
                    )
                else:
                    logger.info(
                        f"    └─ {asset}: {side} {abs(qty):.6f} @ ${entry:.2f} → ${current:.2f} "
                        f"(P&L: ${leg_pnl:+.2f}) ⚠️ NOT ON EXCHANGE"
                    )
        
        logger.info(f"  TOTAL SPREAD P&L: ${total_pnl:+.2f}")
        
        # Show exchange total unrealized P&L for comparison
        if exchange_positions:
            exchange_total_pnl = sum(p.unrealized_pnl for p in exchange_positions.values())
            logger.info(f"  EXCHANGE TOTAL P&L: ${exchange_total_pnl:+.2f}")
            if abs(total_pnl - exchange_total_pnl) > 0.01:
                logger.info(f"  ℹ️ Difference may be due to mark price vs last price, fees, or funding")
        
        logger.info("-" * 60)
