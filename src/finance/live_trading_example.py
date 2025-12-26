#!/usr/bin/env python3
"""
Live Trading Example with Real-Time Data Streaming

This example demonstrates how to use the enhanced CryptoTrader class for:
1. Real-time data streaming from multiple crypto symbols
2. Live trading based on streaming market data
3. Risk management and position monitoring
4. Performance tracking and logging

Usage:
    python live_trading_example.py

Requirements:
    - Valid Alpaca API credentials in environment variables
    - alpaca-py library installed
    - Paper trading account (recommended for testing)
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging
import time
from typing import Any, Dict, List, Optional

from .crypto import CryptoTrader
from .benchmarks import load_benchmarks, is_stale, list_groups, entry_exit_signal, get_weights
from .spread_engine import SpreadSignalEngine, SpreadSignal, SignalType
from .rolling_buffer import RollingCointegrationBuffer, create_buffer_from_trader

from alpaca.trading.enums import OrderSide

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("live_trading.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


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
    
    def update_pnl(self, current_prices: Dict[str, float], current_zscore: float = None):
        """Update current prices and calculate unrealized P&L"""
        self.current_prices = current_prices
        if current_zscore is not None:
            self.current_zscore = current_zscore
        
        # Calculate P&L for each leg
        pnl = 0.0
        for asset in self.assets:
            qty = self.quantities.get(asset, 0)
            entry = self.entry_prices.get(asset, 0)
            current = current_prices.get(asset, entry)
            pnl += qty * (current - entry)
        
        self.unrealized_pnl = pnl


class LiveTradingBot:
    """
    Advanced live trading bot using real-time streaming data

    Features:
    - Multi-symbol real-time data streaming
    - Technical analysis based on streaming data
    - Risk management and position sizing
    - Performance tracking and logging
    - Custom trading strategies
    """

    def __init__(self, paper: bool = True, symbols: List[str] = None):
        """
        Initialize the live trading bot

        Args:
            paper: Whether to use paper trading (recommended for testing)
            symbols: List of crypto symbols to trade (default: major cryptos)
        """
        self.trader = CryptoTrader(paper=paper)

        # Trading symbols
        self.symbols = symbols or ["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD"]

        # Trading parameters
        self.max_position_size = 1000.0  # Maximum USD per position
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.04  # 4% take profit
        self.max_positions = 3  # Maximum concurrent positions

        # State tracking
        self.positions: Dict[str, Position] = {}
        self.signals: List[TradingSignal] = []
        self.running = False
        self.start_time = None

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0

        logger.info(f"Initialized LiveTradingBot with symbols: {self.symbols}")
        logger.info(f"Paper trading: {paper}")

        # Load benchmarks (if available)
        self.benchmarks = None
        try:
            self.benchmarks = load_benchmarks()
            if is_stale(self.benchmarks):
                logger.warning("Benchmarks snapshot is stale (>7 days). Consider recomputing.")
            else:
                logger.info("Loaded benchmarks snapshot for live signals.")
        except Exception as e:
            logger.warning(f"No benchmarks available: {e}")

    def start_streaming(self):
        """Start real-time data streaming"""
        logger.info("Starting real-time data streaming...")

        try:
            # NOTE: Using only "quotes" to minimize WebSocket load and avoid symbol limit errors
            self.trader.start_real_time_streaming(
                symbols=self.symbols, data_types=["quotes"]
            )

            # Wait for streaming to initialize
            time.sleep(3)

            if self.trader.is_streaming_active():
                logger.info("✅ Real-time streaming started successfully")
                return True
            else:
                logger.error("❌ Failed to start real-time streaming")
                return False

        except Exception as e:
            logger.error(f"Error starting streaming: {e}")
            return False

    def get_market_summary(self) -> str:
        """Get a summary of current market conditions"""
        summary = self.trader.get_market_data_summary()

        report = "\n📊 MARKET SUMMARY\n" + "=" * 50 + "\n"

        for symbol in self.symbols:
            if symbol in summary:
                data = summary[symbol]
                latest_price = data["latest_price"]
                buffer_size = data["price_buffer_size"]

                report += f"{symbol}: ${latest_price:.2f} | "
                report += f"Data points: {buffer_size} | "
                report += f"Has bars: {'✅' if data['has_bar_data'] else '❌'} | "
                report += f"Has quotes: {'✅' if data['has_quote_data'] else '❌'}\n"
            else:
                report += f"{symbol}: ❌ No data\n"

        return report

    def analyze_price_momentum(self, symbol: str, lookback: int = 20) -> Dict:
        """
        Analyze price momentum using streaming data

        Args:
            symbol: Crypto symbol to analyze
            lookback: Number of recent data points to consider

        Returns:
            Dictionary with momentum analysis results
        """
        price_history = self.trader.get_price_history(symbol, limit=lookback)

        if len(price_history) < 10:
            return {"signal": "HOLD", "confidence": 0.0, "reason": "Insufficient data"}

        prices = [p["price"] for p in price_history]
        volumes = [p["volume"] for p in price_history]

        # Calculate momentum indicators
        price_change = (prices[-1] - prices[0]) / prices[0]
        volume_avg = sum(volumes) / len(volumes)
        recent_volume = sum(volumes[-5:]) / 5 if len(volumes) >= 5 else volume_avg

        # Simple momentum strategy
        signal = "HOLD"
        confidence = 0.0
        reason = ""

        if price_change > 0.02 and recent_volume > volume_avg * 1.2:
            signal = "BUY"
            confidence = min(0.8, abs(price_change) * 10)
            reason = f"Strong upward momentum ({price_change:.2%}) with high volume"
        elif price_change < -0.02 and recent_volume > volume_avg * 1.2:
            signal = "SELL"
            confidence = min(0.8, abs(price_change) * 10)
            reason = f"Strong downward momentum ({price_change:.2%}) with high volume"
        elif abs(price_change) < 0.01:
            signal = "HOLD"
            confidence = 0.3
            reason = "Low volatility, waiting for clear signal"

        return {
            "signal": signal,
            "confidence": confidence,
            "reason": reason,
            "price_change": price_change,
            "volume_ratio": recent_volume / volume_avg,
            "current_price": prices[-1],
        }

    def generate_trading_signal(self, symbol: str) -> Optional[TradingSignal]:
        """
        Generate trading signal based on real-time data analysis

        Args:
            symbol: Crypto symbol to analyze

        Returns:
            TradingSignal object or None if no signal
        """
        analysis = self.analyze_price_momentum(symbol)

        if analysis["signal"] != "HOLD" and analysis["confidence"] > 0.6:
            return TradingSignal(
                symbol=symbol,
                action=analysis["signal"],
                confidence=analysis["confidence"],
                price=analysis["current_price"],
                reason=analysis["reason"],
                timestamp=datetime.now(),
            )

        return None

    def execute_trade(self, signal: TradingSignal) -> bool:
        """
        Execute a trade based on the signal

        Args:
            signal: TradingSignal object

        Returns:
            True if trade was executed successfully
        """
        try:
            # Check if we already have a position in this symbol
            if signal.symbol in self.positions:
                logger.info(f"Already have position in {signal.symbol}, skipping trade")
                return False

            # Check position limits
            if len(self.positions) >= self.max_positions:
                logger.info(
                    f"Maximum positions ({self.max_positions}) reached, skipping trade"
                )
                return False

            # Calculate position size
            account_value = float(self.trader.acct.equity)
            position_size = min(
                self.max_position_size, account_value * 0.1
            )  # Max 10% of account

            if signal.action == "BUY":
                # Execute buy order
                order = self.trader.buy_market_order(
                    symbol=signal.symbol, notional=position_size
                )

                # Track position
                self.positions[signal.symbol] = Position(
                    symbol=signal.symbol,
                    side="LONG",
                    quantity=position_size / signal.price,
                    entry_price=signal.price,
                    current_price=signal.price,
                    unrealized_pnl=0.0,
                    entry_time=datetime.now(),
                )

                logger.info(
                    f"✅ BUY order executed: {signal.symbol} at ${signal.price:.2f}"
                )
                logger.info(f"   Position size: ${position_size:.2f}")
                logger.info(f"   Reason: {signal.reason}")

            elif signal.action == "SELL":
                # For crypto, we can't short directly, so we'll skip sell signals
                # In a real implementation, you might want to close long positions
                logger.info(
                    f"SELL signal received for {signal.symbol}, but shorting not implemented"
                )
                return False

            self.total_trades += 1
            self.signals.append(signal)
            return True

        except Exception as e:
            logger.error(f"Error executing trade for {signal.symbol}: {e}")
            return False

    def update_positions(self):
        """Update current positions with latest prices"""
        for symbol, position in self.positions.items():
            latest_price = self.trader.get_latest_price(symbol)

            if latest_price:
                position.current_price = latest_price

                if position.side == "LONG":
                    position.unrealized_pnl = (
                        latest_price - position.entry_price
                    ) * position.quantity
                else:
                    position.unrealized_pnl = (
                        position.entry_price - latest_price
                    ) * position.quantity

    def check_exit_conditions(self):
        """Check if any positions should be closed based on stop loss/take profit"""
        positions_to_close = []

        for symbol, position in self.positions.items():
            # Calculate returns
            if position.side == "LONG":
                return_pct = (
                    position.current_price - position.entry_price
                ) / position.entry_price
            else:
                return_pct = (
                    position.entry_price - position.current_price
                ) / position.entry_price

            should_close = False
            close_reason = ""

            # Check stop loss
            if return_pct <= -self.stop_loss_pct:
                should_close = True
                close_reason = f"Stop loss triggered ({return_pct:.2%})"

            # Check take profit
            elif return_pct >= self.take_profit_pct:
                should_close = True
                close_reason = f"Take profit triggered ({return_pct:.2%})"

            if should_close:
                positions_to_close.append((symbol, close_reason))

        # Close positions
        for symbol, reason in positions_to_close:
            self.close_position(symbol, reason)

    def close_position(self, symbol: str, reason: str):
        """
        Close a position

        Args:
            symbol: Symbol to close position for
            reason: Reason for closing the position
        """
        try:
            position = self.positions[symbol]

            # Execute sell order
            order = self.trader.sell_market_order(symbol=symbol, qty=position.quantity)

            # Update performance tracking
            self.total_pnl += position.unrealized_pnl
            if position.unrealized_pnl > 0:
                self.winning_trades += 1

            logger.info(f"🔚 Position closed: {symbol}")
            logger.info(f"   Entry: ${position.entry_price:.2f}")
            logger.info(f"   Exit: ${position.current_price:.2f}")
            logger.info(f"   P&L: ${position.unrealized_pnl:.2f}")
            logger.info(f"   Reason: {reason}")

            # Remove from positions
            del self.positions[symbol]

        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")

    def get_performance_summary(self) -> str:
        """Get performance summary"""
        win_rate = (
            (self.winning_trades / self.total_trades * 100)
            if self.total_trades > 0
            else 0
        )

        summary = f"\n📈 PERFORMANCE SUMMARY\n" + "=" * 50 + "\n"
        summary += f"Total trades: {self.total_trades}\n"
        summary += f"Winning trades: {self.winning_trades}\n"
        summary += f"Win rate: {win_rate:.1f}%\n"
        summary += f"Total P&L: ${self.total_pnl:.2f}\n"
        summary += f"Open positions: {len(self.positions)}\n"

        if self.positions:
            summary += "\nOpen Positions:\n"
            for symbol, pos in self.positions.items():
                summary += f"  {symbol}: {pos.side} ${pos.unrealized_pnl:+.2f}\n"

        return summary

    def run_trading_session(self, duration_minutes: int = 30):
        """
        Run a complete trading session

        Args:
            duration_minutes: Duration of trading session in minutes
        """
        logger.info(f"🤖 Starting live trading session for {duration_minutes} minutes")
        self.running = True
        self.start_time = datetime.now()

        try:
            # Start streaming
            if not self.start_streaming():
                logger.error("Failed to start streaming, exiting")
                return

            # Print account info
            self.trader.print_account_summary()

            end_time = datetime.now() + timedelta(minutes=duration_minutes)
            cycle_count = 0

            while datetime.now() < end_time and self.running:
                cycle_count += 1
                logger.info(f"\n🔄 Trading cycle #{cycle_count}")
                logger.info(f"Time remaining: {end_time - datetime.now()}")

                # Get market summary
                market_summary = self.get_market_summary()
                logger.info(market_summary)

                # Update existing positions
                self.update_positions()

                # Check exit conditions
                self.check_exit_conditions()

                # Generate and execute new signals
                for symbol in self.symbols:
                    signal = self.generate_trading_signal(symbol)
                    if signal:
                        logger.info(
                            f"📊 Signal generated: {signal.action} {signal.symbol} "
                            f"(confidence: {signal.confidence:.2f})"
                        )
                        self.execute_trade(signal)

                # Print performance summary
                if cycle_count % 5 == 0:  # Every 5 cycles
                    logger.info(self.get_performance_summary())

                # Wait before next cycle
                time.sleep(30)  # 30 second cycles

        except KeyboardInterrupt:
            logger.info("\n🛑 Trading session stopped by user")
        except Exception as e:
            logger.error(f"❌ Error in trading session: {e}")
        finally:
            self.running = False

            # Close all positions
            if self.positions:
                logger.info("Closing all remaining positions...")
                for symbol in list(self.positions.keys()):
                    self.close_position(symbol, "Session ended")

            # Final performance summary
            logger.info("\n" + "=" * 60)
            logger.info("🏁 FINAL SESSION RESULTS")
            logger.info("=" * 60)
            logger.info(self.get_performance_summary())

            # Stop streaming
            self.trader.stop_real_time_streaming()
            logger.info("🔚 Trading session completed")


class TradingPartner:
    """
    Main trading interface that combines crypto trading and market data

    This class provides a unified interface for:
    - Real-time crypto trading with streaming data
    - Market data analysis and visualization
    - Portfolio management and risk control
    - Spread trading based on cointegration analysis
    - Simplified trading operations for main.py integration
    """

    def __init__(self, paper: bool = True):
        """
        Initialize the trading partner

        Args:
            paper: Whether to use paper trading (recommended for testing)
        """
        self.crypto_trader = CryptoTrader(paper=paper)
        self.paper = paper
        
        # Symbols to trade (will be set by start_streaming_bot)
        self.symbols: List[str] = []
        
        # Load benchmarks if available
        self.benchmarks: Optional[Dict] = None
        try:
            self.benchmarks = load_benchmarks()
            if is_stale(self.benchmarks):
                logger.warning("⚠️ Benchmark data is stale (>7 days). Consider recomputing.")
        except FileNotFoundError:
            logger.warning("⚠️ No benchmarks file found. Spread trading disabled.")
        
        # Spread trading components
        self.spread_engine: Optional[SpreadSignalEngine] = None
        self.rolling_buffer: Optional[RollingCointegrationBuffer] = None
        
        # Spread position tracking
        self._spread_positions: Dict[str, SpreadPosition] = {}
        
        # Trading parameters for spread trades
        self.spread_notional = 500.0  # USD per spread leg
        self.max_spread_positions = 3
        
        logger.info(f"Initialized TradingPartner (Paper trading: {paper})")

    def get_account(self):
        """Get and display account information"""
        self.crypto_trader.print_account_summary()

    def start_streaming_bot(
        self, 
        symbols: List[str] = None, 
        duration_minutes: int = 30,
        max_stream_symbols: int = 10,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
    ):
        """
        Start the real-time streaming trading bot with spread trading support

        Args:
            symbols: List of crypto symbols (fallback if no benchmarks)
            duration_minutes: Duration of trading session
            max_stream_symbols: Maximum symbols to stream (Alpaca API limit)
            entry_zscore: Z-score threshold for entering positions (default: 2.0)
            exit_zscore: Z-score threshold for exiting positions (default: 0.5)
        """
        if symbols is None:
            symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]

        logger.info(f"🚀 Starting streaming trading bot for {duration_minutes} minutes")
        logger.info(f"Max streaming symbols: {max_stream_symbols}")
        logger.info(f"Z-score thresholds: entry={entry_zscore}, exit={exit_zscore}")

        try:
            # ===== SMART SYMBOL SELECTION =====
            # Only stream symbols needed for cointegration pairs to avoid API limits
            
            all_symbols = []
            
            # Initialize spread engine if benchmarks available
            if self.benchmarks is not None:
                self.spread_engine = SpreadSignalEngine(
                    benchmarks=self.benchmarks,
                    entry_zscore=entry_zscore,
                    exit_zscore=exit_zscore,
                    max_groups=10,
                )
                
                # Get ONLY symbols from cointegration pairs (not full universe)
                all_symbols = self.spread_engine.get_required_symbols()
                logger.info(f"📊 Spread engine initialized with {len(self.spread_engine.groups)} groups")
                logger.info(f"📊 Entry z={entry_zscore}, Exit z={exit_zscore}")
                logger.info(f"📊 Required symbols from cointegration pairs: {all_symbols}")
            
            # Fallback if no spread engine or no symbols
            if not all_symbols:
                all_symbols = symbols[:max_stream_symbols]
                logger.warning(f"⚠️ No cointegration pairs - using fallback symbols: {all_symbols}")
            
            # Apply safety limit to prevent exceeding Alpaca's WebSocket limits
            if len(all_symbols) > max_stream_symbols:
                logger.warning(f"⚠️ Limiting streams from {len(all_symbols)} to {max_stream_symbols} symbols")
                all_symbols = all_symbols[:max_stream_symbols]
            
            self.symbols = all_symbols
            logger.info(f"📡 Will stream {len(all_symbols)} symbols: {all_symbols}")
            
            # Initialize rolling buffer for streaming symbols
            self.rolling_buffer = RollingCointegrationBuffer(
                symbols=all_symbols,
                lookback_bars=500,
            )
            
            # Start real-time streaming
            # NOTE: Using only "quotes" to minimize WebSocket load and avoid symbol limit errors.
            # "quotes" provides bid/ask prices which is sufficient for spread trading.
            self.crypto_trader.start_real_time_streaming(
                all_symbols, ["quotes"]
            )
            
            # Register rolling buffer to receive bar data (for future use when bars are enabled)
            # NOTE: Bar streaming is currently disabled. Uncomment when bars are added back.
            # self.crypto_trader.register_bar_callback(self.rolling_buffer.on_bar)
            
            # Wait for streaming to initialize with data
            logger.info("⏳ Waiting for streaming data...")
            if not self.crypto_trader.wait_for_data(all_symbols, timeout_secs=30):
                logger.warning("⚠️ Timeout waiting for data, continuing anyway...")
            
            if not self.crypto_trader.is_streaming_active():
                logger.error("❌ Failed to start streaming")
                return False

            logger.info("✅ Real-time streaming started successfully")

            # Trading loop
            start_time = datetime.now()
            end_time = start_time + timedelta(minutes=duration_minutes)
            cycle_count = 0

            while datetime.now() < end_time:
                cycle_count += 1
                logger.info(f"\n🔄 Trading cycle #{cycle_count}")

                # Monitor market data (show debug for first 3 cycles)
                self._monitor_market_data(all_symbols, show_debug=(cycle_count <= 3))

                # Execute trading logic
                self._execute_trading_logic(all_symbols)
                
                # Update spread positions
                self._update_spread_positions()
                
                # Log spread position summary
                if self._spread_positions:
                    self._log_spread_positions()

                # Wait before next cycle
                time.sleep(30)  # 30 second cycles

            logger.info("✅ Trading session completed successfully")
            return True

        except KeyboardInterrupt:
            logger.info("🛑 Trading session stopped by user")
        except Exception as e:
            logger.error(f"❌ Error in trading session: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Close all spread positions before stopping
            self._close_all_spread_positions("Session ended")
            self.crypto_trader.stop_real_time_streaming()

        return False

    def _monitor_market_data(self, symbols: List[str], show_debug: bool = False):
        """Monitor real-time market data"""
        print("\n📊 REAL-TIME MARKET DATA")
        print("=" * 60)
        
        # Show connection status
        health = self.crypto_trader.get_connection_health()
        stream_status = "✅ ACTIVE" if health['streaming_active'] else "❌ INACTIVE"
        conn_status = "✅ HEALTHY" if health['connection_healthy'] else "⚠️ WAITING"
        print(f"Stream: {stream_status} | Connection: {conn_status} | Reconnects: {health['reconnect_attempts']}")
        print("-" * 60)

        # Count how many symbols have data
        symbols_with_data = 0
        symbols_waiting = 0
        
        for symbol in symbols:
            price = self.crypto_trader.get_latest_price(symbol)
            ohlc = self.crypto_trader.get_real_time_ohlc(symbol)

            if price and ohlc:
                print(
                    f"{symbol:>10}: ${price:>10.2f} | "
                    f"OHLC: ${ohlc['open']:>8.2f}/${ohlc['high']:>8.2f}/"
                    f"${ohlc['low']:>8.2f}/${ohlc['close']:>8.2f}"
                )
                symbols_with_data += 1
            elif price:
                print(f"{symbol:>10}: ${price:>10.2f} (price only, no OHLC)")
                symbols_with_data += 1
            else:
                print(f"{symbol:>10}: ⏳ Waiting for data...")
                symbols_waiting += 1
        
        print("-" * 60)
        print(f"📈 {symbols_with_data}/{len(symbols)} symbols have data, {symbols_waiting} waiting")
        
        # Show detailed debug info if requested or if no data is coming through
        if show_debug or (symbols_waiting == len(symbols) and len(symbols) > 0):
            print(self.crypto_trader.get_streaming_debug_info())

    def _execute_trading_logic(self, symbols: List[str]):
        """Execute trading logic based on real-time data and spread signals"""
        # Build price map and staleness map from latest streaming data
        price_map = {}
        staleness_map = {}
        
        # Get connection health info for staleness tracking
        health = self.crypto_trader.get_connection_health()
        symbol_health = health.get('symbols', {})
        
        for sym in symbols:
            p = self.crypto_trader.get_latest_price(sym)
            if p is not None:
                price_map[sym] = float(p)
                # Get staleness from connection health
                sym_status = symbol_health.get(sym, {})
                staleness = sym_status.get('last_data_age_secs')
                if staleness is not None:
                    staleness_map[sym] = staleness
                else:
                    staleness_map[sym] = 0.0  # Assume fresh if not tracked
        
        # Log warnings for stale symbols (> 60s)
        stale_symbols = [s for s, age in staleness_map.items() if age > 60]
        if stale_symbols:
            stale_info = {s: f"{staleness_map[s]:.0f}s" for s in stale_symbols}
            logger.warning(f"⚠️ Stale data detected: {stale_info}")
        
        # Check for open positions with stale data and warn
        for group_id, pos in self._spread_positions.items():
            stale_assets = [a for a in pos.assets if staleness_map.get(a, 0) > 60]
            if stale_assets:
                ages = {a: f"{staleness_map.get(a, 0):.0f}s" for a in stale_assets}
                logger.warning(f"⚠️ Position {group_id} has stale assets: {ages}")
        
        # If spread engine is available, use it for signals
        if self.spread_engine is not None:
            # Update engine with latest prices AND staleness info
            self.spread_engine.update_prices(price_map, staleness_map)
            
            # Get signals
            signals = self.spread_engine.get_signals()
            
            for signal in signals:
                # Handle EMERGENCY_EXIT specially - force close position
                if signal.signal_type == SignalType.EMERGENCY_EXIT:
                    logger.warning(f"🚨 EMERGENCY EXIT: {signal.group_id} - Data critically stale!")
                    if signal.group_id in self._spread_positions:
                        self._exit_spread(signal.group_id, price_map, "EMERGENCY: Data staleness")
                    continue
                
                logger.info(f"📊 Spread {signal.group_id}: z={signal.zscore:.2f} -> {signal.signal_type.value} (conf={signal.confidence:.2f})")
                
                # Execute spread trades
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
            
            # Log z-score summary for monitored groups
            zscores = self.spread_engine.get_all_zscores()
            zscore_strs = [f"{gid}: {z:.2f}" for gid, z in zscores.items() if not (z != z)]  # skip NaN
            if zscore_strs:
                logger.info(f"📈 Z-scores: {', '.join(zscore_strs[:5])}")
        else:
            # Fallback: simple per-symbol momentum signals (no execution)
            for symbol in symbols:
                price_history = self.crypto_trader.get_price_history(symbol, limit=20)
                if len(price_history) >= 10:
                    prices = [p['price'] for p in price_history]
                    price_change = (prices[-1] - prices[0]) / prices[0]
                    if price_change > 0.02:
                        logger.info(f"📈 Momentum BUY signal: {symbol} up {price_change:.2%}")
                    elif price_change < -0.02:
                        logger.info(f"📉 Momentum SELL signal: {symbol} down {price_change:.2%}")

    def _enter_spread_long(self, signal: SpreadSignal, price_map: Dict[str, float]) -> bool:
        """
        Enter a long spread position (buy the spread).
        
        For a spread with weights {A: -beta, B: 1.0}:
        - BUY_SPREAD means we expect the spread to increase
        - We buy the asset with positive weight and sell the asset with negative weight
        
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
            
            # Calculate quantities for each leg
            quantities: Dict[str, float] = {}
            entry_prices: Dict[str, float] = {}
            
            for asset in assets:
                if asset not in price_map:
                    logger.error(f"Missing price for {asset}")
                    return False
                
                price = price_map[asset]
                weight = weights.get(asset, 0)
                
                # Calculate quantity based on notional and weight direction
                # Positive weight = buy, negative weight = sell
                qty = (self.spread_notional / price) * (1 if weight > 0 else -1)
                quantities[asset] = qty
                entry_prices[asset] = price
            
            # Execute orders for each leg
            orders = []
            for asset in assets:
                qty = quantities[asset]
                
                if qty > 0:
                    # Buy order
                    order = self.crypto_trader.buy_market_order(
                        symbol=asset,
                        notional=abs(qty) * price_map[asset]
                    )
                    logger.info(f"🟢 BUY {asset}: ${abs(qty) * price_map[asset]:.2f}")
                else:
                    # Sell order (for short leg)
                    order = self.crypto_trader.sell_market_order(
                        symbol=asset,
                        notional=abs(qty) * price_map[asset]
                    )
                    logger.info(f"🔴 SELL {asset}: ${abs(qty) * price_map[asset]:.2f}")
                
                orders.append(order)
            
            # Create spread position record
            position = SpreadPosition(
                group_id=group_id,
                assets=assets,
                side="LONG",
                weights=weights,
                quantities=quantities,
                entry_prices=entry_prices,
                current_prices=price_map.copy(),
                notional=self.spread_notional * len(assets),
                entry_zscore=signal.zscore,
                current_zscore=signal.zscore,
                entry_time=datetime.now(),
            )
            
            self._spread_positions[group_id] = position
            
            # Update spread engine position state
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
        - We sell the asset with positive weight and buy the asset with negative weight
        
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
            
            # Calculate quantities for each leg (opposite of long)
            quantities: Dict[str, float] = {}
            entry_prices: Dict[str, float] = {}
            
            for asset in assets:
                if asset not in price_map:
                    logger.error(f"Missing price for {asset}")
                    return False
                
                price = price_map[asset]
                weight = weights.get(asset, 0)
                
                # For short spread: sell positive weight, buy negative weight (opposite of long)
                qty = (self.spread_notional / price) * (-1 if weight > 0 else 1)
                quantities[asset] = qty
                entry_prices[asset] = price
            
            # Execute orders for each leg
            orders = []
            for asset in assets:
                qty = quantities[asset]
                
                if qty > 0:
                    order = self.crypto_trader.buy_market_order(
                        symbol=asset,
                        notional=abs(qty) * price_map[asset]
                    )
                    logger.info(f"🟢 BUY {asset}: ${abs(qty) * price_map[asset]:.2f}")
                else:
                    order = self.crypto_trader.sell_market_order(
                        symbol=asset,
                        notional=abs(qty) * price_map[asset]
                    )
                    logger.info(f"🔴 SELL {asset}: ${abs(qty) * price_map[asset]:.2f}")
                
                orders.append(order)
            
            # Create spread position record
            position = SpreadPosition(
                group_id=group_id,
                assets=assets,
                side="SHORT",
                weights=weights,
                quantities=quantities,
                entry_prices=entry_prices,
                current_prices=price_map.copy(),
                notional=self.spread_notional * len(assets),
                entry_zscore=signal.zscore,
                current_zscore=signal.zscore,
                entry_time=datetime.now(),
            )
            
            self._spread_positions[group_id] = position
            
            # Update spread engine position state
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
            
            # Execute closing orders (opposite of opening orders)
            for asset in position.assets:
                qty = position.quantities.get(asset, 0)
                current_price = price_map.get(asset, position.entry_prices.get(asset, 0))
                
                if qty > 0:
                    # We're long this leg, so sell to close
                    order = self.crypto_trader.sell_market_order(
                        symbol=asset,
                        notional=abs(qty) * current_price
                    )
                    logger.info(f"🔴 SELL (close) {asset}: ${abs(qty) * current_price:.2f}")
                else:
                    # We're short this leg, so buy to close
                    order = self.crypto_trader.buy_market_order(
                        symbol=asset,
                        notional=abs(qty) * current_price
                    )
                    logger.info(f"🟢 BUY (close) {asset}: ${abs(qty) * current_price:.2f}")
            
            # Calculate final P&L
            position.update_pnl(price_map)
            
            logger.info(f"🔚 Closed spread {group_id}: P&L=${position.unrealized_pnl:.2f} | Reason: {reason}")
            
            # Remove from positions
            del self._spread_positions[group_id]
            
            # Update spread engine position state
            if self.spread_engine:
                self.spread_engine.set_position(group_id, None)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error closing spread {group_id}: {e}")
            return False

    def _update_spread_positions(self) -> None:
        """Update current prices and P&L for all spread positions."""
        if not self._spread_positions:
            return
        
        # Get current prices
        price_map = {}
        for symbol in self.symbols:
            p = self.crypto_trader.get_latest_price(symbol)
            if p is not None:
                price_map[symbol] = float(p)
        
        # Get current z-scores
        zscores = {}
        if self.spread_engine:
            zscores = self.spread_engine.get_all_zscores()
        
        # Update each position
        for group_id, position in self._spread_positions.items():
            current_zscore = zscores.get(group_id)
            position.update_pnl(price_map, current_zscore)

    def _log_spread_positions(self) -> None:
        """Log summary of open spread positions."""
        if not self._spread_positions:
            return
        
        logger.info("\n📋 OPEN SPREAD POSITIONS:")
        logger.info("-" * 50)
        
        total_pnl = 0.0
        for group_id, pos in self._spread_positions.items():
            total_pnl += pos.unrealized_pnl
            z_str = f"{pos.current_zscore:.2f}" if pos.current_zscore == pos.current_zscore else "N/A"
            logger.info(
                f"  {group_id}: {pos.side} | Entry z={pos.entry_zscore:.2f} | "
                f"Current z={z_str} | P&L=${pos.unrealized_pnl:+.2f}"
            )
        
        logger.info(f"  TOTAL P&L: ${total_pnl:+.2f}")
        logger.info("-" * 50)

    def _close_all_spread_positions(self, reason: str = "Session ended") -> None:
        """Close all open spread positions."""
        if not self._spread_positions:
            return
        
        logger.info(f"🔚 Closing all spread positions: {reason}")
        
        # Get current prices
        price_map = {}
        for symbol in self.symbols:
            p = self.crypto_trader.get_latest_price(symbol)
            if p is not None:
                price_map[symbol] = float(p)
        
        # Close each position
        for group_id in list(self._spread_positions.keys()):
            self._exit_spread(group_id, price_map, reason)

    def monitor_data_only(self, symbols: List[str] = None, duration_minutes: int = 10):
        """
        Monitor real-time data without trading

        Args:
            symbols: List of crypto symbols to monitor
            duration_minutes: Duration of monitoring session
        """
        if symbols is None:
            symbols = ["BTC/USD", "ETH/USD"]

        logger.info(f"📡 Starting data monitoring for {duration_minutes} minutes")

        try:
            self.crypto_trader.start_real_time_streaming(symbols, ["trades", "quotes"])
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


def main():
    """Main function to run the live trading example"""

    # Configuration
    PAPER_TRADING = True  # Set to False for live trading (NOT RECOMMENDED FOR TESTING)
    SYMBOLS = ["BTC/USD", "ETH/USD"]  # Symbols to trade
    SESSION_DURATION = 10  # Minutes (start with short sessions for testing)

    # Validate environment
    import os

    if not os.getenv("ALPACA_API_KEY") or not os.getenv("ALPACA_SECRET_KEY"):
        logger.error("❌ Missing Alpaca API credentials!")
        logger.error(
            "Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables"
        )
        return

    # Initialize and run bot
    try:
        bot = LiveTradingBot(paper=PAPER_TRADING, symbols=SYMBOLS)
        bot.run_trading_session(duration_minutes=SESSION_DURATION)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")


if __name__ == "__main__":
    main()
