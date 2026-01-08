#!/usr/bin/env python3

"""
Primary Trading Application with Real-Time Streaming and Cointegration Analysis

This is the main entry point for the crypto trading application.
It provides multiple modes of operation:
1. Real-time streaming trading bot (indefinite or timed)
2. Simple data monitoring
3. Cointegration-based spread trading
4. Benchmark computation

Uses CCXT for exchange-agnostic data streaming and order execution,
enabling native long/short positions for spread trading on futures exchanges.

The exchange is configurable via the EXCHANGE_NAME environment variable.
See EXCHANGE_CONFIG.md for detailed configuration instructions.

Usage:
    # Run indefinitely until keyboard interrupt
    python main.py --mode trade-indefinite
    
    # Run for specific duration
    python main.py --mode trade --duration 60
    
    # Compute fresh benchmarks
    python main.py --mode compute-benchmarks --days 30

Sam Dawley
08/2025
"""

import argparse
from datetime import datetime
import logging
import os
import signal
import sys
import time
import traceback

from src.finance.rolling_buffer import RollingCointegrationBuffer
from src.finance.spread_engine import SpreadSignalEngine

# ===== add src directory to path for imports =====
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# ===== import with error handling =====
try:
    from src.finance.benchmarks import load_benchmarks, is_stale
    from src.finance.trading_bot import TradingPartner
    from src.trading_strategy.compute_benchmarks import (
        compute_benchmarks,
        save_benchmarks,
    )

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nPlease ensure all required dependencies are installed:")
    print("pip install ccxt numpy pandas python-dotenv statsmodels")
    print("\nAlso ensure you have set your environment variables.")
    print("See EXCHANGE_CONFIG.md for configuration instructions.")
    sys.exit(1)

except ValueError as e:
    print(f"❌ Configuration error: {e}")
    print("\nSee EXCHANGE_CONFIG.md for configuration instructions.")
    sys.exit(1)

# global flag for graceful shutdown
shutdown_requested = False


# ==================================================
# LOGGING
# ==================================================


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def _log_health_status(trader: TradingPartner, session_start: datetime, cycle_count: int) -> None:
    """
    Log health status of the trading system to console and log file.
    
    Displays uptime, cycle count, streaming health, symbol staleness,
    spread positions, and account balance.
    
    Args:
        trader: Active TradingPartner instance with open connections
        session_start: Datetime when trading session began
        cycle_count: Number of completed trading cycles
        
    Returns:
        None (outputs to logger)
        
    Example:
        >>> _log_health_status(trader, session_start, cycle_count=42)
        # Outputs health dashboard to console/log
    """
    logger.info("\n" + "=" * 60)
    logger.info("📊 HEALTH STATUS")
    logger.info("=" * 60)
    
    uptime = datetime.now() - session_start
    logger.info(f"Uptime: {uptime}")
    logger.info(f"Cycles completed: {cycle_count}")
    
    # Streaming health
    health = trader.crypto_trader.get_connection_health()
    logger.info(f"Streaming active: {health['streaming_active']}")
    logger.info(f"Connection healthy: {health['connection_healthy']}")
    logger.info(f"Reconnect attempts: {health['reconnect_attempts']}")
    
    # Symbol status
    for symbol, status in health.get('symbols', {}).items():
        age = status.get('last_data_age_secs')
        age_str = f"{age:.0f}s" if age is not None else "N/A"
        healthy = "✅" if status.get('healthy') else "❌"
        logger.info(f"  {symbol}: {healthy} age={age_str}")
    
    # Spread positions
    logger.info(f"Open spread positions: {len(trader._spread_positions)}")
    
    # Account info
    try:
        trader.get_account()
    except Exception as e:
        logger.warning(f"Could not fetch account info: {e}")
    
    logger.info("=" * 60)


def _check_global_emergency(
    trader: TradingPartner,
    stale_threshold_secs: float = 300.0,
    stale_symbol_ratio: float = 0.7,
    connection_unhealthy_cycles: int = 10,
    unhealthy_streak: int = 0,
) -> tuple:
    """
    Check if we should perform a global emergency exit.
    
    Triggers emergency if:
    1. Connection has been unhealthy for N consecutive cycles, OR
    2. More than X% of symbols are stale for extended period
    
    This is a safety mechanism to close all positions and terminate
    when the data feed has completely failed (e.g., exchange WebSocket
    session limits, network issues).
    
    Args:
        trader: Active TradingPartner instance
        stale_threshold_secs: Seconds to consider data stale (default: 300)
        stale_symbol_ratio: Ratio of stale symbols to trigger exit (default: 0.7)
        connection_unhealthy_cycles: Consecutive unhealthy cycles to trigger (default: 10)
        unhealthy_streak: Number of consecutive unhealthy cycles so far
        
    Returns:
        tuple: (should_exit: bool, updated_unhealthy_streak: int, reason: str)
        
    Example:
        >>> should_exit, streak, reason = _check_global_emergency(trader, unhealthy_streak=5)
        >>> if should_exit:
        ...     close_all_and_terminate(reason)
    """
    # ----- IMMEDIATE CHECK: Reconnection exhausted -----
    # If all reconnection attempts have failed, trigger immediate emergency exit
    if trader.crypto_trader.is_reconnection_exhausted():
        reason = "All reconnection attempts exhausted - streaming cannot recover"
        logger.error(f"🚨 GLOBAL EMERGENCY: {reason}")
        return True, unhealthy_streak, reason
    
    health = trader.crypto_trader.get_connection_health(stale_threshold_secs)
    
    # Check overall connection health
    connection_healthy = health.get('connection_healthy', True)
    streaming_active = health.get('streaming_active', True)
    
    if not connection_healthy or not streaming_active:
        unhealthy_streak += 1
        logger.warning(
            f"⚠️ Connection unhealthy for {unhealthy_streak} consecutive cycles "
            f"(streaming_active={streaming_active}, connection_healthy={connection_healthy})"
        )
    else:
        if unhealthy_streak > 0:
            logger.info(f"✅ Connection recovered after {unhealthy_streak} unhealthy cycles")
        unhealthy_streak = 0  # Reset streak on healthy connection
    
    # Trigger exit if connection unhealthy for too many cycles
    if unhealthy_streak >= connection_unhealthy_cycles:
        reason = f"Connection unhealthy for {unhealthy_streak} consecutive cycles"
        logger.error(f"🚨 GLOBAL EMERGENCY: {reason}")
        return True, unhealthy_streak, reason
    
    # Check ratio of stale symbols
    symbol_health = health.get('symbols', {})
    if symbol_health:
        stale_count = sum(1 for s in symbol_health.values() if not s.get('healthy', True))
        total_count = len(symbol_health)
        stale_ratio = stale_count / total_count if total_count > 0 else 0
        
        if stale_ratio >= stale_symbol_ratio:
            reason = f"{stale_count}/{total_count} symbols stale ({stale_ratio:.0%} >= {stale_symbol_ratio:.0%})"
            logger.error(f"🚨 GLOBAL EMERGENCY: {reason}")
            return True, unhealthy_streak, reason
    
    return False, unhealthy_streak, ""


def _log_final_summary(trader: TradingPartner, session_start: datetime, cycle_count: int) -> None:
    """
    Log final summary when trading session ends.
    
    Called during graceful shutdown to report session duration,
    total cycles executed, and final account status.
    
    Args:
        trader: TradingPartner instance (may have closed positions)
        session_start: Datetime when trading session began
        cycle_count: Total trading cycles completed
        
    Returns:
        None (outputs to logger)
        
    Example:
        >>> _log_final_summary(trader, session_start, cycle_count=150)
        # Outputs final summary to console/log
    """
    logger.info("\n" + "=" * 60)
    logger.info("📋 FINAL SESSION SUMMARY")
    logger.info("=" * 60)
    
    duration = datetime.now() - session_start
    logger.info(f"Session duration: {duration}")
    logger.info(f"Total cycles: {cycle_count}")
    
    # Final account status
    try:
        trader.get_account()
    except Exception as e:
        logger.warning(f"Could not fetch final account info: {e}")
    
    logger.info("=" * 60)


# ==================================================
# SYMBOL FILTERING
# ==================================================


def filter_excluded_symbols(symbols: list, exclude: list = None) -> list:
    """
    Remove excluded symbols from a symbol list.
    
    Handles both canonical format (e.g., "BTC") and exchange format (e.g., "BTC/USD:USD").
    Exclusions are matched by base asset (case-insensitive).
    
    Args:
        symbols: List of symbols to filter
        exclude: List of symbols/bases to exclude (e.g., ["DOGE", "SHIB"])
        
    Returns:
        Filtered list with excluded symbols removed
        
    Example:
        >>> filter_excluded_symbols(["BTC/USD:USD", "DOGE/USD:USD"], ["DOGE"])
        ['BTC/USD:USD']
    """
    if not exclude:
        return symbols
    
    # Normalize exclusions to uppercase base symbols
    exclude_bases = set()
    for ex in exclude:
        # Handle both "DOGE" and "DOGE/USD:USD" formats
        base = ex.split('/')[0].upper() if '/' in ex else ex.upper()
        exclude_bases.add(base)
    
    filtered = []
    for sym in symbols:
        # Extract base from symbol
        base = sym.split('/')[0].upper() if '/' in sym else sym.upper()
        if base not in exclude_bases:
            filtered.append(sym)
        else:
            logger.info(f"🚫 Excluding symbol: {sym}")
    
    return filtered


# ==================================================
# LIVE TRADING/MONITORING ROUTINES
# ==================================================


def signal_handler(signum, frame):
    """
    Handle OS shutdown signals (SIGINT, SIGTERM) for graceful shutdown.
    
    Sets global shutdown_requested flag to True, allowing the main
    trading loop to exit cleanly and close positions.
    
    Args:
        signum: Signal number received (e.g., 2 for SIGINT)
        frame: Current stack frame (unused)
        
    Returns:
        None (sets global flag)
        
    Example:
        >>> signal.signal(signal.SIGINT, signal_handler)
        # Ctrl+C now triggers graceful shutdown
    """
    global shutdown_requested
    logger.info(f"\n🛑 Received signal {signum}. Initiating graceful shutdown...")
    shutdown_requested = True


def check_and_refresh_benchmarks(
    symbols: list,
    max_age_days: int = 7,
    days_back: int = 30,
    time_scale: str = "hour",
    max_groups: int = 10,
    p_threshold: float = 0.10,
) -> bool:
    """
    Check if cointegration benchmarks are stale and recompute if needed.
    
    Loads existing benchmarks from data/benchmarks/, checks their age,
    and recomputes if older than max_age_days. Benchmarks contain
    cointegration pairs, hedge ratios, and spread statistics needed
    for live trading signals.
    
    Args:
        symbols: List of crypto symbols (e.g., ['BTC/USDT:USDT', 'ETH/USDT:USDT'])
        max_age_days: Recompute if benchmarks older than this (default: 7)
        days_back: Historical lookback for cointegration analysis (default: 30)
        time_scale: Bar timeframe - 'min', 'hour', or 'day' (default: 'hour')
        max_groups: Max cointegration pairs to keep (default: 10)
        p_threshold: P-value for cointegration test; higher=more pairs (default: 0.10)
        
    Returns:
        True if benchmarks are fresh (existing or newly computed), False on error
        
    Example:
        >>> check_and_refresh_benchmarks(
        ...     symbols=['BTC/USDT:USDT', 'ETH/USDT:USDT', 'LTC/USDT:USDT'],
        ...     max_age_days=7,
        ...     p_threshold=0.15
        ... )
        True  # Benchmarks ready for trading
    """
    try:
        benchmarks = load_benchmarks()
        if not is_stale(benchmarks, max_age_days=max_age_days):
            logger.info(f"✅ Benchmarks are fresh (< {max_age_days} days old)")
            return True
        
        logger.info(f"⚠️ Benchmarks are stale (> {max_age_days} days old). Recomputing...")
        
    except FileNotFoundError:
        logger.info("📊 No benchmarks found. Computing initial benchmarks...")
    
    # Recompute benchmarks
    try:
        payload = compute_benchmarks(
            symbols=symbols,
            days_back=days_back,
            time_scale=time_scale,
            max_groups=max_groups,
            p_threshold=p_threshold,
        )
        out_path = save_benchmarks(payload, "data/benchmarks")
        logger.info(f"✅ Fresh benchmarks saved: {out_path} (p-threshold={p_threshold})")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to compute benchmarks: {e}")
        return False


def run_indefinitely(
    trader: TradingPartner,
    symbols: list,
    lookback_bars: int = 500,
    max_groups: int = 10,
    cycle_seconds: int = 30,
    health_log_minutes: int = 15,
    benchmark_refresh_days: int = 7,
    max_stream_symbols: int = 10,
    entry_zscore: float = 2.0,
    exit_zscore: float = 0.5,
    entry_staleness: float = 30.0,
    exit_staleness: float = 300.0,
    emergency_staleness: float = 900.0,
    recalibrate_interval: int = 10,
    recalibrate_min_obs: int = 50,
    exclude_symbols: list = None,
) -> None:
    """
    Run the live trading bot indefinitely until interrupted (Ctrl+C).
    
    Main trading loop that:
    1. Connects to Binance Futures WebSocket for real-time price streaming
    2. Monitors cointegrated pairs for entry/exit signals via z-scores
    3. Executes spread trades when signals trigger (native long/short)
    4. Periodically recalibrates spread parameters from recent prices
    5. Handles graceful shutdown, closing positions on exit
    
    Uses per-symbol staleness thresholds based on liquidity tiers
    and heartbeat monitoring to detect zombie connections.
    
    Args:
        trader: Initialized TradingPartner with Binance credentials
        symbols: Fallback symbol list if no benchmarks exist
        cycle_seconds: Seconds between trading logic execution (default: 30)
        health_log_minutes: Minutes between health status logs (default: 15)
        benchmark_refresh_days: Days before benchmarks recomputed (default: 7)
        max_stream_symbols: Max WebSocket subscriptions (default: 10)
        entry_zscore: Z-score to trigger entry (default: 2.0)
        exit_zscore: Z-score to trigger exit (default: 0.5)
        entry_staleness: Max data age (sec) for entries (default: 30)
        exit_staleness: Max data age (sec) for exits (default: 300)
        emergency_staleness: Force exit if data older than (default: 900)
        recalibrate_interval: Minutes between spread recalibrations (default: 10)
        recalibrate_min_obs: Min observations before recalibrating (default: 50)
        
    Returns:
        None (runs until interrupted)
        
    Example:
        >>> trader = TradingPartner(paper=True)
        >>> run_indefinitely(
        ...     trader=trader,
        ...     symbols=['BTC/USDT:USDT', 'ETH/USDT:USDT'],
        ...     cycle_seconds=30,
        ...     entry_zscore=2.0
        ... )
        # Runs until Ctrl+C, then closes positions gracefully
    """
    global shutdown_requested
    
    logger.info("=" * 60)
    logger.info("🚀 STARTING INDEFINITE TRADING SESSION")
    logger.info("=" * 60)
    logger.info(f"Cycle interval: {cycle_seconds}s")
    logger.info(f"Health log interval: {health_log_minutes} min")
    logger.info(f"Benchmark refresh: {benchmark_refresh_days} days")
    logger.info(f"Max streaming symbols: {max_stream_symbols}")
    logger.info(f"Z-score thresholds: entry={entry_zscore}, exit={exit_zscore}")

    if recalibrate_interval > 0:
        logger.info(f"Rolling recalibration: every {recalibrate_interval} min (min {recalibrate_min_obs} obs)")

    else:
        logger.info("Rolling recalibration: DISABLED")

    logger.info("Press Ctrl+C to stop gracefully")
    logger.info("=" * 60)
    
    # ----- register signal handlers -----
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # ===== SMART SYMBOL SELECTION =====
    # Only stream symbols that are actually needed for cointegration pairs
    
    all_symbols = []
    
    # Option 1: Get symbols from spread engine (best - uses loaded benchmarks)
    if trader.spread_engine is not None:
        all_symbols = trader.spread_engine.get_required_symbols()
        logger.info(f"📊 Using {len(all_symbols)} symbols from cointegration pairs: {all_symbols}")
    
    # Option 2: Load benchmarks directly and extract symbols (with proper conversion)
    if not all_symbols:
        try:
            from src.finance.benchmarks import get_assets
            
            benchmarks = load_benchmarks()
            groups = benchmarks.get('cointegration_groups', [])

            symbol_set = set()
            for group in groups:
                # Use get_assets() to properly convert canonical symbols to exchange format
                converted_assets = get_assets(group, convert_to_exchange=True)
                symbol_set.update(converted_assets)
            all_symbols = list(symbol_set)
            logger.info(f"📊 Loaded {len(all_symbols)} symbols from {len(groups)} cointegration groups")

        except Exception as e:
            logger.warning(f"Could not load benchmarks: {e}")
    
    # Option 3: Fallback to a minimal set
    if not all_symbols:
        all_symbols = symbols[:max_stream_symbols] if symbols else ["BTC/USDT:USDT", "ETH/USDT:USDT"]
        logger.warning(f"⚠️ No cointegration pairs found - using fallback symbols: {all_symbols}")
    
    # Apply safety limit
    if len(all_symbols) > max_stream_symbols:
        logger.warning(f"⚠️ Limiting streams from {len(all_symbols)} to {max_stream_symbols} symbols")
        all_symbols = all_symbols[:max_stream_symbols]
    
    # Apply exclusion filter
    if exclude_symbols:
        all_symbols = filter_excluded_symbols(all_symbols, exclude_symbols)
    
    logger.info(f"📡 Will stream {len(all_symbols)} symbols: {all_symbols}")
    
    # Initialize spread engine with custom z-score thresholds if not already set
    if trader.spread_engine is None and trader.benchmarks is not None:
        # Get available symbols from exchange for proper filtering
        available_perpetuals = trader.crypto_trader.get_tradeable_symbols()
        
        trader.spread_engine = SpreadSignalEngine(
            benchmarks=trader.benchmarks,
            entry_zscore=entry_zscore,
            exit_zscore=exit_zscore,
            max_groups=max_groups,
            entry_max_staleness_secs=entry_staleness,
            exit_max_staleness_secs=exit_staleness,
            emergency_exit_staleness_secs=emergency_staleness,
            available_symbols=available_perpetuals if available_perpetuals else None,
        )

        logger.info(f"📊 Spread engine initialized: entry z={entry_zscore}, exit z={exit_zscore}")
        logger.info(f"📊 Staleness thresholds: entry={entry_staleness}s, exit={exit_staleness}s, emergency={emergency_staleness}s")
        
        # Update all_symbols from properly filtered spread engine
        filtered_symbols = trader.spread_engine.get_required_symbols()
        if filtered_symbols:
            all_symbols = filtered_symbols
            logger.info(f"📊 Using {len(all_symbols)} filtered symbols from spread engine")

    elif trader.spread_engine is not None:
        trader.spread_engine.entry_zscore = entry_zscore
        trader.spread_engine.exit_zscore = exit_zscore
        trader.spread_engine.entry_max_staleness_secs = entry_staleness
        trader.spread_engine.exit_max_staleness_secs = exit_staleness
        trader.spread_engine.emergency_exit_staleness_secs = emergency_staleness
        logger.info(f"📊 Spread engine thresholds updated: entry z={entry_zscore}, exit z={exit_zscore}")
    
    # Initialize rolling buffer
    trader.rolling_buffer = RollingCointegrationBuffer(
        symbols=all_symbols,
        lookback_bars=lookback_bars,
    )
    
    # Set trader.symbols for P&L updates in _update_spread_positions
    trader.symbols = all_symbols
    
    # ===== start streaming =====
    try:
        logger.info("📡 Starting real-time streaming...")
        # Using "tickers" for Binance Futures streaming
        trader.crypto_trader.start_real_time_streaming(
            all_symbols, ["tickers"]
        )
        
        # Wait for initial data
        logger.info("⏳ Waiting for initial streaming data...")
        if not trader.crypto_trader.wait_for_data(all_symbols, timeout_secs=60):
            logger.warning("⚠️ Timeout waiting for initial data, continuing anyway...")
        
        if not trader.crypto_trader.is_streaming_active():
            logger.error("❌ Streaming failed to start")
            return
        
        logger.info("✅ Streaming active")
        
    except Exception as e:
        logger.error(f"❌ Failed to start streaming: {e}")
        return
    
    # Main trading loop
    cycle_count = 0
    session_start = datetime.now()
    last_health_log = datetime.now()
    last_benchmark_check = datetime.now()
    last_recalibration = datetime.now()
    unhealthy_streak = 0  # Track consecutive unhealthy connection cycles
    
    # Emergency exit parameters
    emergency_stale_threshold = entry_staleness * 2  # 2x entry staleness
    emergency_symbol_ratio = 0.7  # 70% of symbols must be stale
    emergency_unhealthy_cycles = 5  # 5 cycles × cycle_seconds = 2.5 min unhealthy duration
    
    try:
        while not shutdown_requested:
            cycle_count += 1
            cycle_start = datetime.now()
            
            # ===== GLOBAL EMERGENCY CHECK =====
            should_emergency_exit, unhealthy_streak, emergency_reason = _check_global_emergency(
                trader=trader,
                stale_threshold_secs=emergency_stale_threshold,
                stale_symbol_ratio=emergency_symbol_ratio,
                connection_unhealthy_cycles=emergency_unhealthy_cycles,
                unhealthy_streak=unhealthy_streak,
            )
            
            if should_emergency_exit:
                logger.error("=" * 60)
                logger.error("🚨 GLOBAL EMERGENCY EXIT TRIGGERED")
                logger.error(f"Reason: {emergency_reason}")
                logger.error("Closing all positions and terminating...")
                logger.error("=" * 60)
                shutdown_requested = True
                break
            
            # ===== trading cycle =====
            try:
                logger.info(f"\n🔄 Cycle #{cycle_count} | Uptime: {datetime.now() - session_start}")
                
                # ----- monitor, execute, and update positions -----
                # Show debug info for first 3 cycles to help diagnose streaming issues
                trader._monitor_market_data(all_symbols, show_debug=(cycle_count <= 3))
                trader._execute_trading_logic(all_symbols, staleness_threshold=entry_staleness)
                trader._update_spread_positions()
                
                # ----- record prices for rolling recalibration -----
                if trader.spread_engine is not None and recalibrate_interval > 0:
                    price_map = {}
                    for sym in all_symbols:
                        p = trader.crypto_trader.get_latest_price(sym)

                        if p is not None:
                            price_map[sym] = float(p)

                    if price_map:
                        trader.spread_engine.record_prices(price_map)
                
                # ----- logging -----
                if trader._spread_positions:
                    trader._log_spread_positions()
                
            except Exception as e:
                logger.error(f"❌ Error in trading cycle: {e}")
                traceback.print_exc()
            
            # ----- health logging -----
            if (datetime.now() - last_health_log).total_seconds() >= health_log_minutes * 60:
                _log_health_status(trader, session_start, cycle_count)
                last_health_log = datetime.now()
            
            # ----- benchmark refresh check -----
            if (datetime.now() - last_benchmark_check).total_seconds() >= 3600:  # Check hourly
                try:
                    benchmarks = load_benchmarks()
                    if is_stale(benchmarks, max_age_days=benchmark_refresh_days):
                        logger.info("🔄 Benchmarks are stale. Consider restarting with fresh benchmarks.")
                        # Note: We don't auto-refresh during trading to avoid disruption
                except:
                    pass
                last_benchmark_check = datetime.now()
            
            # ----- rolling recalibration -----
            if recalibrate_interval > 0 and trader.spread_engine is not None:
                if (datetime.now() - last_recalibration).total_seconds() >= recalibrate_interval * 60:
                    logger.info("📊 Running rolling recalibration...")
                    results = trader.spread_engine.recalibrate_from_history(
                        min_observations=recalibrate_min_obs
                    )
                    
                    # Log recalibration results
                    successful = sum(1 for r in results.values() if r.get('success'))
                    total = len(results)
                    logger.info(f"📊 Recalibration complete: {successful}/{total} groups updated")
                    
                    for gid, r in results.items():
                        if r.get('success'):
                            mean_shift = r.get('mean_shift', 0)
                            std_change = r.get('std_change_pct', 0)
                            logger.info(f"  {gid}: mean shift={mean_shift:+.4f}, std change={std_change:+.1f}%")
                        else:
                            logger.debug(f"  {gid}: {r.get('reason', 'Failed')}")
                    
                    last_recalibration = datetime.now()
            
            # ----- sleep until next cycle -----
            elapsed = (datetime.now() - cycle_start).total_seconds()
            sleep_time = max(0, cycle_seconds - elapsed)
            
            # ----- interruptible sleep -----
            for _ in range(int(sleep_time * 10)):
                if shutdown_requested:
                    break
                time.sleep(0.1)
    
    except Exception as e:
        logger.error(f"❌ Fatal error in main loop: {e}")
        traceback.print_exc()
    
    finally:
        # ----- graceful shutdown -----
        logger.info("\n" + "=" * 60)
        logger.info("🛑 SHUTTING DOWN")
        logger.info("=" * 60)
        
        trader._close_all_spread_positions("Graceful shutdown")
        trader.crypto_trader.stop_real_time_streaming()

        _log_final_summary(trader, session_start, cycle_count)
        logger.info("✅ Shutdown complete")


# ==================================================
# PRIMARY SUBROUTINE
# ==================================================


def main() -> int:
    """
    Main CLI entry point for the crypto trading application.
    
    Supports multiple modes:
    - trade-indefinite: Run live trading until interrupted (recommended)
    - trade: Run live trading for a fixed duration
    - monitor: Stream prices without trading
    - compute-benchmarks: Compute cointegration pairs
    - account: Display account info
    
    Uses Binance Futures via CCXT for both streaming and trading.
    
    Args:
        None (uses command-line arguments via argparse)
        
    Returns:
        int: Exit code (0=success, 1=error)
        
    Example:
        # From command line:
        $ python main.py --mode monitor --duration 10
        $ python main.py --mode compute-benchmarks --p-threshold 0.15
        $ python main.py --mode trade-indefinite --entry-zscore 2.0 --exit-zscore 0.5
    """
    global crypto_universe
    
    parser = argparse.ArgumentParser(
        description='Crypto Trading Application with Cointegration Analysis (CCXT)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
                # Run trading indefinitely (uses only cointegration pair symbols)
                python main.py --mode trade-indefinite
                
                # Limit streaming to 5 symbols
                python main.py --mode trade-indefinite --max-stream-symbols 5
                
                # Run trading for 60 minutes
                python main.py --mode trade --max-stream-symbols 5 --duration 60
                
                # Monitor data only
                python main.py --mode monitor --duration 10
                
                # Compute fresh benchmarks with default p-value (0.10)
                python main.py --mode compute-benchmarks --days 30 --max-groups 10
                
                # Compute benchmarks with relaxed p-value (finds more pairs)
                python main.py --mode compute-benchmarks --days 30 --max-groups 10 --p-threshold 0.15
                
                # Show account info
                python main.py --mode account
            """
    )

    # ---- primary options -----
    parser.add_argument(
        '--mode', 
        choices=['trade', 'trade-indefinite', 'monitor', 'account', 'compute-benchmarks'], 
        default='monitor', 
        help='Application mode'
    )
    parser.add_argument(
        '--cycle-interval', 
        type=int, 
        default=30,
        help='Seconds between trading cycles (default: 30)'
    )
    parser.add_argument(
        '--duration', 
        type=int, 
        default=10,
        help='Duration in minutes (for timed modes)'
    )
    parser.add_argument(
        '--health-interval', 
        type=int, 
        default=15,
        help='Minutes between health status logs (default: 15)'
    )
    parser.add_argument(
        '--live', 
        action='store_true',
        help='Use live trading (default: testnet/paper trading)'
    )
    parser.add_argument(
        '--lookback-bars',
        type=int,
        default=500,
        help='Lookback bars for historical data (default: 500)'
    )
    parser.add_argument(
        '--symbols', 
        nargs='+', 
        default=None,
        help='Crypto symbols to trade/monitor (default: all supported)'
    )
    parser.add_argument(
        '--exclude-symbols',
        nargs='+',
        default=None,
        help='Symbols to exclude from trading/streaming (e.g., --exclude-symbols DOGE SHIB)'
    )
    
    # ---- benchmarking computation options -----
    parser.add_argument(
        '--days', 
        type=float, 
        default=30,
        help='Lookback days for benchmark computation'
    )
    parser.add_argument(
        '--time-scale', 
        type=str, 
        default='hour',
        choices=['min', 'minute', 'minutes', 'm', 'hour', 'hours', 'h', 'day', 'days', 'd'],
        help='Time scale for historical bars'
    )
    parser.add_argument(
        '--max-groups', 
        type=int, 
        default=10,
        help='Max number of cointegration pairs to keep'
    )
    parser.add_argument(
        '--benchmark-refresh-days',
        type=int,
        default=7,
        help='Days before benchmarks are considered stale'
    )
    parser.add_argument(
        '--p-threshold',
        type=float,
        default=0.10,
        help='P-value threshold for cointegration test (default: 0.10). '
             'Higher values (e.g., 0.15, 0.20) are less strict and find more pairs; '
             'lower values (e.g., 0.05) are more strict.'
    )
    parser.add_argument(
        '--max-stream-symbols',
        type=int,
        default=40,
        help='Maximum symbols to stream simultaneously (default: 40).'
    )
    parser.add_argument(
        '--entry-zscore',
        type=float,
        default=2.0,
        help='Z-score threshold for entering spread positions (default: 2.0). '
             'Lower values (e.g., 1.5) trigger more trades; higher values (e.g., 2.5) are more conservative.'
    )
    parser.add_argument(
        '--exit-zscore',
        type=float,
        default=0.5,
        help='Z-score threshold for exiting spread positions (default: 0.5). '
             'Higher values exit sooner; lower values hold positions longer.'
    )
    
    # ---- staleness thresholds -----
    parser.add_argument(
        '--entry-staleness',
        type=float,
        default=30.0,
        help='Max price age (seconds) for entering new positions (default: 30). '
             'Fresh data is required for entries to avoid bad fills.'
    )
    parser.add_argument(
        '--exit-staleness',
        type=float,
        default=300.0,
        help='Max price age (seconds) for exiting positions (default: 300 = 5 min). '
             'More tolerance for exits to avoid being trapped in positions.'
    )
    parser.add_argument(
        '--emergency-staleness',
        type=float,
        default=900.0,
        help='Price age (seconds) that triggers emergency exit (default: 900 = 15 min). '
             'Forces position close if data becomes critically stale.'
    )
    
    # ---- rolling recalibration options -----
    parser.add_argument(
        '--recalibrate-interval',
        type=int,
        default=10,
        help='Minutes between spread parameter recalibrations (default: 10). '
             'Set to 0 to disable rolling recalibration.'
    )
    parser.add_argument(
        '--recalibrate-min-obs',
        type=int,
        default=50,
        help='Minimum price observations required before recalibration (default: 50). '
             'Lower values recalibrate sooner but may be less stable.'
    )
    
    args = parser.parse_args()
    symbols = args.symbols if args.symbols else crypto_universe
    
    # ----- handle live trading confirmation -----
    paper_trading = not args.live
    if args.live:
        logger.warning("⚠️  LIVE TRADING MODE - Real money at risk!")
        confirm = input("Are you sure you want to trade with real money? (yes/no): ")
        if confirm.lower() != 'yes':
            logger.info("Switching to testnet (paper trading) mode")
            paper_trading = True
    
    # ===== mode determination =====
    if args.mode == 'account':
        trader = TradingPartner(paper=paper_trading)
        logger.info("📊 Displaying account information...")
        trader.get_account()
        return 0
    
    # ----- recompute cointegration benchmarks -----
    elif args.mode == 'compute-benchmarks':
        # Apply exclusion filter to symbols
        filtered_symbols = filter_excluded_symbols(symbols, args.exclude_symbols)
        
        logger.info("🧮 Computing pairwise cointegration benchmarks")
        logger.info(f"Symbols: {filtered_symbols}")
        logger.info(f"Lookback: {args.days} days, time scale: {args.time_scale}, max-groups: {args.max_groups}")
        logger.info(f"P-value threshold: {args.p_threshold} (higher = more pairs found)")
        
        try:
            payload = compute_benchmarks(
                symbols=filtered_symbols,
                days_back=args.days,
                time_scale=args.time_scale,
                max_groups=args.max_groups,
                p_threshold=args.p_threshold,
            )
            out_path = save_benchmarks(payload, "data/benchmarks")
            logger.info(f"✅ Benchmarks saved: {out_path}")
            logger.info(f"📊 Found {len(payload.get('cointegration_groups', []))} cointegration groups")
            return 0
        except Exception as e:
            logger.error(f"❌ Failed to compute benchmarks: {e}")
            traceback.print_exc()
            return 1
    
    # ----- monitor prices without trading -----
    elif args.mode == 'monitor':
        # Apply exclusion filter to symbols
        filtered_symbols = filter_excluded_symbols(symbols, args.exclude_symbols)
        
        trader = TradingPartner(paper=paper_trading)
        logger.info("📡 Starting monitoring mode...")
        success = trader.monitor_data_only(filtered_symbols, args.duration)
        return 0 if success else 1
    
    # ----- live trading for a specified duration -----
    elif args.mode == "trade":
        check_and_refresh_benchmarks(
            symbols=symbols,
            max_age_days=args.benchmark_refresh_days,
            days_back=args.days,
            time_scale=args.time_scale,
            max_groups=args.max_groups,
            p_threshold=args.p_threshold,
        )
        
        # Apply exclusion filter to symbols
        filtered_symbols = filter_excluded_symbols(symbols, args.exclude_symbols)
        
        trader = TradingPartner(paper=paper_trading)
        logger.info("🤖 Starting timed trading mode...")

        success = trader.start_streaming_bot(
            symbols=filtered_symbols, 
            lookback_bars=args.lookback_bars,
            cycle_interval=args.cycle_interval,
            duration_minutes=args.duration,
            max_stream_symbols=args.max_stream_symbols,
            entry_zscore=args.entry_zscore,
            exit_zscore=args.exit_zscore,
        )
        
        return 0 if success else 1
    
    # ----- indefinite trading (runs until interrupted) -----
    elif args.mode == "trade-indefinite":
        check_and_refresh_benchmarks(
            symbols=symbols,
            max_age_days=args.benchmark_refresh_days,
            days_back=args.days,
            time_scale=args.time_scale,
            max_groups=args.max_groups,
            p_threshold=args.p_threshold,
        )
        
        trader = TradingPartner(paper=paper_trading)
        logger.info("🤖 Starting indefinite trading mode...")

        run_indefinitely(
            trader=trader,
            symbols=symbols,
            lookback_bars=args.lookback_bars,
            max_groups=args.max_groups,
            cycle_seconds=args.cycle_interval,
            health_log_minutes=args.health_interval,
            benchmark_refresh_days=args.benchmark_refresh_days,
            max_stream_symbols=args.max_stream_symbols,
            entry_zscore=args.entry_zscore,
            exit_zscore=args.exit_zscore,
            entry_staleness=args.entry_staleness,
            exit_staleness=args.exit_staleness,
            emergency_staleness=args.emergency_staleness,
            recalibrate_interval=args.recalibrate_interval,
            recalibrate_min_obs=args.recalibrate_min_obs,
            exclude_symbols=args.exclude_symbols,
        )

        return 0
    
    return 0


if __name__ == "__main__":
    
    # Import symbols from configured exchange (see src/finance/__init__.py)
    from src.finance import CRYPTO_TICKERS, EXCHANGE_NAME, EXCHANGE_CONFIG
    
    crypto_universe = CRYPTO_TICKERS
    logger.info(f"🔧 Using {EXCHANGE_CONFIG['name']} with {len(crypto_universe)} symbols")

    exit_code = main()
    sys.exit(exit_code)
