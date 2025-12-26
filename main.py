#!/usr/bin/env python3

"""
Primary Trading Application with Real-Time Streaming and Cointegration Analysis

This is the main entry point for the crypto trading application.
It provides multiple modes of operation:
1. Real-time streaming trading bot (indefinite or timed)
2. Simple data monitoring
3. Cointegration-based spread trading
4. Benchmark computation

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

# ===== add src directory to path for imports =====
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# ===== import with error handling =====
try:
    from src.finance.live_trading_example import TradingPartner
    from src.finance.benchmarks import load_benchmarks, is_stale
    from src.trading_strategy.compute_benchmarks import (
        compute_benchmarks,
        save_benchmarks,
    )

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nPlease ensure all required dependencies are installed:")
    print("pip install alpaca-py numpy pandas matplotlib python-dotenv statsmodels")
    print("\nAlso ensure you have set your environment variables:")
    print("export ALPACA_API_KEY='your_key'")
    print("export ALPACA_SECRET_KEY='your_secret'")
    sys.exit(1)

except ValueError as e:
    print("❌ Missing required environment variables!")
    print("\nPlease set the following environment variables:")
    print("export ALPACA_API_KEY='your_api_key'")
    print("export ALPACA_SECRET_KEY='your_secret_key'")
    print("export ALPACA_API_BASE_URL='https://paper-api.alpaca.markets'  # For paper trading")
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
    """Log health status of the trading system."""
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


def _log_final_summary(trader: TradingPartner, session_start: datetime, cycle_count: int) -> None:
    """Log final session summary."""
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
# LIVE TRADING/MONITORING ROUTINES
# ==================================================


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
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
    Check if benchmarks are stale and recompute if needed.
    
    Args:
        symbols: List of symbols for benchmark computation
        max_age_days: Maximum age before benchmarks are considered stale
        days_back: Lookback period for cointegration analysis
        time_scale: Time scale for bars
        max_groups: Maximum number of cointegration groups to keep
        p_threshold: P-value threshold for cointegration test (higher = more pairs)
        
    Returns:
        True if benchmarks are fresh (either already fresh or successfully refreshed)
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
    cycle_seconds: int = 30,
    health_log_minutes: int = 15,
    benchmark_refresh_days: int = 7,
    max_stream_symbols: int = 10,
    entry_zscore: float = 2.0,
    exit_zscore: float = 0.5,
    entry_staleness: float = 30.0,
    exit_staleness: float = 300.0,
    emergency_staleness: float = 900.0,
) -> None:
    """
    Run the trading bot indefinitely until interrupted.
    
    Args:
        trader: TradingPartner instance
        symbols: List of symbols to trade (used as fallback if no benchmarks)
        cycle_seconds: Seconds between trading cycles
        health_log_minutes: Minutes between health status logs
        benchmark_refresh_days: Days after which to refresh benchmarks
        max_stream_symbols: Maximum number of symbols to stream (Alpaca API limit)
        entry_zscore: Z-score threshold for entering positions (default: 2.0)
        entry_staleness: Max price age (secs) for new entries (default: 30)
        exit_staleness: Max price age (secs) for exits (default: 300)
        emergency_staleness: Force exit after this staleness (default: 900)
        exit_zscore: Z-score threshold for exiting positions (default: 0.5)
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
    logger.info("Press Ctrl+C to stop gracefully")
    logger.info("=" * 60)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # ===== SMART SYMBOL SELECTION =====
    # Only stream symbols that are actually needed for cointegration pairs
    # This prevents "symbol limit exceeded" errors from Alpaca API
    
    all_symbols = []
    
    # Option 1: Get symbols from spread engine (best - uses loaded benchmarks)
    if trader.spread_engine is not None:
        all_symbols = trader.spread_engine.get_required_symbols()
        logger.info(f"📊 Using {len(all_symbols)} symbols from cointegration pairs: {all_symbols}")
    
    # Option 2: Load benchmarks directly and extract symbols
    if not all_symbols:
        try:
            benchmarks = load_benchmarks()
            groups = benchmarks.get('cointegration_groups', [])
            # Extract unique symbols from cointegration groups
            symbol_set = set()
            for group in groups:
                symbol_set.update(group.get('assets', []))
            all_symbols = list(symbol_set)
            logger.info(f"📊 Loaded {len(all_symbols)} symbols from {len(groups)} cointegration groups")
        except Exception as e:
            logger.warning(f"Could not load benchmarks: {e}")
    
    # Option 3: Fallback to a minimal set
    if not all_symbols:
        all_symbols = symbols[:max_stream_symbols] if symbols else ["BTC/USD", "ETH/USD"]
        logger.warning(f"⚠️ No cointegration pairs found - using fallback symbols: {all_symbols}")
    
    # Apply safety limit to prevent exceeding Alpaca's WebSocket limits
    if len(all_symbols) > max_stream_symbols:
        logger.warning(f"⚠️ Limiting streams from {len(all_symbols)} to {max_stream_symbols} symbols")
        all_symbols = all_symbols[:max_stream_symbols]
    
    logger.info(f"📡 Will stream {len(all_symbols)} symbols: {all_symbols}")
    
    # Initialize spread engine with custom z-score thresholds if not already set
    if trader.spread_engine is None and trader.benchmarks is not None:
        from src.finance.spread_engine import SpreadSignalEngine
        trader.spread_engine = SpreadSignalEngine(
            benchmarks=trader.benchmarks,
            entry_zscore=entry_zscore,
            exit_zscore=exit_zscore,
            max_groups=10,
            entry_max_staleness_secs=entry_staleness,
            exit_max_staleness_secs=exit_staleness,
            emergency_exit_staleness_secs=emergency_staleness,
        )
        logger.info(f"📊 Spread engine initialized: entry z={entry_zscore}, exit z={exit_zscore}")
        logger.info(f"📊 Staleness thresholds: entry={entry_staleness}s, exit={exit_staleness}s, emergency={emergency_staleness}s")
    elif trader.spread_engine is not None:
        # Update existing spread engine thresholds
        trader.spread_engine.entry_zscore = entry_zscore
        trader.spread_engine.exit_zscore = exit_zscore
        trader.spread_engine.entry_max_staleness_secs = entry_staleness
        trader.spread_engine.exit_max_staleness_secs = exit_staleness
        trader.spread_engine.emergency_exit_staleness_secs = emergency_staleness
        logger.info(f"📊 Spread engine thresholds updated: entry z={entry_zscore}, exit z={exit_zscore}")
    
    # Initialize rolling buffer
    from src.finance.rolling_buffer import RollingCointegrationBuffer
    trader.rolling_buffer = RollingCointegrationBuffer(
        symbols=all_symbols,
        lookback_bars=500,
    )
    
    # Start streaming
    try:
        logger.info("📡 Starting real-time streaming...")
        # NOTE: Using only "quotes" to minimize WebSocket load and avoid symbol limit errors.
        # "quotes" provides bid/ask prices which is sufficient for spread trading.
        # To add more data types, append to this list: ["quotes", "trades", "bars"]
        trader.crypto_trader.start_real_time_streaming(
            all_symbols, ["quotes"]
        )
        
        # Register rolling buffer callback (for future use when bars are enabled)
        # NOTE: Bar streaming is currently disabled. Uncomment when bars are added back.
        # trader.crypto_trader.register_bar_callback(trader.rolling_buffer.on_bar)
        
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
    
    try:
        while not shutdown_requested:
            cycle_count += 1
            cycle_start = datetime.now()
            
            # ===== trading cycle =====
            try:
                logger.info(f"\n🔄 Cycle #{cycle_count} | Uptime: {datetime.now() - session_start}")
                
                # ----- monitor, execute, and update positions -----
                # Show debug info for first 3 cycles to help diagnose streaming issues
                trader._monitor_market_data(all_symbols, show_debug=(cycle_count <= 3))
                trader._execute_trading_logic(all_symbols)
                trader._update_spread_positions()
                
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
# PRIMIARY SUBROUTINE
# ==================================================


def main() -> int:
    """
    Main application entry point with command-line interface
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    global crypto_universe
    
    parser = argparse.ArgumentParser(
        description='Crypto Trading Application with Cointegration Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
                # Run trading indefinitely (uses only cointegration pair symbols)
                python main.py --mode trade-indefinite
                
                # Limit streaming to 5 symbols (if hitting Alpaca limits)
                python main.py --mode trade-indefinite --max-stream-symbols 5
                
                # Run trading for 60 minutes
                python main.py --mode trade --duration 60
                
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
        '--symbols', 
        nargs='+', 
        default=None,
        help='Crypto symbols to trade/monitor (default: all supported)'
    )
    parser.add_argument(
        '--duration', 
        type=int, 
        default=10,
        help='Duration in minutes (for timed modes)'
    )
    parser.add_argument(
        '--live', 
        action='store_true',
        help='Use live trading (default: paper trading)'
    )
    parser.add_argument(
        '--cycle-interval', 
        type=int, 
        default=30,
        help='Seconds between trading cycles (default: 30)'
    )
    parser.add_argument(
        '--health-interval', 
        type=int, 
        default=15,
        help='Minutes between health status logs (default: 15)'
    )
    
    # ---- benchmarking computation options -----
    parser.add_argument(
        '--days', 
        type=int, 
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
        default=10,
        help='Maximum symbols to stream simultaneously (default: 10). '
             'Alpaca API has limits on concurrent WebSocket subscriptions. '
             'Reduce this if you get "symbol limit exceeded" errors.'
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
    
    args = parser.parse_args()
    symbols = args.symbols if args.symbols else crypto_universe
    
    # ----- handle live trading confirmation -----
    paper_trading = not args.live
    if args.live:
        logger.warning("⚠️  LIVE TRADING MODE - Real money at risk!")
        confirm = input("Are you sure you want to trade with real money? (yes/no): ")
        if confirm.lower() != 'yes':
            logger.info("Switching to paper trading mode")
            paper_trading = True
    
    # ===== mode determination =====
    if args.mode == 'account':
        trader = TradingPartner(paper=paper_trading)
        logger.info("📊 Displaying account information...")
        trader.get_account()
        return 0
    
    # ----- recompute cointegration benchmarks -----
    elif args.mode == 'compute-benchmarks':
        logger.info("🧮 Computing pairwise cointegration benchmarks")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Lookback: {args.days} days, time scale: {args.time_scale}, max-groups: {args.max_groups}")
        logger.info(f"P-value threshold: {args.p_threshold} (higher = more pairs found)")
        
        try:
            payload = compute_benchmarks(
                symbols=symbols,
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
        trader = TradingPartner(paper=paper_trading)
        logger.info("📡 Starting monitoring mode...")
        success = trader.monitor_data_only(symbols[:10], args.duration)
        return 0 if success else 1
    
    # ----- live trading for a specified duration -----
    elif args.mode == 'trade':
        # Check/refresh benchmarks before trading
        check_and_refresh_benchmarks(
            symbols=symbols,
            max_age_days=args.benchmark_refresh_days,
            days_back=args.days,
            time_scale=args.time_scale,
            max_groups=args.max_groups,
            p_threshold=args.p_threshold,
        )
        
        trader = TradingPartner(paper=paper_trading)
        logger.info("🤖 Starting timed trading mode...")
        success = trader.start_streaming_bot(
            symbols=symbols, 
            duration_minutes=args.duration,
            max_stream_symbols=args.max_stream_symbols,
            entry_zscore=args.entry_zscore,
            exit_zscore=args.exit_zscore,
        )
        return 0 if success else 1
    
    # ----- indefinite trading (runs until interrupted) -----
    elif args.mode == 'trade-indefinite':
        # check/refresh benchmarks before trading
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
            cycle_seconds=args.cycle_interval,
            health_log_minutes=args.health_interval,
            benchmark_refresh_days=args.benchmark_refresh_days,
            max_stream_symbols=args.max_stream_symbols,
            entry_zscore=args.entry_zscore,
            exit_zscore=args.exit_zscore,
            entry_staleness=args.entry_staleness,
            exit_staleness=args.exit_staleness,
            emergency_staleness=args.emergency_staleness,
        )
        return 0
    
    return 0


if __name__ == "__main__":
    crypto_universe = [
        "AAVE/USD",
        "AVAX/USD",
        "BAT/USD",
        "BCH/USD",
        "BTC/USD",
        "CRV/USD",
        "DOGE/USD",
        "DOT/USD",
        "ETH/USD",
        "GRT/USD",
        "LINK/USD",
        "LTC/USD",
        "MKR/USD",
        "SHIB/USD",
        "SUSHI/USD",
        "UNI/USD",
        "USDC/USD",
        "USDT/USD",
        "XTZ/USD",
        "YFI/USD",
    ]
    exit_code = main()
    sys.exit(exit_code)
