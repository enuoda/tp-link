#!/usr/bin/env python3

"""
Primary Trading Application with Real-Time Streaming

This is the main entry point for the crypto trading application.
It provides multiple modes of operation:
1. Real-time streaming trading bot
2. Simple data monitoring
3. Historical data analysis
4. Testing and development tools

Sam Dawley
08/2025
"""

import os
import sys
import time
import logging
import argparse
# from datetime import datetime, timedelta
# from typing import List, Optional

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import with error handling
try:
    from src.finance.live_trading_example import TradingPartner
    from src.trading_strategy.compute_benchmarks import (
        compute_benchmarks,
        save_benchmarks,
    )

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nPlease ensure all required dependencies are installed:")
    print("pip install alpaca-py numpy pandas matplotlib python-dotenv")
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# def testing():
#     """Testing function for development and debugging"""
#     logger.info("🧪 Running testing function...")
    
#     try:
#         from polygon import RESTClient
        
#         # Note: This API key appears to be hardcoded for testing
#         # In production, use environment variables
#         client = RESTClient("qadVu6Vm9lfDpYrHy0e4xNG0wAYrnLQq")
        
#         types = client.get_ticker_types(asset_class="crypto", locale="us")
#         print("Crypto ticker types:", types)
        
#         return True
#     except Exception as e:
#         logger.error(f"Testing failed: {e}")
#         return False

def main() -> int:
    """
    Main application entry point with command-line interface
    
    Returns:
        Exit code (0 for success, 1 for error)
    """

    global crypto_universe
    
    parser = argparse.ArgumentParser(description='Crypto Trading Application')
    parser.add_argument('--mode', choices=['trade', 'monitor', 'test', 'account', 'compute-benchmarks'], 
                       default='monitor', help='Application mode')
    parser.add_argument('--symbols', nargs='+', 
                       default=crypto_universe,
                       help='Crypto symbols to trade/monitor')
    parser.add_argument('--duration', type=int, default=10,
                       help='Duration in minutes')
    parser.add_argument('--live', action='store_true',
                       help='Use live trading (default: paper trading)')
    # Benchmark computation options
    parser.add_argument('--days', type=int, default=30,
                        help='Lookback days for benchmark computation')
    parser.add_argument('--time-scale', type=str, default='hour',
                        choices=['min', 'minute', 'minutes', 'm', 'hour', 'hours', 'h', 'day', 'days', 'd'],
                        help='Time scale for historical bars')
    parser.add_argument('--max-groups', type=int, default=10,
                        help='Max number of cointegration pairs to keep')
    
    args = parser.parse_args()
    
    # Environment validation is handled by the import process above
    
    # Initialize trading partner
    paper_trading = not args.live
    if args.live:
        logger.warning("⚠️  LIVE TRADING MODE - Real money at risk!")
        confirm = input("Are you sure you want to trade with real money? (yes/no): ")
        if confirm.lower() != 'yes':
            logger.info("Switching to paper trading mode")
            paper_trading = True
    
    # try:
    trader = TradingPartner(paper=paper_trading)
    
    if args.mode == 'account':
        logger.info("📊 Displaying account information...")
        trader.get_account()
        
    elif args.mode == 'trade':
        logger.info("🤖 Starting trading mode...")
        success = trader.start_streaming_bot(args.symbols, args.duration)
        return 0 if success else 1
        
    elif args.mode == 'monitor':
        logger.info("📡 Starting monitoring mode...")
        success = trader.monitor_data_only(crypto_universe[:10], args.duration)
        return 0 if success else 1

    elif args.mode == 'compute-benchmarks':
        # Use provided symbols if given, otherwise default to crypto_universe
        syms = args.symbols if args.symbols else crypto_universe
        logger.info("🧮 Computing pairwise cointegration benchmarks")
        logger.info(f"Symbols: {syms}")
        logger.info(f"Lookback: {args.days} days, time scale: {args.time_scale}, max-groups: {args.max_groups}")

        try:
            payload = compute_benchmarks(
                symbols=syms,
                days_back=args.days,
                time_scale=args.time_scale,
                max_groups=args.max_groups,
            )
            out_path = save_benchmarks(payload, "data/benchmarks")
            logger.info(f"✅ Benchmarks saved: {out_path}")
            return 0
        except Exception as e:
            logger.error(f"❌ Failed to compute benchmarks: {e}")
            return 1
        
    # elif args.mode == 'test':
    #     logger.info("🧪 Running tests...")
    #     success = testing()
    #     return 0 if success else 1
    
    return 0
        
    # except Exception as e:
    #     logger.error(f"❌ Application error: {e}")
    #     return 1

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
