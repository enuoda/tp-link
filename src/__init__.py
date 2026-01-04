#!/usr/bin/env python3
"""
tp-link: Cryptocurrency Trading Bot

This package provides automated cryptocurrency trading using cointegration-based
spread trading strategies. The exchange is configurable via environment variables.

Configuration:
    EXCHANGE_NAME: Exchange to use (default: 'kraken')
        - 'kraken': Kraken Futures (US-friendly, futures with shorting)
        - 'binance': Binance Futures (international only)
        - 'binanceus': Binance.US (US, spot only)
        - 'bybit': Bybit (international)
    
    {EXCHANGE}_API_KEY: API key for the exchange
    {EXCHANGE}_SECRET_KEY: Secret key for the exchange
    {EXCHANGE}_DEMO: Use demo/testnet mode (default: 'true')

Example .env file for Kraken:
    EXCHANGE_NAME=kraken
    KRAKEN_API_KEY=your_api_key
    KRAKEN_SECRET_KEY=your_secret_key
    KRAKEN_DEMO=true

See EXCHANGE_CONFIG.md for detailed documentation on switching exchanges.
"""

import os

from dotenv import load_dotenv

# Load environment variables from .env file
try:
    load_dotenv()
except Exception as e:
    print(f"Warning: Could not load '.env' file: {e}", flush=True)

# Get the configured exchange (default to Kraken for US users)
EXCHANGE_NAME = os.getenv("EXCHANGE_NAME", "kraken").lower()

# Build environment variable names based on exchange
_exchange_upper = EXCHANGE_NAME.upper()
API_KEY_VAR = f"{_exchange_upper}_API_KEY"
SECRET_KEY_VAR = f"{_exchange_upper}_SECRET_KEY"
DEMO_VAR = f"{_exchange_upper}_DEMO"

# Validate required environment variables
REQUIRED_VARS = [API_KEY_VAR, SECRET_KEY_VAR]
MISSING_VARS = [var for var in REQUIRED_VARS if not os.getenv(var)]

if MISSING_VARS:
    raise ValueError(
        f"\n❌ Missing required environment variables for {EXCHANGE_NAME}:\n"
        f"   {MISSING_VARS}\n\n"
        f"Please set the following in your .env file or environment:\n"
        f"   export EXCHANGE_NAME={EXCHANGE_NAME}\n"
        f"   export {API_KEY_VAR}='your_api_key'\n"
        f"   export {SECRET_KEY_VAR}='your_secret_key'\n"
        f"   export {DEMO_VAR}='true'  # Optional, defaults to true\n\n"
        f"See EXCHANGE_CONFIG.md for more information."
    )

print(f"🔧 Configured for {EXCHANGE_NAME.upper()} exchange")
