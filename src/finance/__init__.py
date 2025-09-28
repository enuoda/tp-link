#!/bin/bash
import os
import sys

from dotenv import load_dotenv


# ===== environment variables =====

try:
    load_dotenv()

except Exception as e:

    print(f"Caught Exception trying to load '.env'", flush=True)
    sys.exit(1)

REQUIRED_VARS = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY", "POLYGON_API_KEY"]
MISSING_VARS = [var for var in REQUIRED_VARS if not os.getenv(var)]

if MISSING_VARS:
    raise ValueError(
        f"Missing required environment variables: {MISSING_VARS}", flush=True
    )

# ===== tickers =====

CRYPTO_TICKERS = [
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

BTC_PAIRS = ["BCH/USD", "ETH/USD", "LTC/USD", "UNI/USD"]
USDT_PAIRS = [
    "AAVE/USD",
    "BCH/USD",
    "BTC/USD",
    "DOGE/USD",
    "ETH/USD",
    "LINK/USD",
    "LTC/USD",
    "SUSHI/USD",
    "UNI/USD",
    "YFI/USD",
]
USDC_PAIRS = [
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
    "XTZ/USD",
    "YFI/USD",
]
USD_PAIRS = [
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
