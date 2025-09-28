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
    raise ValueError(f"Missing required environment variables: {MISSING_VARS}", flush=True)

# ===== tickers =====

CRYPTO_TICKERS = [
    "BTC",
    "ETH",
    "USDT",
    "XRP",
    "BNB",
    "SOL",
    "USDC",
    "DOGE",
    "STETH",
    "TRX",
    "WTRX",
    "ADA",
    "WSTETH",
    "HYPE32196",
    "USDE29470",
    "LINK",
    "WBETH",
    "WBTC",
    "WETH",
    "AVAX",
    "XLM",
    "SUI20947",
    "BCH",
    "WEETH",
    "AETHWETH"   
]
