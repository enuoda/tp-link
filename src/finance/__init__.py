#!/bin/bash
# ===== environment variables =====

import os

from dotenv import load_dotenv

try:
    load_dotenv()

except Exception as e:
    print(f"Caught Exception trying to load '.env'", flush=True)

REQUIRED_VARS = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY"]
MISSING_VARS = [var for var in REQUIRED_VARS if not os.getenv(var)]

if MISSING_VARS:
    raise ValueError

# ===== temporal constants =====

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

NOW = datetime.now(ZoneInfo("America/New_York"))
PAST_N_YEARS = dict((n, timedelta(days=n * 365)) for n in range(10))

TIME_FRAMES = {
    "min": TimeFrame(amount=1, unit=TimeFrameUnit.Minute),
    "hour": TimeFrame(amount=1, unit=TimeFrameUnit.Hour),
    "day": TimeFrame(amount=1, unit=TimeFrameUnit.Day),
    "week": TimeFrame(amount=1, unit=TimeFrameUnit.Week),
    "month": TimeFrame(amount=1, unit=TimeFrameUnit.Month),
}

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
