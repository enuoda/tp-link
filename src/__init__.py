#!/bin/bash

import os
import sys

from dotenv import load_dotenv

from src.finance import algo, store

try:
    load_dotenv()

except Exception as e:

    print(f"Caught Exception trying to load '.env'", flush=True)
    sys.exit(1)

REQUIRED_VARS = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY", "POLYGON_API_KEY"]
MISSING_VARS = [var for var in REQUIRED_VARS if not os.getenv(var)]

if MISSING_VARS:
    raise ValueError(f"Missing required environment variables: {MISSING_VARS}", flush=True)
