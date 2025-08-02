"""
Primary subroutines

Sam Dawley
08/2025
"""

# misc
import os
import sys
from typing import Tuple

from dotenv import load_dotenv

# numerics
import numpy as np

# trading
from alpaca.trading.client import TradingClient


# ----- global/environment variables -----

load_dotenv() # defaults to ./.env
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# ==================================================
# MAIN ROUTINES
# ================================================== 


def main() -> int:
    """
    DOCSTRING
    """

    # ----- trading client -----

    trade_client = TradingClient(api_key=API_KEY, secret_key=API_SECRET_KEY, paper=True)
    account_data = trade_client.get_account()
    print('${} is available as buying power.'.format(account_data.buying_power))

    return 0


if __name__ == "__main__":
    main()