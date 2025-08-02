"""

"""

# misc
import os
import sys
from typing import Tuple

from dotenv import load_dotenv
import requests

# numerics
import numpy as np

# trading
from alpaca.trading.client import TradingClient


# ----- global/environment variables -----

load_dotenv()
API_KEY = os.environ["ALPACA_API_KEY"] 
API_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]

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
    print(account_data)

    return 0


if __name__ == "__main__":
    main()