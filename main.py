#!/bin/bash
"""
Primary subroutines

Sam Dawley
08/2025
"""

from src.finance.trade import TradingPartner

# ==================================================
# MAIN ROUTINES
# ==================================================


def testing():

    from polygon import RESTClient

    client = RESTClient("qadVu6Vm9lfDpYrHy0e4xNG0wAYrnLQq")

    types = client.get_ticker_types(asset_class="crypto", locale="us")

    print(types)

    return


def main() -> int:
    """
    DOCSTRING
    """

    # ----- trading client -----

    trader = TradingPartner()
    # trader.get_account()

    trader.market.get_company_profile("APPL")

    return 0


if __name__ == "__main__":
    main()
