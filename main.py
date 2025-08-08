"""
Primary subroutines

Sam Dawley
08/2025
"""

from src.trade import TradingPartner

# ==================================================
# MAIN ROUTINES
# ================================================== 


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