"""
Primary subroutines

Sam Dawley
08/2025
"""


from src.store import Market
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

    mark = Market()
    print(mark.get_company_profile(symbol="APPL"))



    return 0


if __name__ == "__main__":
    main()