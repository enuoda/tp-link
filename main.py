"""
Primary subroutines

Sam Dawley
08/2025
"""


from src.trade import TradeConfig

# ==================================================
# MAIN ROUTINES
# ================================================== 


def main() -> int:
    """
    DOCSTRING
    """

    # ----- trading client -----

    trader = TradeConfig()
    trader.get_account()



    return 0


if __name__ == "__main__":
    main()