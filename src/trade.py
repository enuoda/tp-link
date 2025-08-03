"""
Classes used for initialized trader

Sam Dawley
08/2025
"""
import os

from dotenv import load_dotenv

# trading client
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass

# trading framework and algorithms
import backtrader as bt

# ==================================================
# TRADER
# ==================================================

class TradingPartner:

    # ===== Initialization =====

    def __init__(self, api_key: str | None = None, api_secret_key: str | None = None, paper: bool = True) -> None:
        
        if isinstance(api_key, type(None)) or isinstance(api_secret_key, type(None)):
            load_dotenv()
            self.api_key = os.getenv("ALPACA_API_KEY")
            self.api_secret_key = os.getenv("ALPACA_SECRET_KEY")

        else:
            self.api_key = api_key
            self.api_secret_key = api_secret_key

        self.trader = TradingClient(self.api_key, self.api_secret_key, paper=paper)

        return 
    
    def __repr__(self) -> str:
        return f"Trading Partner"
    

    # ===== private functions =====

    # ===== public functions =====

    def get_account(self) -> None:
        for a in self.trader.get_account():
            print(f"{a[0]:<30s}: {a[1]}")

        return
            


if __name__ == "__main__":
    ...