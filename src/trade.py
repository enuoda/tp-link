#! /bin/bash

# Sam Dawley
# 08/2025

import os

from dotenv import load_dotenv

# trading client
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass

# brain of trader 
import backtrader as bt
import yfinance as yf

# trading framework and algorithms
from src.algo import SimpleMovingAverage

# ==================================================
# TRADER
# ==================================================

class TradingPartner:
    """
    Class which defines our trader
    """
    ticker = "MSFT"

    def __init__(
            self,
            apca_api_key: str | None = None,
            apca_api_secret_key: str | None = None,
            apca_paper: bool = True,
            **cerebral_kwargs
        ) -> None:
        """
        Parameters:
        -----------
            api_key : str
                public API key for use with Alpaca
                Assumed that this is stored in tp-link/.env
            api_secret_key : str
                secret API key for use with Alpaca
                Assumed that this is stored in tp-link/.env
            paper : bool
                whether or not use to paper-market with Alpaca
            **cerebral_kwargs : dict
                optional parametrs to load with backtrader.Cerebro
                See https://www.backtrader.com/docu/cerebro/#reference

        """
        
        # ----- brain of trading client ----- 

        self.cerebro = bt.Cerebro(**cerebral_kwargs)

        # ----- body of trading client -----

        if isinstance(apca_api_key, type(None)) or isinstance(apca_api_secret_key, type(None)):
            load_dotenv()
            self.api_key = os.getenv("ALPACA_API_KEY")
            self.api_secret_key = os.getenv("ALPACA_SECRET_KEY")

        else:
            self.api_key = apca_api_key
            self.api_secret_key = apca_api_secret_key

        self.trader = TradingClient(self.api_key, self.api_secret_key, paper=apca_paper)

        # ----- loading market data -----

        self.dat = yf.Ticker(self.ticker)
        print(self.dat.actions)


        return 
    
    def __repr__(self) -> str:
        return f"Trading Partner"
    

    # ===== private functions =====


    # ===== public functions =====


    def get_account(self) -> None:
        for a in self.trader.get_account():
            print(f"{a[0]:<30s}: {a[1]}")

        return


    def implement_strategy(self, strat: bt.Strategy, *args, **kwargs) -> int:
        """
        Parameters:
        -----------
            strat : backtrader.Strategy
                ...
        
        Returns:
        --------
            : int
                Index with which addition of other objects can be referenced
        """
        return self.cerebro.addstrategy(strat, *args, **kwargs)
            


if __name__ == "__main__":


    trad = TradingPartner()
