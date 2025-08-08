#! /bin/bash

# Sam Dawley
# 08/2025

import os

from dotenv import load_dotenv

# trading client
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass

# trading algorithms
import backtrader as bt
from .algo import SMACrossover, TheStrat
from .store import Market

import yfinance as yf

# ==================================================
# TRADER
# ==================================================

class TradingPartner:
    """
    Class which defines our trader
    """

    load_dotenv()
    apca_api_key = os.getenv("ALPACA_API_KEY")
    apca_api_secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    apca_paper = True

    # ==================================================
    # INITIALIZATION
    # ==================================================

    def __init__(self, **cerebral_kwargs) -> None:
        """
        Instantiates the trinity:
            The father (cerebro), son (trader), and holy spirit (market)

        Parameters:
        -----------
            **cerebral_kwargs : dict
                optional parametrs to load with backtrader.Cerebro
                See https://www.backtrader.com/docu/cerebro/#reference
        """         

        self.cerebro = bt.Cerebro(**cerebral_kwargs)
        self.trader = TradingClient(self.apca_api_key, self.apca_api_secret_key, paper=self.apca_paper)
        self.market = Market()

        dat = yf.Ticker("MSFT")

        return 
    
    def __repr__(self) -> str:
        return f"Trading Partner"
    
    # ==============================
    # DEPARTMENT OF THE INTERIOR
    # ==============================

    # ==============================
    # DEPARTMENT OF THE EXTERIOR
    # ==============================

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
            
    def implement_data(data) -> bool:

        # bt.BacktraderCSVData()

        return

if __name__ == "__main__":


    trad = TradingPartner()
