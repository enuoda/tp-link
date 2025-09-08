#!/bin/bash

# Sam Dawley
# 08/2025

import datetime
from datetime import date
from pathlib import Path
import os
import requests
import sys
import time

from dotenv import load_dotenv
import pandas as pd

# trading client
from alpaca.trading.client import TradingClient

# trading algorithms
import backtrader as bt
# from algo import SMACrossover, TheStrat
from store import Market

import yfinance as yf


# ==================================================
# TRADER
# ==================================================

class TradingPartner:
    """ An interesting docstring """

    try:
        load_dotenv()
    except Exception as e:
        print(f"Caught Execption trying to load .env file: {e}", flush=True)
        sys.exit(1)
    
    # ===== INITIALIZATION =====

    def __init__(
            self,
            alpaca_api_key: str = None,
            alpaca_secret_key: str = None,
            alpaca_base_url: str = None,
            **cerebral_kwargs
        ) -> None:
        """
        Instantiates the trinity:
            The father (cerebro), son (trader), and holy spirit (market)

        Parameters:
        -----------
            polygon_api_key : str
                Polygon API key
            polygon_base_url : str
                Polygon base url
            **cerebral_kwargs : dict
                optional parametrs to load with backtrader.Cerebro
                See https://www.backtrader.com/docu/cerebro/#reference
        """

        if isinstance(alpaca_api_key, type(None)):
            try:
                load_dotenv()
                self.alpaca_api_key = os.getenv("ALPACA_API_KEY")
            except Exception as e:
                print(f"Caught Exception trying to load '.env': {e}", flush=True)
        else:
            self.alpaca_api_key = alpaca_api_key

        if isinstance(alpaca_secret_key, type(None)):
            try:
                load_dotenv()
                self.alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY")
            except Exception as e:
                print(f"Caught Exception trying to load '.env': {e}", flush=True)
        else:
            self.alpaca_secret_key = alpaca_secret_key

        if isinstance(alpaca_base_url, type(None)):
            self.alpaca_base_url = "https://paper-api.alpaca.markets"
            self.alpaca_paper = True
        else:
            self.alpaca_base_url = alpaca_base_url

        self.cerebro = bt.Cerebro(**cerebral_kwargs)

        self.trader = TradingClient(self.alpaca_api_key, self.alpaca_secret_key, paper=self.alpaca_paper)


        self.market = Market()

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
    
# ==================================================
# Auxiliary Functions
# ==================================================




if __name__ == "__main__":

    tp = TradingPartner()
    data = tp.collect_ticker_data("GOOGL")

    display_sample_data(data)
