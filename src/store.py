#! /bin/bash

# Sam Dawley
# 08/2025

import os

import yfinance as yf

# ==================================================
# CLASS FOR STORING MARKET DATA
# ==================================================

class Market:
    """
    Class for storing and manipulating market data
    """

    def __init__(self, ticker: str | list = "MSFT") -> None:

        # ----- initialize tickers
        if isinstance(ticker, str):
            self.ticker = ticker
            
        elif isinstance(ticker, list):

            self.tickers = ticker
            self.ticker = self.tickers[0]

        return
    

    def __repr__(self) -> str:
        return f""
    
    # ===== private functions =====

    # ===== public functions =====




if __name__ == "__main__":

    mark = Market()