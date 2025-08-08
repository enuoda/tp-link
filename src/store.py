#! /bin/bash

# Sam Dawley
# 08/2025

import os
import websocket

import pandas as pd

from dotenv import load_dotenv

# import finnhub
# from finnhub.exceptions import FinnhubAPIException

import alpaca

class Market:
    """
    Class for storing and manipulating market data

    Finnhub utilizes JSON files and so nearly all arguments and
    return values of the finnhub class methods are strings

    References:
    -----------
        - List of supported exchange codes:
            https://docs.google.com/spreadsheets/d/1I3pBxjfXB056-g_JYf_6o3Rns3BV2kMGG1nCatb91ls/edit?pli=1&gid=0#gid=0
    """

    # ----- authentication -----

    # load_dotenv()
    # finn_api_key = os.getenv("FINNHUB_API_KEY")
    # finn_api_secret_key = os.getenv("FINNHUB_SECRET_KEY")

    # ==================================================
    # INITIALIZATION
    # ==================================================
    
    def __init__(self) -> None:

        # self.client = finnhub.Client(api_key=self.finn_api_key)
        
        return
    

    def __repr__(self) -> str:
        return f""
    
    # ==============================
    # DEPARTMENT OF THE INTERIOR
    # ==============================
    def _get_stock_symbols(self, exchange: str, mic: str = None, security_type: str = None, currency: str = None) -> list:
        """
        Returns:
        --------
            : list
                list of supported stocks
        
        References:
        -----------
            For attributes of return value, see 'Stock Symbol':
                https://finnhub.io/docs/api/stock-symbols
        """
        # return self.client.stock_symbols(exchange, mic=mic, security_type=security_type, currency=currency)
        return
    
    # ==============================
    # DEPARTMENT OF THE EXTERIOR
    # ==============================

    def get_company_profile(self, symbol: str, isin: str = None, cusip: str = None) -> None:
        """
        DOCSTRING
        """
        # try:
        #     return self.client.company_profile(symbol=symbol, isin=isin, cusip=cusip)

        # except FinnhubAPIException:
        #     return self.client.company_profile2(symbol=symbol, isin=isin, cusip=cusip)
        return

    def stream_trades(self) -> None:
        """
        DOCSTRING
        """
        return

if __name__ == "__main__":

    mark = Market()