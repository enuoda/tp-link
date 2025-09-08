#!/bin/bash

# Sam Dawley
# 08/2025

import os
from datetime import date
import requests
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

    load_dotenv()

    apca_api_key = os.getenv("ALPACA_API_KEY")
    apca_api_secret_key = os.getenv("ALPACA_SECRET_KEY")
    apca_paper = True

    polygon_api_key = os.getenv("POLYGON_API_KEY")
    polygon_base_url = os.getenv("POLYGON_BASE_URL")
    
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
        
        # for use with alpaca
        self.trader = TradingClient(self.apca_api_key, self.apca_api_secret_key, paper=self.apca_paper)
        
        # for use with polygon
        self.session = requests.Session()
        self.polygon_params = {
            "adjusted": "true",  # Adjust for splits and dividends
            "sort": "asc",       # Sort by timestamp ascending
            "limit": 120,      # Limit results
            "apikey": self.polygon_api_key
        }

        self.market = Market()

        return 
    
    def __repr__(self) -> str:
        return f"Trading Partner"
    
    # ==============================
    # DEPARTMENT OF THE INTERIOR
    # ==============================

    def _format_bars_data(self, results):
        """
        Convert Polygon API results to pandas DataFrame
        """
        # Create DataFrame from results
        df = pd.DataFrame(results)
        
        # Convert timestamp (milliseconds) to datetime
        df['datetime'] = pd.to_datetime(df['t'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        # Rename columns to standard OHLCV format
        column_mapping = {
            'o': 'Open',
            'h': 'High',
            'l': 'Low',
            'c': 'Close',
            'v': 'Volume',
            'vw': 'VWAP',  # Volume Weighted Average Price
            'n': 'Transactions'  # Number of transactions
        }
        
        df.rename(columns=column_mapping, inplace=True)
        
        # Keep only the columns we want
        columns_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
        if 'VWAP' in df.columns:
            columns_to_keep.append('VWAP')
        if 'Transactions' in df.columns:
            columns_to_keep.append('Transactions')
        
        df = df[columns_to_keep]
        
        return df

    # ==============================
    # DEPARTMENT OF THE EXTERIOR
    # ==============================

    def get_account(self) -> None:
        for a in self.trader.get_account():
            print(f"{a[0]:<30s}: {a[1]}")

        return
    
    
    def collect_ticker_data(
            self, 
            ticker: str,
            start_date: date | str = "2025-01-01",
            end_date: date | str = "2025-06-01",
            interval: str = "day"
        ):
        """
        Collect history of certain ticker 

        Parameters:
        -----------
            ticker : str
                ticker of stock
            start_date : str
                'YYYY-MM-DD' formatted date
            end_date : str
                'YYYY-MM-DD' formatted date
            interval : str
                interval over which data is recorded
                '1d', '1h', '5m', etc.
        
        Returns:
        --------
            : yf.Ticker.history
        """

        # ----- metadata -----

        ticker = ticker.upper()

        if isinstance(end_date, type(None)):
            end_date = datetime.now().isoformat()

        if isinstance(start_date, type(None)):
            start_date = (datetime.now() - timedelta(days=30)).isoformat()

        # ----- collect data -----

        multiplier = 1
        endpoint = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{interval}/{start_date}/{end_date}"
        
        url = self.polygon_base_url + endpoint
        response = self.session.get(url, params=self.polygon_params)

        if response.status_code == 429:
            print("Rate limit hit. Waiting 60 seconds...")
            time.sleep(60)
            response = self.session.get(url, params=self.polygon_params)
        
        response.raise_for_status()
        data = response.json()
        df = self._format_bars_data(data['results'])

        # tick = yf.Ticker(ticker)
        # data = tick.history(start=start_date, end=end_date, interval=interval)

        # if data.empty:
        #     print(f"No data found for {ticker} between {start_date} and {end_date}.", flush=True)
        #     return None
        
        return df


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

def display_sample_data(data, num_rows=5):
    """
    Display sample of the collected data
    """
    if isinstance(data, type(None)):
        print("No data to display.")
        return
    
    print(f"\n=== First {num_rows} Records ===")
    print(data.head(num_rows).round(2))
    
    if len(data) > num_rows * 2:
        print(f"\n=== Last {num_rows} Records ===")
        print(data.tail(num_rows).round(2))

    return

if __name__ == "__main__":

    tp = TradingPartner()
    data = tp.collect_ticker_data("GOOGL")

    display_sample_data(data)
