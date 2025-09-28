#! /bin/bash
"""
List of supported exchange codes:
    https://docs.google.com/spreadsheets/d/1I3pBxjfXB056-g_JYf_6o3Rns3BV2kMGG1nCatb91ls/edit?pli=1&gid=0#gid=0

Sam Dawley
08/2025
"""

import datetime
from datetime import date
from pathlib import Path
import os
import requests
import time
from typing import Generator

import pandas as pd
from urllib3.exceptions import MaxRetryError 

from polygon import RESTClient

from . import CRYPTO_TICKERS


class Market:
    """ Class for storing and manipulating market data """

    __slots__ = ("polygon_api_key", "polygon_base_url", "session", "polygon_params", "client")
    

    def __init__(self, polygon_api_key: str = None) -> None:
        """
        Parameters:
        -----------
            polygon_api_key : str
                Polygon API key, available at "https://api.polygon.io"
        """
        if isinstance(polygon_api_key, type(None)):
            self.polygon_api_key = os.getenv("POLYGON_API_KEY")
        else:
            self.polygon_api_key = polygon_api_key

        # print(f"Polygon API secret key: {self.polygon_api_key}", flush=True)

        self.polygon_base_url = "https://api.polygon.io"
        self.session = requests.Session()
        self.polygon_params = {
            "adjusted": "true",  # Adjust for splits and dividends
            "sort": "asc",       # Sort by timestamp ascending
            "limit": 120,      # Limit results
            "apikey": self.polygon_api_key
        }

        self.client = RESTClient(self.polygon_api_key)
        
        return
    

    def __repr__(self) -> str:
        return f"Hello, world!"


    def _format_bars_data(self, results: pd.Series) -> pd.DataFrame:
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



    def _single_ticker_data(
            self,
            tick: str,
            start_date: date | str = "2025-01-01",
            end_date: date | str = "2025-06-01",
            interval: str = "day"
        ) -> pd.DataFrame:
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
            : pd.DataFrame
                ...
        """
        tick = tick.upper()

        if isinstance(end_date, type(None)):
            end_date = datetime.now().isoformat()

        if isinstance(start_date, type(None)):
            start_date = (datetime.now() - timedelta(days=30)).isoformat()

        # ----- collect data -----


        multiplier = 1
        endpoint = f"/v2/aggs/ticker/{tick}/range/{multiplier}/{interval}/{start_date}/{end_date}"
        
        url = self.polygon_base_url + endpoint
        response = self.session.get(url, params=self.polygon_params)

        if response.status_code == 429:
            print("Rate limit hit. Waiting 60 seconds...", flush=True)
            time.sleep(60)
            response = self.session.get(url, params=self.polygon_params)
        
        response.raise_for_status()
        return response.json()
    
    def _multiple_ticker_data(
            self,
            tick: list,
            start_date: date | str = "2025-01-01",
            end_date: date | str = "2025-06-01",
            interval: str = "day"
        ) -> pd.DataFrame:
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
            : pd.DataFrame
                ...
        """
        data = []
        for t in tick:
            tick_history = self._single_ticker_data(t, start_date, end_date, interval)
            results = tick_history.get("results", None)

            if isinstance(results, pd.DataFrame): 
                data.append(self._format_bars_data(tick_history))

            elif isinstance(results, type(None)):
                tick_label = tick_history.get("ticker", None)
                tick_count = tick_history.get("resultsCount", None)
                print(f"Ticker {tick_label} has {tick_count} results", flush=True)


        for d in data:
            print(d)

        return

    def collect_ticker_data(
            self, 
            ticker: str,
            start_date: date | str = "2025-01-01",
            end_date: date | str = "2025-06-01",
            interval: str = "day"
        ) -> pd.DataFrame:
        """
        Collect history of certain ticker 

        Parameters:
            ticker : str
                ticker of stock
            start_date : str
                'YYYY-MM-DD' formatted date
            end_date : str
                'YYYY-MM-DD' formatted date
            interval : str
                interval over which data is recorded
                '1d', '1h', '5m', etc.
        """
        if isinstance(ticker, str):
            return self._single_ticker_data(ticker, start_date, end_date, interval)
        
        elif isinstance(ticker, list):
            return self._multiple_ticker_data(ticker, start_date, end_date, interval)
        
        elif isinstance(ticker, type(Generator)):
            print("generator type")
        
        return None

    def list_market_tickers(self, market: str) -> Generator[str]:
        """
        Generator for listing crypto symbols available with Polygon
        Elements returned by the generator are Ticker() with attributes:
            'active' (True), 'cik', 'composite_figi', 'currency_name',
            'currency_symbol', 'base_currency_symbol', 'base_currency_name',
            'delisted_utc', 'last_updated_utc', 'locale', 'market',
            'name', 'primary_exchange', 'share_class_figi', 'ticker',
            'type', 'source_feed',

        """
        tickers = []

        for t in self.client.list_tickers(
            market=market, #"crypto",
            active="true",
            order="asc",
            limit="100",
            sort="ticker",
        ):
            print(t.ticker)
            # tickers.append(t)
            
            try:
                tickers.append(t)

            except MaxRetryError as e:
                print(f"Caught URLlib Exception: {e}", flush=True)
                return tickers

        



# ==================================================
# AUXILIARY FUNCTIONS
# ==================================================


def display_sample_data(data: pd.DataFrame, num_rows: int=5) -> None:
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


def save_to_csv(data: pd.DataFrame, filename: Path=None) -> None:
    """
    Save data to CSV file
    """
    if data is None or data.empty:
        print("No data to save.", flush=True)
        return
    
    if filename is None:
        timestamp = datetime.datetime.now().isoformat()
        filename = f"GOOGL_polygon_data_{timestamp}.csv"
    
    try:
        data.to_csv(filename)
        print(f"\nData saved to: {filename}", flush=True)

    except Exception as e:
        print(f"Error saving data: {e}", flush=True)

    return


if __name__ == "__main__":

    mark = Market()
    crypto_ticker_generator = mark.list_market_tickers("crypto")
    crypto_tickers = [c.ticker for c in crypto_ticker_generator]

    # data = mark.collect_ticker_data(CRYPTO_TICKERS)
    # data = mark.collect_ticker_data(crypto_tickers)


    # display_sample_data(data)
    # save_to_csv(data)