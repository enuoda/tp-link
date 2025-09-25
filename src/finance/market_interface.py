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

import pandas as pd

from polygon import RESTClient


class Market:
    """ Class for storing and manipulating market data """
    

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

        print(f"Polygon API secret key: {self.polygon_api_key}", flush=True)

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
            print("Rate limit hit. Waiting 60 seconds...", flush=True)
            time.sleep(60)
            response = self.session.get(url, params=self.polygon_params)
        
        response.raise_for_status()
        data = response.json()
        df = self._format_bars_data(data["results"])
        
        return df



    def collect_crypto_data(self):
        tickers = []
        for t in self.client.list_tickers(
            market="crypto",
            active="true",
            order="asc",
            limit="100",
            sort="ticker",
        ):
            tickers.append(t)
        print(tickers)

        return 
        



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
    # data = mark.collect_ticker_data("GOOGL")
    # display_sample_data(data)
    # save_to_csv(data)

    mark.collect_crypto_data()