#! /bin/bash
"""
List of supported exchange codes:
    https://docs.google.com/spreadsheets/d/1I3pBxjfXB056-g_JYf_6o3Rns3BV2kMGG1nCatb91ls/edit?pli=1&gid=0#gid=0

Sam Dawley
08/2025
"""

from collections import namedtuple
from contextlib import nullcontext
from dataclasses import dataclass
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

PolygonAddress = namedtuple(
    "PolygonAddress", ["address1", "address2", "city", "postal_code", "state"]
)
PolygonBranding = namedtuple("PolygonBranding", ["icon_url", "logo_url"])


@dataclass
class PolygonTicker:
    """See: https://polygon.io/docs/rest/stocks/tickers/ticker-overview"""

    active: bool
    address: PolygonAddress
    branding: PolygonBranding
    cik: str
    currency_name: str
    delisted_utc: str
    description: str
    homepage_url: str
    list_date: str
    locale: list
    market: list
    market_cap: float
    name: str
    phone_number: str
    primary_exchange: str
    round_lot: str
    share_class_figi: str
    share_class_shares_outstanding: float
    sic_code: str
    sic_description: str
    ticker: str
    ticker_root: str
    ticker_suffix: str
    total_employees: int
    type: str
    weighted_shares_outstanding: float


class Market:
    """Class for storing and manipulating market data"""

    __slots__ = (
        "polygon_api_key",
        "polygon_base_url",
        "session",
        "polygon_params",
        "client",
    )

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

        self.client = RESTClient(self.polygon_api_key)
        self.polygon_base_url = "https://api.polygon.io"
        self.session = requests.Session()
        self.polygon_params = {
            "adjusted": "true",  # Adjust for splits and dividends
            "sort": "asc",  # Sort by timestamp ascending
            "limit": 120,  # Limit results
            "apikey": self.polygon_api_key,
        }

        return

    def _polygon_to_df(self, results) -> pd.DataFrame:
        """
        Convert Polygon API results to pandas DataFrame

        Parameters:
            results
                return value of self._single_ticker_data()
        """
        df = pd.DataFrame(results)

        # timestamp (milliseconds) to datetime
        df["datetime"] = pd.to_datetime(df["t"], unit="ms")
        df.set_index("datetime", inplace=True)

        # ----- standard OHLCV format -----

        column_mapping = {
            "o": "Open",
            "h": "High",
            "l": "Low",
            "c": "Close",
            "v": "Volume",
            "vw": "VWAP",  # Volume Weighted Average Price
            "n": "Transactions",  # Number of transactions
        }

        df.rename(columns=column_mapping, inplace=True)

        # Keep only the columns we want
        columns_to_keep = ["Open", "High", "Low", "Close", "Volume"]

        if "VWAP" in df.columns:
            columns_to_keep.append("VWAP")

        if "Transactions" in df.columns:
            columns_to_keep.append("Transactions")

        df = df[columns_to_keep]

        return df

    def _single_ticker_data(
        self,
        tick: str,
        start_date: date | str = "2025-01-01",
        end_date: date | str = "2025-06-01",
        interval: str = "day",
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
        interval: str = "day",
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

        return

    def collect_ticker_data(
        self,
        ticker: str,
        start_date: date | str = "2025-01-01",
        end_date: date | str = "2025-06-01",
        interval: str = "day",
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

    def list_market_tickers(self, market: str, outfile: str = None) -> Generator[str]:
        """
        Generator for listing crypto symbols available with Polygon
        Elements returned by the generator are Ticker() with attributes:
            'active' (True), 'cik', 'composite_figi', 'currency_name',
            'currency_symbol', 'base_currency_symbol', 'base_currency_name',
            'delisted_utc', 'last_updated_utc', 'locale', 'market',
            'name', 'primary_exchange', 'share_class_figi', 'ticker',
            'type', 'source_feed',

        """

        ticker_attributes = [vk for vk in vars(PolygonTicker).values()][-1]
        header = "# " + ",".join(ticker_attributes) + "\n"

        df = pd.DataFrame(columns=ticker_attributes)

        # if isinstance(outfile, type(None)):
        #     cm = nullcontext(outfile)
        # else:
        #     cm = open(outfile, "w")

        # with cm as f:
        # f.write(header)

        details = self.client.get_ticker_details("BTC")

        print(details)
        return

        count = 0
        ticker_data = []
        for t in self.client.list_tickers(
            market=market, active="true", sort="ticker"
        ):  # "crypto",
            print(t)
            ticker_data.append(vars(t))
            # print(vars(t).keys())

            count += 1

            if count > 5:
                break

        df = pd.DataFrame(ticker_data)
        print(df[["ticker"]])
        return df

    def list_crypto_tickers_by_price(self, limit: int = None) -> pd.DataFrame:
        """
        List all crypto tickers and sort by current price (highest to lowest)

        Args:
            limit: Optional limit on number of tickers to process (useful for testing)

        Returns:
            DataFrame with crypto tickers sorted by price
        """

        # Step 1: Get all crypto tickers
        print("Fetching crypto tickers...")
        ticker_data = []
        count = 0

        for ticker in self.client.list_tickers(
            market="crypto",
            active="true",
            order="asc",
            limit=1000,  # Max per request
            sort="ticker",
        ):
            ticker_info = {
                "ticker": ticker.ticker,
                "name": ticker.name,
                "currency_symbol": getattr(ticker, "currency_symbol", None),
                "base_currency_symbol": getattr(ticker, "base_currency_symbol", None),
                "market": ticker.market,
                "active": ticker.active,
                "type": ticker.type,
            }
            ticker_data.append(ticker_info)
            count += 1

            # Optional limit for testing
            if limit and count >= limit:
                break

        print(f"Found {len(ticker_data)} crypto tickers")

        # Step 2: Get price data for each ticker
        print("Fetching price data...")
        prices_data = []

        for i, ticker_info in enumerate(ticker_data):
            ticker_symbol = ticker_info["ticker"]

            try:
                # Method 1: Get last trade (most current price)
                last_trade = self.client.get_last_trade(ticker_symbol)
                if last_trade:
                    price = last_trade.price
                    timestamp = last_trade.timestamp
                else:
                    # Method 2: Fallback to previous close
                    prev_close = self.client.get_previous_close(ticker_symbol)
                    if prev_close and len(prev_close) > 0:
                        price = prev_close[0].close
                        timestamp = prev_close[0].timestamp
                    else:
                        price = None
                        timestamp = None

                ticker_info["price"] = price
                ticker_info["last_updated"] = timestamp
                prices_data.append(ticker_info)

            except Exception as e:
                print(f"Error fetching price for {ticker_symbol}: {e}")
                ticker_info["price"] = None
                ticker_info["last_updated"] = None
                prices_data.append(ticker_info)

            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(ticker_data)} tickers...")

            # Rate limiting - Polygon has API rate limits
            time.sleep(0.1)  # Adjust based on your plan limits

        # Step 3: Create DataFrame and sort by price
        df = pd.DataFrame(prices_data)

        # Remove tickers without price data
        df_with_prices = df.dropna(subset=["price"])

        # Sort by price (highest first)
        df_sorted = df_with_prices.sort_values("price", ascending=False)

        print(f"\nTop 10 crypto tickers by price:")
        with pd.option_context("display.max_columns", 8, "display.width", None):
            print(df_sorted.head(10)[["ticker", "name", "price", "currency_symbol"]])

        return df_sorted

    def list_crypto_tickers_by_prev_close(self, limit: int = None) -> pd.DataFrame:
        """
        List crypto tickers sorted by previous close price
        This method is often more reliable as it uses aggregated daily data
        """

        print("Fetching crypto tickers...")
        ticker_data = []
        count = 0

        for ticker in self.client.list_tickers(
            market="crypto", active="true", order="asc", limit=1000, sort="ticker"
        ):
            ticker_data.append(
                {
                    "ticker": ticker.ticker,
                    "name": ticker.name,
                    "currency_symbol": getattr(ticker, "currency_symbol", None),
                    "base_currency_symbol": getattr(ticker, "base_currency_symbol", None),
                }
            )
            count += 1
            if limit and count >= limit:
                break

        print(f"Fetching previous close data for {len(ticker_data)} tickers...")

        # Get previous close for all tickers in batch if possible
        prices_data = []
        for i, ticker_info in enumerate(ticker_data):
            try:
                prev_close = self.client.get_previous_close(ticker_info["ticker"])
                if prev_close and len(prev_close) > 0:
                    ticker_info.update(
                        {
                            "close_price": prev_close[0].close,
                            "volume": prev_close[0].volume,
                            "timestamp": prev_close[0].timestamp,
                        }
                    )
                else:
                    ticker_info.update(
                        {"close_price": None, "volume": None, "timestamp": None}
                    )
                prices_data.append(ticker_info)

            except Exception as e:
                print(f"Error for {ticker_info['ticker']}: {e}")
                ticker_info.update(
                    {"close_price": None, "volume": None, "timestamp": None}
                )
                prices_data.append(ticker_info)

            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(ticker_data)} tickers...")

            time.sleep(0.05)  # Rate limiting

        df = pd.DataFrame(prices_data)
        df_with_prices = df.dropna(subset=["close_price"])
        df_sorted = df_with_prices.sort_values("close_price", ascending=False)

        print(f"\nTop 10 crypto tickers by previous close price:")
        with pd.option_context("display.max_columns", 6):
            print(df_sorted.head(10)[["ticker", "name", "close_price", "volume"]])

        return df_sorted


# ==================================================
# AUXILIARY FUNCTIONS
# ==================================================


def save_to_csv(data: pd.DataFrame, filename: Path = None) -> None:
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
    mark.list_crypto_tickers_by_prev_close()

    # crypto_ticker_generator = mark.list_market_tickers("crypto", "tmp.txt")
    # crypto_tickers = [c.ticker for c in crypto_ticker_generator]

    # single_ticker_data = mark.collect_ticker_data("GOOGL")
    # data = mark._polygon_to_df(single_ticker_data)

    # display_sample_data(data)
    # save_to_csv(data)
