#!/usr/bin/env python3
"""
Cryptocurrency Data Retrieval via CCXT

This module provides functions for fetching historical cryptocurrency data
using the CCXT library. The exchange is configurable via environment variables.

The trading interface (streaming, order execution) is in ccxt_trader.py.

Supported Exchanges:
    - Kraken Futures (default for US users)
    - Binance Futures (international)
    - Binance.US (US spot only)
    - Bybit (international)

Configuration:
    Set EXCHANGE_NAME environment variable to switch exchanges.
    See src/finance/__init__.py for full configuration details.

References:
    CCXT Documentation: https://docs.ccxt.com/

Sam Dawley
"""
from datetime import datetime, timedelta
from typing import List, Tuple, Union
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import ccxt

from . import (
    CRYPTO_TICKERS,
    # TIME_FRAMES,
    EXCHANGE_NAME,
    EXCHANGE_CONFIG,
    API_KEY,
    SECRET_KEY,
    DEMO_MODE,
)


# ==================================================
# CCXT CLIENT UTILITIES
# ==================================================


def _get_exchange_client():
    """
    Create and return a CCXT exchange client based on configuration.
    
    Uses the EXCHANGE_NAME environment variable to determine which
    exchange to connect to. Credentials are loaded from environment
    variables named {EXCHANGE}_API_KEY and {EXCHANGE}_SECRET_KEY.
    
    Returns:
        ccxt.Exchange: Configured exchange client
        
    Raises:
        ValueError: If exchange is not supported
        
    Example:
        >>> client = _get_exchange_client()
        >>> client.fetch_ticker('BTC/USD:USD')
    """
    ccxt_id = EXCHANGE_CONFIG["ccxt_id"]
    
    if not hasattr(ccxt, ccxt_id):
        raise ValueError(f"CCXT does not support exchange: {ccxt_id}")
    
    exchange_class = getattr(ccxt, ccxt_id)
    
    # ----- build configuration -----
    config = {"enableRateLimit": True}
    
    if API_KEY and SECRET_KEY:
        config["apiKey"] = API_KEY
        config["secret"] = SECRET_KEY
    
    if EXCHANGE_NAME == "binance":
        config["options"] = {'defaultType': 'future'}

    elif EXCHANGE_NAME == "bybit":
        config["options"] = {'defaultType': 'swap'}
    
    exchange = exchange_class(config)
    
    # ----- control paper trading versus live trading -----
    if DEMO_MODE and API_KEY and SECRET_KEY:
        try:
            exchange.set_sandbox_mode(True)
            print(f"📋 Using {EXCHANGE_CONFIG['name']} DEMO/TESTNET mode (paper trading)")

        except Exception as e:
            print(f"⚠️ Could not enable sandbox mode: {e}")
    
    return exchange


def _get_max_limit() -> int:
    """Get the maximum OHLCV limit for the configured exchange."""
    limits = {
        "kraken": 720,
        "binance": 1500,
        "binanceus": 1000,
        "bybit": 1000,
    }
    return limits.get(EXCHANGE_NAME, 500)


def _normalize_timeframe(frequency: str) -> str:
    """
    Normalize timeframe string to CCXT format.
    
    Args:
        frequency: Timeframe string (e.g., '1h', 'hour', '1m', 'min')
        
    Returns:
        CCXT-compatible timeframe string (e.g., '1h', '1m', '1d')
    """
    # If already in CCXT format, return as-is
    if frequency in ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']:
        return frequency
    
    # Map common aliases
    freq_lower = frequency.lower().strip()
    
    if freq_lower in ['min', 'minute', 'minutes', 'm']:
        return '1m'
    if freq_lower in ['5min', '5m', '5minutes']:
        return '5m'
    if freq_lower in ['15min', '15m', '15minutes']:
        return '15m'
    if freq_lower in ['hour', 'hours', 'h', '1hour']:
        return '1h'
    if freq_lower in ['4hour', '4h', '4hours']:
        return '4h'
    if freq_lower in ['day', 'days', 'd', '1day']:
        return '1d'
    if freq_lower in ['week', 'weeks', 'w', '1week']:
        return '1w'
    if freq_lower in ['month', 'months', '1month']:
        return '1M'
    
    # Default to 1h if unknown
    print(f"⚠️ Unknown timeframe '{frequency}', defaulting to '1h'")
    return '1h'


def _timeframe_to_seconds(timeframe: str) -> int:
    """Convert CCXT timeframe to seconds."""
    tf = _normalize_timeframe(timeframe)
    multipliers = {
        'm': 60,
        'h': 3600,
        'd': 86400,
        'w': 604800,
        'M': 2592000,  # ~30 days
    }

    amount = int(tf[:-1]) if tf[:-1].isdigit() else 1
    return amount * multipliers.get(tf[-1], 3600)


# ==================================================
# DATA RETRIEVAL FUNCTIONS
# ==================================================


def fetch_crypto_data_for_cointegration(
    symbols: List[str],
    days_back: int = 7,
    frequency: str = "1h",
) -> Tuple[List[np.ndarray], List[str], pd.DatetimeIndex]:
    """
    Fetch crypto data for cointegration analysis with proper time alignment.
    
    Uses the configured exchange via CCXT for data retrieval.
    
    Parameters:
    -----------
    symbols : List[str]
        List of crypto symbols to fetch (format depends on exchange)
        - Kraken: ['BTC/USD:USD', 'ETH/USD:USD']
        - Binance: ['BTC/USDT:USDT', 'ETH/USDT:USDT']
    days_back : int
        Number of days back to fetch data (default: 7)
    frequency : str
        Data frequency - CCXT timeframe string (default: '1h')
        
    Returns:
    --------
    price_arrays : List[np.ndarray]
        List of numpy arrays containing close prices for each symbol (time-aligned)
    sorted_symbols : List[str]
        List of symbols in the same order as price_arrays
    timestamps : pd.DatetimeIndex
        Common timestamps for all symbols
    """
    end_time = datetime.now(ZoneInfo("UTC"))
    start_time = end_time - timedelta(days=days_back)
    
    df, ohlcv, sorted_symbols = retrieve_crypto_data(
        symbols=symbols,
        start_time=start_time,
        end_time=end_time,
        frequency=frequency,
        limit=None
    )
    
    if df.empty:
        return [], [], pd.DatetimeIndex([])
    
    # ----- find common timestamps across all symbols -----
    all_timestamps = set()
    for symbol in sorted_symbols:
        try:
            symbol_timestamps = df.xs(symbol, level="symbol").index
            all_timestamps.update(symbol_timestamps)
        except KeyError:
            continue
    
    # ----- sort timestamps and create common time index -----
    common_timestamps = pd.DatetimeIndex(sorted(all_timestamps))
    
    price_arrays = []
    for symbol in sorted_symbols:
        try:
            symbol_data = df.xs(symbol, level="symbol")

            # ----- reindex to common timestamps without forward-filling; leave NaNs for missing points -----
            aligned_data = symbol_data.reindex(common_timestamps)
            close_prices = aligned_data['close'].values
            price_arrays.append(close_prices)

        except KeyError:
            price_arrays.append(np.full(len(common_timestamps), np.nan))
    
    return price_arrays, list(sorted_symbols), common_timestamps


def retrieve_crypto_data(
    symbols: Union[str, List[str]],
    start_time: datetime,
    end_time: datetime,
    frequency: str = "1h",
    limit: int = None,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Retrieve crypto data from the configured exchange via CCXT.

    Args:
        symbols: Single symbol or list of symbols (format depends on exchange)
        start_time: Start datetime for data retrieval
        end_time: End datetime for data retrieval
        frequency: CCXT timeframe string (e.g., '1h', '1m', '1d')
        limit: Maximum number of bars to fetch per symbol (None for auto)

    Returns:
        df : pd.DataFrame
            Multi-indexed DataFrame with columns: ['open', 'high', 'low', 'close', 'volume']
            Index: MultiIndex of (symbol, timestamp)
        ohlcv : np.ndarray, shape (n_symbols, n_columns, n_times)
            Symbols indexed in order of 'sorted_symbols'
        sorted_symbols : List[str]
            Symbols in DataFrame order
    """
    if isinstance(symbols, str):
        symbols = [symbols]
    
    exchange = _get_exchange_client()
    tf = _normalize_timeframe(frequency)
    max_limit = _get_max_limit()

    if limit is None:
        tf_seconds = _timeframe_to_seconds(tf)
        duration_seconds = (end_time - start_time).total_seconds()
        limit = int(duration_seconds / tf_seconds) + 10  # add buffer
        limit = min(limit, max_limit)
    
    since_ms = int(start_time.timestamp() * 1000)
    
    all_data = []
    successful_symbols = []

    for symbol in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=tf,
                since=since_ms,
                limit=limit,
            )
            
            if not ohlcv:
                print(f"⚠️ No data returned for {symbol}")
                continue
            
            df_symbol = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df_symbol['timestamp'] = pd.to_datetime(df_symbol['timestamp'], unit='ms', utc=True)
            df_symbol['symbol'] = symbol
            df_symbol.set_index(['symbol', 'timestamp'], inplace=True)
            
            # ----- filter to requested time range -----
            df_symbol = df_symbol[
                (df_symbol.index.get_level_values('timestamp') >= start_time) &
                (df_symbol.index.get_level_values('timestamp') <= end_time)
            ]
            
            all_data.append(df_symbol)
            successful_symbols.append(symbol)
            
        except Exception as e:
            print(f"⚠️ Failed to fetch data for {symbol}: {e}")
            continue
    
    if not all_data:
        return pd.DataFrame(), np.array([]), []
    
    # ----- combine all data -----
    df = pd.concat(all_data)
    sorted_symbols = successful_symbols
    
    if len(sorted_symbols) != len(symbols):
        missing_symbols = set(symbols) - set(sorted_symbols)
        print(f"⚠️ Missing data for symbols: {missing_symbols}")
    
    min_timestamps = min(len(df.xs(symbol, level="symbol")) for symbol in sorted_symbols)
    
    ohlcv_list = []
    for symbol in sorted_symbols:
        symbol_data = df.xs(symbol, level="symbol").iloc[:min_timestamps]
        ohlcv_list.append(symbol_data.values)
    ohlcv_array = np.array(ohlcv_list).transpose(0, 2, 1)
    
    return df, np.squeeze(ohlcv_array), sorted_symbols


def fetch_latest_prices(symbols: List[str]) -> dict:
    """
    Fetch latest prices for multiple symbols.
    
    Args:
        symbols: List of symbols (format depends on configured exchange)
        
    Returns:
        dict: Symbol -> latest price mapping
    """
    exchange = _get_exchange_client()
    
    prices = {}
    for symbol in symbols:
        try:
            ticker = exchange.fetch_ticker(symbol)
            prices[symbol] = float(ticker['last'])
        except Exception as e:
            print(f"⚠️ Failed to fetch price for {symbol}: {e}")
            continue
    
    return prices


def fetch_multiple_ohlcv(
    symbols: List[str],
    timeframe: str = '1h',
    limit: int = 100,
) -> dict:
    """
    Fetch OHLCV data for multiple symbols.
    
    Args:
        symbols: List of symbols
        timeframe: CCXT timeframe string
        limit: Number of candles per symbol
        
    Returns:
        dict: Symbol -> DataFrame mapping
    """
    exchange = _get_exchange_client()
    tf = _normalize_timeframe(timeframe)
    
    result = {}
    for symbol in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=tf,
                limit=limit,
            )
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            
            result[symbol] = df
            
        except Exception as e:
            print(f"⚠️ Failed to fetch OHLCV for {symbol}: {e}")
            continue
    
    return result


def get_exchange_info() -> dict:
    """
    Get information about the configured exchange.
    
    Returns:
        dict: Exchange configuration and status
    """
    return {
        "name": EXCHANGE_CONFIG["name"],
        "exchange_id": EXCHANGE_NAME,
        "ccxt_id": EXCHANGE_CONFIG["ccxt_id"],
        "has_futures": EXCHANGE_CONFIG["has_futures"],
        "us_available": EXCHANGE_CONFIG["us_available"],
        "demo_mode": DEMO_MODE,
        "symbol_format": EXCHANGE_CONFIG["symbol_format"],
        "available_symbols": len(CRYPTO_TICKERS),
    }
