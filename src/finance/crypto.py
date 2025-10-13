#!/bin/bash
"""
References
    alpaca.data.historical source code:
        https://github.com/alpacahq/alpaca-py/blob/master/alpaca/data/historical/crypto.py

    supported cryptocurrencies
        https://alpaca.markets/support/what-are-the-supported-coins-pairs
"""

# stdlib
import asyncio
from collections import defaultdict, deque
from datetime import datetime
from itertools import zip_longest
import os
import threading
from typing import Any, Dict, List, Tuple, Union, Optional, Callable
from zoneinfo import ZoneInfo

# numerics
import numpy as np
import pandas as pd

# visualization
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

# trading
from alpaca.data.models.bars import Bar, BarSet
from alpaca.data.enums import CryptoFeed
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.live import CryptoDataStream
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from alpaca.trading.enums import (
    OrderClass,
    OrderSide,
    OrderType,
    PositionIntent,
    TimeInForce,
)
from alpaca.trading.client import TradingClient
from alpaca.trading.models import Asset, ClosePositionResponse, Order, Position
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    OptionLegRequest,
    OrderRequest,
    StopLimitOrderRequest,
    StopLossRequest,
    StopOrderRequest,
    TakeProfitRequest,
    TrailingStopOrderRequest,
)

from . import CRYPTO_TICKERS

N_CPUS = os.cpu_count()

# ==================================================
# FUNCTIONS FOR INTERACTING WITH THE MARKET
# ==================================================


def get_historical_bars(
    client,
    symbol: str,
    start: str,
    end: str = datetime.now(ZoneInfo("America/New_York")),
    interval: TimeFrameUnit = TimeFrameUnit.Day,
) -> pd.DataFrame:
    req = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(amount=1, unit=interval),
        start=start,
        end=end,
        limit=None,
    )
    return client.get_crypto_bars(req).df


def retrieve_crypto_data(
    symbols: Union[str, List[str]],
    start_time,
    end_time,
    frequency: TimeFrameUnit = TimeFrame(1, TimeFrameUnit.Hour),
    limit: int = None,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Retrieve crypto data and store in user-friendly manner

    Returns:
        df : pd.DataFrame
            columns: ['open' 'high' 'low' 'close' 'volume' 'trade_count' 'vwap']
            n_columns := number of columns in this dataframe
        ohlcv : jnp.ndarray, shape (n_symbols, n_columns, n_times)
            symbols are indexed in the order they appear in 'sorted_symbols'
        sorted_symbols : List[str]
            symbols in DataFrame not necessarily ordered in same the
            same way they are ordered in 'symbols' parameter
            'sorted_symbols' provides this ordering, i.e., symbol corresponding
            to sorted_symbols[i] corresponds to data in ohlcv[i]
    """
    client = CryptoHistoricalDataClient()

    req = CryptoBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=frequency,
        start=start_time,
        end=end_time,
        limit=limit,
    )

    data = client.get_crypto_bars(req).df

    # ----- make numpy-friendly -----

    sorted_symbols = data.index.get_level_values("symbol").unique()
    n_timestamps = len(data.loc[symbols[0]])
    n_columns = len(data.columns)
    ohlcv = data.values.reshape(len(symbols), n_timestamps, n_columns).transpose(0, 2, 1)

    # squeeze to remove redundant outer dimension if only one symbol is requested
    return data, np.squeeze(ohlcv), sorted_symbols


# ==================================================
# TRADING INTERFACE
# ==================================================


class CryptoTrader(TradingClient):
    """
    References:
        Order requests
            https://alpaca.markets/sdks/python/api_reference/trading/requests.html#orderrequest
    """

    __slots__ = ("client", "acct", "acct_config", "crypto_universe")

    def __init__(self, paper: bool = True) -> None:
        self.trade_client = TradingClient(
            api_key=os.getenv("ALPACA_API_KEY"),
            secret_key=os.getenv("ALPACA_SECRET_KEY"),
            url_override=os.getenv("ALPACA_API_BASE_URL"),
            paper=paper,
        )

        self.data_client = CryptoHistoricalDataClient(
            api_key=os.getenv("ALPACA_API_KEY"),
            secret_key=os.getenv("ALPACA_SECRET_KEY"),
            url_override=os.getenv("ALPACA_API_BASE_URL"),
        )

        # Initialize streaming client for real-time data
        # Note: WebSocket client needs different URL format than HTTP API
        api_base_url = os.getenv("ALPACA_API_BASE_URL", "https://paper-api.alpaca.markets")
        if "paper-api" in api_base_url:
            # Convert HTTP URL to WebSocket URL for paper trading
            ws_url = api_base_url.replace("https://", "wss://")
        else:
            # For live trading, use the live WebSocket URL
            ws_url = "wss://stream.data.alpaca.markets"
        
        self.stream_client = CryptoDataStream(
            api_key=os.getenv("ALPACA_API_KEY"),
            secret_key=os.getenv("ALPACA_SECRET_KEY"),
            url_override=ws_url,
        )

        self.data_feed = CryptoFeed.US
        self.default_timeframe = TimeFrame(1, TimeFrameUnit.Minute)

        # Real-time data storage
        self._latest_prices: Dict[str, float] = {}
        self._latest_bars: Dict[str, Bar] = {}
        self._latest_quotes: Dict[str, Any] = {}
        self._latest_trades: Dict[str, Any] = {}

        # Data buffers for efficient access
        self._price_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._bar_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Streaming control
        self._streaming_active = False
        self._stream_thread = None
        self._stream_lock = threading.Lock()

        # check trading account
        # You can check definition of each field in the following documents
        # ref. https://docs.alpaca.markets/docs/account-plans
        # ref. https://docs.alpaca.markets/reference/getaccount-1
        self.acct = self.trade_client.get_account()

        # check account configuration
        # ref. https://docs.alpaca.markets/reference/getaccountconfig-1
        self.acct_config = self.trade_client.get_account_configurations()

        self.crypto_universe = CRYPTO_TICKERS

    def print_account_summary(self) -> None:
        """Print account summary"""
        print("Account Summary:")
        print(f"\t- Cash            : {self.acct.cash}")
        print(f"\t- Buying Power    : {self.acct.buying_power}")
        print(f"\t- Equity          : {self.acct.equity}")
        print(f"\t- Portfolio Value : {self.acct.portfolio_value}")

    # @override
    def submit_order(self, order_data: OrderRequest) -> Union[Order, Dict[str, Any]]:
        return super().submit_order(order_data)

    # ==================================================
    # submitting MARKET orders
    # ==================================================

    def buy_market_order(
        self,
        symbol: str,
        notional: float = None,  # quantity in # of shares
        qty: float = None,  # quantity in USD
        extended_hours: float = None,
        client_order_id: str = None,
        legs: List[OptionLegRequest] = None,
        take_profit: TakeProfitRequest = None,
        stop_loss: StopLossRequest = None,
        position_intent: PositionIntent = None,
    ) -> Order:
        """Submit market order"""
        if isinstance(notional, type(None)) and isinstance(qty, type(None)):
            raise ValueError("Either notional or qty must be provided")

        req = MarketOrderRequest(
            symbol=symbol,
            notional=notional,
            qty=qty,
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC,
            extended_hours=extended_hours,
            client_order_id=client_order_id,
            order_class=OrderClass.SIMPLE,
            legs=legs,
            take_profit=take_profit,
            stop_loss=stop_loss,
            position_intent=position_intent,
        )
        return self.trade_client.submit_order(req)

    def sell_market_order(
        self,
        symbol: str,
        notional: float = None,  # quantity in # of shares
        qty: float = None,  # quantity in USD
        extended_hours: float = None,
        client_order_id: str = None,
        legs: List[OptionLegRequest] = None,
        take_profit: TakeProfitRequest = None,
        stop_loss: StopLossRequest = None,
        position_intent: PositionIntent = None,
    ) -> Order:
        """Submit market order"""
        if isinstance(notional, type(None)) and isinstance(qty, type(None)):
            raise ValueError("Either notional or qty must be provided")

        req = MarketOrderRequest(
            symbol=symbol,
            notional=notional,
            qty=qty,
            side=OrderSide.SELL,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC,
            extended_hours=extended_hours,
            client_order_id=client_order_id,
            order_class=OrderClass.SIMPLE,
            legs=legs,
            take_profit=take_profit,
            stop_loss=stop_loss,
            position_intent=position_intent,
        )
        return self.trade_client.submit_order(req)

    def combined_market_order(
        self,
        symbols: List[str],
        notionals: List[float] = None,  # quantity in # of shares
        qtys: List[float] = None,  # quantity in USD
        sides: List[OrderSide] = None,
        extended_hours: float = None,
        client_order_id: str = None,
        legs: List[OptionLegRequest] = None,
        take_profit: TakeProfitRequest = None,
        stop_loss: StopLossRequest = None,
        position_intent: PositionIntent = None,
    ) -> List[Order]:
        """Submit multiple market orders"""
        if isinstance(notionals, type(None)) and isinstance(qtys, type(None)):
            raise ValueError("Either notionals or qtys must be provided")

        # Validate that all lists have the same length
        lengths = [len(symbols), len(sides)]
        if notionals is not None:
            lengths.append(len(notionals))
        if qtys is not None:
            lengths.append(len(qtys))

        if not all(length == lengths[0] for length in lengths):
            raise ValueError("Length of symbols, sides, and quantities must be the same")

        orders = []
        for i, (symbol, side) in enumerate(zip(symbols, sides)):
            notional = notionals[i] if notionals is not None else None
            qty = qtys[i] if qtys is not None else None
            req = MarketOrderRequest(
                symbol=symbol,
                notional=notional,
                qty=qty,
                side=side,
                type=OrderType.MARKET,
                time_in_force=TimeInForce.GTC,
                extended_hours=extended_hours,
                client_order_id=client_order_id,
                order_class=OrderClass.SIMPLE,
                legs=legs,
                take_profit=take_profit,
                stop_loss=stop_loss,
                position_intent=position_intent,
            )

            orders.append(self.trade_client.submit_order(req))
        return orders

    # ==================================================
    # submitting LIMIT orders
    # ==================================================

    def buy_limit_order(
        self,
        symbol: str,
        notional: float = None,  # quantity in # of shares
        qty: float = None,  # quantity in USD
        extended_hours: float = None,
        client_order_id: str = None,
        legs: List[OptionLegRequest] = None,
        take_profit: TakeProfitRequest = None,
        stop_loss: StopLossRequest = None,
        position_intent: PositionIntent = None,
        limit_price: float = None,
    ) -> Order:
        """Submit buy limit order"""
        if isinstance(notional, type(None)) and isinstance(qty, type(None)):
            raise ValueError("Either notional or qty must be provided")

        req = LimitOrderRequest(
            symbol=symbol,
            notional=notional,
            qty=qty,
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.GTC,
            extended_hours=extended_hours,
            client_order_id=client_order_id,
            order_class=OrderClass.SIMPLE,
            legs=legs,
            take_profit=take_profit,
            stop_loss=stop_loss,
            position_intent=position_intent,
            limit_price=limit_price,
        )
        return self.trade_client.submit_order(req)

    def sell_limit_order(
        self,
        symbol: str,
        notional: float = None,  # quantity in # of shares
        qty: float = None,  # quantity in USD
        extended_hours: float = None,
        client_order_id: str = None,
        legs: List[OptionLegRequest] = None,
        take_profit: TakeProfitRequest = None,
        stop_loss: StopLossRequest = None,
        position_intent: PositionIntent = None,
        limit_price: float = None,
    ) -> Order:
        """Submit sell limit order"""
        if isinstance(notional, type(None)) and isinstance(qty, type(None)):
            raise ValueError("Either notional or qty must be provided")

        req = LimitOrderRequest(
            symbol=symbol,
            notional=notional,
            qty=qty,
            side=OrderSide.SELL,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.GTC,
            extended_hours=extended_hours,
            client_order_id=client_order_id,
            order_class=OrderClass.SIMPLE,
            legs=legs,
            take_profit=take_profit,
            stop_loss=stop_loss,
            position_intent=position_intent,
            limit_price=limit_price,
        )
        return self.trade_client.submit_order(req)

    def combined_limit_order(
        self,
        symbols: List[str],
        notionals: List[float] = None,  # quantity in # of shares
        qtys: List[float] = None,  # quantity in USD
        sides: List[OrderSide] = None,
        limit_prices: List[float] = None,
        extended_hours: float = None,
        client_order_id: str = None,
        legs: List[OptionLegRequest] = None,
        take_profit: TakeProfitRequest = None,
        stop_loss: StopLossRequest = None,
        position_intent: PositionIntent = None,
    ) -> List[Order]:
        """Submit multiple limit orders"""
        if isinstance(notionals, type(None)) and isinstance(qtys, type(None)):
            raise ValueError("Either notionals or qtys must be provided")

        # Validate that all lists have the same length
        lengths = [len(symbols), len(sides), len(limit_prices)]
        if notionals is not None:
            lengths.append(len(notionals))
        if qtys is not None:
            lengths.append(len(qtys))

        if not all(length == lengths[0] for length in lengths):
            raise ValueError(
                "Length of symbols, sides, limit_prices, and quantities must be the same"
            )

        orders = []
        for i, (symbol, side, limit_price) in enumerate(
            zip(symbols, sides, limit_prices)
        ):
            notional = notionals[i] if notionals is not None else None
            qty = qtys[i] if qtys is not None else None

            req = LimitOrderRequest(
                symbol=symbol,
                notional=notional,
                qty=qty,
                side=side,
                type=OrderType.LIMIT,
                time_in_force=TimeInForce.GTC,
                extended_hours=extended_hours,
                client_order_id=client_order_id,
                order_class=OrderClass.SIMPLE,
                legs=legs,
                take_profit=take_profit,
                stop_loss=stop_loss,
                position_intent=position_intent,
                limit_price=limit_price,
            )

            orders.append(self.trade_client.submit_order(req))
        return orders

    # ==================================================
    # submitting STOP ORDERS orders
    # ==================================================

    def buy_stop_order(
        self,
        symbol: str,
        stop_price: float,
        notional: float = None,  # quantity in # of shares
        qty: float = None,  # quantity in USD
        extended_hours: float = None,
        client_order_id: str = None,
        legs: List[OptionLegRequest] = None,
        take_profit: TakeProfitRequest = None,
        stop_loss: StopLossRequest = None,
        position_intent: PositionIntent = None,
    ) -> Order:
        """Submit buy stop order"""
        if isinstance(notional, type(None)) and isinstance(qty, type(None)):
            raise ValueError("Either notional or qty must be provided")

        req = StopOrderRequest(
            symbol=symbol,
            stop_price=stop_price,
            notional=notional,
            qty=qty,
            side=OrderSide.BUY,
            type=OrderType.STOP,
            time_in_force=TimeInForce.GTC,
            extended_hours=extended_hours,
            client_order_id=client_order_id,
            order_class=OrderClass.SIMPLE,
            legs=legs,
            take_profit=take_profit,
            stop_loss=stop_loss,
            position_intent=position_intent,
        )
        return self.trade_client.submit_order(req)

    def sell_stop_order(
        self,
        symbol: str,
        stop_price: float,
        notional: float = None,  # quantity in # of shares
        qty: float = None,  # quantity in USD
        extended_hours: float = None,
        client_order_id: str = None,
        legs: List[OptionLegRequest] = None,
        take_profit: TakeProfitRequest = None,
        stop_loss: StopLossRequest = None,
        position_intent: PositionIntent = None,
    ) -> Order:
        """Submit sell stop order"""
        if isinstance(notional, type(None)) and isinstance(qty, type(None)):
            raise ValueError("Either notional or qty must be provided")

        req = StopOrderRequest(
            symbol=symbol,
            stop_price=stop_price,
            notional=notional,
            qty=qty,
            side=OrderSide.SELL,
            type=OrderType.STOP,
            time_in_force=TimeInForce.GTC,
            extended_hours=extended_hours,
            client_order_id=client_order_id,
            order_class=OrderClass.SIMPLE,
            legs=legs,
            take_profit=take_profit,
            stop_loss=stop_loss,
            position_intent=position_intent,
        )
        return self.trade_client.submit_order(req)

    def combined_stop_order(
        self,
        symbols: List[str],
        stop_prices: List[float],
        notionals: List[float] = None,  # quantity in # of shares
        qtys: List[float] = None,  # quantity in USD
        sides: List[OrderSide] = None,
        extended_hours: float = None,
        client_order_id: str = None,
        legs: List[OptionLegRequest] = None,
        take_profit: TakeProfitRequest = None,
        stop_loss: StopLossRequest = None,
        position_intent: PositionIntent = None,
    ) -> List[Order]:
        """Submit multiple stop orders"""
        if isinstance(notionals, type(None)) and isinstance(qtys, type(None)):
            raise ValueError("Either notionals or qtys must be provided")

        # Validate that all lists have the same length
        lengths = [len(symbols), len(stop_prices), len(sides)]
        if notionals is not None:
            lengths.append(len(notionals))
        if qtys is not None:
            lengths.append(len(qtys))

        if not all(length == lengths[0] for length in lengths):
            raise ValueError(
                "Length of symbols, stop_prices, sides, and quantities must be the same"
            )

        orders = []
        for i, (symbol, stop_price, side) in enumerate(zip(symbols, stop_prices, sides)):
            notional = notionals[i] if notionals is not None else None
            qty = qtys[i] if qtys is not None else None

            req = StopOrderRequest(
                symbol=symbol,
                stop_price=stop_price,
                notional=notional,
                qty=qty,
                side=side,
                type=OrderType.STOP,
                time_in_force=TimeInForce.GTC,
                extended_hours=extended_hours,
                client_order_id=client_order_id,
                order_class=OrderClass.SIMPLE,
                legs=legs,
                take_profit=take_profit,
                stop_loss=stop_loss,
                position_intent=position_intent,
            )

            orders.append(self.trade_client.submit_order(req))
        return orders

    # ==================================================
    # submitting STOP LIMIT orders
    # ==================================================

    def buy_stop_limit_order(
        self,
        symbol: str,
        stop_price: float,
        limit_price: float,
        notional: float = None,  # quantity in # of shares
        qty: float = None,  # quantity in USD
        extended_hours: float = None,
        client_order_id: str = None,
        legs: List[OptionLegRequest] = None,
        take_profit: TakeProfitRequest = None,
        stop_loss: StopLossRequest = None,
        position_intent: PositionIntent = None,
    ) -> Order:
        """Submit buy stop-limit order"""
        if isinstance(notional, type(None)) and isinstance(qty, type(None)):
            raise ValueError("Either notional or qty must be provided")

        req = StopLimitOrderRequest(
            symbol=symbol,
            stop_price=stop_price,
            limit_price=limit_price,
            notional=notional,
            qty=qty,
            side=OrderSide.BUY,
            type=OrderType.STOP_LIMIT,
            time_in_force=TimeInForce.GTC,
            extended_hours=extended_hours,
            client_order_id=client_order_id,
            order_class=OrderClass.SIMPLE,
            legs=legs,
            take_profit=take_profit,
            stop_loss=stop_loss,
            position_intent=position_intent,
        )
        return self.trade_client.submit_order(req)

    def sell_stop_limit_order(
        self,
        symbol: str,
        stop_price: float,
        limit_price: float,
        notional: float = None,  # quantity in # of shares
        qty: float = None,  # quantity in USD
        extended_hours: float = None,
        client_order_id: str = None,
        legs: List[OptionLegRequest] = None,
        take_profit: TakeProfitRequest = None,
        stop_loss: StopLossRequest = None,
        position_intent: PositionIntent = None,
    ) -> Order:
        """Submit sell stop-limit order"""
        if isinstance(notional, type(None)) and isinstance(qty, type(None)):
            raise ValueError("Either notional or qty must be provided")

        req = StopLimitOrderRequest(
            symbol=symbol,
            stop_price=stop_price,
            limit_price=limit_price,
            notional=notional,
            qty=qty,
            side=OrderSide.SELL,
            type=OrderType.STOP_LIMIT,
            time_in_force=TimeInForce.GTC,
            extended_hours=extended_hours,
            client_order_id=client_order_id,
            order_class=OrderClass.SIMPLE,
            legs=legs,
            take_profit=take_profit,
            stop_loss=stop_loss,
            position_intent=position_intent,
        )
        return self.trade_client.submit_order(req)

    def combined_stop_limit_order(
        self,
        symbols: List[str],
        stop_prices: List[float],
        limit_prices: List[float],
        notionals: List[float] = None,  # quantity in # of shares
        qtys: List[float] = None,  # quantity in USD
        sides: List[OrderSide] = None,
        extended_hours: float = None,
        client_order_id: str = None,
        legs: List[OptionLegRequest] = None,
        take_profit: TakeProfitRequest = None,
        stop_loss: StopLossRequest = None,
        position_intent: PositionIntent = None,
    ) -> List[Order]:
        """Submit multiple stop-limit orders"""
        if isinstance(notionals, type(None)) and isinstance(qtys, type(None)):
            raise ValueError("Either notionals or qtys must be provided")

        # Validate that all lists have the same length
        lengths = [len(symbols), len(stop_prices), len(limit_prices), len(sides)]
        if notionals is not None:
            lengths.append(len(notionals))
        if qtys is not None:
            lengths.append(len(qtys))

        if not all(length == lengths[0] for length in lengths):
            raise ValueError(
                "Length of symbols, stop_prices, limit_prices, sides, and quantities must be the same"
            )

        orders = []
        for i, (symbol, stop_price, limit_price, side) in enumerate(
            zip(symbols, stop_prices, limit_prices, sides)
        ):
            notional = notionals[i] if notionals is not None else None
            qty = qtys[i] if qtys is not None else None

            req = StopLimitOrderRequest(
                symbol=symbol,
                stop_price=stop_price,
                limit_price=limit_price,
                notional=notional,
                qty=qty,
                side=side,
                type=OrderType.STOP_LIMIT,
                time_in_force=TimeInForce.GTC,
                extended_hours=extended_hours,
                client_order_id=client_order_id,
                order_class=OrderClass.SIMPLE,
                legs=legs,
                take_profit=take_profit,
                stop_loss=stop_loss,
                position_intent=position_intent,
            )

            orders.append(self.trade_client.submit_order(req))
        return orders

    # ==================================================
    # submitting TRAILING STOP orders
    # ==================================================

    def buy_trailing_stop_order(
        self,
        symbol: str,
        stop_price: float,
        limit_price: float,
        notional: float = None,  # quantity in # of shares
        qty: float = None,  # quantity in USD
        extended_hours: float = None,
        client_order_id: str = None,
        legs: List[OptionLegRequest] = None,
        take_profit: TakeProfitRequest = None,
        stop_loss: StopLossRequest = None,
        position_intent: PositionIntent = None,
        trail_price: float = None,
        trail_percent: float = None,
    ) -> Order:
        """Submit buy trailing stop order"""
        if isinstance(notional, type(None)) and isinstance(qty, type(None)):
            raise ValueError("Either notional or qty must be provided")

        req = TrailingStopOrderRequest(
            symbol=symbol,
            stop_price=stop_price,
            limit_price=limit_price,
            notional=notional,
            qty=qty,
            side=OrderSide.BUY,
            type=OrderType.TRAILING_STOP,
            time_in_force=TimeInForce.GTC,
            extended_hours=extended_hours,
            client_order_id=client_order_id,
            order_class=OrderClass.SIMPLE,
            legs=legs,
            take_profit=take_profit,
            stop_loss=stop_loss,
            position_intent=position_intent,
            trail_price=trail_price,
            trail_percent=trail_percent,
        )
        return self.trade_client.submit_order(req)

    def sell_trailing_stop_order(
        self,
        symbol: str,
        stop_price: float,
        limit_price: float,
        notional: float = None,  # quantity in # of shares
        qty: float = None,  # quantity in USD
        extended_hours: float = None,
        client_order_id: str = None,
        legs: List[OptionLegRequest] = None,
        take_profit: TakeProfitRequest = None,
        stop_loss: StopLossRequest = None,
        position_intent: PositionIntent = None,
        trail_price: float = None,
        trail_percent: float = None,
    ) -> Order:
        """Submit sell trailing stop order"""
        if isinstance(notional, type(None)) and isinstance(qty, type(None)):
            raise ValueError("Either notional or qty must be provided")

        req = TrailingStopOrderRequest(
            symbol=symbol,
            stop_price=stop_price,
            limit_price=limit_price,
            notional=notional,
            qty=qty,
            side=OrderSide.SELL,
            type=OrderType.TRAILING_STOP,
            time_in_force=TimeInForce.GTC,
            extended_hours=extended_hours,
            client_order_id=client_order_id,
            order_class=OrderClass.SIMPLE,
            legs=legs,
            take_profit=take_profit,
            stop_loss=stop_loss,
            position_intent=position_intent,
            trail_price=trail_price,
            trail_percent=trail_percent,
        )
        return self.trade_client.submit_order(req)

    def combined_trailing_stop_order(
        self,
        symbols: List[str],
        stop_prices: List[float],
        limit_prices: List[float],
        notionals: List[float] = None,  # quantity in # of shares
        qtys: List[float] = None,  # quantity in USD
        sides: List[OrderSide] = None,
        trail_prices: List[float] = None,
        trail_percents: List[float] = None,
        extended_hours: float = None,
        client_order_id: str = None,
        legs: List[OptionLegRequest] = None,
        take_profit: TakeProfitRequest = None,
        stop_loss: StopLossRequest = None,
        position_intent: PositionIntent = None,
    ) -> List[Order]:
        """Submit multiple trailing stop orders"""
        if isinstance(notionals, type(None)) and isinstance(qtys, type(None)):
            raise ValueError("Either notionals or qtys must be provided")

        # Validate that all lists have the same length
        lengths = [len(symbols), len(stop_prices), len(limit_prices), len(sides)]
        if notionals is not None:
            lengths.append(len(notionals))
        if qtys is not None:
            lengths.append(len(qtys))
        if trail_prices is not None:
            lengths.append(len(trail_prices))
        if trail_percents is not None:
            lengths.append(len(trail_percents))

        if not all(length == lengths[0] for length in lengths):
            raise ValueError(
                "Length of symbols, stop_prices, limit_prices, sides, quantities, and trailing parameters must be the same"
            )

        orders = []
        for i, (symbol, stop_price, limit_price, side) in enumerate(
            zip(symbols, stop_prices, limit_prices, sides)
        ):
            notional = notionals[i] if notionals is not None else None
            qty = qtys[i] if qtys is not None else None
            trail_price = trail_prices[i] if trail_prices is not None else None
            trail_percent = trail_percents[i] if trail_percents is not None else None

            req = TrailingStopOrderRequest(
                symbol=symbol,
                stop_price=stop_price,
                limit_price=limit_price,
                notional=notional,
                qty=qty,
                side=side,
                type=OrderType.TRAILING_STOP,
                time_in_force=TimeInForce.GTC,
                extended_hours=extended_hours,
                client_order_id=client_order_id,
                order_class=OrderClass.SIMPLE,
                legs=legs,
                take_profit=take_profit,
                stop_loss=stop_loss,
                position_intent=position_intent,
                trail_price=trail_price,
                trail_percent=trail_percent,
            )

            orders.append(self.trade_client.submit_order(req))
        return orders

    # ===== gather information on assets =====

    # @# @override
    def get_asset(symbol_or_asset_id: str) -> Union[Asset, Dict[str, Any]]:
        """
        Return type has the following parameters:
            id : UUID
                unique id of asset
            asset_class : AssetClass
                The name of the asset class.
            exchange : AssetExchange
                Which exchange this asset is available through.
            symbol : str
                The symbol identifier of the asset.
            name : str
                The name of the asset.
            status : AssetStatus
                The active status of the asset.
            tradable : bool
                Whether the asset can be traded.
            marginable : bool
                Whether the asset can be traded on margin.
            shortable : bool
                Whether the asset can be shorted.
            easy_to_borrow : bool
                When shorting, whether the asset is easy to borrow
            fractionable : bool
                Whether fractional shares are available
            attributes : Optional[List[str]]
                One of ptp_no_exception or ptp_with_exception.
                It will include unique characteristics of the asset here.
        """
        return super().get_asset(symbol_or_asset_id)

    def get_asset_price(self, symbol: str) -> float:
        """Get latest price for asset"""
        return self.get_asset(symbol).price

    # ==================================================
    # retrieving market data
    # ==================================================

    # @# @override
    def get_crypto_bars(
        self,
        symbol_or_symbols: Union[str, List[str]],
        timeframe: TimeFrame = None,
        start: datetime = None,
        end: datetime = None,
        limit: int = None,
    ) -> Union[Dict[str, Bar]]:
        if isinstance(timeframe, type(None)):
            timeframe = self.default_timeframe

        req = CryptoBarsRequest(
            symbol_or_symbols=symbol_or_symbols,
            timeframe=timeframe,
            start=start,
            end=end,
            limit=limit,
            sort=None,
        )
        return self.data_client.get_crypto_bars(req, feed=self.data_feed)

    # @override
    def get_crypto_latest_bar(
        self,
        symbol_or_symbols: str,
        timeframe: TimeFrame,
        start: datetime = None,
        end: datetime = None,
        limit: int = None,
    ) -> Union[BarSet, Dict[str, Any]]:
        req = CryptoBarsRequest(
            symbol_or_symbols=symbol_or_symbols,
            timeframe=timeframe,
            start=start,
            end=end,
            limit=limit,
            sort=None,
        )
        return self.data_client.get_crypto_bars(req, feed=self.data_feed)

    def retrieve_crypto_data(
        self,
        symbol_or_symbols: Union[str, List[str]],
        timeframe: TimeFrame = None,
        start: datetime = None,
        end: datetime = None,
        limit: int = None,
    ) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """
        Retrieve crypto data and store in user-friendly manner

        Returns:
            df : pd.DataFrame
                columns: ['open' 'high' 'low' 'close' 'volume' 'trade_count' 'vwap']
                n_columns := number of columns in this dataframe
            ohlcv : jnp.ndarray, shape (n_symbols, n_columns, n_times)
                symbols are indexed in the order they appear in 'sorted_symbols'
            sorted_symbols : List[str]
                symbols in DataFrame not necessarily ordered in same the
                same way they are ordered in 'symbols' parameter
                'sorted_symbols' provides this ordering, i.e., symbol corresponding
                to sorted_symbols[i] corresponds to data in ohlcv[i]
        """
        data = self.get_crypto_bars(symbol_or_symbols, timeframe, start, end, limit)

        # ----- make numpy-friendly -----

        sorted_symbols = data.index.get_level_values("symbol").unique()
        ohlcv = data.values.reshape(
            len(symbols), len(data.loc[symbols[0]]), len(data.columns)
        )

        # squeeze to remove redundant outer dimension if only one symbol is requested
        return data, np.squeeze(ohlcv.transpose(0, 2, 1)), sorted_symbols

    def retrieve_latest_crypto_data(
        self,
        symbol_or_symbols: str,
        timeframe: TimeFrame,
        start: datetime = None,
        end: datetime = None,
        limit: int = None,
    ) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        data = self.get_crypto_latest_bar(symbol_or_symbols, timeframe, start, end, limit)

        # ----- make numpy-friendly -----

        sorted_symbols = data.index.get_level_values("symbol").unique()
        ohlcv = data.values.reshape(
            len(sorted_symbols), len(data.loc[sorted_symbols[0]]), len(data.columns)
        )

        # squeeze to remove redundant outer dimension if only one symbol is requested
        return data, np.squeeze(ohlcv.transpose(0, 2, 1)), sorted_symbols

    # ==================================================
    # REAL-TIME STREAMING DATA METHODS
    # ==================================================

    async def _trade_callback(self, trade_data):
        """Handle incoming trade data"""
        symbol = trade_data.symbol
        price = float(trade_data.price)

        with self._stream_lock:
            self._latest_trades[symbol] = trade_data
            self._latest_prices[symbol] = price
            self._price_buffer[symbol].append(
                {
                    "price": price,
                    "volume": float(trade_data.size),
                    "timestamp": trade_data.timestamp,
                }
            )

    async def _quote_callback(self, quote_data):
        """Handle incoming quote data"""
        symbol = quote_data.symbol

        with self._stream_lock:
            self._latest_quotes[symbol] = quote_data
            # Update latest price with mid price
            if quote_data.bid_price and quote_data.ask_price:
                mid_price = (
                    float(quote_data.bid_price) + float(quote_data.ask_price)
                ) / 2
                self._latest_prices[symbol] = mid_price

    async def _bar_callback(self, bar_data):
        """Handle incoming bar data"""
        symbol = bar_data.symbol

        with self._stream_lock:
            self._latest_bars[symbol] = bar_data
            self._bar_buffer[symbol].append(
                {
                    "open": float(bar_data.open),
                    "high": float(bar_data.high),
                    "low": float(bar_data.low),
                    "close": float(bar_data.close),
                    "volume": float(bar_data.volume),
                    "timestamp": bar_data.timestamp,
                }
            )

    def start_real_time_streaming(
        self, symbols: List[str], data_types: List[str] = None
    ) -> None:
        """
        Start real-time streaming for specified symbols and data types

        Args:
            symbols: List of crypto symbols to stream (e.g., ['BTC/USD', 'ETH/USD'])
            data_types: List of data types to stream ['trades', 'quotes', 'bars'].
                       Defaults to all types if None.
        """
        if data_types is None:
            data_types = ["trades", "quotes", "bars"]

        def run_stream():
            asyncio.set_event_loop(asyncio.new_event_loop())
            loop = asyncio.get_event_loop()

            # Subscribe to data streams
            if "trades" in data_types:
                for symbol in symbols:
                    self.stream_client.subscribe_trades(self._trade_callback, symbol)

            if "quotes" in data_types:
                for symbol in symbols:
                    self.stream_client.subscribe_quotes(self._quote_callback, symbol)

            if "bars" in data_types:
                for symbol in symbols:
                    self.stream_client.subscribe_bars(self._bar_callback, symbol)

            self._streaming_active = True
            loop.run_until_complete(self.stream_client.run())

        if not self._streaming_active:
            self._stream_thread = threading.Thread(target=run_stream, daemon=True)
            self._stream_thread.start()
            print(f"Started real-time streaming for {symbols}")

    def stop_real_time_streaming(self) -> None:
        """
        Stop real-time streaming; streaming client doesn't have a
        direct stop method (thread will terminate when connection is closed)
        """
        if self._streaming_active:
            self._streaming_active = False
            print("Real-time streaming stopped")

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get the latest price for a symbol from real-time data

        Args:
            symbol: Crypto symbol (e.g., 'BTC/USD')

        Returns:
            Latest price or None if not available
        """
        with self._stream_lock:
            return self._latest_prices.get(symbol)

    def get_latest_bar(self, symbol: str) -> Optional[Bar]:
        """
        Get the latest bar for a symbol from real-time data

        Args:
            symbol: Crypto symbol (e.g., 'BTC/USD')

        Returns:
            Latest bar or None if not available
        """
        with self._stream_lock:
            return self._latest_bars.get(symbol)

    def get_latest_quote(self, symbol: str) -> Optional[Any]:
        """
        Get the latest quote for a symbol from real-time data

        Args:
            symbol: Crypto symbol (e.g., 'BTC/USD')

        Returns:
            Latest quote or None if not available
        """
        with self._stream_lock:
            return self._latest_quotes.get(symbol)

    def get_price_history(self, symbol: str, limit: int = 100) -> List[Dict]:
        """
        Get recent price history from the buffer

        Args:
            symbol: Crypto symbol (e.g., 'BTC/USD')
            limit: Maximum number of data points to return

        Returns:
            List of price data dictionaries
        """
        with self._stream_lock:
            buffer_data = list(self._price_buffer.get(symbol, []))
            return buffer_data[-limit:] if buffer_data else []

    def get_bar_history(self, symbol: str, limit: int = 50) -> List[Dict]:
        """
        Get recent bar history from the buffer

        Args:
            symbol: Crypto symbol (e.g., 'BTC/USD')
            limit: Maximum number of bars to return

        Returns:
            List of bar data dictionaries
        """
        with self._stream_lock:
            buffer_data = list(self._bar_buffer.get(symbol, []))
            return buffer_data[-limit:] if buffer_data else []

    def get_real_time_ohlc(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Get real-time OHLC data for a symbol

        Args:
            symbol: Crypto symbol (e.g., 'BTC/USD')

        Returns:
            Dictionary with 'open', 'high', 'low', 'close', 'volume' or None
        """
        with self._stream_lock:
            bar = self._latest_bars.get(symbol)
            if bar:
                return {
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                    "timestamp": bar.timestamp,
                }
            return None

    def get_all_streaming_symbols(self) -> List[str]:
        """
        Get list of all symbols currently being streamed

        Returns:
            List of symbols with active data streams
        """
        with self._stream_lock:
            return list(
                set(
                    list(self._latest_prices.keys())
                    + list(self._latest_bars.keys())
                    + list(self._latest_quotes.keys())
                    + list(self._latest_trades.keys())
                )
            )

    def is_streaming_active(self) -> bool:
        """Check if real-time streaming is active"""
        return self._streaming_active

    def add_streaming_callback(
        self, symbol: str, data_type: str, callback: Callable
    ) -> None:
        """
        Add a custom callback for streaming data

        Args:
            symbol: Crypto symbol (e.g., 'BTC/USD')
            data_type: Type of data ('trades', 'quotes', 'bars')
            callback: Async function to handle the data
        """
        if data_type == "trades":
            self.stream_client.subscribe_trades(callback, symbol)
        elif data_type == "quotes":
            self.stream_client.subscribe_quotes(callback, symbol)
        elif data_type == "bars":
            self.stream_client.subscribe_bars(callback, symbol)
        else:
            raise ValueError(
                f"Invalid data_type: {data_type}. Must be 'trades', 'quotes', or 'bars'"
            )

    def get_market_data_summary(self) -> Dict[str, Dict]:
        """
        Get a summary of all available real-time market data

        Returns:
            Dictionary with symbol as key and data summary as value
        """
        with self._stream_lock:
            summary = {}
            all_symbols = self.get_all_streaming_symbols()

            for symbol in all_symbols:
                summary[symbol] = {
                    "latest_price": self._latest_prices.get(symbol),
                    "has_bar_data": symbol in self._latest_bars,
                    "has_quote_data": symbol in self._latest_quotes,
                    "has_trade_data": symbol in self._latest_trades,
                    "price_buffer_size": len(self._price_buffer.get(symbol, [])),
                    "bar_buffer_size": len(self._bar_buffer.get(symbol, [])),
                }

            return summary


# ==================================================
# PLOTTING
# ==================================================


def plot_df(df: pd.DataFrame) -> None:

    styles = {
        "open": {"color": "red", "linestyle": "-", "linewidth": 1.5},
        "high": {"color": "blue", "linestyle": "-", "linewidth": 1.5},
        "low": {"color": "green", "linestyle": "-", "linewidth": 1.5},
        "close": {"color": "magenta", "linestyle": "-", "linewidth": 1.5},
        "vwap": {"color": "black", "linestyle": "--", "linewidth": 2},
    }

    # Assuming your dataframe is called 'df'
    # Get the first symbol's data
    first_symbol = df.index.get_level_values("symbol")[0]
    symbol_data = df.xs(first_symbol, level="symbol")

    # Create figure and GridSpec layout
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 3, figure=fig, height_ratios=[2, 1, 1, 1], width_ratios=[1, 1, 1])

    # Main OHLC + VWAP plot (top row, spanning all columns)
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.plot(
        symbol_data.index, symbol_data["open"], label="Open", **styles["open"], alpha=0.8
    )
    ax_main.plot(
        symbol_data.index, symbol_data["high"], label="High", **styles["high"], alpha=0.8
    )
    ax_main.plot(
        symbol_data.index, symbol_data["low"], label="Low", **styles["low"], alpha=0.8
    )
    ax_main.plot(
        symbol_data.index,
        symbol_data["close"],
        label="Close",
        **styles["close"],
        alpha=0.8,
    )
    ax_main.plot(symbol_data.index, symbol_data["vwap"], label="VWAP", **styles["vwap"])
    ax_main.set_ylabel("Price", fontweight="bold")
    ax_main.legend(loc="upper left", ncol=5)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_title(f"Price Data for {first_symbol}", fontweight="bold", fontsize=14)

    # Volume plot (second row, spanning all columns)
    ax_volume = fig.add_subplot(gs[1, :], sharex=ax_main)
    ax_volume.bar(symbol_data.index, symbol_data["volume"], alpha=0.7, color="steelblue")
    ax_volume.grid(True, alpha=0.3)
    ax_volume.set_title("Volume", fontweight="bold")

    # Individual metric plots (bottom two rows)
    # Row 3: Open, High, Low
    ax_open = fig.add_subplot(gs[2, 0], sharex=ax_main)
    ax_open.plot(symbol_data.index, symbol_data["open"], **styles["open"])
    ax_open.grid(True, alpha=0.3)
    ax_open.set_title("Open Price", fontweight="bold")

    ax_high = fig.add_subplot(gs[2, 1], sharex=ax_main)
    ax_high.plot(symbol_data.index, symbol_data["high"], **styles["high"])
    ax_high.grid(True, alpha=0.3)
    ax_high.set_title("High Price", fontweight="bold")

    ax_low = fig.add_subplot(gs[2, 2], sharex=ax_main)
    ax_low.plot(symbol_data.index, symbol_data["low"], **styles["low"])
    ax_low.grid(True, alpha=0.3)
    ax_low.set_title("Low Price", fontweight="bold")

    # Row 4: Close, Trade Count, and a combined volume/trade_count plot
    ax_close = fig.add_subplot(gs[3, 0], sharex=ax_main)
    ax_close.plot(symbol_data.index, symbol_data["close"], **styles["close"])
    ax_close.grid(True, alpha=0.3)
    ax_close.set_title("Close Price", fontweight="bold")

    ax_trades = fig.add_subplot(gs[3, 1], sharex=ax_main)
    ax_trades.plot(
        symbol_data.index, symbol_data["trade_count"], color="orange", linewidth=1.5
    )
    ax_trades.grid(True, alpha=0.3)
    ax_trades.set_title("Number of Trades", fontweight="bold")

    # Combined volume and trade count (dual y-axis)
    ax_combined = fig.add_subplot(gs[3, 2], sharex=ax_main)
    ax_combined_twin = ax_combined.twinx()

    # Volume on left y-axis
    vol_line = ax_combined.plot(
        symbol_data.index,
        symbol_data["volume"],
        color="steelblue",
        linewidth=1.5,
        label="Volume",
    )
    ax_combined.set_ylabel("Volume", color="steelblue", fontweight="bold")
    ax_combined.tick_params(axis="y", labelcolor="steelblue")

    # Trade count on right y-axis
    trade_line = ax_combined_twin.plot(
        symbol_data.index,
        symbol_data["trade_count"],
        color="orange",
        linewidth=1.5,
        label="Trade Count",
    )
    ax_combined_twin.set_ylabel("Trade Count", color="orange", fontweight="bold")
    ax_combined_twin.tick_params(axis="y", labelcolor="orange")

    ax_combined.grid(True, alpha=0.3)
    ax_combined.set_title("Volume vs Trade Count", fontweight="bold")

    # Format x-axis labels only for bottom row
    for ax in [ax_close, ax_trades, ax_combined]:
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("Time", fontweight="bold")

    # Hide x-axis labels for upper plots to avoid clutter
    plt.setp(ax_main.get_xticklabels(), visible=False)
    plt.setp(ax_volume.get_xticklabels(), visible=False)
    plt.setp(ax_open.get_xticklabels(), visible=False)
    plt.setp(ax_high.get_xticklabels(), visible=False)
    plt.setp(ax_low.get_xticklabels(), visible=False)

    # Add overall title
    # fig.suptitle(
    #     f"Comprehensive Time Series Analysis for {first_symbol}",
    #     fontsize=16,
    #     fontweight="bold",
    #     y=0.98,
    # )

    plt.tight_layout()
    plt.subplots_adjust(top=0.94)  # Make room for the main title

    savename = f"{symbol_data}"
    # plt.savefig(savename, dpi=512)
    plt.show()

    return


def compare_symbols(
    df: pd.DataFrame, symbols: List[str], figsize: tuple = (18, 14)
) -> None:
    """
    Create detailed comparison plots for multiple symbols from the dataframe.

    Parameters:
    -----------
    df : pandas.DataFrame
        Multi-indexed dataframe with ('symbol', 'timestamp') index
    symbols : list
        List of symbols to compare (must exist in the dataframe)
    figsize : tuple
        Figure size (width, height)

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    """

    # Validate symbols exist in dataframe
    available_symbols = df.index.get_level_values("symbol").unique()
    missing_symbols = [s for s in symbols if s not in available_symbols]
    if missing_symbols:
        raise ValueError(f"Symbols not found in dataframe: {missing_symbols}")

    # Extract data for each symbol
    symbol_data = {}
    for symbol in symbols:
        symbol_data[symbol] = df.xs(symbol, level="symbol")

    # Create figure and GridSpec layout
    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = GridSpec(
        4,
        2,
        figure=fig,
        height_ratios=[2, 1, 1, 1],
        width_ratios=[1, 1],
        hspace=0.3,
        wspace=0.25,
    )

    # Define colors for each symbol
    colors = plt.cm.Set1(np.linspace(0, 1, len(symbols)))
    symbol_colors = dict(zip(symbols, colors))

    # 1. Main comparison plot - All OHLC + VWAP (top row, spanning both columns)
    ax_main = fig.add_subplot(gs[0, :])

    for symbol in symbols:
        data = symbol_data[symbol]
        color = symbol_colors[symbol]

        # Plot OHLC with slightly different line styles for distinction
        ax_main.plot(
            data.index,
            data["open"],
            label=f"{symbol} Open",
            alpha=0.7,
            linewidth=1.5,
            color=color,
            linestyle="-",
        )
        ax_main.plot(
            data.index,
            data["high"],
            label=f"{symbol} High",
            alpha=0.7,
            linewidth=1.5,
            color=color,
            linestyle=":",
        )
        ax_main.plot(
            data.index,
            data["low"],
            label=f"{symbol} Low",
            alpha=0.7,
            linewidth=1.5,
            color=color,
            linestyle="--",
        )
        ax_main.plot(
            data.index,
            data["close"],
            label=f"{symbol} Close",
            alpha=0.8,
            linewidth=2,
            color=color,
            linestyle="-",
        )
        ax_main.plot(
            data.index,
            data["vwap"],
            label=f"{symbol} VWAP",
            alpha=0.9,
            linewidth=2,
            color=color,
            linestyle="-.",
        )

    ax_main.set_ylabel("Price", fontweight="bold", fontsize=12)
    ax_main.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=1, fontsize=9)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_title("Price Comparison - All Symbols", fontweight="bold", fontsize=14)

    # Format x-axis with proper tick marks
    ax_main.tick_params(axis="x", rotation=45, which="both")
    if hasattr(list(symbol_data.values())[0].index, "to_pydatetime"):
        ax_main.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        ax_main.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    ax_main.xaxis.set_tick_params(which="major", length=6, width=1.5)
    ax_main.xaxis.set_tick_params(which="minor", length=3, width=1)

    # 2. Individual price component comparisons (remaining rows)
    price_components = [
        ("Open Prices", "open"),
        ("High Prices", "high"),
        ("Low Prices", "low"),
        ("Close Prices", "close"),
        ("VWAP", "vwap"),
    ]

    # Positions for subplots: (1,0), (1,1), (2,0), (2,1), (3,:)
    subplot_positions = [(1, 0), (1, 1), (2, 0), (2, 1), (3, slice(None))]

    for i, ((title, column), pos) in enumerate(zip(price_components, subplot_positions)):
        if pos[1] == slice(None):  # VWAP spans both columns
            ax = fig.add_subplot(gs[pos[0], pos[1]], sharex=ax_main)
        else:
            ax = fig.add_subplot(gs[pos[0], pos[1]], sharex=ax_main)

        # Plot each symbol for this price component
        for symbol in symbols:
            data = symbol_data[symbol]
            color = symbol_colors[symbol]

            ax.plot(
                data.index,
                data[column],
                label=symbol,
                color=color,
                linewidth=2,
                alpha=0.8,
            )

        ax.set_ylabel(title, fontweight="bold")
        ax.legend(loc="upper left" if i < 4 else "upper right")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{title} Comparison", fontweight="bold")

        # Only show x-axis labels on bottom row
        if pos[0] != 3:  # Not bottom row
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.tick_params(axis="x", rotation=45)
            ax.set_xlabel("Time", fontweight="bold")

    # Add overall title
    symbols_str = ", ".join(symbols)
    fig.suptitle(
        f"Multi-Symbol Comparison: {symbols_str}", fontsize=16, fontweight="bold", y=0.96
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, right=0.85)  # Make room for title and legend
    plt.show()

    return


def compare_symbols_normalized(
    df: pd.DataFrame, symbols: List[str], figsize: tuple = (16, 12)
) -> None:
    """
    Create normalized comparison plots showing percentage changes from the first data point.
    This is useful for comparing symbols with very different price ranges.

    Parameters:
    -----------
    df : pandas.DataFrame
        Multi-indexed dataframe with ('symbol', 'timestamp') index
    symbols : list
        List of symbols to compare
    figsize : tuple
        Figure size (width, height)

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    """

    # Extract and normalize data for each symbol
    symbol_data = {}
    for symbol in symbols:
        data = df.xs(symbol, level="symbol")
        # Normalize to percentage change from first value
        normalized_data = data.copy()
        for col in ["open", "high", "low", "close", "vwap"]:
            first_val = data[col].iloc[0]
            normalized_data[col] = ((data[col] - first_val) / first_val) * 100
        symbol_data[symbol] = normalized_data

    # Create figure and GridSpec layout
    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[2, 1, 1],
        width_ratios=[1, 1],
        hspace=0.3,
        wspace=0.25,
    )

    # Define colors for each symbol
    colors = plt.cm.Set1(np.linspace(0, 1, len(symbols)))
    symbol_colors = dict(zip(symbols, colors))

    # Main normalized comparison plot
    ax_main = fig.add_subplot(gs[0, :])

    for symbol in symbols:
        data = symbol_data[symbol]
        color = symbol_colors[symbol]

        ax_main.plot(
            data.index,
            data["close"],
            label=f"{symbol} Close",
            color=color,
            linewidth=2.5,
            alpha=0.8,
        )
        ax_main.plot(
            data.index,
            data["vwap"],
            label=f"{symbol} VWAP",
            color=color,
            linewidth=2,
            alpha=0.7,
            linestyle="--",
        )

    ax_main.axhline(y=0, color="black", linestyle="-", alpha=0.5, linewidth=1)
    ax_main.set_ylabel("Percentage Change (%)", fontweight="bold", fontsize=12)
    ax_main.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax_main.grid(True, alpha=0.3)
    # ax_main.set_title('Normalized Price Comparison (% Change from Start)',
    #  fontweight='bold', fontsize=14)

    # Format x-axis
    ax_main.tick_params(axis="x", rotation=45)
    if hasattr(list(symbol_data.values())[0].index, "to_pydatetime"):
        ax_main.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    # Individual normalized components
    components = [
        ("Close % Change", "close"),
        ("VWAP % Change", "vwap"),
        ("High % Change", "high"),
        ("Low % Change", "low"),
    ]
    positions = [(1, 0), (1, 1), (2, 0), (2, 1)]

    for (title, column), pos in zip(components, positions):
        if column == "vwap":
            ls = "--"
        else:
            ls = "-"

        ax = fig.add_subplot(gs[pos[0], pos[1]], sharex=ax_main)

        for symbol in symbols:
            data = symbol_data[symbol]
            color = symbol_colors[symbol]
            ax.plot(
                data.index,
                data[column],
                label=symbol,
                color=color,
                linestyle=ls,
                linewidth=2,
                alpha=0.8,
            )

        ax.axhline(y=0, color="black", linestyle="-", alpha=0.5, linewidth=0.8)
        ax.set_ylabel("Change (%)", fontweight="bold")
        # ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontweight="bold")

        if pos[0] != 2:  # Not bottom row
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.tick_params(axis="x", rotation=45)
            ax.set_xlabel("Time", fontweight="bold")

    symbols_str = ", ".join(symbols)
    fig.suptitle(
        f"Normalized Multi-Symbol Comparison: {symbols_str}",
        fontsize=16,
        fontweight="bold",
        y=0.96,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, right=0.85)
    plt.show()

    return


if __name__ == "__main__":

    from . import CRYPTO_TICKERS, BTC_PAIRS, USDT_PAIRS, USDC_PAIRS, USD_PAIRS
    from . import NOW, PAST_N_YEARS, TIME_FRAMES

    # ===== specify data to be retrieved =====

    symbols = USDT_PAIRS
    start_time = NOW - PAST_N_YEARS[2]
    end_time = NOW
    frequency = TIME_FRAMES["day"]
    limit = None  # no limit to number of data points to store

    df, ohlcv, sort_symbols = retrieve_crypto_data(
        symbols=symbols,
        start_time=start_time,
        end_time=end_time,
        frequency=frequency,
        limit=limit,
    )

    # plot_df(df)
