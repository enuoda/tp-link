#!/usr/bin/env python3
"""
CCXT Futures Trader

Exchange-agnostic futures trading and streaming via CCXT library.
Provides native long/short positions for spread trading strategies.

The exchange is configurable via environment variables. See __init__.py
for configuration details.

Features:
    - WebSocket streaming for real-time price data
    - Futures trading with native long/short positions
    - Testnet/demo support for paper trading
    - Automatic reconnection with exponential backoff
    - Heartbeat monitoring for zombie connection detection

Supported Exchanges:
    - Kraken Futures (default, US-friendly)
    - Binance Futures (international)
    - Bybit (international)

References:
    CCXT Documentation: https://docs.ccxt.com/

Sam Dawley
"""

# stdlib
import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# numerics
import numpy as np
import pandas as pd

# ccxt
import ccxt
import ccxt.pro as ccxtpro

from zoneinfo import ZoneInfo

from . import (
    CRYPTO_TICKERS,
    EXCHANGE_NAME,
    EXCHANGE_CONFIG,
    API_KEY,
    SECRET_KEY,
    DEMO_MODE,
    set_runtime_symbol_map,
)


# ==================================================
# DATA CLASSES
# ==================================================


@dataclass
class FuturesOrder:
    """Represents a futures order response"""
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    type: str  # 'market', 'limit', etc.
    amount: float
    price: Optional[float]
    cost: float
    status: str
    timestamp: datetime
    info: Dict[str, Any]


@dataclass
class FuturesPosition:
    """Represents an open futures position"""
    symbol: str
    side: str  # 'long' or 'short'
    contracts: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    leverage: int
    liquidation_price: Optional[float]
    margin_type: str  # 'cross' or 'isolated'
    info: Dict[str, Any]


# ==================================================
# CCXT FUTURES TRADER
# ==================================================


class CCXTFuturesTrader:
    """
    Exchange-agnostic futures trader using CCXT for streaming and trading.
    
    The exchange is determined by the EXCHANGE_NAME environment variable.
    
    Provides:
        - Real-time WebSocket streaming (tickers, trades)
        - Futures trading with native long/short positions
        - Position and balance monitoring
        - Demo/testnet support for paper trading
        - Automatic reconnection with exponential backoff
        - Heartbeat monitoring
    
    Example:
        >>> trader = CCXTFuturesTrader(testnet=True)
        >>> trader.start_real_time_streaming(['BTC/USD:USD', 'ETH/USD:USD'])
        >>> trader.wait_for_data(['BTC/USD:USD'])
        >>> price = trader.get_latest_price('BTC/USD:USD')
        >>> order = trader.open_long('BTC/USD:USD', notional=100.0)
    """

    def __init__(
        self,
        testnet: bool = True,
        default_leverage: int = 1,
        margin_mode: str = "cross",
        max_reconnect_attempts: int = 5,
        reconnect_delay_base: float = 1.0,
        heartbeat_interval: float = 30.0,
        heartbeat_timeout_multiplier: float = 3.0,
        price_buffer_maxlen: int = 1000,
        bar_buffer_maxlen: int = 100,
    ) -> None:
        """
        Initialize the CCXT Futures Trader.
        
        The exchange is determined by the EXCHANGE_NAME environment variable.
        
        Args:
            testnet: Use demo/testnet mode (default: True for safety)
            default_leverage: Default leverage for positions (default: 1x)
            margin_mode: 'cross' or 'isolated' margin (default: 'cross')
            max_reconnect_attempts: Max WebSocket reconnection attempts
            reconnect_delay_base: Base delay for exponential backoff (seconds)
            heartbeat_interval: Expected data interval for heartbeat (seconds)
            heartbeat_timeout_multiplier: Multiplier for heartbeat timeout
            price_buffer_maxlen: Max entries in price history buffer
            bar_buffer_maxlen: Max entries in bar history buffer
        """
        self.testnet = testnet
        self.default_leverage = default_leverage
        self.margin_mode = margin_mode
        self.exchange_name = EXCHANGE_NAME
        self.exchange_config = EXCHANGE_CONFIG
        
        # ----- configuration (avoid hardcoding) -----
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_delay_base = reconnect_delay_base
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_timeout_multiplier = heartbeat_timeout_multiplier
        self._price_buffer_maxlen = price_buffer_maxlen
        self._bar_buffer_maxlen = bar_buffer_maxlen
        
        # ----- determine quote currency for balance lookups -----
        self._quote_currency = EXCHANGE_CONFIG.get("quote_currency", "USD")
        
        # ----- initialize exchange clients -----
        self.exchange = self._create_rest_client()
        self.ws_exchange = self._create_ws_client()
        
        # ----- enable demo/testnet if requested -----
        if testnet or DEMO_MODE:
            try:
                self.exchange.set_sandbox_mode(True)
                self.ws_exchange.set_sandbox_mode(True)
                print(f"📋 Using {EXCHANGE_CONFIG['name']} DEMO/TESTNET mode", flush=True)

            except Exception as e:
                print(f"⚠️ Could not enable sandbox mode: {e}", flush=True)
        else:
            print(f"⚠️ Using {EXCHANGE_CONFIG['name']} PRODUCTION mode", flush=True)
        
        # ----- real-time data storage -----
        self._latest_bars: Dict[str, Dict] = {}
        self._latest_prices: Dict[str, float] = {}
        self._latest_tickers: Dict[str, Dict] = {}
        self._latest_trades: Dict[str, Any] = {}
        
        # ----- data buffers for efficient access -----
        self._price_buffer: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self._price_buffer_maxlen)
        )
        self._bar_buffer: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self._bar_buffer_maxlen)
        )
        
        # ----- streaming control -----
        self._streaming_active = False
        self._stream_thread: Optional[threading.Thread] = None
        self._stream_lock = threading.Lock()
        
        # ----- enhanced streaming management -----
        self._streaming_symbols: List[str] = []
        self._streaming_data_types: List[str] = []
        self._last_data_received: Dict[str, datetime] = {}
        self._reconnect_attempts = 0
        self._connection_healthy = threading.Event()
        self._stop_streaming = threading.Event()
        
        # ----- heartbeat monitoring -----
        self._last_any_data_received: Optional[datetime] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_stop = threading.Event()
        
        # ----- debug tracking -----
        self._debug_streaming = True
        self._data_received_count: Dict[str, int] = {}
        self._first_data_logged: set = set()
        
        # ----- account info -----
        self._balance: Optional[Dict] = None
        self._positions: Dict[str, FuturesPosition] = {}
        
        # ----- load markets -----
        try:
            self.exchange.load_markets()
            print(f"✅ Loaded {len(self.exchange.markets)} {EXCHANGE_CONFIG['name']} markets")
            
            # Set runtime symbol map for accurate canonical → exchange conversion
            symbol_map = self.get_canonical_to_exchange_map()
            set_runtime_symbol_map(symbol_map)
        except Exception as e:
            print(f"⚠️ Failed to load markets: {e}")
        
        # ----- get available perpetual symbols dynamically -----
        # Use ALL available perpetuals from the exchange directly
        # This automatically handles testnet (limited symbols) vs production (full list)
        available_perpetuals = self.get_tradeable_symbols()
        
        if available_perpetuals:
            print(f"📊 Found {len(available_perpetuals)} perpetual contracts on {EXCHANGE_CONFIG['name']}", flush=True)
            # Use exchange symbols directly - no filtering from static list
            self.crypto_universe = available_perpetuals
            print(f"✅ Using all {len(self.crypto_universe)} available symbols from exchange", flush=True)
            
            # Log the symbols for debugging
            if len(self.crypto_universe) <= 20:
                print(f"📋 Available symbols: {self.crypto_universe}", flush=True)
            else:
                print(f"📋 Available symbols (first 20): {self.crypto_universe[:20]}...", flush=True)
        else:
            # Fallback to static list if dynamic fetch failed
            print(f"⚠️ Could not fetch symbols dynamically, falling back to static list", flush=True)
            self.crypto_universe = self._validate_symbols(CRYPTO_TICKERS)
            
            if len(self.crypto_universe) < len(CRYPTO_TICKERS):
                unavailable = set(CRYPTO_TICKERS) - set(self.crypto_universe)
                print(f"⚠️ {len(unavailable)} symbols not available on {EXCHANGE_CONFIG['name']}: {unavailable}", flush=True)


    def _validate_symbols(self, symbols: List[str]) -> List[str]:
        """
        Validate symbols against exchange's available markets.
        
        Args:
            symbols: List of symbols to validate
            
        Returns:
            List of symbols that are available on the exchange
        """
        if not hasattr(self.exchange, 'markets') or not self.exchange.markets:
            print("⚠️ Markets not loaded, cannot validate symbols", flush=True)
            return symbols
        
        available = []
        for symbol in symbols:
            if symbol in self.exchange.markets:
                available.append(symbol)
            else:
                base = symbol.split('/')[0] if '/' in symbol else symbol
                # Look for matching perpetual contracts
                matches = [
                    m for m in self.exchange.markets.keys()
                    if m.startswith(base + '/') and self._is_perpetual_contract(m)
                ]

                if matches:
                    print(f"ℹ️ Symbol {symbol} not found, using {matches[0]}", flush=True)
                    available.append(matches[0])
                # else symbol is not available
        
        return available


    def _is_perpetual_contract(self, symbol: str) -> bool:
        """
        Check if a symbol is a perpetual/futures contract.
        
        Args:
            symbol: Market symbol
            
        Returns:
            True if the symbol is a perpetual contract
        """
        if not hasattr(self.exchange, 'markets') or symbol not in self.exchange.markets:
            # Use heuristic: CCXT unified format for derivatives includes ":"
            return ':' in symbol
        
        market = self.exchange.markets[symbol]
        return (
            market.get('type') in ('swap', 'future') or
            market.get('swap', False) or
            market.get('future', False) or
            ':' in symbol
        )


    def get_available_markets(self) -> List[str]:
        """
        Get list of all available markets on the exchange.
        
        Returns:
            List of market symbols
        """
        if hasattr(self.exchange, 'markets') and self.exchange.markets:
            return list(self.exchange.markets.keys())
        return []


    def is_symbol_available(self, symbol: str) -> bool:
        """
        Check if a symbol is available on the exchange.
        
        Args:
            symbol: Symbol to check
            
        Returns:
            True if symbol is available
        """
        if hasattr(self.exchange, 'markets') and self.exchange.markets:
            return symbol in self.exchange.markets
        return True  # Assume available if markets not loaded


    def get_canonical_to_exchange_map(self) -> Dict[str, str]:
        """
        Create a mapping from canonical symbols (e.g., "BTC") to exchange symbols.
        
        This is useful for converting benchmark symbols (stored in canonical format)
        to the exact format the exchange expects.
        
        Returns:
            Dict mapping canonical base asset to exchange symbol
            
        Example:
            >>> mapping = trader.get_canonical_to_exchange_map()
            >>> mapping.get("BTC")  # Returns "BTC/USD:USD" or whatever Kraken uses
        """
        if not hasattr(self.exchange, 'markets') or not self.exchange.markets:
            return {}
        
        mapping = {}
        for symbol in self.exchange.markets:
            if self._is_perpetual_contract(symbol):
                base = symbol.split('/')[0].upper() if '/' in symbol else symbol.upper()
                if base == "XBT":
                    base = "BTC"
                if base not in mapping:
                    mapping[base] = symbol
        
        return mapping


    def get_tradeable_symbols(self) -> List[str]:
        """
        Get only perpetual futures symbols available for trading.
        
        Filters the exchange's markets to return only perpetual/swap contracts
        that are suitable for spread trading with native long/short positions.
        
        Returns:
            List of perpetual futures symbols in CCXT unified format
            
        Example:
            >>> trader = CCXTFuturesTrader(testnet=True)
            >>> symbols = trader.get_tradeable_symbols()
            >>> print(f"Available perpetuals: {symbols[:5]}")
        """
        if not hasattr(self.exchange, 'markets') or not self.exchange.markets:
            return self.crypto_universe
        
        perpetuals = []
        for symbol in self.exchange.markets:
            if self._is_perpetual_contract(symbol):
                perpetuals.append(symbol)
        
        return sorted(perpetuals)


    def _create_rest_client(self):
        """Create the REST API exchange client."""
        ccxt_id = EXCHANGE_CONFIG["ccxt_id"]
        
        if not hasattr(ccxt, ccxt_id):
            raise ValueError(f"CCXT does not support exchange: {ccxt_id}")
        
        exchange_class = getattr(ccxt, ccxt_id)
        
        config = {
            'apiKey': API_KEY,
            'secret': SECRET_KEY,
            'enableRateLimit': True,
        }
        
        # Exchange-specific options
        if EXCHANGE_NAME == "binance":
            config['options'] = {'defaultType': 'future'}
        elif EXCHANGE_NAME == "bybit":
            config['options'] = {'defaultType': 'swap'}
        
        return exchange_class(config)


    def _create_ws_client(self):
        """Create the WebSocket exchange client."""
        ccxt_ws_id = EXCHANGE_CONFIG["ccxt_ws_id"]
        
        if not hasattr(ccxtpro, ccxt_ws_id):
            raise ValueError(f"CCXT Pro does not support WebSocket for: {ccxt_ws_id}")
        
        ws_class = getattr(ccxtpro, ccxt_ws_id)
        
        config = {
            'apiKey': API_KEY,
            'secret': SECRET_KEY,
            'enableRateLimit': True,
        }
        
        # Exchange-specific options
        if EXCHANGE_NAME == "binance":
            config['options'] = {'defaultType': 'future'}
        elif EXCHANGE_NAME == "bybit":
            config['options'] = {'defaultType': 'swap'}
        
        return ws_class(config)


    # ==================================================
    # ACCOUNT INFORMATION
    # ==================================================


    def print_account_summary(self) -> None:
        """Print account summary"""
        try:
            balance = self.get_futures_balance()
            
            # Get balance for the quote currency
            quote = self._quote_currency
            quote_balance = balance.get(quote, {})
            total = float(quote_balance.get('total', 0))
            free = float(quote_balance.get('free', 0))
            used = float(quote_balance.get('used', 0))
            
            print("Account Summary:", flush=True)
            print(f"\t- Total Balance   : {total:.2f} {quote}", flush=True)
            print(f"\t- Available       : {free:.2f} {quote}", flush=True)
            print(f"\t- In Positions    : {used:.2f} {quote}", flush=True)
            
            # Show open positions
            positions = self.get_all_positions()
            if positions:
                print(f"\t- Open Positions  : {len(positions)}", flush=True)
                for pos in positions:
                    pnl_str = f"+{pos.unrealized_pnl:.2f}" if pos.unrealized_pnl >= 0 else f"{pos.unrealized_pnl:.2f}"
                    print(f"\t    {pos.symbol}: {pos.side.upper()} {pos.contracts} @ {pos.entry_price:.2f} (PnL: {pnl_str})", flush=True)
            else:
                print(f"\t- Open Positions  : 0", flush=True)
                
        except Exception as e:
            print(f"❌ Failed to get account summary: {e}", flush=True)


    def get_futures_balance(self) -> Dict[str, Any]:
        """
        Get futures account balance.
        
        Returns:
            dict: Balance by asset with 'free', 'used', 'total' for each
            
        Example:
            >>> balance = trader.get_futures_balance()
            >>> print(f"USD: {balance['USD']['free']}")
        """
        try:
            balance = self.exchange.fetch_balance()
            self._balance = balance
            return balance
        except Exception as e:
            print(f"❌ Failed to fetch balance: {e}", flush=True)
            return {}


    def get_available_balance(self, asset: str = None) -> float:
        """
        Get available (free) balance for an asset.
        
        Args:
            asset: Asset symbol (default: exchange's quote currency)
            
        Returns:
            float: Available balance
        """
        if asset is None:
            asset = self._quote_currency
        balance = self.get_futures_balance()
        return float(balance.get(asset, {}).get('free', 0))


    # ==================================================
    # POSITION MANAGEMENT
    # ==================================================


    def get_position(self, symbol: str) -> Optional[FuturesPosition]:
        """
        Get current position for a symbol.
        
        Args:
            symbol: Futures symbol (format depends on exchange)
            
        Returns:
            FuturesPosition or None if no position
        """
        try:
            positions = self.exchange.fetch_positions([symbol])
            
            for p in positions:
                # Use CCXT unified fields directly, NOT p["info"]
                contracts = float(p.get('contracts') or 0)
                if contracts != 0:
                    return FuturesPosition(
                        symbol=p['symbol'],
                        side=p.get('side', 'long'),
                        contracts=abs(contracts),
                        entry_price=float(p.get('entryPrice') or 0),
                        mark_price=float(p.get('markPrice') or 0),
                        unrealized_pnl=float(p.get('unrealizedPnl') or 0),
                        leverage=int(p.get('leverage') or 1),
                        liquidation_price=float(p['liquidationPrice']) if p.get('liquidationPrice') else None,
                        margin_type=p.get('marginMode', 'cross'),
                        info=p.get('info', {}),
                    )
            
            return None
            
        except Exception as e:
            print(f"❌ Failed to fetch position for {symbol}: {e}", flush=True)
            return None


    def get_all_positions(self) -> List[FuturesPosition]:
        """
        Get all open positions.
        
        Returns:
            List of FuturesPosition objects
        """
        try:
            positions = self.exchange.fetch_positions()
            result = []
            
            for p in positions:
                # Use CCXT unified fields directly, NOT p["info"]
                contracts = float(p.get('contracts') or 0)
                if contracts != 0:
                    result.append(FuturesPosition(
                        symbol=p['symbol'],
                        side=p.get('side', 'long'),
                        contracts=abs(contracts),
                        entry_price=float(p.get('entryPrice') or 0),
                        mark_price=float(p.get('markPrice') or 0),
                        unrealized_pnl=float(p.get('unrealizedPnl') or 0),
                        leverage=int(p.get('leverage') or 1),
                        liquidation_price=float(p['liquidationPrice']) if p.get('liquidationPrice') else None,
                        margin_type=p.get('marginMode', 'cross'),
                        info=p.get('info', {}),
                    ))
            
            return result
            
        except Exception as e:
            print(f"❌ Failed to fetch positions: {e}", flush=True)
            return []


    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        Set leverage for a symbol.
        
        Args:
            symbol: Futures symbol
            leverage: Leverage multiplier
            
        Returns:
            bool: True if successful or not needed
        """
        # Check if exchange supports setLeverage
        if not self.exchange.has.get('setLeverage', False):
            # Exchange doesn't support leverage setting
            # This is fine - use exchange/contract defaults
            return True
        
        try:
            self.exchange.set_leverage(leverage, symbol)
            print(f"✅ Set leverage for {symbol} to {leverage}x", flush=True)
            return True
        except Exception as e:
            # Kraken returns error for non-flexible futures - treat as non-fatal
            error_str = str(e).lower()
            if 'not_flexible' in error_str or 'contract_not_flexible' in error_str:
                # Contract doesn't support leverage setting - use defaults
                return True
            print(f"❌ Failed to set leverage for {symbol}: {e}", flush=True)
            return False


    def set_margin_mode(self, symbol: str, mode: str = "cross") -> bool:
        """
        Set margin mode for a symbol.
        
        Args:
            symbol: Futures symbol
            mode: 'cross' or 'isolated'
            
        Returns:
            bool: True if successful or not needed
        """
        # Check if exchange supports setMarginMode
        if not self.exchange.has.get('setMarginMode', False):
            # Exchange doesn't support margin mode setting (e.g., Kraken Futures)
            # This is fine - use exchange defaults
            return True
        
        try:
            self.exchange.set_margin_mode(mode, symbol)
            print(f"✅ Set margin mode for {symbol} to {mode}", flush=True)
            return True
        except ccxt.MarginModeAlreadySet:
            # Already set to this mode, that's fine
            return True
        except Exception as e:
            print(f"❌ Failed to set margin mode for {symbol}: {e}", flush=True)
            return False


    # ==================================================
    # TRADING - OPEN POSITIONS
    # ==================================================


    def open_long(
        self,
        symbol: str,
        notional: float = None,
        qty: float = None,
        leverage: int = None,
    ) -> Optional[FuturesOrder]:
        """
        Open a long (buy) futures position.
        
        Args:
            symbol: Futures symbol
            notional: USD value to buy (alternative to qty)
            qty: Quantity in base asset (alternative to notional)
            leverage: Leverage to use (default: self.default_leverage)
            
        Returns:
            FuturesOrder or None on failure
            
        Example:
            >>> order = trader.open_long('BTC/USD:USD', notional=100.0)
        """
        if notional is None and qty is None:
            raise ValueError("Either notional or qty must be provided")
        
        try:
            # Set leverage if specified
            lev = leverage if leverage is not None else self.default_leverage
            self.set_leverage(symbol, lev)
            self.set_margin_mode(symbol, self.margin_mode)
            
            # Calculate quantity from notional if needed
            if qty is None:
                ticker = self.exchange.fetch_ticker(symbol)
                price = float(ticker['last'])
                qty = notional / price
            
            # Round to valid precision
            market = self.exchange.market(symbol)
            qty = self.exchange.amount_to_precision(symbol, qty)
            
            # Submit market buy order
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side='buy',
                amount=float(qty),
            )
            
            print(f"🟢 Opened LONG {symbol}: qty={qty}", flush=True)
            
            return FuturesOrder(
                id=order['id'],
                symbol=order['symbol'],
                side=order['side'],
                type=order['type'],
                amount=float(order.get('amount', qty)),
                price=float(order['price']) if order.get('price') else None,
                cost=float(order.get('cost', 0)),
                status=order['status'],
                timestamp=datetime.fromtimestamp(order['timestamp'] / 1000, tz=ZoneInfo("UTC")),
                info=order.get('info', {}),
            )
            
        except Exception as e:
            print(f"❌ Failed to open long {symbol}: {e}", flush=True)
            return None


    def open_short(
        self,
        symbol: str,
        notional: float = None,
        qty: float = None,
        leverage: int = None,
    ) -> Optional[FuturesOrder]:
        """
        Open a short (sell) futures position.
        
        Args:
            symbol: Futures symbol
            notional: USD value to short (alternative to qty)
            qty: Quantity in base asset (alternative to notional)
            leverage: Leverage to use (default: self.default_leverage)
            
        Returns:
            FuturesOrder or None on failure
            
        Example:
            >>> order = trader.open_short('BTC/USD:USD', notional=100.0)
        """
        if notional is None and qty is None:
            raise ValueError("Either notional or qty must be provided")
        
        try:
            # Set leverage if specified
            lev = leverage if leverage is not None else self.default_leverage
            self.set_leverage(symbol, lev)
            self.set_margin_mode(symbol, self.margin_mode)
            
            # Calculate quantity from notional if needed
            if qty is None:
                ticker = self.exchange.fetch_ticker(symbol)
                price = float(ticker['last'])
                qty = notional / price
            
            # Round to valid precision
            market = self.exchange.market(symbol)
            qty = self.exchange.amount_to_precision(symbol, qty)
            
            # Submit market sell order
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side='sell',
                amount=float(qty),
            )
            
            print(f"🔴 Opened SHORT {symbol}: qty={qty}", flush=True)
            
            return FuturesOrder(
                id=order['id'],
                symbol=order['symbol'],
                side=order['side'],
                type=order['type'],
                amount=float(order.get('amount', qty)),
                price=float(order['price']) if order.get('price') else None,
                cost=float(order.get('cost', 0)),
                status=order['status'],
                timestamp=datetime.fromtimestamp(order['timestamp'] / 1000, tz=ZoneInfo("UTC")),
                info=order.get('info', {}),
            )
            
        except Exception as e:
            print(f"❌ Failed to open short {symbol}: {e}")
            return None


    def close_position(self, symbol: str) -> Optional[FuturesOrder]:
        """
        Close an existing position for a symbol.
        
        Args:
            symbol: Futures symbol
            
        Returns:
            FuturesOrder or None on failure/no position
            
        Example:
            >>> order = trader.close_position('BTC/USD:USD')
        """
        try:
            position = self.get_position(symbol)
            
            if position is None:
                print(f"⚠️ No position to close for {symbol}", flush=True)
                return None
            
            # Close by submitting opposite order
            close_side = 'sell' if position.side == 'long' else 'buy'
            qty = position.contracts
            
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=close_side,
                amount=qty,
                params={'reduceOnly': True},
            )
            
            pnl_str = f"+{position.unrealized_pnl:.2f}" if position.unrealized_pnl >= 0 else f"{position.unrealized_pnl:.2f}"
            print(f"🔚 Closed {position.side.upper()} {symbol}: qty={qty}, PnL={pnl_str}", flush=True)
            
            return FuturesOrder(
                id=order['id'],
                symbol=order['symbol'],
                side=order['side'],
                type=order['type'],
                amount=float(order.get('amount', qty)),
                price=float(order['price']) if order.get('price') else None,
                cost=float(order.get('cost', 0)),
                status=order['status'],
                timestamp=datetime.fromtimestamp(order['timestamp'] / 1000, tz=ZoneInfo("UTC")),
                info=order.get('info', {}),
            )
            
        except Exception as e:
            print(f"❌ Failed to close position {symbol}: {e}", flush=True)
            return None


    def close_all_positions(self) -> List[FuturesOrder]:
        """
        Close all open positions.
        
        Returns:
            List of FuturesOrder for each closed position
        """
        orders = []
        positions = self.get_all_positions()
        
        for pos in positions:
            order = self.close_position(pos.symbol)
            if order:
                orders.append(order)
        
        return orders


    # ==================================================
    # MARKET DATA - HISTORICAL
    # ==================================================


    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1h',
        limit: int = 500,
        since: datetime = None,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.
        
        Args:
            symbol: Futures symbol
            timeframe: Candle timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            limit: Number of candles to fetch
            since: Start time (optional)
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            since_ms = int(since.timestamp() * 1000) if since else None
            
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                since=since_ms,
            )
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"❌ Failed to fetch OHLCV for {symbol}: {e}", flush=True)
            return pd.DataFrame()


    # ==================================================
    # REAL-TIME STREAMING
    # ==================================================


    def start_real_time_streaming(
        self,
        symbols: List[str],
        data_types: List[str] = None,
    ) -> bool:
        """
        Start real-time WebSocket streaming for futures prices.
        
        Connects to the configured exchange's WebSocket and streams live market data.
        Includes automatic reconnection with exponential backoff and
        heartbeat monitoring to detect zombie connections.
        
        Args:
            symbols: Futures symbols to stream
            data_types: Data types to subscribe - any of ['tickers', 'trades'].
                        Defaults to ['tickers'].
        
        Returns:
            bool: True if streaming started, False on error
            
        Example:
            >>> trader.start_real_time_streaming(['BTC/USD:USD', 'ETH/USD:USD'])
            >>> trader.wait_for_data(['BTC/USD:USD'])
            >>> price = trader.get_latest_price('BTC/USD:USD')
        """
        if data_types is None:
            data_types = ["tickers"]
        
        # ----- store for potential reconnection -----
        self._streaming_symbols = symbols
        self._streaming_data_types = data_types
        self._stop_streaming.clear()
        self._connection_healthy.clear()
        
        def run_stream_with_reconnect() -> None:
            while not self._stop_streaming.is_set():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    self._streaming_active = True
                    self._reconnect_attempts = 0
                    
                    print(f"✅ Started real-time streaming for {symbols}", flush=True)
                    print(f"📡 Subscribing to data types: {data_types}", flush=True)
                    
                    self._start_heartbeat_monitor()
                    
                    # Run the async streaming
                    loop.run_until_complete(self._stream_loop(symbols, data_types))
                    
                except Exception as e:
                    self._streaming_active = False
                    self._connection_healthy.clear()
                    
                    if self._stop_streaming.is_set():
                        print("Streaming stopped by user request", flush=True)
                        break
                    
                    self._reconnect_attempts += 1
                    if self._reconnect_attempts > self._max_reconnect_attempts:
                        print(f"❌ Max reconnection attempts ({self._max_reconnect_attempts}) exceeded. Giving up.", flush=True)
                        break
                    
                    # ----- exponential backoff -----
                    delay = self._reconnect_delay_base * (2 ** (self._reconnect_attempts - 1))
                    delay = min(delay, 60.0)
                    print(f"⚠️ Stream disconnected: {e}. Reconnecting in {delay:.1f}s (attempt {self._reconnect_attempts}/{self._max_reconnect_attempts})", flush=True)
                    
                    # ----- wait before reconnecting (interruptible) -----
                    if self._stop_streaming.wait(timeout=delay):
                        break
                
                finally:
                    try:
                        loop.close()
                    except:
                        pass
            
            self._streaming_active = False
            print("🔚 Streaming thread exited", flush=True)
        
        if not self._streaming_active:
            self._stream_thread = threading.Thread(
                target=run_stream_with_reconnect,
                daemon=True,
                name="CCXTStreamThread"
            )
            self._stream_thread.start()
            return True
        else:
            print("Streaming already active", flush=True)
            return True


    async def _stream_loop(self, symbols: List[str], data_types: List[str]) -> None:
        """Main async streaming loop"""
        try:
            while not self._stop_streaming.is_set():
                tasks = []
                
                for symbol in symbols:
                    if "tickers" in data_types:
                        tasks.append(self._watch_ticker(symbol))
                    if "trades" in data_types:
                        tasks.append(self._watch_trades(symbol))
                
                if tasks:
                    # Run all watchers concurrently
                    await asyncio.gather(*tasks)
                else:
                    await asyncio.sleep(0.1)
                    
        except asyncio.CancelledError:
            pass
        finally:
            await self.ws_exchange.close()


    async def _watch_ticker(self, symbol: str) -> None:
        """Watch ticker updates for a symbol"""
        try:
            while not self._stop_streaming.is_set():
                ticker = await self.ws_exchange.watch_ticker(symbol)
                
                with self._stream_lock:
                    self._latest_tickers[symbol] = ticker
                    
                    if ticker.get('last'):
                        price = float(ticker['last'])
                        self._latest_prices[symbol] = price
                        
                        self._price_buffer[symbol].append({
                            'price': price,
                            'timestamp': datetime.now(ZoneInfo("UTC")),
                        })
                        
                        self._data_received_count[symbol] = self._data_received_count.get(symbol, 0) + 1
                        
                        if self._debug_streaming and symbol not in self._first_data_logged:
                            self._first_data_logged.add(symbol)
                            print(f"📥 First TICKER data received for {symbol}: ${price:.2f}", flush=True)
                    
                    now = datetime.now(ZoneInfo("UTC"))
                    self._last_data_received[symbol] = now
                    self._last_any_data_received = now
                    self._connection_healthy.set()
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if not self._stop_streaming.is_set():
                print(f"Error in ticker stream for {symbol}: {e}", flush=True)
                raise


    async def _watch_trades(self, symbol: str) -> None:
        """Watch trade updates for a symbol"""
        try:
            while not self._stop_streaming.is_set():
                trades = await self.ws_exchange.watch_trades(symbol)
                
                if trades:
                    latest_trade = trades[-1]
                    
                    with self._stream_lock:
                        self._latest_trades[symbol] = latest_trade
                        
                        if latest_trade.get('price'):
                            price = float(latest_trade['price'])
                            self._latest_prices[symbol] = price
                            
                            self._price_buffer[symbol].append({
                                'price': price,
                                'volume': float(latest_trade.get('amount', 0)),
                                'timestamp': datetime.now(ZoneInfo("UTC")),
                            })
                            
                            self._data_received_count[symbol] = self._data_received_count.get(symbol, 0) + 1
                            
                            if self._debug_streaming and symbol not in self._first_data_logged:
                                self._first_data_logged.add(symbol)
                                print(f"📥 First TRADE data received for {symbol}: ${price:.2f}", flush=True)
                        
                        now = datetime.now(ZoneInfo("UTC"))
                        self._last_data_received[symbol] = now
                        self._last_any_data_received = now
                        self._connection_healthy.set()
                        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if not self._stop_streaming.is_set():
                print(f"Error in trade stream for {symbol}: {e}", flush=True)
                raise


    def stop_real_time_streaming(self) -> None:
        """
        Stop real-time streaming gracefully.
        
        Stops the WebSocket connection, heartbeat monitor, and clears all
        cached price data. Safe to call multiple times.
        """
        if self._streaming_active or self._stream_thread is not None:
            print("Stopping real-time streaming...")
            self._stop_streaming.set()
            self._streaming_active = False
            
            self._stop_heartbeat_monitor()
            
            # ----- give the thread a moment to clean up -----
            if self._stream_thread is not None and self._stream_thread.is_alive():
                self._stream_thread.join(timeout=5.0)
            
            # ----- clear state -----
            self._streaming_symbols = []
            self._streaming_data_types = []
            self._connection_healthy.clear()
            self._last_any_data_received = None
            print("✅ Real-time streaming stopped")


    # ==================================================
    # HEARTBEAT MONITOR
    # ==================================================


    def _start_heartbeat_monitor(self) -> None:
        """
        Start the heartbeat monitoring thread.
        
        The heartbeat monitor watches for "zombie connections" - situations where
        the WebSocket appears connected but no data is flowing. If no data is
        received from ANY symbol for an extended period, it triggers a reconnect.
        """
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return  # already running
        
        self._heartbeat_stop.clear()
        
        def monitor():
            print("💓 Heartbeat monitor started")
            while not self._heartbeat_stop.is_set():
                
                for _ in range(int(self._heartbeat_interval)):
                    if self._heartbeat_stop.is_set():
                        break
                    time.sleep(1.0)
                
                if self._heartbeat_stop.is_set():
                    break
                
                if not self._streaming_active:
                    continue
                
                with self._stream_lock:
                    last_data = self._last_any_data_received
                
                if last_data is None:
                    continue
                
                age = (datetime.now(ZoneInfo("UTC")) - last_data).total_seconds()
                timeout = self._heartbeat_interval * self._heartbeat_timeout_multiplier
                
                if age > timeout:
                    print(f"💔 Heartbeat timeout: No data from ANY symbol for {age:.0f}s "
                          f"(threshold: {timeout:.0f}s). Triggering reconnect...")
                    self._trigger_heartbeat_reconnect()
            
            print("💓 Heartbeat monitor stopped")
        
        self._heartbeat_thread = threading.Thread(
            target=monitor,
            daemon=True,
            name="HeartbeatMonitor"
        )
        self._heartbeat_thread.start()


    def _stop_heartbeat_monitor(self) -> None:
        """Stop the heartbeat monitoring thread."""
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=5.0)
        self._heartbeat_thread = None


    def _trigger_heartbeat_reconnect(self) -> None:
        """
        Trigger a reconnection due to heartbeat timeout.
        
        This attempts to gracefully restart the WebSocket connection
        when a zombie connection is detected.
        """
        if not self._streaming_active:
            return
        
        # Save current config
        symbols = self._streaming_symbols.copy()
        data_types = self._streaming_data_types.copy()
        
        if not symbols:
            print("⚠️ Cannot reconnect - no symbols configured")
            return
        
        print(f"🔄 Heartbeat reconnect: Restarting stream for {len(symbols)} symbols...")
        
        try:
            self._streaming_active = False
            self._stop_streaming.set()
            
            if self._stream_thread is not None and self._stream_thread.is_alive():
                self._stream_thread.join(timeout=5.0)
            
            time.sleep(2.0)
            
            # ----- restart streaming -----
            self._stop_streaming.clear()
            self._reconnect_attempts += 1
            
            # ----- start streaming again -----
            self.start_real_time_streaming(symbols, data_types)
            
            print(f"✅ Heartbeat reconnect successful")
            
        except Exception as e:
            print(f"❌ Heartbeat reconnect failed: {e}")
            self._reconnect_attempts += 1


    # ==================================================
    # STREAMING DATA ACCESS
    # ==================================================


    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get the latest cached price from WebSocket stream.
        
        Returns the most recent price received via streaming. Note that
        this may be stale if no recent trades/quotes occurred. Use
        get_connection_health() to check data freshness.
        
        Args:
            symbol: Futures symbol
            
        Returns:
            float: Latest price, or None if no data received yet
        """
        with self._stream_lock:
            return self._latest_prices.get(symbol)


    def get_latest_bar(self, symbol: str) -> Optional[Dict]:
        """
        Get the latest bar for a symbol from real-time data.
        
        Args:
            symbol: Futures symbol
            
        Returns:
            Dict with OHLCV data or None if not available
        """
        with self._stream_lock:
            return self._latest_bars.get(symbol)


    def get_real_time_ohlc(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Get real-time OHLC data for a symbol.
        
        Args:
            symbol: Futures symbol
            
        Returns:
            Dictionary with 'open', 'high', 'low', 'close', 'volume' or None
        """
        with self._stream_lock:
            ticker = self._latest_tickers.get(symbol).get('info')
            if ticker:
                return {
                    "open": float(ticker.get('open', 0)),
                    "high": float(ticker.get('high', 0)),
                    "low": float(ticker.get('low', 0)),
                    "close": float(ticker.get('last', 0)),
                    "volume": float(ticker.get('quoteVolume', 0)),
                    "timestamp": datetime.now(ZoneInfo("UTC")),
                }
            return None


    def get_price_history(self, symbol: str, limit: int = 100) -> List[Dict]:
        """
        Get recent price history from the buffer.
        
        Args:
            symbol: Futures symbol
            limit: Maximum number of data points to return
            
        Returns:
            List of price data dictionaries
        """
        with self._stream_lock:
            buffer_data = list(self._price_buffer.get(symbol, []))
            return buffer_data[-limit:] if buffer_data else []


    def get_bar_history(self, symbol: str, limit: int = 50) -> List[Dict]:
        """
        Get recent bar history from the buffer.
        
        Args:
            symbol: Futures symbol
            limit: Maximum number of bars to return
            
        Returns:
            List of bar data dictionaries
        """
        with self._stream_lock:
            buffer_data = list(self._bar_buffer.get(symbol, []))
            return buffer_data[-limit:] if buffer_data else []


    def get_all_streaming_symbols(self) -> List[str]:
        """
        Get list of all symbols currently being streamed.
        
        Returns:
            List of symbols with active data streams
        """
        with self._stream_lock:
            return list(set(
                list(self._latest_prices.keys()) +
                list(self._latest_bars.keys()) +
                list(self._latest_tickers.keys()) +
                list(self._latest_trades.keys())
            ))


    def is_streaming_active(self) -> bool:
        """Check if real-time streaming is active"""
        return self._streaming_active


    def wait_for_data(
        self,
        symbols: List[str] = None,
        timeout_secs: float = 30.0,
    ) -> bool:
        """
        Block until price data is received for specified symbols.
        
        Call after start_real_time_streaming() to ensure data is flowing
        before executing trading logic. Polls every 100ms until all
        symbols have received at least one price update.
        
        Args:
            symbols: Symbols to wait for (default: all streaming symbols)
            timeout_secs: Max seconds to wait (default: 30)
            
        Returns:
            bool: True if all symbols received data, False on timeout
        """
        if symbols is None:
            symbols = self._streaming_symbols
        
        if not symbols:
            return True
        
        start_time = datetime.now()
        check_interval = 0.1  # seconds
        
        while (datetime.now() - start_time).total_seconds() < timeout_secs:
            with self._stream_lock:
                all_have_data = all(
                    s in self._latest_prices or s in self._latest_bars
                    for s in symbols
                )
            
            if all_have_data:
                return True
            
            time.sleep(check_interval)
        
        # ----- report which symbols are missing data -----
        with self._stream_lock:
            missing = [
                s for s in symbols
                if s not in self._latest_prices and s not in self._latest_bars
            ]
        
        print(f"⚠️ Timeout waiting for data. Missing symbols: {missing}")
        return False


    # ==================================================
    # CONNECTION HEALTH & DEBUG
    # ==================================================


    def get_connection_health(self, staleness: float = 60.0) -> Dict[str, Any]:
        """
        Get comprehensive health status of the streaming connection.
        
        Returns staleness info per symbol, heartbeat status, and connection
        state. Use to detect data issues and stale prices before trading.
        
        Args:
            staleness: Threshold in seconds to consider data stale (default: 60)
            
        Returns:
            dict: Health metrics with structure:
                {
                    'streaming_active': bool,
                    'connection_healthy': bool,
                    'reconnect_attempts': int,
                    'symbols': {symbol: {'last_data_age_secs': float, 'healthy': bool}},
                    'heartbeat': {'healthy': bool, 'last_any_data_age_secs': float}
                }
        """
        with self._stream_lock:
            now = datetime.now(ZoneInfo("UTC"))
            symbol_status = {}
            
            for s in self._streaming_symbols:
                last_received = self._last_data_received.get(s)
                
                if last_received:
                    age_secs = (now - last_received).total_seconds()
                    symbol_status[s] = {
                        "last_data_age_secs": age_secs,
                        "healthy": age_secs < staleness,
                        "has_price": s in self._latest_prices,
                        "has_bar": s in self._latest_bars,
                    }
                else:
                    symbol_status[s] = {
                        "last_data_age_secs": None,
                        "healthy": False,
                        "has_price": s in self._latest_prices,
                        "has_bar": s in self._latest_bars,
                    }
            
            # ----- calculate heartbeat status -----
            heartbeat_age = None
            heartbeat_healthy = False
            if self._last_any_data_received is not None:
                heartbeat_age = (now - self._last_any_data_received).total_seconds()
                timeout = self._heartbeat_interval * self._heartbeat_timeout_multiplier
                heartbeat_healthy = heartbeat_age < timeout
            
            return {
                "streaming_active": self._streaming_active,
                "connection_healthy": self._connection_healthy.is_set(),
                "reconnect_attempts": self._reconnect_attempts,
                "exchange": EXCHANGE_CONFIG["name"],
                "symbols": symbol_status,
                "heartbeat": {
                    "last_any_data_age_secs": heartbeat_age,
                    "healthy": heartbeat_healthy,
                    "interval_secs": self._heartbeat_interval,
                    "timeout_secs": self._heartbeat_interval * self._heartbeat_timeout_multiplier,
                    "monitor_active": self._heartbeat_thread is not None and self._heartbeat_thread.is_alive(),
                },
            }


    def get_streaming_debug_info(self) -> str:
        """
        Get detailed debug information about streaming status.
        
        Returns:
            Formatted string with debug information
        """
        with self._stream_lock:
            lines = [
                "=" * 60,
                "🔍 STREAMING DEBUG INFO",
                "=" * 60,
                f"Exchange: {EXCHANGE_CONFIG['name']}",
                f"Streaming active flag: {self._streaming_active}",
                f"Stream thread alive: {self._stream_thread.is_alive() if self._stream_thread else 'No thread'}",
                f"Connection healthy event: {self._connection_healthy.is_set()}",
                f"Stop streaming event: {self._stop_streaming.is_set()}",
                f"Reconnect attempts: {self._reconnect_attempts}",
                f"Subscribed symbols: {self._streaming_symbols}",
                f"Subscribed data types: {self._streaming_data_types}",
                "",
                "DATA RECEPTION:",
            ]
            
            for s in self._streaming_symbols:
                count = self._data_received_count.get(s, 0)
                price = self._latest_prices.get(s)
                last_recv = self._last_data_received.get(s)
                
                if last_recv:
                    age = (datetime.now(ZoneInfo("UTC")) - last_recv).total_seconds()
                    age_str = f"{age:.1f}s ago"
                else:
                    age_str = "never"
                
                price_str = f"${price:.2f}" if price else "N/A"
                lines.append(f"  {s}: {count} msgs, price={price_str}, last={age_str}")
            
            lines.append("=" * 60)
            return "\n".join(lines)


    def get_market_data_summary(self) -> Dict[str, Dict]:
        """
        Get a summary of all available real-time market data.
        
        Returns:
            Dictionary with symbol as key and data summary as value
        """
        with self._stream_lock:
            summary = {}
            all_symbols = self.get_all_streaming_symbols()
            
            for s in all_symbols:
                summary[s] = {
                    "latest_price": self._latest_prices.get(s),
                    "has_bar_data": s in self._latest_bars,
                    "has_ticker_data": s in self._latest_tickers,
                    "has_trade_data": s in self._latest_trades,
                    "price_buffer_size": len(self._price_buffer.get(s, [])),
                    "bar_buffer_size": len(self._bar_buffer.get(s, [])),
                }
            
            return summary


    def get_exchange_info(self) -> Dict[str, Any]:
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
            "quote_currency": self._quote_currency,
            "available_symbols": len(self.crypto_universe),
        }
