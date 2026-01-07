#!/usr/bin/env python3
"""
Finance Module Configuration

This module provides exchange-agnostic configuration for cryptocurrency trading.
The exchange can be configured via the EXCHANGE_NAME environment variable.

Supported Exchanges:
    - kraken (Kraken Futures) - Default, US-friendly
    - binance (Binance Futures) - International only
    - binanceus (Binance.US) - US spot only, no futures
    - bybit (Bybit) - Not available in US

Environment Variables:
    EXCHANGE_NAME: Exchange to use (default: 'kraken')
    {EXCHANGE}_API_KEY: API key for the exchange
    {EXCHANGE}_SECRET_KEY: Secret key for the exchange
    {EXCHANGE}_DEMO: Use demo/testnet mode (default: 'true')

Example:
    export EXCHANGE_NAME=kraken
    export KRAKEN_API_KEY=your_key
    export KRAKEN_SECRET_KEY=your_secret
    export KRAKEN_DEMO=true
"""

# ===== environment variables =====

from datetime import datetime, timedelta
import os
from typing import Dict, List
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

try:
    load_dotenv()
except Exception as e:
    print(f"Caught Exception trying to load '.env'", flush=True)


# ==================================================
# TRADING BOT DEFAULT PARAMETERS
# ==================================================    
    
MAX_SPREAD_NOTIONAL = 200.0 # maxmimum $ amount to bet on a spread position
MIN_SPREAD_NOTIONAL = 20.0 # minimum $ amount to bet on a spread position
MAX_SPREAD_POSITIONS = 10 # maximum number of allowed spread positions open at once


# ==================================================
# EXCHANGE CONFIGURATION
# ==================================================

# Supported exchanges and their CCXT identifiers
SUPPORTED_EXCHANGES = {
    "kraken": {
        "ccxt_id": "krakenfutures",
        "ccxt_ws_id": "krakenfutures",
        "name": "Kraken Futures",
        "has_futures": True,
        "us_available": True,
        "symbol_format": "BTC/USD:USD",  # Example format
        "quote_currency": "USD",
        "settlement_currency": "USD",
    },
    "binance": {
        "ccxt_id": "binance",
        "ccxt_ws_id": "binance",
        "name": "Binance Futures",
        "has_futures": True,
        "us_available": False,
        "symbol_format": "BTC/USDT:USDT",
        "quote_currency": "USDT",
        "settlement_currency": "USDT",
    },
    "binanceus": {
        "ccxt_id": "binanceus",
        "ccxt_ws_id": "binanceus",
        "name": "Binance.US",
        "has_futures": False,
        "us_available": True,
        "symbol_format": "BTC/USD",
        "quote_currency": "USD",
        "settlement_currency": None,
    },
    "bybit": {
        "ccxt_id": "bybit",
        "ccxt_ws_id": "bybit",
        "name": "Bybit",
        "has_futures": True,
        "us_available": False,
        "symbol_format": "BTC/USDT:USDT",
        "quote_currency": "USDT",
        "settlement_currency": "USDT",
    },
}

# Get configured exchange (default to Kraken for US users)
EXCHANGE_NAME = os.getenv("EXCHANGE_NAME", "kraken").lower()

if EXCHANGE_NAME not in SUPPORTED_EXCHANGES:
    raise ValueError(
        f"Unsupported exchange: {EXCHANGE_NAME}. "
        f"Supported: {list(SUPPORTED_EXCHANGES.keys())}"
    )

EXCHANGE_CONFIG = SUPPORTED_EXCHANGES[EXCHANGE_NAME]

# Build environment variable names based on exchange
_exchange_upper = EXCHANGE_NAME.upper()
API_KEY_VAR = f"{_exchange_upper}_API_KEY"
SECRET_KEY_VAR = f"{_exchange_upper}_SECRET_KEY"
DEMO_VAR = f"{_exchange_upper}_DEMO"

# Validate required environment variables
REQUIRED_VARS = [API_KEY_VAR, SECRET_KEY_VAR]
MISSING_VARS = [var for var in REQUIRED_VARS if not os.getenv(var)]

if MISSING_VARS:
    raise ValueError(
        f"Missing required environment variables for {EXCHANGE_CONFIG['name']}: {MISSING_VARS}\n"
        f"Please set:\n"
        f"  export {API_KEY_VAR}='your_api_key'\n"
        f"  export {SECRET_KEY_VAR}='your_secret_key'\n"
        f"  export {DEMO_VAR}='true'  # Optional, defaults to true"
    )

# Get credentials
API_KEY = os.getenv(API_KEY_VAR)
SECRET_KEY = os.getenv(SECRET_KEY_VAR)
DEMO_MODE = os.getenv(DEMO_VAR, "true").lower() == "true"

print(f"📊 Configured exchange: {EXCHANGE_CONFIG['name']}")
print(f"📋 Demo/Testnet mode: {DEMO_MODE}")


# ==================================================
# EXCHANGE-SPECIFIC SYMBOL CONFIGURATIONS
# ==================================================

# Symbol mappings per exchange
# These are the perpetual futures symbols available on each exchange
# NOTE: CCXT uses standard BTC symbol for Kraken Futures (not XBT)

EXCHANGE_SYMBOLS = {
    # Kraken Futures - CCXT unified format uses BTC (not XBT)
    # Multi-collateral perpetuals in CCXT format
    "kraken": [
        "BTC/USD:USD",   # Bitcoin
        "ETH/USD:USD",   # Ethereum
        "SOL/USD:USD",   # Solana
        "XRP/USD:USD",   # Ripple
        "LTC/USD:USD",   # Litecoin
        "BCH/USD:USD",   # Bitcoin Cash
        "AVAX/USD:USD",  # Avalanche
        "LINK/USD:USD",  # Chainlink
        "DOT/USD:USD",   # Polkadot
        "ATOM/USD:USD",  # Cosmos
        "MATIC/USD:USD", # Polygon
        "DOGE/USD:USD",  # Dogecoin
        "ADA/USD:USD",   # Cardano
    ],
    # Binance Futures - USDT-margined perpetuals
    "binance": [
        "AAVE/USDT:USDT",
        "AVAX/USDT:USDT",
        "BCH/USDT:USDT",
        "BTC/USDT:USDT",
        "CRV/USDT:USDT",
        "DOGE/USDT:USDT",
        "DOT/USDT:USDT",
        "ETH/USDT:USDT",
        "GRT/USDT:USDT",
        "LINK/USDT:USDT",
        "LTC/USDT:USDT",
        "MKR/USDT:USDT",
        "SHIB/USDT:USDT",
        "SUSHI/USDT:USDT",
        "UNI/USDT:USDT",
        "XTZ/USDT:USDT",
        "YFI/USDT:USDT",
    ],
    # Binance.US - Spot only, no futures
    "binanceus": [
        "BTC/USD",
        "ETH/USD",
        "LTC/USD",
        "BCH/USD",
        "LINK/USD",
        "UNI/USD",
        "AAVE/USD",
        "DOT/USD",
        "SOL/USD",
        "DOGE/USD",
    ],
    # Bybit - USDT-margined perpetuals
    "bybit": [
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        "SOL/USDT:USDT",
        "XRP/USDT:USDT",
        "DOGE/USDT:USDT",
        "AVAX/USDT:USDT",
        "LINK/USDT:USDT",
        "DOT/USDT:USDT",
        "MATIC/USDT:USDT",
        "UNI/USDT:USDT",
    ],
}

# Kraken alias mappings (CCXT uses standard BTC, but Kraken web UI shows XBT)
# These are kept for reference but NOT used in symbol conversion since CCXT normalizes to BTC
CANONICAL_TO_KRAKEN = {
    # "BTC": "XBT",  # Not needed - CCXT uses BTC for Kraken Futures
}
# Kraken -> Canonical: Convert any legacy XBT symbols back to standard
KRAKEN_TO_CANONICAL = {
    "XBT": "BTC",  # Convert Kraken's XBT back to BTC
}

# Get symbols for the configured exchange
CRYPTO_TICKERS = EXCHANGE_SYMBOLS.get(EXCHANGE_NAME, [])


# ==================================================
# TEMPORAL CONSTANTS
# ==================================================

NOW = datetime.now(ZoneInfo("America/New_York"))
PAST_N_YEARS = dict((n, timedelta(days=n * 365)) for n in range(10))

# CCXT-compatible timeframe strings (universal across exchanges)
TIME_FRAMES = {
    "min": "1m",
    "minute": "1m",
    "5min": "5m",
    "15min": "15m",
    "hour": "1h",
    "4hour": "4h",
    "day": "1d",
    "week": "1w",
    "month": "1M",
}


# ==================================================
# SYMBOL CONVERSION UTILITIES
# ==================================================

# Runtime symbol mapping - populated by CCXTFuturesTrader when markets are loaded
# Maps canonical base assets (e.g., "BTC") to actual exchange symbols (e.g., "BTC/USD:USD")
_RUNTIME_SYMBOL_MAP: Dict[str, str] = {}


def set_runtime_symbol_map(mapping: Dict[str, str]) -> None:
    """
    Set the runtime symbol mapping from canonical to exchange format.
    
    This is called by CCXTFuturesTrader after loading markets to ensure
    canonical_to_exchange() returns the exact symbols the exchange expects.
    
    Args:
        mapping: Dict mapping canonical base (e.g., "BTC") to exchange symbol
    """
    global _RUNTIME_SYMBOL_MAP
    _RUNTIME_SYMBOL_MAP = mapping.copy()
    print(f"✅ Set runtime symbol map with {len(mapping)} symbols")


def get_runtime_symbol_map() -> Dict[str, str]:
    """Get the current runtime symbol mapping."""
    return _RUNTIME_SYMBOL_MAP.copy()


def clear_runtime_symbol_map() -> None:
    """Clear the runtime symbol mapping."""
    global _RUNTIME_SYMBOL_MAP
    _RUNTIME_SYMBOL_MAP = {}


# Canonical base assets supported across exchanges
# These are the "common denominator" symbols used in benchmarks
CANONICAL_BASES = [
    "AAVE", "ADA", "ATOM", "AVAX", "BAT", "BCH", "BTC", "CRV",
    "DOGE", "DOT", "ETH", "GRT", "LINK", "LTC", "MATIC", "MKR",
    "SHIB", "SOL", "SUSHI", "UNI", "USDC", "USDT", "XRP", "XTZ", "YFI",
]


def canonical_to_exchange(canonical: str, exchange: str = None) -> str:
    """
    Convert a canonical symbol to exchange-specific format.
    
    If a runtime symbol map has been set (via set_runtime_symbol_map),
    uses the actual exchange symbols. Otherwise constructs the symbol
    based on exchange configuration.
    
    Canonical format is just the base asset (e.g., "BTC", "ETH").
    Exchange format varies:
        - Kraken: "BTC/USD:USD"
        - Binance: "BTC/USDT:USDT"
        - Binance.US: "BTC/USD"
        - Bybit: "BTC/USDT:USDT"
    
    Args:
        canonical: Base asset symbol (e.g., "BTC", "ETH")
        exchange: Exchange name (default: current EXCHANGE_NAME)
        
    Returns:
        Exchange-specific symbol string
        
    Example:
        >>> canonical_to_exchange("BTC", "kraken")
        'BTC/USD:USD'
        >>> canonical_to_exchange("ETH", "binance")
        'ETH/USDT:USDT'
    """
    if exchange is None:
        exchange = EXCHANGE_NAME
    
    exchange = exchange.lower()
    base = canonical.upper().strip()
    
    # Handle case where full symbol was passed (extract base)
    if "/" in base:
        base = base.split("/")[0]
    
    # Use runtime symbol map if available (most accurate)
    if _RUNTIME_SYMBOL_MAP and base in _RUNTIME_SYMBOL_MAP:
        return _RUNTIME_SYMBOL_MAP[base]
    
    # Handle exchange-specific symbol aliases (e.g., Kraken uses XBT for BTC)
    if exchange == "kraken" and base in CANONICAL_TO_KRAKEN:
        base = CANONICAL_TO_KRAKEN[base]
    
    config = SUPPORTED_EXCHANGES.get(exchange, {})
    quote = config.get("quote_currency", "USD")
    settlement = config.get("settlement_currency")
    
    if settlement:
        return f"{base}/{quote}:{settlement}"
    else:
        return f"{base}/{quote}"


def exchange_to_canonical(symbol: str, exchange: str = None) -> str:
    """
    Convert an exchange-specific symbol to canonical format.
    
    Extracts just the base asset from any exchange format.
    Handles exchange-specific aliases (e.g., Kraken's XBT -> BTC).
    
    Args:
        symbol: Exchange-specific symbol (e.g., "XBT/USD:USD", "ETH/USDT:USDT")
        exchange: Exchange name for alias resolution (default: current EXCHANGE_NAME)
        
    Returns:
        Canonical base asset (e.g., "BTC", "ETH")
        
    Example:
        >>> exchange_to_canonical("XBT/USD:USD")
        'BTC'
        >>> exchange_to_canonical("ETH/USDT:USDT")
        'ETH'
    """
    if not symbol:
        return ""
    
    if exchange is None:
        exchange = EXCHANGE_NAME
    
    # Extract base from "BASE/QUOTE:SETTLEMENT" or "BASE/QUOTE"
    base = symbol.split("/")[0].upper().strip()
    
    # Handle exchange-specific aliases (e.g., Kraken XBT -> BTC canonical)
    if exchange.lower() == "kraken" and base in KRAKEN_TO_CANONICAL:
        base = KRAKEN_TO_CANONICAL[base]
    
    return base


def convert_symbol_list(symbols: list, target_exchange: str = None, source_exchange: str = None) -> list:
    """
    Convert a list of symbols to the target exchange format.
    
    Args:
        symbols: List of symbols in any format
        target_exchange: Target exchange (default: current EXCHANGE_NAME)
        source_exchange: Source exchange for alias resolution (default: current EXCHANGE_NAME)
        
    Returns:
        List of symbols in target exchange format
    """
    if target_exchange is None:
        target_exchange = EXCHANGE_NAME
    if source_exchange is None:
        source_exchange = EXCHANGE_NAME
    
    result = []
    for sym in symbols:
        canonical = exchange_to_canonical(sym, source_exchange)
        exchange_sym = canonical_to_exchange(canonical, target_exchange)
        result.append(exchange_sym)
    
    return result


def symbol_to_exchange_format(symbol: str, exchange: str = None, source_exchange: str = None) -> str:
    """
    Ensure a symbol is in the correct format for the specified exchange.
    
    Handles symbols in any format (canonical, old Alpaca, or other exchange).
    
    Args:
        symbol: Symbol in any format
        exchange: Target exchange (default: current EXCHANGE_NAME)
        source_exchange: Source exchange for alias resolution (default: current EXCHANGE_NAME)
        
    Returns:
        Symbol in correct exchange format
    """
    if source_exchange is None:
        source_exchange = exchange if exchange else EXCHANGE_NAME
    canonical = exchange_to_canonical(symbol, source_exchange)
    return canonical_to_exchange(canonical, exchange)


def get_canonical_symbols() -> list:
    """
    Get canonical (base asset) versions of all configured exchange symbols.
    
    Returns:
        List of canonical base assets for the current exchange
    """
    return [exchange_to_canonical(s) for s in CRYPTO_TICKERS]


def is_symbol_available(canonical: str, exchange: str = None) -> bool:
    """
    Check if a canonical symbol is available on the specified exchange.
    
    Args:
        canonical: Base asset (e.g., "BTC")
        exchange: Exchange to check (default: current EXCHANGE_NAME)
        
    Returns:
        True if symbol is available on the exchange
    """
    if exchange is None:
        exchange = EXCHANGE_NAME
    
    exchange_symbols = EXCHANGE_SYMBOLS.get(exchange.lower(), [])
    available_bases = {exchange_to_canonical(s) for s in exchange_symbols}
    
    return canonical.upper() in available_bases


def get_common_symbols(exchanges: list = None) -> list:
    """
    Get canonical symbols available on all specified exchanges.
    
    Args:
        exchanges: List of exchange names (default: all supported)
        
    Returns:
        List of canonical base assets available on all exchanges
    """
    if exchanges is None:
        exchanges = list(SUPPORTED_EXCHANGES.keys())
    
    # Get canonical symbols for each exchange
    base_sets = []
    for ex in exchanges:
        ex_symbols = EXCHANGE_SYMBOLS.get(ex.lower(), [])
        bases = {exchange_to_canonical(s) for s in ex_symbols}
        base_sets.append(bases)
    
    if not base_sets:
        return []
    
    # Find intersection
    common = base_sets[0]
    for s in base_sets[1:]:
        common = common.intersection(s)
    
    return sorted(list(common))


# ==================================================
# DYNAMIC SYMBOL FETCHING
# ==================================================


def fetch_available_symbols(
    exchange_name: str = None,
    testnet: bool = None,
    filter_perpetual: bool = True,
) -> List[str]:
    """
    Dynamically fetch available symbols from the exchange.
    
    This queries the actual exchange (testnet or production) to get
    the list of currently tradeable symbols. This is the recommended
    way to get symbols rather than using the hardcoded EXCHANGE_SYMBOLS.
    
    Args:
        exchange_name: Exchange to query (default: current EXCHANGE_NAME)
        testnet: Use testnet/sandbox mode (default: from DEMO_MODE)
        filter_perpetual: Only return perpetual/futures contracts
        
    Returns:
        List of available symbols in CCXT unified format
        
    Example:
        >>> symbols = fetch_available_symbols()
        >>> print(f"Available on {EXCHANGE_NAME}: {len(symbols)} symbols")
        >>> for s in symbols[:5]:
        ...     print(f"  {s}")
    """
    # Import ccxt here to avoid circular imports at module load time
    import ccxt
    
    if exchange_name is None:
        exchange_name = EXCHANGE_NAME
    
    if testnet is None:
        testnet = DEMO_MODE
    
    config = SUPPORTED_EXCHANGES.get(exchange_name.lower())
    if not config:
        raise ValueError(f"Unsupported exchange: {exchange_name}")
    
    ccxt_id = config["ccxt_id"]
    
    if not hasattr(ccxt, ccxt_id):
        raise ValueError(f"CCXT does not support exchange: {ccxt_id}")
    
    exchange_class = getattr(ccxt, ccxt_id)
    
    # ----- minimal client for market fetching (no auth needed) -----
    client_config = {'enableRateLimit': True}
    
    if exchange_name.lower() == "binance":
        client_config["options"] = {'defaultType': 'future'}
    elif exchange_name.lower() == "bybit":
        client_config["options"] = {'defaultType': 'swap'}
    
    exchange = exchange_class(client_config)
    
    if testnet:
        try:
            exchange.set_sandbox_mode(True)
        except Exception:
            pass
    
    try:
        exchange.load_markets()
    except Exception as e:
        print(f"⚠️ Failed to load markets from {config['name']}: {e}")
        return []
    
    # Filter to perpetual/futures contracts if requested
    available = []
    for symbol, market in exchange.markets.items():
        if filter_perpetual:
            is_perpetual = (
                market.get('type') in ('swap', 'future') or
                market.get('swap', False) or
                market.get('future', False) or
                ':' in symbol  # CCXT unified format for derivatives
            )
            if is_perpetual:
                available.append(symbol)
        else:
            available.append(symbol)
    
    return sorted(available)


def get_dynamic_crypto_tickers(
    testnet: bool = None,
    fallback_to_static: bool = True,
) -> List[str]:
    """
    Get crypto tickers, preferring dynamic fetch from exchange.
    
    Attempts to fetch available symbols from the exchange at runtime.
    Falls back to static EXCHANGE_SYMBOLS if fetch fails.
    
    Args:
        testnet: Use testnet mode (default: DEMO_MODE)
        fallback_to_static: Use static list if fetch fails (default: True)
        
    Returns:
        List of available symbols for trading
        
    Example:
        >>> tickers = get_dynamic_crypto_tickers()
        >>> print(f"Available: {tickers}")
    """
    try:
        symbols = fetch_available_symbols(testnet=testnet)
        if symbols:
            print(f"✅ Dynamically loaded {len(symbols)} symbols from {EXCHANGE_CONFIG['name']}")
            return symbols
    except Exception as e:
        print(f"⚠️ Could not fetch symbols dynamically: {e}")
    
    if fallback_to_static:
        print(f"📋 Using static symbol list ({len(CRYPTO_TICKERS)} symbols)")
        return CRYPTO_TICKERS
    
    return []


# ==================================================
# STALENESS PROFILES
# ==================================================


def _build_staleness_profiles():
    """Build staleness profiles for the configured exchange."""
    symbols = CRYPTO_TICKERS
    
    # Categorize by known liquidity tiers
    high_liquidity_bases = {"BTC", "ETH"}
    medium_liquidity_bases = {"LTC", "LINK", "SOL", "DOT", "AVAX", "BCH", "XRP", "ADA"}
    
    high = []
    medium = []
    low = []
    
    for symbol in symbols:
        base = symbol.split("/")[0]
        if base in high_liquidity_bases:
            high.append(symbol)
        elif base in medium_liquidity_bases:
            medium.append(symbol)
        else:
            low.append(symbol)
    
    return {
    "high_liquidity": {
            "symbols": high,
            "entry_staleness": 15.0,
            "exit_staleness": 60.0,
            "emergency_staleness": 300.0,
    },
    "medium_liquidity": {
            "symbols": medium,
            "entry_staleness": 60.0,
            "exit_staleness": 300.0,
            "emergency_staleness": 900.0,
    },
    "low_liquidity": {
            "symbols": low,
            "entry_staleness": 180.0,
            "exit_staleness": 600.0,
            "emergency_staleness": 1800.0,
        },
    }


STALENESS_PROFILES = _build_staleness_profiles()

DEFAULT_STALENESS = {
    "entry_staleness": 60.0,
    "exit_staleness": 300.0,
    "emergency_staleness": 900.0,
}


def get_staleness_for_symbol(symbol: str) -> dict:
    """
    Get staleness thresholds for a specific symbol based on its liquidity tier.
    
    Args:
        symbol: Crypto symbol (e.g., 'BTC/USD:USD' for Kraken)
        
    Returns:
        Dict with 'entry', 'exit', and 'emergency' staleness thresholds in seconds
    """
    for profile in STALENESS_PROFILES.values():
        if symbol in profile["symbols"]:
            return {
                "entry": profile["entry_staleness"],
                "exit": profile["exit_staleness"],
                "emergency": profile["emergency_staleness"],
            }
    return {
        "entry": DEFAULT_STALENESS["entry_staleness"],
        "exit": DEFAULT_STALENESS["exit_staleness"],
        "emergency": DEFAULT_STALENESS["emergency_staleness"],
    }


def get_staleness_for_pair(assets: list) -> dict:
    """
    Get staleness thresholds for a pair of assets.
    Uses the MOST CONSERVATIVE (shortest) threshold from either asset.
    
    Args:
        assets: List of crypto symbols
        
    Returns:
        Dict with 'entry', 'exit', and 'emergency' staleness thresholds
    """
    if not assets:
        return get_staleness_for_symbol("")
    
    all_thresholds = [get_staleness_for_symbol(asset) for asset in assets]
    
    return {
        "entry": min(t["entry"] for t in all_thresholds),
        "exit": min(t["exit"] for t in all_thresholds),
        "emergency": min(t["emergency"] for t in all_thresholds),
    }


def get_liquidity_tier(symbol: str) -> str:
    """
    Get the liquidity tier name for a symbol.
    
    Args:
        symbol: Crypto symbol
        
    Returns:
        'high_liquidity', 'medium_liquidity', 'low_liquidity', or 'unknown'
    """
    for tier_name, profile in STALENESS_PROFILES.items():
        if symbol in profile["symbols"]:
            return tier_name
    return "unknown"


# ==================================================
# MODULE EXPORTS
# ==================================================

__all__ = [
    # Exchange configuration
    "EXCHANGE_NAME",
    "EXCHANGE_CONFIG",
    "SUPPORTED_EXCHANGES",
    "API_KEY",
    "SECRET_KEY",
    "DEMO_MODE",
    # Constants
    "NOW",
    "PAST_N_YEARS",
    "TIME_FRAMES",
    "CRYPTO_TICKERS",
    "EXCHANGE_SYMBOLS",
    # Symbol conversion
    "CANONICAL_BASES",
    "CANONICAL_TO_KRAKEN",
    "KRAKEN_TO_CANONICAL",
    "canonical_to_exchange",
    "exchange_to_canonical",
    "convert_symbol_list",
    "symbol_to_exchange_format",
    "get_canonical_symbols",
    "is_symbol_available",
    "get_common_symbols",
    # Runtime symbol mapping
    "set_runtime_symbol_map",
    "get_runtime_symbol_map",
    "clear_runtime_symbol_map",
    # Dynamic symbol fetching
    "fetch_available_symbols",
    "get_dynamic_crypto_tickers",
    # Staleness profiles
    "STALENESS_PROFILES",
    "DEFAULT_STALENESS",
    "get_staleness_for_symbol",
    "get_staleness_for_pair",
    "get_liquidity_tier",
]
