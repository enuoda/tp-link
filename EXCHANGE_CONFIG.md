# Exchange Configuration Guide

This document explains how to configure and switch between different cryptocurrency exchanges in the tp-link trading bot.

## Overview

The trading bot uses the [CCXT](https://docs.ccxt.com/) library to provide a unified interface for multiple cryptocurrency exchanges. This allows you to:

1. **Switch exchanges** by changing a single environment variable
2. **Test strategies** on different exchanges without code changes
3. **Migrate between exchanges** if regulations or preferences change

## Supported Exchanges

| Exchange | ID | Futures | Shorting | US Available | Symbol Format |
|----------|-----|---------|----------|--------------|---------------|
| **Kraken Futures** | `kraken` | ✅ | ✅ | ✅ | `BTC/USD:USD` |
| **Binance Futures** | `binance` | ✅ | ✅ | ❌ | `BTC/USDT:USDT` |
| **Binance.US** | `binanceus` | ❌ | ❌ | ✅ | `BTC/USD` |
| **Bybit** | `bybit` | ✅ | ✅ | ❌ | `BTC/USDT:USDT` |

### Exchange Details

#### Kraken Futures (Recommended for US Users)
- **Best for**: US residents who need futures trading with shorting capability
- **Demo/Testnet**: Available at [demo-futures.kraken.com](https://demo-futures.kraken.com)
- **Quote Currency**: USD
- **Symbol Note**: Kraken uses `XBT` instead of `BTC` for Bitcoin. The system handles this automatically.
- **Pros**: US-friendly, regulated, good liquidity
- **Cons**: Smaller selection of trading pairs than Binance

#### Binance Futures
- **Best for**: International users seeking maximum liquidity and pair selection
- **Demo/Testnet**: Available at [testnet.binancefuture.com](https://testnet.binancefuture.com)
- **Quote Currency**: USDT
- **Pros**: Most liquid, most pairs, advanced order types
- **Cons**: Not available to US residents

#### Binance.US
- **Best for**: US users who only need spot trading
- **Demo/Testnet**: None available
- **Quote Currency**: USD
- **Pros**: US-regulated
- **Cons**: No futures, no shorting, limited pairs

#### Bybit
- **Best for**: International users as alternative to Binance
- **Demo/Testnet**: Available at [testnet.bybit.com](https://testnet.bybit.com)
- **Quote Currency**: USDT
- **Pros**: Good liquidity, competitive fees
- **Cons**: Not available to US residents

## Configuration

### Step 1: Choose Your Exchange

Set the `EXCHANGE_NAME` environment variable:

```bash
export EXCHANGE_NAME=kraken    # Kraken Futures (default)
export EXCHANGE_NAME=binance   # Binance Futures
export EXCHANGE_NAME=binanceus # Binance.US
export EXCHANGE_NAME=bybit     # Bybit
```

### Step 2: Set API Credentials

Each exchange requires its own set of API credentials. The environment variable names are based on the exchange name:

#### For Kraken:
```bash
export KRAKEN_API_KEY="your_api_key"
export KRAKEN_SECRET_KEY="your_secret_key"
export KRAKEN_DEMO="true"  # Use testnet (optional, default: true)
```

#### For Binance:
```bash
export BINANCE_API_KEY="your_api_key"
export BINANCE_SECRET_KEY="your_secret_key"
export BINANCE_DEMO="true"  # Use testnet (optional, default: true)
```

#### For Binance.US:
```bash
export BINANCEUS_API_KEY="your_api_key"
export BINANCEUS_SECRET_KEY="your_secret_key"
# Note: Binance.US does not have a testnet
```

#### For Bybit:
```bash
export BYBIT_API_KEY="your_api_key"
export BYBIT_SECRET_KEY="your_secret_key"
export BYBIT_DEMO="true"  # Use testnet (optional, default: true)
```

### Step 3: Create/Update Your .env File

Create a `.env` file in the project root:

```bash
# .env - Example for Kraken Futures

# Exchange selection
EXCHANGE_NAME=kraken

# Kraken API credentials
KRAKEN_API_KEY=your_kraken_api_key
KRAKEN_SECRET_KEY=your_kraken_secret_key
KRAKEN_DEMO=true
```

## Switching Exchanges

To switch from one exchange to another:

1. **Update your `.env` file** with the new exchange name and credentials
2. **Restart the trading bot** - the new configuration will be loaded automatically

Example - switching from Kraken to Binance:

```bash
# Before (.env)
EXCHANGE_NAME=kraken
KRAKEN_API_KEY=xxx
KRAKEN_SECRET_KEY=xxx

# After (.env)
EXCHANGE_NAME=binance
BINANCE_API_KEY=yyy
BINANCE_SECRET_KEY=yyy
BINANCE_DEMO=true
```

**Note**: Symbol formats differ between exchanges. The bot automatically uses the correct symbols for each exchange from the pre-configured symbol lists.

## Symbol Formats

Different exchanges use different symbol formats:

| Exchange | Example Symbol | Format Description |
|----------|----------------|-------------------|
| Kraken | `BTC/USD:USD` | `BASE/QUOTE:SETTLE` (USD settled) |
| Binance | `BTC/USDT:USDT` | `BASE/QUOTE:SETTLE` (USDT settled) |
| Binance.US | `BTC/USD` | `BASE/QUOTE` (spot only) |
| Bybit | `BTC/USDT:USDT` | `BASE/QUOTE:SETTLE` (USDT settled) |

The `:SETTLE` suffix indicates the settlement currency for perpetual futures contracts.

### Automatic Symbol Conversion

The system uses **canonical format** (just the base asset, e.g., `BTC`, `ETH`) for storing benchmarks and automatically converts to/from exchange-specific formats as needed.

This means:
- **Benchmarks are portable** - computed on one exchange, usable on any exchange
- **No manual symbol editing** - just change `EXCHANGE_NAME` and symbols adapt
- **Old data works** - benchmarks with old Alpaca format (`BTC/USD`) are auto-converted

Example conversions:
```python
from src.finance import canonical_to_exchange, exchange_to_canonical

# Convert canonical to exchange format
canonical_to_exchange("BTC", "kraken")   # → "XBT/USD:USD" (note: Kraken uses XBT)
canonical_to_exchange("BTC", "binance")  # → "BTC/USDT:USDT"
canonical_to_exchange("BTC", "binanceus") # → "BTC/USD"

# Convert any format to canonical
exchange_to_canonical("XBT/USD:USD", "kraken")  # → "BTC" (converts XBT back to BTC)
exchange_to_canonical("ETH/USDT:USDT")          # → "ETH"
exchange_to_canonical("BTC/USD")                # → "BTC"
exchange_to_canonical("LINK")                   # → "LINK"
```

**Note on Kraken's XBT**: Kraken uses `XBT` instead of `BTC` for Bitcoin. The symbol conversion functions automatically handle this:
- When sending to Kraken: `BTC` → `XBT`
- When receiving from Kraken: `XBT` → `BTC`

This means your benchmarks can use `BTC` (canonical) and they will automatically work with Kraken.

### Benchmark Symbol Storage

Benchmarks are stored in **canonical format** (version 1.2+) for portability:

```json
{
  "version": "1.2",
  "computed_on_exchange": "kraken",
  "universe": ["BTC", "ETH", "LINK", "SOL"],
  "cointegration_groups": [
    {
      "id": "BTC-ETH",
      "assets": ["BTC", "ETH"],
      "vectors": [{
        "weights": {"BTC": -0.5, "ETH": 1.0}
      }]
    }
  ]
}
```

When the trading bot loads benchmarks:
1. Symbols are automatically converted to the current exchange format
2. Price lookups use exchange-specific symbols
3. Orders are submitted with exchange-specific symbols

**Note**: If you have old benchmark files (version 1.1 or earlier with exchange-specific symbols), they will still work - the system can convert from any symbol format.

## Getting API Keys

### Kraken Futures
1. Create account at [futures.kraken.com](https://futures.kraken.com)
2. Navigate to Settings → API
3. Create a new API key with required permissions
4. For demo trading, use [demo-futures.kraken.com](https://demo-futures.kraken.com)

### Binance Futures
1. Create account at [binance.com](https://binance.com)
2. Navigate to API Management
3. Create API key with Futures permissions
4. For testnet, create keys at [testnet.binancefuture.com](https://testnet.binancefuture.com)

### Binance.US
1. Create account at [binance.us](https://binance.us)
2. Navigate to API Management
3. Create API key with trading permissions

### Bybit
1. Create account at [bybit.com](https://bybit.com)
2. Navigate to API Management
3. Create API key with trading permissions
4. For testnet, create keys at [testnet.bybit.com](https://testnet.bybit.com)

## Testing Your Configuration

Run the following to verify your exchange configuration:

```bash
python -c "from src.finance.crypto import get_exchange_info; print(get_exchange_info())"
```

Expected output:
```python
{
    'name': 'Kraken Futures',
    'exchange_id': 'kraken',
    'ccxt_id': 'krakenfutures',
    'has_futures': True,
    'us_available': True,
    'demo_mode': True,
    'symbol_format': 'BTC/USD:USD',
    'available_symbols': 15
}
```

## Architecture

The exchange abstraction is implemented in three key files:

1. **`src/finance/__init__.py`**: Exchange configuration and symbol definitions
   - `EXCHANGE_CONFIG`: Selected exchange configuration
   - `CRYPTO_TICKERS`: Available symbols for the exchange
   - `SUPPORTED_EXCHANGES`: All supported exchange configurations

2. **`src/finance/crypto.py`**: Historical data retrieval
   - `_get_exchange_client()`: Creates CCXT REST client
   - `fetch_crypto_data_for_cointegration()`: Gets data for analysis

3. **`src/finance/ccxt_trader.py`**: Trading and streaming
   - `CCXTFuturesTrader`: Main trading interface
   - Handles WebSocket streaming and order execution

## Adding a New Exchange

To add support for a new exchange:

1. **Update `SUPPORTED_EXCHANGES`** in `src/finance/__init__.py`:

```python
SUPPORTED_EXCHANGES = {
    # ... existing exchanges ...
    "newexchange": {
        "ccxt_id": "newexchange",        # CCXT exchange ID
        "ccxt_ws_id": "newexchange",     # CCXT Pro WebSocket ID
        "name": "New Exchange",          # Display name
        "has_futures": True,             # Supports futures trading
        "us_available": False,           # Available in US
        "symbol_format": "BTC/USDT",     # Example symbol format
        "quote_currency": "USDT",        # Default quote currency
        "settlement_currency": "USDT",   # For futures (or None for spot)
    },
}
```

2. **Add symbols** to `EXCHANGE_SYMBOLS`:

```python
EXCHANGE_SYMBOLS = {
    # ... existing exchanges ...
    "newexchange": [
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        # ... more symbols
    ],
}
```

3. **Add exchange-specific options** if needed in `crypto.py` and `ccxt_trader.py`:

```python
def _create_rest_client(self):
    # ...
    if EXCHANGE_NAME == "newexchange":
        config['options'] = {'specific': 'options'}
```

4. **Test thoroughly** with the exchange's testnet before live trading.

## Troubleshooting

### "Missing required environment variables"
Ensure your `.env` file has the correct variable names for your exchange:
```bash
# Check which variables are expected
echo $EXCHANGE_NAME  # Should show your exchange name
```

### "CCXT does not support exchange"
The exchange ID may be incorrect. Check [CCXT documentation](https://docs.ccxt.com/#exchanges) for valid exchange IDs.

### "Service unavailable from a restricted location"
You're trying to access an exchange that's blocked in your region. Try:
- Using a different exchange (e.g., Kraken for US users)
- Ensuring you're using the testnet (set `{EXCHANGE}_DEMO=true`)

### "Exchange does not support futures"
Some exchanges (like Binance.US) only support spot trading. For shorting capability, use an exchange that supports futures.

## Best Practices

1. **Always start with demo/testnet** - Set `{EXCHANGE}_DEMO=true` until you're confident
2. **Never commit API keys** - Keep `.env` in your `.gitignore`
3. **Test data retrieval first** - Verify historical data works before live trading
4. **Monitor connection health** - Use `get_connection_health()` to detect issues
5. **Keep credentials separate** - Use different API keys for different exchanges

## Security Recommendations

1. **Restrict API permissions** - Only enable the permissions you need
2. **Use IP whitelisting** - Restrict API access to known IPs
3. **Rotate keys regularly** - Replace API keys periodically
4. **Use testnet for development** - Never develop against production
5. **Monitor for unauthorized access** - Review exchange security logs

