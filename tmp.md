# Live Trading System

A quantitative cryptocurrency trading system using cointegration-based pairs trading via the Alpaca API.

## Overview

This system identifies statistically cointegrated cryptocurrency pairs, monitors their spread in real-time, and executes mean-reversion trades when the spread deviates significantly from its historical mean (measured by z-score).

**Key Features:**
- Real-time WebSocket streaming from Alpaca
- Cointegration-based pairs trading signals
- Per-symbol staleness thresholds based on liquidity
- Heartbeat monitoring for zombie connection detection
- Graceful shutdown with position closing
- Paper trading support for testing

---

## Directory Structure

```
tp-link/
├── main.py                          # Primary entry point (CLI)
├── data/
│   └── benchmarks/                  # Saved cointegration benchmarks
└── src/
    ├── __init__.py                  # Environment variables loader
    ├── finance/
    │   ├── __init__.py              # Constants, tickers, staleness profiles
    │   ├── benchmarks.py            # Load/save cointegration benchmarks
    │   ├── crypto.py                # CryptoTrader - Alpaca API interface
    │   ├── trading_bot.py  # TradingPartner - trading logic
    │   ├── rolling_buffer.py        # Real-time data buffer
    │   └── spread_engine.py         # Signal generation engine
    └── trading_strategy/
        ├── compute_benchmarks.py    # Cointegration computation
        └── cointegration_utils.py   # Statistical test utilities
```

---

## File Descriptions

### `main.py`
The command-line interface and main entry point. Supports multiple modes:
- `trade-indefinite`: Run live trading until Ctrl+C
- `trade`: Run live trading for a fixed duration
- `monitor`: Stream prices without trading
- `compute-benchmarks`: Compute cointegration pairs
- `account`: Display Alpaca account info

### `src/finance/__init__.py`
Contains:
- `CRYPTO_TICKERS`: List of 20 supported cryptocurrencies
- `TIME_FRAMES`: Mapping of time scale strings to Alpaca TimeFrame objects
- `STALENESS_PROFILES`: Per-symbol staleness thresholds by liquidity tier
- `get_staleness_for_symbol()`, `get_staleness_for_pair()`: Staleness utilities

### `src/finance/crypto.py`
The `CryptoTrader` class - core interface to Alpaca API:
- WebSocket streaming with auto-reconnection
- Heartbeat monitoring for zombie connections
- Price caching and staleness tracking
- Order submission (market, limit, stop, trailing stop)
- Historical data retrieval

### `src/finance/trading_bot.py`
The `TradingPartner` class - orchestrates trading:
- Connects streaming to signal engine
- Executes spread trades based on signals
- Manages spread positions (entry, exit, P&L tracking)
- Handles emergency exits for stale data

### `src/finance/spread_engine.py`
The `SpreadSignalEngine` class - generates trading signals:
- Computes z-scores from live prices and benchmark parameters
- Emits `BUY_SPREAD`, `SELL_SPREAD`, `EXIT`, `EMERGENCY_EXIT` signals
- Uses per-pair staleness thresholds based on asset liquidity

### `src/finance/benchmarks.py`
Utilities for cointegration benchmark files:
- `load_benchmarks()`: Load from JSON
- `save_benchmarks()`: Save to JSON
- `is_stale()`: Check if benchmarks need refresh
- `get_weights()`, `get_spread_params()`: Extract trading parameters

### `src/finance/rolling_buffer.py`
`RollingCointegrationBuffer` for accumulating streaming bar data:
- Thread-safe data storage
- Time-aligned arrays for analysis
- Staleness tracking per symbol

### `src/trading_strategy/compute_benchmarks.py`
Functions to compute cointegration benchmarks:
- Fetches historical data
- Runs pairwise cointegration tests
- Computes hedge ratios and spread statistics
- Saves results to JSON

### `src/trading_strategy/cointegration_utils.py`
Statistical utilities:
- `run_pairwise_cointegration()`: Engle-Granger test
- `check_higher_order_cointegration()`: Johansen test

---

<!-- ## Environment Setup

### Required Environment Variables

```bash
export ALPACA_API_KEY='your_api_key'
export ALPACA_SECRET_KEY='your_secret_key'
```

Or create a `.env` file in the project root:

```
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
```

### Dependencies

```bash
pip install alpaca-py numpy pandas statsmodels python-dotenv matplotlib
``` -->

---

## Usage Examples

### 1. Compute Cointegration Benchmarks

Before trading, you need cointegration benchmarks. This writes `data/benchmarks/benchmarks_YYYY-WW.json` and updates `data/benchmarks/benchmarks_latest.json`. Live trading will automatically load the latest benchmarks and evaluate spread z-scores. If no benchmarks are present or are stale (>7 days), the bot falls back to a simple momentum preview.

Example:

```bash
# Basic benchmark computation (30 days, hourly bars)
python main.py --mode compute-benchmarks

# With custom parameters
python main.py --mode compute-benchmarks --days 30 --time-scale hour --max-groups 40 --p-threshold 0.15
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--days` | 30 | Historical lookback period |
| `--time-scale` | hour | Bar timeframe (min, hour, day) |
| `--max-groups` | 10 | Max cointegration pairs to keep |
| `--p-threshold` | 0.10 | P-value threshold (higher = more pairs) |

---

### 2. Monitor Prices (No Trading)

Test WebSocket connectivity and data flow:

```bash
# Monitor for 10 minutes (default)
python main.py --mode monitor

# Monitor specific symbols for 30 minutes
python main.py --mode monitor --symbols BTC/USD ETH/USD LTC/USD --duration 30
```

---

### 3. Live Trading (Indefinite)

Run the trading bot until you press Ctrl+C:

```bash
# Basic indefinite trading (paper mode)
python main.py --mode trade-indefinite

# With custom z-score thresholds (more conservative)
python main.py --mode trade-indefinite --entry-zscore 2.5 --exit-zscore 0.3

# With custom staleness thresholds
python main.py --mode trade-indefinite --entry-staleness 60 --exit-staleness 600 --emergency-staleness 1800
```

**Trading Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--entry-zscore` | 2.0 | Z-score to trigger entry (lower = more trades) |
| `--exit-zscore` | 0.5 | Z-score to trigger exit (higher = exit sooner) |
| `--cycle-interval` | 30 | Seconds between trading cycles |
| `--health-interval` | 15 | Minutes between health logs |
| `--max-stream-symbols` | 10 | Max WebSocket subscriptions |

**Staleness Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--entry-staleness` | 30 | Max data age (sec) for entries |
| `--exit-staleness` | 300 | Max data age (sec) for exits |
| `--emergency-staleness` | 900 | Force exit if data older than this |

**Rolling Recalibration Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--recalibrate-interval` | 10 | Minutes between spread recalibrations |
| `--recalibrate-min-obs` | 50 | Min price observations before recalibrating |

```bash
# With rolling recalibration every 5 minutes
python main.py --mode trade-indefinite --recalibrate-interval 5

# Disable rolling recalibration
python main.py --mode trade-indefinite --recalibrate-interval 0
```

---

### 4. Live Trading (Fixed Duration)

Run for a specific time:

```bash
# Trade for 60 minutes
python main.py --mode trade --duration 60

# Trade for 2 hours with custom settings
python main.py --mode trade --duration 120 --entry-zscore 1.8 --max-stream-symbols 5
```

---

### 5. Check Account Info

```bash
python main.py --mode account
```

---

### 6. Live Trading (Real Money) ⚠️

**Warning: This uses real money!**

```bash
python main.py --mode trade-indefinite --live
```

You will be prompted to confirm before proceeding.

---

## Trading Logic

### Signal Generation

1. **Z-Score Calculation**: For each cointegrated pair, compute the spread and its z-score:
   ```
   spread = price_A * weight_A + price_B * weight_B
   z-score = (spread - mean) / std
   ```

2. **Entry Signals**:
   - `z-score >= entry_threshold` → `SELL_SPREAD` (expect spread to decrease)
   - `z-score <= -entry_threshold` → `BUY_SPREAD` (expect spread to increase)

3. **Exit Signals**:
   - `|z-score| <= exit_threshold` → `EXIT` (spread reverted to mean)

4. **Emergency Exit**:
   - Data staleness exceeds threshold → `EMERGENCY_EXIT`

### Staleness Tiers

Staleness thresholds vary by asset liquidity:

| Tier | Assets | Entry | Exit | Emergency |
|------|--------|-------|------|-----------|
| High | BTC, ETH, DOGE, SHIB | 15s | 60s | 5 min |
| Medium | LTC, LINK, AVAX, DOT | 60s | 5 min | 15 min |
| Low | BAT, YFI, SUSHI, CRV | 3 min | 10 min | 30 min |

For pairs, the **most conservative** (shortest) threshold is used.

### Rolling Recalibration

Over time, the spread's mean and standard deviation can drift from the values computed during benchmarking. This causes z-scores to become systematically biased (e.g., always negative), reducing trading effectiveness.

**Rolling recalibration** automatically updates the spread parameters from recent price observations:

1. Each trading cycle, current prices are recorded into a rolling history
2. Every N minutes (default: 10), the spread mean/std are recalculated
3. Z-scores then reflect current market conditions, not stale benchmarks

**Example output:**
```
📊 Running rolling recalibration...
📊 Recalibration complete: 3/3 groups updated
  BTC_ETH: mean shift=+12.4500, std change=-2.3%
  LTC_LINK: mean shift=-8.2100, std change=+1.1%
  DOT_AVAX: mean shift=+3.0800, std change=+0.5%
```

**When to use:**
- If z-scores are consistently one-sided (always positive or always negative)
- For long-running trading sessions (hours/days)
- When market conditions are volatile

---

### Example Output

```
2025-01-15 10:30:00 - INFO - 🚀 STARTING INDEFINITE TRADING SESSION
2025-01-15 10:30:01 - INFO - ✅ Started real-time streaming for ['BTC/USD', 'ETH/USD']
2025-01-15 10:30:05 - INFO - 💓 Heartbeat monitor started
2025-01-15 10:30:35 - INFO - 📊 Spread BTC_ETH: z=2.34 -> SELL_SPREAD (conf=0.58)
2025-01-15 10:30:35 - INFO - 📈 SELL_SPREAD executed for BTC_ETH
```

---

## Graceful Shutdown

Press **Ctrl+C** to initiate graceful shutdown:

1. Stops accepting new signals
2. Closes all open spread positions
3. Stops WebSocket streaming
4. Logs final session summary

---

## Troubleshooting

### "symbol limit exceeded (405)"
Reduce `--max-stream-symbols` to stay under Alpaca's WebSocket limit.

### No cointegration pairs found
- Increase `--p-threshold` (e.g., 0.15 or 0.20)
- Increase `--days` for more historical data
- Cryptocurrencies may not be cointegrated in current market conditions

### "Waiting for data..." messages
Normal for low-liquidity assets. The system uses staleness thresholds to handle this automatically.

### Heartbeat timeout / reconnection
The system auto-reconnects on zombie connections. If persistent, check your network connection.

---

## License

MIT License - See LICENSE file for details.

