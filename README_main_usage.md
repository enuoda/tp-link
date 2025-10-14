# Main Trading Application Usage Guide

The enhanced `main.py` file now provides a comprehensive command-line interface for real-time crypto trading with streaming data. This guide explains how to use all the available features.

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Set up API credentials
export ALPACA_API_KEY="your_api_key_here"
export ALPACA_SECRET_KEY="your_secret_key_here"
export ALPACA_API_BASE_URL="https://paper-api.alpaca.markets"  # For paper trading

# Run the application
python main.py --help
```

### 2. Basic Usage Examples

```bash
# Monitor real-time data (safest option)
python main.py --mode monitor

# Check account information
python main.py --mode account

# Run trading bot with default settings
python main.py --mode trade

# Test API connections
python main.py --mode test
```

## 📋 Available Modes

### 🔍 **Monitor Mode** (Default)
Real-time data monitoring without trading - perfect for learning and testing.

```bash
# Basic monitoring
python main.py --mode monitor

# Monitor specific symbols for 15 minutes
python main.py --mode monitor --symbols BTC/USD ETH/USD SOL/USD --duration 15

# Monitor with custom symbols
python main.py --mode monitor --symbols ADA/USD MATIC/USD --duration 5
```

**Features:**
- Real-time price updates
- No trading execution
- Safe for testing
- Shows streaming data quality

### 🤖 **Trade Mode**
Full trading bot with real-time streaming and automated trading logic.

```bash
# Start trading bot (paper trading)
python main.py --mode trade

# Trade specific symbols for 30 minutes
python main.py --mode trade --symbols BTC/USD ETH/USD --duration 30

# Live trading (REAL MONEY - use with caution!)
python main.py --mode trade --live
```

**Features:**
- Real-time streaming data
- Automated trading signals
- Risk management
- Position monitoring
- Performance tracking

### 📊 **Account Mode**
Display account information and portfolio status.

```bash
python main.py --mode account
```

**Shows:**
- Account balance
- Buying power
- Equity
- Portfolio value

### 🧪 **Test Mode**
Test API connections and data feeds.

```bash
python main.py --mode test
```

**Tests:**
- API connectivity
- Data feed access
- Symbol availability

## ⚙️ Command-Line Options

### Basic Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--mode` | Application mode | `monitor` | `--mode trade` |
| `--symbols` | Crypto symbols to monitor/trade | `['BTC/USD', 'ETH/USD']` | `--symbols BTC/USD ETH/USD SOL/USD` |
| `--duration` | Duration in minutes | `10` | `--duration 30` |
| `--live` | Use live trading (real money) | `False` (paper trading) | `--live` |

### Advanced Usage Examples

```bash
# Monitor multiple symbols for 20 minutes
python main.py --mode monitor --symbols BTC/USD ETH/USD SOL/USD ADA/USD --duration 20

# Trade with custom duration and symbols
python main.py --mode trade --symbols BTC/USD ETH/USD --duration 60

# Test with live trading (requires confirmation)
python main.py --mode trade --live --symbols BTC/USD --duration 5

# Quick account check
python main.py --mode account
```

## 🛡️ Safety Features

### Paper Trading (Default)
- **All modes default to paper trading**
- No real money at risk
- Identical interface to live trading
- Perfect for testing strategies

### Live Trading Protection
- Requires explicit `--live` flag
- Interactive confirmation prompt
- Clear warnings about real money risk
- Easy to cancel and switch to paper trading

### Error Handling
- Comprehensive error handling
- Graceful shutdown procedures
- Detailed logging to `trading.log`
- Connection recovery mechanisms

## 📈 Trading Logic

The trading bot implements a simple momentum-based strategy:

### Signal Generation
- **BUY Signal**: Price increases > 2% with sufficient data
- **SELL Signal**: Price decreases > 2% with sufficient data
- **HOLD**: Price change < 2% or insufficient data

### Risk Management
- Position size limits
- Stop-loss protection (configurable)
- Maximum position limits
- Account equity protection

### Data Requirements
- Minimum 10 data points for signals
- Real-time price updates
- Volume confirmation
- Historical price analysis

## 📊 Output Examples

### Monitor Mode Output
```
📡 Starting data monitoring for 10 minutes
✅ Real-time streaming started successfully

🕐 14:30:15
BTC/USD: $43,250.50
ETH/USD: $2,650.75

🕐 14:30:20
BTC/USD: $43,255.20
ETH/USD: $2,652.10
```

### Trade Mode Output
```
🤖 Starting trading mode...
🚀 Starting streaming trading bot for 30 minutes
✅ Real-time streaming started successfully

🔄 Trading cycle #1

📊 REAL-TIME MARKET DATA
==================================================
BTC/USD: $43250.50 | OHLC: $43200.00/$43260.00/$43180.00/$43250.50
ETH/USD: $2650.75 | OHLC: $2645.00/$2655.00/$2640.00/$2650.75

🟢 BUY signal: BTC/USD up 2.15%
```

## 🔧 Configuration

### Environment Variables
```bash
# Required for all modes
export ALPACA_API_KEY="your_key"
export ALPACA_SECRET_KEY="your_secret"

# Optional - defaults to paper trading URL
export ALPACA_API_BASE_URL="https://paper-api.alpaca.markets"
```

### Trading Parameters
The trading logic can be customized by modifying the `_execute_trading_logic` method in `main.py`:

```python
# Current thresholds (can be adjusted)
BUY_THRESHOLD = 0.02    # 2% price increase
SELL_THRESHOLD = -0.02  # 2% price decrease
MIN_DATA_POINTS = 10    # Minimum data for signals
```

## 📝 Logging

All activities are logged to `trading.log` with timestamps:
- Trading decisions
- Market data updates
- Error messages
- Performance metrics

## 🚨 Important Notes

### Before Trading
1. **Always start with paper trading**
2. **Test your strategy thoroughly**
3. **Understand the risks involved**
4. **Never risk more than you can afford to lose**

### API Limits
- Streaming data doesn't count against API limits
- Historical data requests do count
- Monitor your usage in the Alpaca dashboard

### Market Hours
- Crypto markets are 24/7
- Some data feeds may have maintenance windows
- Check Alpaca status page for outages

## 🆘 Troubleshooting

### Common Issues

1. **"Missing API credentials"**
   ```bash
   export ALPACA_API_KEY="your_key"
   export ALPACA_SECRET_KEY="your_secret"
   ```

2. **"Streaming failed to start"**
   - Check network connectivity
   - Verify API credentials
   - Ensure paper trading URL is correct

3. **"No data received"**
   - Wait a few seconds for initialization
   - Check symbol format (use 'BTC/USD' not 'BTCUSD')
   - Verify market hours

### Debug Mode
Enable detailed logging:
```bash
# Set log level to DEBUG
export PYTHONPATH=/path/to/your/project
python -c "import logging; logging.basicConfig(level=logging.DEBUG)" main.py --mode monitor
```

## 🎯 Next Steps

1. **Start with monitoring** - Get familiar with real-time data
2. **Test trading logic** - Use paper trading to test strategies
3. **Customize signals** - Modify trading logic for your needs
4. **Add risk management** - Implement proper position sizing
5. **Monitor performance** - Track results and optimize

## 📚 Benchmarks (weekly)

Run a weekly job to compute cointegration benchmarks and persist JSON snapshots.

```bash
# Option A: via module
python3 -m src.trading_strategy.compute_benchmarks \
  --symbols BTC/USD ETH/USD SOL/USD ADA/USD \
  --days 30 --time-scale hour --max-groups 10

# Option B: via main.py mode
python3 main.py --mode compute-benchmarks \
  --symbols BTC/USD ETH/USD SOL/USD ADA/USD \
  --days 30 --time-scale hour --max-groups 10
```

This writes `data/benchmarks/benchmarks_YYYY-WW.json` and updates `data/benchmarks/benchmarks_latest.json`.
Live trading will automatically load the latest benchmarks and evaluate spread z-scores
for up to 10 configured groups. If no benchmarks are present or are stale (>7 days),
the bot falls back to a simple momentum preview.

## 📁 File Structure

The main.py application is organized with proper separation of concerns:

- **`main.py`** - Command-line interface and application entry point
- **`src/finance/live_trading_example.py`** - Contains the `TradingPartner` class and advanced trading logic
- **`src/finance/crypto.py`** - Enhanced `CryptoTrader` class with real-time streaming
- **`src/finance/simple_streaming_example.py`** - Basic examples and learning tools

This structure keeps the main.py clean while providing access to all the powerful trading functionality.

## ⚠️ Disclaimer

This software is for educational purposes only. Cryptocurrency trading involves significant risk. Always:

- Start with paper trading
- Never risk more than you can afford to lose
- Understand the risks involved
- Consider consulting with financial professionals
- Comply with all applicable laws and regulations
