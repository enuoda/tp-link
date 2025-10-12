# Real-Time Crypto Trading Examples

This directory contains comprehensive examples demonstrating how to use the enhanced `CryptoTrader` class for live crypto trading with real-time data streaming.

## Files Overview

### 📁 Core Files
- **`crypto.py`** - Enhanced CryptoTrader class with real-time streaming capabilities
- **`live_trading_example.py`** - Complete trading bot with advanced features
- **`simple_streaming_example.py`** - Basic examples for learning and testing

### 📊 Features Demonstrated

#### Real-Time Streaming
- WebSocket-based data streaming (no polling)
- Multi-symbol monitoring
- Instant data access with buffering
- Custom callback functions

#### Trading Capabilities
- Market order execution
- Position management
- Risk management (stop-loss, take-profit)
- Performance tracking

#### Data Analysis
- Price momentum analysis
- Volume analysis
- Technical indicators
- Signal generation

## Quick Start

### 1. Setup Environment

```bash
# Install required packages
pip install alpaca-py numpy pandas matplotlib

# Set up environment variables
export ALPACA_API_KEY="your_api_key_here"
export ALPACA_SECRET_KEY="your_secret_key_here"
export ALPACA_API_BASE_URL="https://paper-api.alpaca.markets"  # For paper trading
```

### 2. Run Simple Example

```bash
# Start with the simple example to test your setup
python simple_streaming_example.py
```

### 3. Run Advanced Trading Bot

```bash
# Run the complete trading bot (paper trading)
python live_trading_example.py
```

## Example Usage Patterns

### Basic Streaming Setup

```python
from crypto import CryptoTrader

# Initialize trader
trader = CryptoTrader(paper=True)

# Start streaming
trader.start_real_time_streaming(['BTC/USD', 'ETH/USD'])

# Get instant data
price = trader.get_latest_price('BTC/USD')
ohlc = trader.get_real_time_ohlc('BTC/USD')
```

### Custom Data Processing

```python
# Custom callback for processing trade data
async def my_trade_handler(trade_data):
    if float(trade_data.size) > 100:
        print(f"Large trade: {trade_data.size} at ${trade_data.price}")

# Add custom callback
trader.add_streaming_callback('BTC/USD', 'trades', my_trade_handler)
```

### Trading Execution

```python
# Execute trades based on signals
signal = generate_trading_signal('BTC/USD')
if signal.action == 'BUY':
    order = trader.buy_market_order('BTC/USD', notional=100)
```

## Configuration Options

### Trading Parameters
- `max_position_size`: Maximum USD per position
- `stop_loss_pct`: Stop-loss percentage
- `take_profit_pct`: Take-profit percentage
- `max_positions`: Maximum concurrent positions

### Streaming Parameters
- `symbols`: List of crypto symbols to monitor
- `data_types`: Types of data to stream (`['trades', 'quotes', 'bars']`)
- `buffer_size`: Size of data buffers (configurable)

## Safety Features

### Paper Trading
- All examples default to paper trading
- No real money at risk during testing
- Identical API interface to live trading

### Risk Management
- Position size limits
- Stop-loss and take-profit orders
- Maximum position limits
- Account equity protection

### Error Handling
- Comprehensive error handling and logging
- Graceful shutdown procedures
- Connection recovery mechanisms

## Performance Benefits

### Efficiency Comparison

| Method | Latency | API Calls | Data Freshness |
|--------|---------|-----------|----------------|
| **Polling** | 500ms-2s | Every request | Stale |
| **WebSocket Streaming** | <50ms | Once at start | Real-time |

### Key Advantages
- **10-40x faster** than polling-based approaches
- **Real-time data** with minimal latency
- **No API rate limits** for streaming data
- **Efficient memory usage** with circular buffers
- **Multi-symbol support** without performance degradation

## Monitoring and Logging

### Built-in Logging
- All trading activities are logged
- Performance metrics tracked
- Error handling with detailed logs
- Market data summaries

### Performance Tracking
- Win/loss ratios
- Total P&L tracking
- Trade execution statistics
- Real-time position monitoring

## Advanced Features

### Custom Strategies
The examples show how to implement:
- Momentum-based trading
- Volume analysis
- Multi-timeframe analysis
- Custom signal generation

### Data Buffering
- Circular buffers for price history
- OHLC data buffering
- Configurable buffer sizes
- Thread-safe data access

### Real-time Analysis
- Instant price momentum calculation
- Volume profile analysis
- Technical indicator computation
- Signal confidence scoring

## Troubleshooting

### Common Issues

1. **Streaming fails to start**
   - Check API credentials
   - Verify network connectivity
   - Ensure paper trading URL is correct

2. **No data received**
   - Wait a few seconds for initialization
   - Check symbol format (use 'BTC/USD' not 'BTCUSD')
   - Verify market hours

3. **High memory usage**
   - Reduce buffer sizes
   - Limit number of symbols
   - Clear old data periodically

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Next Steps

1. **Start with paper trading** - Never test with real money
2. **Modify the examples** - Adapt to your trading strategy
3. **Add risk management** - Implement proper position sizing
4. **Backtest your strategy** - Test on historical data first
5. **Monitor performance** - Track metrics and optimize

## Disclaimer

⚠️ **Important**: These examples are for educational purposes only. Cryptocurrency trading involves significant risk. Always:

- Start with paper trading
- Never risk more than you can afford to lose
- Understand the risks involved
- Consider consulting with financial professionals
- Comply with all applicable laws and regulations

## Support

For issues or questions:
1. Check the logs for error messages
2. Verify your API credentials
3. Test with the simple example first
4. Review the Alpaca API documentation
5. Check your account status and permissions
