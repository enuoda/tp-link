#!/usr/bin/env python3
"""
Simple Real-Time Streaming Example

This is a minimal example showing how to use the enhanced CryptoTrader class
for real-time data streaming and basic trading operations.

Perfect for:
- Learning how to use the streaming features
- Testing your API credentials
- Understanding the data flow
- Quick prototyping

Usage:
    python simple_streaming_example.py
"""

import time
import logging
from datetime import datetime
from crypto import CryptoTrader

# Simple logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def simple_streaming_demo():
    """Simple demonstration of real-time streaming capabilities"""
    
    print("🚀 Simple Crypto Streaming Demo")
    print("=" * 50)
    
    # Initialize trader (paper trading for safety)
    trader = CryptoTrader(paper=True)
    
    # Symbols to stream
    symbols = ['BTC/USD', 'ETH/USD']
    
    try:
        # Start streaming
        print("📡 Starting real-time data streaming...")
        trader.start_real_time_streaming(symbols, ['trades', 'quotes', 'bars'])
        
        # Wait for data to start flowing
        time.sleep(3)
        
        if not trader.is_streaming_active():
            print("❌ Streaming failed to start. Check your API credentials.")
            return
        
        print("✅ Streaming started successfully!")
        print(f"📊 Monitoring: {', '.join(symbols)}")
        print("\n" + "=" * 50)
        
        # Monitor data for 2 minutes
        for cycle in range(24):  # 24 cycles × 5 seconds = 2 minutes
            print(f"\n🔄 Cycle {cycle + 1}/24 - {datetime.now().strftime('%H:%M:%S')}")
            
            for symbol in symbols:
                # Get latest price
                price = trader.get_latest_price(symbol)
                
                # Get OHLC data
                ohlc = trader.get_real_time_ohlc(symbol)
                
                if price and ohlc:
                    print(f"{symbol:>8}: ${price:>8.2f} | "
                          f"OHLC: ${ohlc['open']:>6.2f}/${ohlc['high']:>6.2f}/"
                          f"${ohlc['low']:>6.2f}/${ohlc['close']:>6.2f}")
                elif price:
                    print(f"{symbol:>8}: ${price:>8.2f} (price only)")
                else:
                    print(f"{symbol:>8}: Waiting for data...")
            
            # Show data buffer status
            summary = trader.get_market_data_summary()
            total_points = sum(data['price_buffer_size'] for data in summary.values())
            print(f"📈 Total data points received: {total_points}")
            
            time.sleep(5)  # Wait 5 seconds between cycles
        
        print("\n" + "=" * 50)
        print("✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        # Clean up
        trader.stop_real_time_streaming()
        print("🔚 Streaming stopped")

def price_momentum_example():
    """Example of using streaming data for simple momentum analysis"""
    
    print("\n🎯 Price Momentum Analysis Example")
    print("=" * 50)
    
    trader = CryptoTrader(paper=True)
    symbol = 'BTC/USD'
    
    try:
        # Start streaming for one symbol
        trader.start_real_time_streaming([symbol], ['trades'])
        time.sleep(3)
        
        if not trader.is_streaming_active():
            print("❌ Streaming failed to start")
            return
        
        print(f"📊 Analyzing momentum for {symbol}")
        print("Collecting data for analysis...")
        
        # Collect data for 1 minute
        time.sleep(60)
        
        # Analyze price momentum
        price_history = trader.get_price_history(symbol, limit=100)
        
        if len(price_history) >= 20:
            prices = [p['price'] for p in price_history]
            volumes = [p['volume'] for p in price_history]
            
            # Calculate momentum
            price_change = (prices[-1] - prices[0]) / prices[0]
            avg_volume = sum(volumes) / len(volumes)
            recent_volume = sum(volumes[-10:]) / 10 if len(volumes) >= 10 else avg_volume
            
            print(f"\n📈 Momentum Analysis Results:")
            print(f"   Price change: {price_change:+.2%}")
            print(f"   Volume ratio: {recent_volume/avg_volume:.2f}x")
            print(f"   Data points: {len(price_history)}")
            
            # Simple signal generation
            if price_change > 0.01 and recent_volume > avg_volume * 1.1:
                print("🟢 Signal: STRONG BUY (upward momentum + high volume)")
            elif price_change < -0.01 and recent_volume > avg_volume * 1.1:
                print("🔴 Signal: STRONG SELL (downward momentum + high volume)")
            else:
                print("🟡 Signal: HOLD (waiting for clear direction)")
        else:
            print("❌ Insufficient data for analysis")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        trader.stop_real_time_streaming()

def custom_callback_example():
    """Example of using custom callbacks for data processing"""
    
    print("\n⚡ Custom Callback Example")
    print("=" * 50)
    
    trader = CryptoTrader(paper=True)
    symbol = 'ETH/USD'
    
    # Custom callback function
    async def large_trade_alert(trade_data):
        """Alert when large trades occur"""
        if float(trade_data.size) > 50:  # Large trade threshold
            print(f"🚨 LARGE TRADE: {symbol} - {trade_data.size} ETH at ${trade_data.price}")
    
    try:
        # Start streaming
        trader.start_real_time_streaming([symbol], ['trades'])
        
        # Add custom callback
        trader.add_streaming_callback(symbol, 'trades', large_trade_alert)
        
        print(f"👂 Listening for large trades in {symbol}...")
        print("Monitoring for 60 seconds...")
        
        # Monitor for 1 minute
        time.sleep(60)
        
        print("✅ Custom callback monitoring completed")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        trader.stop_real_time_streaming()

def main():
    """Run all examples"""
    
    print("🤖 CryptoTrader Real-Time Streaming Examples")
    print("=" * 60)
    
    # Check for API credentials
    import os
    if not os.getenv("ALPACA_API_KEY"):
        print("❌ Missing ALPACA_API_KEY environment variable")
        print("Please set your Alpaca API credentials before running this example")
        return
    
    try:
        # Run examples
        simple_streaming_demo()
        price_momentum_example()
        custom_callback_example()
        
        print("\n🎉 All examples completed successfully!")
        print("\nNext steps:")
        print("1. Modify the examples to suit your trading strategy")
        print("2. Add proper risk management")
        print("3. Test with paper trading before going live")
        print("4. Monitor your trades and performance")
        
    except KeyboardInterrupt:
        print("\n🛑 Examples stopped by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
