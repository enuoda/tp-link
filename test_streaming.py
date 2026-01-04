#!/usr/bin/env python3
"""
Simple test script to verify streaming connectivity.
Run this to diagnose streaming issues.

Usage:
    python test_streaming.py
"""

import os
import sys
import time

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_streaming():
    """Test streaming connectivity with minimal setup."""
    
    print("=" * 60)
    print("🔍 STREAMING CONNECTIVITY TEST")
    print("=" * 60)
    
    # Check environment variables
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    ALPACA_API_BASE_URL="https://paper-api.alpaca.markets"
    
    if not api_key or not secret_key:
        print("❌ Missing environment variables!")
        print("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY")
        return False
    
    print(f"✅ API Key found: {api_key[:8]}...")
    print(f"✅ Secret Key found: {secret_key[:8]}...")
    print()
    
    # Import trader
    try:
        from src.finance.crypto import CryptoTrader
        print("✅ CryptoTrader imported successfully")
    except Exception as e:
        print(f"❌ Failed to import CryptoTrader: {e}")
        return False
    
    # Initialize trader
    try:
        trader = CryptoTrader(paper=True)
        print("✅ CryptoTrader initialized (paper mode)")
        print(f"   Account equity: ${float(trader.acct.equity):,.2f}")
    except Exception as e:
        print(f"❌ Failed to initialize CryptoTrader: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    
    # Test symbols
    test_symbols = ["BTC/USD", "ETH/USD"]
    print(f"📡 Testing streaming for: {test_symbols}")
    print("-" * 60)
    
    # Start streaming
    try:
        print("Starting streaming...")
        trader.start_real_time_streaming(test_symbols, ["trades", "quotes"])
        print("Streaming thread started")
    except Exception as e:
        print(f"❌ Failed to start streaming: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Wait for data with detailed progress
    print()
    print("⏳ Waiting for streaming data (60 seconds max)...")
    print("-" * 60)
    
    start_time = time.time()
    max_wait = 60  # seconds
    check_interval = 2  # seconds
    
    while (time.time() - start_time) < max_wait:
        elapsed = time.time() - start_time
        
        # Check streaming status
        active = trader.is_streaming_active()
        healthy = trader._connection_healthy.is_set()
        
        print(f"\n[{elapsed:.0f}s] Streaming active: {active}, Healthy: {healthy}")
        
        # Check for prices
        has_all_data = True
        for symbol in test_symbols:
            price = trader.get_latest_price(symbol)
            if price:
                print(f"  ✅ {symbol}: ${price:.2f}")
            else:
                print(f"  ⏳ {symbol}: No data yet")
                has_all_data = False
        
        if has_all_data:
            print()
            print("=" * 60)
            print("✅ STREAMING TEST PASSED!")
            print("=" * 60)
            print("Data is being received successfully.")
            trader.stop_real_time_streaming()
            return True
        
        # Show debug info
        print(f"  Thread alive: {trader._stream_thread.is_alive() if trader._stream_thread else 'No thread'}")
        print(f"  Reconnect attempts: {trader._reconnect_attempts}")
        
        time.sleep(check_interval)
    
    print()
    print("=" * 60)
    print("❌ STREAMING TEST FAILED!")
    print("=" * 60)
    print("No data received within 60 seconds.")
    print()
    print("Debug info:")
    print(trader.get_streaming_debug_info())
    
    # Cleanup
    trader.stop_real_time_streaming()
    
    print()
    print("Possible causes:")
    print("1. Network connectivity issues")
    print("2. API credentials may be invalid")
    print("3. Alpaca service may be down")
    print("4. Rate limiting (if you've made many requests)")
    print()
    print("Try:")
    print("- Check your internet connection")
    print("- Verify API credentials at https://alpaca.markets/")
    print("- Check Alpaca status at https://status.alpaca.markets/")
    
    return False


if __name__ == "__main__":
    try:
        success = test_streaming()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

