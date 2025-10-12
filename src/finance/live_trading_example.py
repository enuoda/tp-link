#!/usr/bin/env python3
"""
Live Trading Example with Real-Time Data Streaming

This example demonstrates how to use the enhanced CryptoTrader class for:
1. Real-time data streaming from multiple crypto symbols
2. Live trading based on streaming market data
3. Risk management and position monitoring
4. Performance tracking and logging

Usage:
    python live_trading_example.py

Requirements:
    - Valid Alpaca API credentials in environment variables
    - alpaca-py library installed
    - Paper trading account (recommended for testing)
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import json

from crypto import CryptoTrader
from alpaca.trading.enums import OrderSide

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """Data class for trading signals"""
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 to 1.0
    price: float
    reason: str
    timestamp: datetime

@dataclass
class Position:
    """Data class for position tracking"""
    symbol: str
    side: str  # 'LONG', 'SHORT'
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    entry_time: datetime

class LiveTradingBot:
    """
    Advanced live trading bot using real-time streaming data
    
    Features:
    - Multi-symbol real-time data streaming
    - Technical analysis based on streaming data
    - Risk management and position sizing
    - Performance tracking and logging
    - Custom trading strategies
    """
    
    def __init__(self, paper: bool = True, symbols: List[str] = None):
        """
        Initialize the live trading bot
        
        Args:
            paper: Whether to use paper trading (recommended for testing)
            symbols: List of crypto symbols to trade (default: major cryptos)
        """
        self.trader = CryptoTrader(paper=paper)
        
        # Trading symbols
        self.symbols = symbols or ['BTC/USD', 'ETH/USD', 'SOL/USD', 'ADA/USD']
        
        # Trading parameters
        self.max_position_size = 1000.0  # Maximum USD per position
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.04  # 4% take profit
        self.max_positions = 3  # Maximum concurrent positions
        
        # State tracking
        self.positions: Dict[str, Position] = {}
        self.signals: List[TradingSignal] = []
        self.running = False
        self.start_time = None
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        
        logger.info(f"Initialized LiveTradingBot with symbols: {self.symbols}")
        logger.info(f"Paper trading: {paper}")
    
    def start_streaming(self):
        """Start real-time data streaming"""
        logger.info("Starting real-time data streaming...")
        
        try:
            self.trader.start_real_time_streaming(
                symbols=self.symbols,
                data_types=['trades', 'quotes', 'bars']
            )
            
            # Wait for streaming to initialize
            time.sleep(3)
            
            if self.trader.is_streaming_active():
                logger.info("✅ Real-time streaming started successfully")
                return True
            else:
                logger.error("❌ Failed to start real-time streaming")
                return False
                
        except Exception as e:
            logger.error(f"Error starting streaming: {e}")
            return False
    
    def get_market_summary(self) -> str:
        """Get a summary of current market conditions"""
        summary = self.trader.get_market_data_summary()
        
        report = "\n📊 MARKET SUMMARY\n" + "="*50 + "\n"
        
        for symbol in self.symbols:
            if symbol in summary:
                data = summary[symbol]
                latest_price = data['latest_price']
                buffer_size = data['price_buffer_size']
                
                report += f"{symbol}: ${latest_price:.2f} | "
                report += f"Data points: {buffer_size} | "
                report += f"Has bars: {'✅' if data['has_bar_data'] else '❌'} | "
                report += f"Has quotes: {'✅' if data['has_quote_data'] else '❌'}\n"
            else:
                report += f"{symbol}: ❌ No data\n"
        
        return report
    
    def analyze_price_momentum(self, symbol: str, lookback: int = 20) -> Dict:
        """
        Analyze price momentum using streaming data
        
        Args:
            symbol: Crypto symbol to analyze
            lookback: Number of recent data points to consider
            
        Returns:
            Dictionary with momentum analysis results
        """
        price_history = self.trader.get_price_history(symbol, limit=lookback)
        
        if len(price_history) < 10:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Insufficient data'}
        
        prices = [p['price'] for p in price_history]
        volumes = [p['volume'] for p in price_history]
        
        # Calculate momentum indicators
        price_change = (prices[-1] - prices[0]) / prices[0]
        volume_avg = sum(volumes) / len(volumes)
        recent_volume = sum(volumes[-5:]) / 5 if len(volumes) >= 5 else volume_avg
        
        # Simple momentum strategy
        signal = 'HOLD'
        confidence = 0.0
        reason = ""
        
        if price_change > 0.02 and recent_volume > volume_avg * 1.2:
            signal = 'BUY'
            confidence = min(0.8, abs(price_change) * 10)
            reason = f"Strong upward momentum ({price_change:.2%}) with high volume"
        elif price_change < -0.02 and recent_volume > volume_avg * 1.2:
            signal = 'SELL'
            confidence = min(0.8, abs(price_change) * 10)
            reason = f"Strong downward momentum ({price_change:.2%}) with high volume"
        elif abs(price_change) < 0.01:
            signal = 'HOLD'
            confidence = 0.3
            reason = "Low volatility, waiting for clear signal"
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'price_change': price_change,
            'volume_ratio': recent_volume / volume_avg,
            'current_price': prices[-1]
        }
    
    def generate_trading_signal(self, symbol: str) -> Optional[TradingSignal]:
        """
        Generate trading signal based on real-time data analysis
        
        Args:
            symbol: Crypto symbol to analyze
            
        Returns:
            TradingSignal object or None if no signal
        """
        analysis = self.analyze_price_momentum(symbol)
        
        if analysis['signal'] != 'HOLD' and analysis['confidence'] > 0.6:
            return TradingSignal(
                symbol=symbol,
                action=analysis['signal'],
                confidence=analysis['confidence'],
                price=analysis['current_price'],
                reason=analysis['reason'],
                timestamp=datetime.now()
            )
        
        return None
    
    def execute_trade(self, signal: TradingSignal) -> bool:
        """
        Execute a trade based on the signal
        
        Args:
            signal: TradingSignal object
            
        Returns:
            True if trade was executed successfully
        """
        try:
            # Check if we already have a position in this symbol
            if signal.symbol in self.positions:
                logger.info(f"Already have position in {signal.symbol}, skipping trade")
                return False
            
            # Check position limits
            if len(self.positions) >= self.max_positions:
                logger.info(f"Maximum positions ({self.max_positions}) reached, skipping trade")
                return False
            
            # Calculate position size
            account_value = float(self.trader.acct.equity)
            position_size = min(self.max_position_size, account_value * 0.1)  # Max 10% of account
            
            if signal.action == 'BUY':
                # Execute buy order
                order = self.trader.buy_market_order(
                    symbol=signal.symbol,
                    notional=position_size
                )
                
                # Track position
                self.positions[signal.symbol] = Position(
                    symbol=signal.symbol,
                    side='LONG',
                    quantity=position_size / signal.price,
                    entry_price=signal.price,
                    current_price=signal.price,
                    unrealized_pnl=0.0,
                    entry_time=datetime.now()
                )
                
                logger.info(f"✅ BUY order executed: {signal.symbol} at ${signal.price:.2f}")
                logger.info(f"   Position size: ${position_size:.2f}")
                logger.info(f"   Reason: {signal.reason}")
                
            elif signal.action == 'SELL':
                # For crypto, we can't short directly, so we'll skip sell signals
                # In a real implementation, you might want to close long positions
                logger.info(f"SELL signal received for {signal.symbol}, but shorting not implemented")
                return False
            
            self.total_trades += 1
            self.signals.append(signal)
            return True
            
        except Exception as e:
            logger.error(f"Error executing trade for {signal.symbol}: {e}")
            return False
    
    def update_positions(self):
        """Update current positions with latest prices"""
        for symbol, position in self.positions.items():
            latest_price = self.trader.get_latest_price(symbol)
            
            if latest_price:
                position.current_price = latest_price
                
                if position.side == 'LONG':
                    position.unrealized_pnl = (latest_price - position.entry_price) * position.quantity
                else:
                    position.unrealized_pnl = (position.entry_price - latest_price) * position.quantity
    
    def check_exit_conditions(self):
        """Check if any positions should be closed based on stop loss/take profit"""
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            # Calculate returns
            if position.side == 'LONG':
                return_pct = (position.current_price - position.entry_price) / position.entry_price
            else:
                return_pct = (position.entry_price - position.current_price) / position.entry_price
            
            should_close = False
            close_reason = ""
            
            # Check stop loss
            if return_pct <= -self.stop_loss_pct:
                should_close = True
                close_reason = f"Stop loss triggered ({return_pct:.2%})"
            
            # Check take profit
            elif return_pct >= self.take_profit_pct:
                should_close = True
                close_reason = f"Take profit triggered ({return_pct:.2%})"
            
            if should_close:
                positions_to_close.append((symbol, close_reason))
        
        # Close positions
        for symbol, reason in positions_to_close:
            self.close_position(symbol, reason)
    
    def close_position(self, symbol: str, reason: str):
        """
        Close a position
        
        Args:
            symbol: Symbol to close position for
            reason: Reason for closing the position
        """
        try:
            position = self.positions[symbol]
            
            # Execute sell order
            order = self.trader.sell_market_order(
                symbol=symbol,
                qty=position.quantity
            )
            
            # Update performance tracking
            self.total_pnl += position.unrealized_pnl
            if position.unrealized_pnl > 0:
                self.winning_trades += 1
            
            logger.info(f"🔚 Position closed: {symbol}")
            logger.info(f"   Entry: ${position.entry_price:.2f}")
            logger.info(f"   Exit: ${position.current_price:.2f}")
            logger.info(f"   P&L: ${position.unrealized_pnl:.2f}")
            logger.info(f"   Reason: {reason}")
            
            # Remove from positions
            del self.positions[symbol]
            
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
    
    def get_performance_summary(self) -> str:
        """Get performance summary"""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        summary = f"\n📈 PERFORMANCE SUMMARY\n" + "="*50 + "\n"
        summary += f"Total trades: {self.total_trades}\n"
        summary += f"Winning trades: {self.winning_trades}\n"
        summary += f"Win rate: {win_rate:.1f}%\n"
        summary += f"Total P&L: ${self.total_pnl:.2f}\n"
        summary += f"Open positions: {len(self.positions)}\n"
        
        if self.positions:
            summary += "\nOpen Positions:\n"
            for symbol, pos in self.positions.items():
                summary += f"  {symbol}: {pos.side} ${pos.unrealized_pnl:+.2f}\n"
        
        return summary
    
    def run_trading_session(self, duration_minutes: int = 30):
        """
        Run a complete trading session
        
        Args:
            duration_minutes: Duration of trading session in minutes
        """
        logger.info(f"🤖 Starting live trading session for {duration_minutes} minutes")
        self.running = True
        self.start_time = datetime.now()
        
        try:
            # Start streaming
            if not self.start_streaming():
                logger.error("Failed to start streaming, exiting")
                return
            
            # Print account info
            self.trader.print_account_summary()
            
            end_time = datetime.now() + timedelta(minutes=duration_minutes)
            cycle_count = 0
            
            while datetime.now() < end_time and self.running:
                cycle_count += 1
                logger.info(f"\n🔄 Trading cycle #{cycle_count}")
                logger.info(f"Time remaining: {end_time - datetime.now()}")
                
                # Get market summary
                market_summary = self.get_market_summary()
                logger.info(market_summary)
                
                # Update existing positions
                self.update_positions()
                
                # Check exit conditions
                self.check_exit_conditions()
                
                # Generate and execute new signals
                for symbol in self.symbols:
                    signal = self.generate_trading_signal(symbol)
                    if signal:
                        logger.info(f"📊 Signal generated: {signal.action} {signal.symbol} "
                                  f"(confidence: {signal.confidence:.2f})")
                        self.execute_trade(signal)
                
                # Print performance summary
                if cycle_count % 5 == 0:  # Every 5 cycles
                    logger.info(self.get_performance_summary())
                
                # Wait before next cycle
                time.sleep(30)  # 30 second cycles
                
        except KeyboardInterrupt:
            logger.info("\n🛑 Trading session stopped by user")
        except Exception as e:
            logger.error(f"❌ Error in trading session: {e}")
        finally:
            self.running = False
            
            # Close all positions
            if self.positions:
                logger.info("Closing all remaining positions...")
                for symbol in list(self.positions.keys()):
                    self.close_position(symbol, "Session ended")
            
            # Final performance summary
            logger.info("\n" + "="*60)
            logger.info("🏁 FINAL SESSION RESULTS")
            logger.info("="*60)
            logger.info(self.get_performance_summary())
            
            # Stop streaming
            self.trader.stop_real_time_streaming()
            logger.info("🔚 Trading session completed")

class TradingPartner:
    """
    Main trading interface that combines crypto trading and market data
    
    This class provides a unified interface for:
    - Real-time crypto trading with streaming data
    - Market data analysis and visualization
    - Portfolio management and risk control
    - Simplified trading operations for main.py integration
    """
    
    def __init__(self, paper: bool = True):
        """
        Initialize the trading partner
        
        Args:
            paper: Whether to use paper trading (recommended for testing)
        """
        self.crypto_trader = CryptoTrader(paper=paper)
        self.paper = paper
        
        logger.info(f"Initialized TradingPartner (Paper trading: {paper})")
    
    def get_account(self):
        """Get and display account information"""
        self.crypto_trader.print_account_summary()
    
    def start_streaming_bot(self, symbols: List[str] = None, duration_minutes: int = 30):
        """
        Start the real-time streaming trading bot
        
        Args:
            symbols: List of crypto symbols to trade (default: major cryptos)
            duration_minutes: Duration of trading session
        """
        if symbols is None:
            symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD']
        
        logger.info(f"🚀 Starting streaming trading bot for {duration_minutes} minutes")
        logger.info(f"Symbols: {symbols}")
        
        try:
            # Start real-time streaming
            self.crypto_trader.start_real_time_streaming(symbols, ['trades', 'quotes', 'bars'])
            time.sleep(3)  # Wait for streaming to initialize
            
            if not self.crypto_trader.is_streaming_active():
                logger.error("❌ Failed to start streaming")
                return False
            
            logger.info("✅ Real-time streaming started successfully")
            
            # Trading loop
            start_time = datetime.now()
            end_time = start_time + timedelta(minutes=duration_minutes)
            cycle_count = 0
            
            while datetime.now() < end_time:
                cycle_count += 1
                logger.info(f"\n🔄 Trading cycle #{cycle_count}")
                
                # Monitor market data
                self._monitor_market_data(symbols)
                
                # Execute trading logic
                self._execute_trading_logic(symbols)
                
                # Wait before next cycle
                time.sleep(30)  # 30 second cycles
            
            logger.info("✅ Trading session completed successfully")
            return True
            
        except KeyboardInterrupt:
            logger.info("🛑 Trading session stopped by user")
        except Exception as e:
            logger.error(f"❌ Error in trading session: {e}")
        finally:
            self.crypto_trader.stop_real_time_streaming()
        
        return False
    
    def _monitor_market_data(self, symbols: List[str]):
        """Monitor real-time market data"""
        print("\n📊 REAL-TIME MARKET DATA")
        print("=" * 50)
        
        for symbol in symbols:
            price = self.crypto_trader.get_latest_price(symbol)
            ohlc = self.crypto_trader.get_real_time_ohlc(symbol)
            
            if price and ohlc:
                print(f"{symbol:>8}: ${price:>8.2f} | "
                      f"OHLC: ${ohlc['open']:>6.2f}/${ohlc['high']:>6.2f}/"
                      f"${ohlc['low']:>6.2f}/${ohlc['close']:>6.2f}")
            elif price:
                print(f"{symbol:>8}: ${price:>8.2f} (price only)")
            else:
                print(f"{symbol:>8}: Waiting for data...")
    
    def _execute_trading_logic(self, symbols: List[str]):
        """Execute trading logic based on real-time data"""
        for symbol in symbols:
            # Simple momentum-based trading logic
            price_history = self.crypto_trader.get_price_history(symbol, limit=20)
            
            if len(price_history) >= 10:
                prices = [p['price'] for p in price_history]
                price_change = (prices[-1] - prices[0]) / prices[0]
                
                # Simple trading signals
                if price_change > 0.02:  # 2% up
                    logger.info(f"🟢 BUY signal: {symbol} up {price_change:.2%}")
                    # Uncomment to execute trades:
                    # self.crypto_trader.buy_market_order(symbol, notional=100)
                elif price_change < -0.02:  # 2% down
                    logger.info(f"🔴 SELL signal: {symbol} down {price_change:.2%}")
                    # Uncomment to execute trades:
                    # self.crypto_trader.sell_market_order(symbol, notional=100)
    
    def monitor_data_only(self, symbols: List[str] = None, duration_minutes: int = 10):
        """
        Monitor real-time data without trading
        
        Args:
            symbols: List of crypto symbols to monitor
            duration_minutes: Duration of monitoring session
        """
        if symbols is None:
            symbols = ['BTC/USD', 'ETH/USD']
        
        logger.info(f"📡 Starting data monitoring for {duration_minutes} minutes")
        
        try:
            self.crypto_trader.start_real_time_streaming(symbols, ['trades', 'quotes'])
            time.sleep(3)
            
            if not self.crypto_trader.is_streaming_active():
                logger.error("❌ Failed to start streaming")
                return False
            
            end_time = datetime.now() + timedelta(minutes=duration_minutes)
            
            while datetime.now() < end_time:
                print(f"\n🕐 {datetime.now().strftime('%H:%M:%S')}")
                
                for symbol in symbols:
                    price = self.crypto_trader.get_latest_price(symbol)
                    if price:
                        print(f"{symbol}: ${price:.2f}")
                    else:
                        print(f"{symbol}: No data")
                
                time.sleep(5)  # Update every 5 seconds
            
            logger.info("✅ Data monitoring completed")
            return True
            
        except KeyboardInterrupt:
            logger.info("🛑 Monitoring stopped by user")
        except Exception as e:
            logger.error(f"❌ Error in monitoring: {e}")
        finally:
            self.crypto_trader.stop_real_time_streaming()
        
        return False

def main():
    """Main function to run the live trading example"""
    
    # Configuration
    PAPER_TRADING = True  # Set to False for live trading (NOT RECOMMENDED FOR TESTING)
    SYMBOLS = ['BTC/USD', 'ETH/USD']  # Symbols to trade
    SESSION_DURATION = 10  # Minutes (start with short sessions for testing)
    
    # Validate environment
    import os
    if not os.getenv("ALPACA_API_KEY") or not os.getenv("ALPACA_SECRET_KEY"):
        logger.error("❌ Missing Alpaca API credentials!")
        logger.error("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        return
    
    # Initialize and run bot
    try:
        bot = LiveTradingBot(paper=PAPER_TRADING, symbols=SYMBOLS)
        bot.run_trading_session(duration_minutes=SESSION_DURATION)
    except Exception as e:
        logger.error(f"Fatal error: {e}")

if __name__ == "__main__":
    main()
