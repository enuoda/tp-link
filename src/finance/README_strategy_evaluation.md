# Strategy Evaluation Toolkit

This module provides comprehensive tools for evaluating trading strategy performance, including risk-adjusted returns, drawdown analysis, and other key performance indicators.

## 📁 Files

- **`strategy_evaluation.py`** - Main module with all evaluation functions
- **`strategy_evaluation_example.py`** - Example usage and demonstrations
- **`README_strategy_evaluation.md`** - This documentation

## 🎯 Key Features

### Core Metrics
- **Sharpe Ratio** - Risk-adjusted return measurement
- **Sortino Ratio** - Downside deviation focused ratio
- **Calmar Ratio** - Return vs maximum drawdown
- **Value at Risk (VaR)** - Potential loss at confidence level
- **Conditional VaR (CVaR)** - Expected loss beyond VaR threshold

### Comprehensive Analysis
- **Drawdown Analysis** - Maximum, average, and duration metrics
- **Trade Statistics** - Win rate, profit factor, average win/loss
- **Higher Moments** - Skewness and kurtosis analysis
- **Performance Visualization** - Charts and plots

## 🚀 Quick Start

### Basic Usage

```python
from strategy_evaluation import calculate_sharpe_ratio, calculate_comprehensive_metrics

# Your strategy returns (daily, weekly, etc.)
returns = [0.01, -0.005, 0.02, 0.015, ...]  # Example returns

# Calculate Sharpe ratio
sharpe_ratio, details = calculate_sharpe_ratio(returns)
print(f"Sharpe Ratio: {sharpe_ratio:.3f}")

# Calculate comprehensive metrics
metrics = calculate_comprehensive_metrics(returns)
print(f"Annualized Return: {metrics.annualized_return:.2%}")
print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
```

### Advanced Analysis

```python
from strategy_evaluation import (
    calculate_comprehensive_metrics,
    plot_strategy_performance,
    print_strategy_report
)

# Calculate all metrics
metrics = calculate_comprehensive_metrics(returns, trades=trade_returns)

# Print detailed report
print_strategy_report(metrics, "My Trading Strategy")

# Create performance plots
plot_strategy_performance(returns, "Strategy Performance")
```

## 📊 Available Functions

### Core Metric Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `calculate_sharpe_ratio()` | Sharpe ratio with related quantities | (ratio, details_dict) |
| `calculate_sortino_ratio()` | Sortino ratio focusing on downside | (ratio, details_dict) |
| `calculate_calmar_ratio()` | Calmar ratio (return/max drawdown) | (ratio, details_dict) |
| `calculate_drawdown_metrics()` | Comprehensive drawdown analysis | metrics_dict |
| `calculate_var_cvar()` | Value at Risk and Conditional VaR | (var, cvar, details_dict) |
| `calculate_trade_statistics()` | Trade-level performance metrics | metrics_dict |

### Comprehensive Analysis

| Function | Description | Returns |
|----------|-------------|---------|
| `calculate_comprehensive_metrics()` | All metrics in one call | StrategyMetrics object |
| `plot_strategy_performance()` | Performance visualization plots | None (shows plots) |
| `print_strategy_report()` | Formatted performance report | None (prints report) |

## 📈 StrategyMetrics Class

The `StrategyMetrics` dataclass contains all calculated metrics:

```python
@dataclass
class StrategyMetrics:
    # Basic returns
    total_return: float
    annualized_return: float
    avg_daily_return: float
    
    # Risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Drawdown metrics
    max_drawdown: float
    max_drawdown_duration: int
    avg_drawdown: float
    
    # Trade statistics
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Additional metrics
    var_95: float
    cvar_95: float
    skewness: float
    kurtosis: float
```

## 🎯 Usage Examples

### Example 1: Basic Sharpe Ratio Analysis

```python
import numpy as np
from strategy_evaluation import calculate_sharpe_ratio

# Generate sample returns
returns = np.random.normal(0.001, 0.02, 252)  # 252 days

# Calculate Sharpe ratio
sharpe, details = calculate_sharpe_ratio(returns, risk_free_rate=0.02)

print(f"Sharpe Ratio: {sharpe:.3f}")
print(f"Annualized Return: {details['annualized_return']:.2%}")
print(f"Volatility: {details['volatility']:.2%}")
```

### Example 2: Strategy Comparison

```python
from strategy_evaluation import calculate_comprehensive_metrics

# Compare two strategies
strategy_a_returns = [0.01, -0.005, 0.02, ...]
strategy_b_returns = [0.008, -0.003, 0.015, ...]

metrics_a = calculate_comprehensive_metrics(strategy_a_returns)
metrics_b = calculate_comprehensive_metrics(strategy_b_returns)

print(f"Strategy A Sharpe: {metrics_a.sharpe_ratio:.3f}")
print(f"Strategy B Sharpe: {metrics_b.sharpe_ratio:.3f}")

# Find better strategy
better = "A" if metrics_a.sharpe_ratio > metrics_b.sharpe_ratio else "B"
print(f"Better strategy: {better}")
```

### Example 3: Real Trading Data Analysis

```python
import pandas as pd
from strategy_evaluation import calculate_comprehensive_metrics, print_strategy_report

# Load your trading results
df = pd.read_csv('trading_results.csv')
returns = df['daily_returns'].values
trades = df['trade_returns'].dropna().values

# Calculate comprehensive metrics
metrics = calculate_comprehensive_metrics(
    returns=returns,
    trades=trades,
    risk_free_rate=0.025,  # 2.5% risk-free rate
    periods_per_year=252   # Daily data
)

# Print detailed report
print_strategy_report(metrics, "My Crypto Trading Strategy")
```

## 📊 Performance Interpretation

### Sharpe Ratio Guidelines
- **> 2.0**: Excellent
- **1.5 - 2.0**: Very Good  
- **1.0 - 1.5**: Good
- **0.5 - 1.0**: Acceptable
- **< 0.5**: Poor

### Risk Metrics
- **Volatility**: Lower is generally better (all else equal)
- **Max Drawdown**: Should be acceptable for your risk tolerance
- **VaR**: Helps understand potential losses

### Trade Statistics
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Average Win/Loss**: Size of typical wins vs losses

## 🔧 Configuration Options

### Risk-Free Rate
Default is 2% annually. Adjust based on current market conditions:
```python
sharpe, _ = calculate_sharpe_ratio(returns, risk_free_rate=0.025)  # 2.5%
```

### Trading Periods
Adjust for your data frequency:
```python
# Daily data
metrics = calculate_comprehensive_metrics(returns, periods_per_year=252)

# Weekly data  
metrics = calculate_comprehensive_metrics(returns, periods_per_year=52)

# Monthly data
metrics = calculate_comprehensive_metrics(returns, periods_per_year=12)
```

### Confidence Levels
For VaR calculations:
```python
var_99, cvar_99, _ = calculate_var_cvar(returns, confidence_level=0.99)  # 99% VaR
var_95, cvar_95, _ = calculate_var_cvar(returns, confidence_level=0.95)  # 95% VaR
```

## 📈 Visualization

The module provides comprehensive performance plots:

```python
from strategy_evaluation import plot_strategy_performance

# Create performance plots
plot_strategy_performance(
    returns=returns,
    title="My Trading Strategy Performance",
    save_path="strategy_performance.png"  # Optional: save plot
)
```

Plots include:
- Cumulative returns over time
- Drawdown analysis
- Return distribution histogram
- Rolling Sharpe ratio

## 🧪 Testing

Run the example script to see the toolkit in action:

```bash
python strategy_evaluation_example.py
```

This will demonstrate:
- Basic metric calculations
- Strategy comparisons
- Comprehensive analysis
- Real-world usage examples

## 📋 Requirements

- **numpy** - Numerical calculations
- **pandas** - Data handling
- **matplotlib** - Visualization (optional)

Install with:
```bash
pip install numpy pandas matplotlib
```

## ⚠️ Important Notes

1. **Data Quality**: Ensure your returns data is clean (no NaN values, proper format)
2. **Risk-Free Rate**: Use current market rates for accurate Sharpe ratios
3. **Periods**: Match the periods_per_year parameter to your data frequency
4. **Interpretation**: Consider multiple metrics together, not just Sharpe ratio alone

## 🎯 Integration with Trading System

This module integrates seamlessly with your existing trading system:

```python
# In your trading bot
from strategy_evaluation import calculate_comprehensive_metrics

# After collecting trading results
def evaluate_strategy_performance(self):
    returns = self.get_strategy_returns()
    trades = self.get_trade_results()
    
    metrics = calculate_comprehensive_metrics(returns, trades)
    
    # Log performance
    self.logger.info(f"Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
    self.logger.info(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    
    return metrics
```

The strategy evaluation toolkit provides everything you need to thoroughly analyze and compare your trading strategies!
