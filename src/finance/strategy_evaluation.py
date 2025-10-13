#!/usr/bin/env python3

"""
Strategy Evaluation and Performance Metrics

This module provides comprehensive tools for evaluating trading strategies,
including risk-adjusted returns, drawdown analysis, and other key performance indicators.

Author: Sam Dawley
Date: 08/2025
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

@dataclass
class StrategyMetrics:
    """
    Data class for storing comprehensive strategy performance metrics
    """
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
    var_95: float  # Value at Risk (95%)
    cvar_95: float  # Conditional Value at Risk (95%)
    skewness: float
    kurtosis: float

def calculate_sharpe_ratio(
    returns: Union[List[float], np.ndarray, pd.Series],
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate the Sharpe ratio and related quantities for a trading strategy.
    
    The Sharpe ratio measures risk-adjusted returns by dividing the excess return
    over the risk-free rate by the volatility of returns.
    
    Args:
        returns: Array-like object containing strategy returns
        risk_free_rate: Annual risk-free rate (default: 2%)
        periods_per_year: Number of trading periods per year (default: 252 for daily)
        
    Returns:
        Tuple containing:
        - Sharpe ratio (float)
        - Dictionary with related quantities:
          - 'excess_return': Annualized excess return over risk-free rate
          - 'volatility': Annualized volatility
          - 'mean_return': Mean period return
          - 'std_return': Standard deviation of returns
          - 'risk_free_return': Risk-free return per period
          - 'total_return': Total cumulative return
          - 'annualized_return': Annualized return
    """
    
    # Convert to numpy array for calculations
    if isinstance(returns, pd.Series):
        returns = returns.values
    elif isinstance(returns, list):
        returns = np.array(returns)
    
    # Remove any NaN values
    returns = returns[~np.isnan(returns)]
    
    if len(returns) == 0:
        raise ValueError("No valid returns data provided")
    
    # Calculate basic statistics
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)  # Sample standard deviation
    total_return = np.prod(1 + returns) - 1
    
    # Annualized metrics
    periods = len(returns)
    annualized_return = (1 + total_return) ** (periods_per_year / periods) - 1
    annualized_volatility = std_return * np.sqrt(periods_per_year)
    
    # Risk-free return per period
    risk_free_return_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    # Excess return
    excess_return_per_period = mean_return - risk_free_return_per_period
    annualized_excess_return = excess_return_per_period * periods_per_year
    
    # Sharpe ratio
    if std_return == 0:
        sharpe_ratio = 0.0 if mean_return == risk_free_return_per_period else np.inf
    else:
        sharpe_ratio = excess_return_per_period / std_return * np.sqrt(periods_per_year)
    
    # Compile related quantities
    related_quantities = {
        'excess_return': annualized_excess_return,
        'volatility': annualized_volatility,
        'mean_return': mean_return,
        'std_return': std_return,
        'risk_free_return': risk_free_return_per_period,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'periods': periods
    }
    
    return sharpe_ratio, related_quantities

def calculate_sortino_ratio(
    returns: Union[List[float], np.ndarray, pd.Series],
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate the Sortino ratio, which focuses on downside deviation instead of total volatility.
    
    Args:
        returns: Array-like object containing strategy returns
        risk_free_rate: Annual risk-free rate (default: 2%)
        periods_per_year: Number of trading periods per year (default: 252 for daily)
        
    Returns:
        Tuple containing:
        - Sortino ratio (float)
        - Dictionary with related quantities
    """
    
    # Convert to numpy array
    if isinstance(returns, pd.Series):
        returns = returns.values
    elif isinstance(returns, list):
        returns = np.array(returns)
    
    returns = returns[~np.isnan(returns)]
    
    if len(returns) == 0:
        raise ValueError("No valid returns data provided")
    
    # Calculate basic statistics
    mean_return = np.mean(returns)
    risk_free_return_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_return_per_period = mean_return - risk_free_return_per_period
    
    # Calculate downside deviation (standard deviation of negative returns)
    negative_returns = returns[returns < 0]
    if len(negative_returns) == 0:
        downside_deviation = 0.0
        sortino_ratio = np.inf if excess_return_per_period > 0 else 0.0
    else:
        downside_deviation = np.std(negative_returns, ddof=1)
        if downside_deviation == 0:
            sortino_ratio = np.inf if excess_return_per_period > 0 else 0.0
        else:
            sortino_ratio = excess_return_per_period / downside_deviation * np.sqrt(periods_per_year)
    
    related_quantities = {
        'downside_deviation': downside_deviation * np.sqrt(periods_per_year),
        'downside_deviation_period': downside_deviation,
        'excess_return': excess_return_per_period * periods_per_year,
        'negative_returns_count': len(negative_returns),
        'downside_frequency': len(negative_returns) / len(returns)
    }
    
    return sortino_ratio, related_quantities

def calculate_drawdown_metrics(
    returns: Union[List[float], np.ndarray, pd.Series]
) -> Dict[str, float]:
    """
    Calculate comprehensive drawdown metrics for a trading strategy.
    
    Args:
        returns: Array-like object containing strategy returns
        
    Returns:
        Dictionary containing:
        - 'max_drawdown': Maximum drawdown percentage
        - 'max_drawdown_duration': Maximum drawdown duration in periods
        - 'avg_drawdown': Average drawdown percentage
        - 'drawdown_periods': Number of periods in drawdown
        - 'recovery_factor': Ratio of total return to max drawdown
    """
    
    # Convert to numpy array
    if isinstance(returns, pd.Series):
        returns = returns.values
    elif isinstance(returns, list):
        returns = np.array(returns)
    
    returns = returns[~np.isnan(returns)]
    
    if len(returns) == 0:
        raise ValueError("No valid returns data provided")
    
    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + returns)
    
    # Calculate running maximum (peak)
    running_max = np.maximum.accumulate(cumulative_returns)
    
    # Calculate drawdown as percentage
    drawdown = (cumulative_returns - running_max) / running_max
    
    # Maximum drawdown
    max_drawdown = np.min(drawdown)
    
    # Drawdown duration analysis
    in_drawdown = drawdown < 0
    drawdown_periods = np.sum(in_drawdown)
    
    # Find maximum drawdown duration
    max_duration = 0
    current_duration = 0
    
    for is_dd in in_drawdown:
        if is_dd:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0
    
    # Average drawdown (only periods in drawdown)
    if drawdown_periods > 0:
        avg_drawdown = np.mean(drawdown[in_drawdown])
    else:
        avg_drawdown = 0.0
    
    # Recovery factor (total return / max drawdown)
    total_return = cumulative_returns[-1] - 1
    if max_drawdown != 0:
        recovery_factor = abs(total_return / max_drawdown)
    else:
        recovery_factor = np.inf
    
    return {
        'max_drawdown': max_drawdown,
        'max_drawdown_duration': max_duration,
        'avg_drawdown': avg_drawdown,
        'drawdown_periods': drawdown_periods,
        'drawdown_frequency': drawdown_periods / len(returns),
        'recovery_factor': recovery_factor,
        'total_return': total_return
    }

def calculate_calmar_ratio(
    returns: Union[List[float], np.ndarray, pd.Series],
    periods_per_year: int = 252
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate the Calmar ratio, which is the annualized return divided by maximum drawdown.
    
    Args:
        returns: Array-like object containing strategy returns
        periods_per_year: Number of trading periods per year (default: 252 for daily)
        
    Returns:
        Tuple containing:
        - Calmar ratio (float)
        - Dictionary with related quantities
    """
    
    # Get drawdown metrics
    drawdown_metrics = calculate_drawdown_metrics(returns)
    
    # Calculate annualized return
    total_return = drawdown_metrics['total_return']
    periods = len(returns) if not isinstance(returns, (list, np.ndarray)) else len(returns)
    
    if isinstance(returns, (list, np.ndarray)):
        periods = len(returns)
    else:
        periods = len(returns)
    
    annualized_return = (1 + total_return) ** (periods_per_year / periods) - 1
    
    # Calmar ratio
    max_drawdown = abs(drawdown_metrics['max_drawdown'])
    if max_drawdown == 0:
        calmar_ratio = np.inf if annualized_return > 0 else 0.0
    else:
        calmar_ratio = annualized_return / max_drawdown
    
    related_quantities = {
        'annualized_return': annualized_return,
        'max_drawdown': drawdown_metrics['max_drawdown'],
        'total_return': total_return,
        'periods': periods
    }
    
    return calmar_ratio, related_quantities

def calculate_var_cvar(
    returns: Union[List[float], np.ndarray, pd.Series],
    confidence_level: float = 0.95
) -> Tuple[float, float, Dict[str, float]]:
    """
    Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR).
    
    Args:
        returns: Array-like object containing strategy returns
        confidence_level: Confidence level for VaR calculation (default: 0.95)
        
    Returns:
        Tuple containing:
        - VaR (float)
        - CVaR (float)
        - Dictionary with related quantities
    """
    
    # Convert to numpy array
    if isinstance(returns, pd.Series):
        returns = returns.values
    elif isinstance(returns, list):
        returns = np.array(returns)
    
    returns = returns[~np.isnan(returns)]
    
    if len(returns) == 0:
        raise ValueError("No valid returns data provided")
    
    # Calculate VaR (negative of the percentile)
    var_percentile = (1 - confidence_level) * 100
    var = -np.percentile(returns, var_percentile)
    
    # Calculate CVaR (expected value of returns below VaR threshold)
    threshold = -var
    tail_returns = returns[returns <= threshold]
    
    if len(tail_returns) == 0:
        cvar = var
    else:
        cvar = -np.mean(tail_returns)
    
    related_quantities = {
        'confidence_level': confidence_level,
        'tail_observations': len(tail_returns),
        'tail_probability': len(tail_returns) / len(returns),
        'threshold_return': threshold
    }
    
    return var, cvar, related_quantities

def calculate_trade_statistics(
    trades: Union[List[float], np.ndarray, pd.Series]
) -> Dict[str, float]:
    """
    Calculate trade-level statistics.
    
    Args:
        trades: Array-like object containing individual trade returns
        
    Returns:
        Dictionary containing trade statistics
    """
    
    # Convert to numpy array
    if isinstance(trades, pd.Series):
        trades = trades.values
    elif isinstance(trades, list):
        trades = np.array(trades)
    
    trades = trades[~np.isnan(trades)]
    
    if len(trades) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0
        }
    
    # Basic statistics
    total_trades = len(trades)
    winning_trades = trades[trades > 0]
    losing_trades = trades[trades < 0]
    
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
    
    # Average win/loss
    avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0.0
    avg_loss = np.mean(losing_trades) if len(losing_trades) > 0 else 0.0
    
    # Profit factor
    total_wins = np.sum(winning_trades) if len(winning_trades) > 0 else 0.0
    total_losses = abs(np.sum(losing_trades)) if len(losing_trades) > 0 else 0.0
    
    if total_losses == 0:
        profit_factor = np.inf if total_wins > 0 else 0.0
    else:
        profit_factor = total_wins / total_losses
    
    # Largest win/loss
    largest_win = np.max(trades) if len(trades) > 0 else 0.0
    largest_loss = np.min(trades) if len(trades) > 0 else 0.0
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'largest_win': largest_win,
        'largest_loss': largest_loss,
        'total_wins': total_wins,
        'total_losses': total_losses
    }

def calculate_comprehensive_metrics(
    returns: Union[List[float], np.ndarray, pd.Series],
    trades: Optional[Union[List[float], np.ndarray, pd.Series]] = None,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
    confidence_level: float = 0.95
) -> StrategyMetrics:
    """
    Calculate comprehensive strategy performance metrics.
    
    Args:
        returns: Array-like object containing strategy returns
        trades: Optional array of individual trade returns
        risk_free_rate: Annual risk-free rate (default: 2%)
        periods_per_year: Number of trading periods per year (default: 252 for daily)
        confidence_level: Confidence level for VaR calculation (default: 0.95)
        
    Returns:
        StrategyMetrics object containing all calculated metrics
    """
    
    # Convert to numpy array
    if isinstance(returns, pd.Series):
        returns = returns.values
    elif isinstance(returns, list):
        returns = np.array(returns)
    
    returns = returns[~np.isnan(returns)]
    
    if len(returns) == 0:
        raise ValueError("No valid returns data provided")
    
    # Basic return metrics
    total_return = np.prod(1 + returns) - 1
    periods = len(returns)
    annualized_return = (1 + total_return) ** (periods_per_year / periods) - 1
    avg_daily_return = np.mean(returns)
    
    # Risk metrics
    volatility = np.std(returns, ddof=1) * np.sqrt(periods_per_year)
    
    # Sharpe ratio
    sharpe_ratio, _ = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
    
    # Sortino ratio
    sortino_ratio, _ = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
    
    # Calmar ratio
    calmar_ratio, _ = calculate_calmar_ratio(returns, periods_per_year)
    
    # Drawdown metrics
    drawdown_metrics = calculate_drawdown_metrics(returns)
    
    # VaR and CVaR
    var_95, cvar_95, _ = calculate_var_cvar(returns, confidence_level)
    
    # Higher moments
    skewness = float(pd.Series(returns).skew()) if len(returns) > 2 else 0.0
    kurtosis = float(pd.Series(returns).kurtosis()) if len(returns) > 2 else 0.0
    
    # Trade statistics (if provided)
    if trades is not None:
        trade_stats = calculate_trade_statistics(trades)
        total_trades = trade_stats['total_trades']
        win_rate = trade_stats['win_rate']
        avg_win = trade_stats['avg_win']
        avg_loss = trade_stats['avg_loss']
        profit_factor = trade_stats['profit_factor']
    else:
        total_trades = 0
        win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        profit_factor = 0.0
    
    return StrategyMetrics(
        # Basic returns
        total_return=total_return,
        annualized_return=annualized_return,
        avg_daily_return=avg_daily_return,
        
        # Risk metrics
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        
        # Drawdown metrics
        max_drawdown=drawdown_metrics['max_drawdown'],
        max_drawdown_duration=drawdown_metrics['max_drawdown_duration'],
        avg_drawdown=drawdown_metrics['avg_drawdown'],
        
        # Trade statistics
        total_trades=total_trades,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        
        # Additional metrics
        var_95=var_95,
        cvar_95=cvar_95,
        skewness=skewness,
        kurtosis=kurtosis
    )

def plot_strategy_performance(
    returns: Union[List[float], np.ndarray, pd.Series],
    title: str = "Strategy Performance Analysis",
    save_path: Optional[str] = None
) -> None:
    """
    Create comprehensive performance visualization plots.
    
    Args:
        returns: Array-like object containing strategy returns
        title: Title for the plot
        save_path: Optional path to save the plot
    """
    
    # Convert to pandas Series for easier handling
    if isinstance(returns, (list, np.ndarray)):
        returns = pd.Series(returns)
    
    returns = returns.dropna()
    
    if len(returns) == 0:
        raise ValueError("No valid returns data provided")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. Cumulative returns
    cumulative_returns = (1 + returns).cumprod()
    axes[0, 0].plot(cumulative_returns.index, cumulative_returns.values)
    axes[0, 0].set_title('Cumulative Returns')
    axes[0, 0].set_ylabel('Cumulative Return')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Drawdown
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
    axes[0, 1].set_title('Drawdown')
    axes[0, 1].set_ylabel('Drawdown %')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Return distribution
    axes[1, 0].hist(returns.values, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.4f}')
    axes[1, 0].set_title('Return Distribution')
    axes[1, 0].set_xlabel('Daily Returns')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Rolling Sharpe ratio (30-day window)
    if len(returns) >= 30:
        rolling_sharpe = returns.rolling(window=30).mean() / returns.rolling(window=30).std() * np.sqrt(252)
        axes[1, 1].plot(rolling_sharpe.index, rolling_sharpe.values)
        axes[1, 1].set_title('30-Day Rolling Sharpe Ratio')
        axes[1, 1].set_ylabel('Sharpe Ratio')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor rolling Sharpe', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('30-Day Rolling Sharpe Ratio')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def print_strategy_report(
    metrics: StrategyMetrics,
    strategy_name: str = "Strategy"
) -> None:
    """
    Print a comprehensive strategy performance report.
    
    Args:
        metrics: StrategyMetrics object containing performance metrics
        strategy_name: Name of the strategy for the report
    """
    
    print(f"\n{'='*60}")
    print(f"STRATEGY PERFORMANCE REPORT: {strategy_name}")
    print(f"{'='*60}")
    
    print(f"\n📈 RETURN METRICS:")
    print(f"  Total Return:           {metrics.total_return:>10.2%}")
    print(f"  Annualized Return:      {metrics.annualized_return:>10.2%}")
    print(f"  Average Daily Return:   {metrics.avg_daily_return:>10.4f}")
    
    print(f"\n📊 RISK METRICS:")
    print(f"  Volatility:             {metrics.volatility:>10.2%}")
    print(f"  Sharpe Ratio:           {metrics.sharpe_ratio:>10.2f}")
    print(f"  Sortino Ratio:          {metrics.sortino_ratio:>10.2f}")
    print(f"  Calmar Ratio:           {metrics.calmar_ratio:>10.2f}")
    
    print(f"\n📉 DRAWDOWN METRICS:")
    print(f"  Maximum Drawdown:       {metrics.max_drawdown:>10.2%}")
    print(f"  Max DD Duration:        {metrics.max_drawdown_duration:>10} periods")
    print(f"  Average Drawdown:       {metrics.avg_drawdown:>10.2%}")
    
    if metrics.total_trades > 0:
        print(f"\n💰 TRADE STATISTICS:")
        print(f"  Total Trades:          {metrics.total_trades:>10}")
        print(f"  Win Rate:              {metrics.win_rate:>10.2%}")
        print(f"  Average Win:           {metrics.avg_win:>10.2%}")
        print(f"  Average Loss:          {metrics.avg_loss:>10.2%}")
        print(f"  Profit Factor:         {metrics.profit_factor:>10.2f}")
    
    print(f"\n🎯 ADDITIONAL METRICS:")
    print(f"  VaR (95%):              {metrics.var_95:>10.2%}")
    print(f"  CVaR (95%):             {metrics.cvar_95:>10.2%}")
    print(f"  Skewness:               {metrics.skewness:>10.2f}")
    print(f"  Kurtosis:               {metrics.kurtosis:>10.2f}")
    
    print(f"\n{'='*60}")

# Example usage and testing
if __name__ == "__main__":
    # Generate sample data for testing
    np.random.seed(42)
    n_periods = 252  # One year of daily data
    
    # Create sample returns with some trend and volatility
    returns = np.random.normal(0.001, 0.02, n_periods)  # 0.1% daily return, 2% daily vol
    returns[50:60] += 0.05  # Add a positive trend period
    returns[150:160] -= 0.08  # Add a negative trend period
    
    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(returns)
    
    # Print report
    print_strategy_report(metrics, "Sample Strategy")
    
    # Calculate individual metrics
    sharpe, sharpe_details = calculate_sharpe_ratio(returns)
    sortino, sortino_details = calculate_sortino_ratio(returns)
    calmar, calmar_details = calculate_calmar_ratio(returns)
    var_95, cvar_95, var_details = calculate_var_cvar(returns)
    
    print(f"\n📋 DETAILED METRICS:")
    print(f"Sharpe Ratio: {sharpe:.3f}")
    print(f"Sortino Ratio: {sortino:.3f}")
    print(f"Calmar Ratio: {calmar:.3f}")
    print(f"VaR (95%): {var_95:.3f}")
    print(f"CVaR (95%): {cvar_95:.3f}")
    
    # Create performance plots
    try:
        plot_strategy_performance(returns, "Sample Strategy Performance")
    except Exception as e:
        print(f"Note: Could not create plots: {e}")
