#!/usr/bin/env python3
"""
Strategy Evaluation Example

This script demonstrates how to use the strategy evaluation functions
to analyze trading strategy performance.

Usage:
    python strategy_evaluation_example.py
"""

import numpy as np
import pandas as pd

from strategy_evaluation import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_comprehensive_metrics,
    plot_strategy_performance,
    print_strategy_report,
)


def generate_sample_strategy_data(
    n_periods: int = 252, strategy_type: str = "trend_following"
) -> np.ndarray:
    """
    Generate sample strategy returns for testing

    Args:
        n_periods: Number of periods to generate
        strategy_type: Type of strategy to simulate

    Returns:
        Array of strategy returns
    """

    np.random.seed(42)

    if strategy_type == "trend_following":
        # Trend-following strategy: higher returns with higher volatility
        returns = np.random.normal(0.0015, 0.025, n_periods)
        # Add some trend periods
        returns[50:80] += np.linspace(0, 0.02, 30)  # Uptrend
        returns[120:140] -= np.linspace(0, 0.015, 20)  # Downtrend

    elif strategy_type == "mean_reversion":
        # Mean reversion strategy: lower volatility, more consistent returns
        returns = np.random.normal(0.0008, 0.015, n_periods)
        # Add mean reversion effects
        for i in range(10, n_periods - 10):
            if abs(returns[i]) > 0.03:  # Large move
                returns[i + 1 : i + 5] *= -0.3  # Reversion effect

    elif strategy_type == "momentum":
        # Momentum strategy: periods of high returns followed by consolidation
        returns = np.random.normal(0.0012, 0.02, n_periods)
        # Add momentum bursts
        momentum_periods = [30, 90, 180, 220]
        for start in momentum_periods:
            if start + 15 < n_periods:
                returns[start : start + 15] += 0.01  # Momentum burst

    else:  # "random"
        # Random walk strategy
        returns = np.random.normal(0.001, 0.02, n_periods)

    return returns


def demonstrate_basic_metrics():
    """Demonstrate basic metric calculations"""

    print("🔍 DEMONSTRATING BASIC STRATEGY METRICS")
    print("=" * 50)

    # Generate sample data
    returns = generate_sample_strategy_data(252, "trend_following")

    # Calculate Sharpe ratio
    sharpe, sharpe_details = calculate_sharpe_ratio(returns)
    print(f"\n📊 Sharpe Ratio Analysis:")
    print(f"  Sharpe Ratio: {sharpe:.3f}")
    print(f"  Annualized Return: {sharpe_details['annualized_return']:.2%}")
    print(f"  Volatility: {sharpe_details['volatility']:.2%}")
    print(f"  Excess Return: {sharpe_details['excess_return']:.2%}")

    # Calculate Sortino ratio
    sortino, sortino_details = calculate_sortino_ratio(returns)
    print(f"\n📈 Sortino Ratio Analysis:")
    print(f"  Sortino Ratio: {sortino:.3f}")
    print(f"  Downside Deviation: {sortino_details['downside_deviation']:.2%}")
    print(f"  Downside Frequency: {sortino_details['downside_frequency']:.2%}")

    # Calculate Calmar ratio
    calmar, calmar_details = calculate_calmar_ratio(returns)
    print(f"\n📉 Calmar Ratio Analysis:")
    print(f"  Calmar Ratio: {calmar:.3f}")
    print(f"  Annualized Return: {calmar_details['annualized_return']:.2%}")
    print(f"  Max Drawdown: {calmar_details['max_drawdown']:.2%}")


def compare_strategies():
    """Compare different strategy types"""

    print("\n\n🔄 STRATEGY COMPARISON")
    print("=" * 50)

    strategies = {
        "Trend Following": "trend_following",
        "Mean Reversion": "mean_reversion",
        "Momentum": "momentum",
        "Random Walk": "random",
    }

    results = {}

    for name, strategy_type in strategies.items():
        returns = generate_sample_strategy_data(252, strategy_type)
        metrics = calculate_comprehensive_metrics(returns)
        results[name] = metrics

        print(f"\n📋 {name} Strategy:")
        print(f"  Sharpe Ratio:    {metrics.sharpe_ratio:>6.2f}")
        print(f"  Sortino Ratio:   {metrics.sortino_ratio:>6.2f}")
        print(f"  Calmar Ratio:    {metrics.calmar_ratio:>6.2f}")
        print(f"  Max Drawdown:    {metrics.max_drawdown:>6.2%}")
        print(f"  Volatility:      {metrics.volatility:>6.2%}")

    # Find best strategy by Sharpe ratio
    best_strategy = max(results.keys(), key=lambda x: results[x].sharpe_ratio)
    print(f"\n🏆 Best Strategy by Sharpe Ratio: {best_strategy}")
    print(f"   Sharpe Ratio: {results[best_strategy].sharpe_ratio:.3f}")


def demonstrate_comprehensive_analysis():
    """Demonstrate comprehensive strategy analysis"""

    print("\n\n📊 COMPREHENSIVE STRATEGY ANALYSIS")
    print("=" * 50)

    # Generate sample data
    returns = generate_sample_strategy_data(500, "trend_following")

    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(returns)

    # Print detailed report
    print_strategy_report(metrics, "Sample Trading Strategy")

    # Create performance plots
    try:
        print("\n📈 Generating performance plots...")
        plot_strategy_performance(returns, "Sample Strategy Performance Analysis")
    except Exception as e:
        print(f"Note: Could not create plots: {e}")


def demonstrate_real_world_usage():
    """Demonstrate how to use with real trading data"""

    print("\n\n🌍 REAL-WORLD USAGE EXAMPLE")
    print("=" * 50)

    # Simulate real trading scenario
    print("Simulating real trading scenario with:")
    print("- 1 year of daily trading data")
    print("- Paper trading results")
    print("- Risk-free rate of 2.5%")

    # Generate realistic trading data
    returns = generate_sample_strategy_data(252, "trend_following")

    # Calculate metrics with custom parameters
    sharpe, _ = calculate_sharpe_ratio(returns, risk_free_rate=0.025)
    sortino, _ = calculate_sortino_ratio(returns, risk_free_rate=0.025)

    print(f"\n📊 Performance Results:")
    print(f"  Sharpe Ratio (2.5% risk-free): {sharpe:.3f}")
    print(f"  Sortino Ratio (2.5% risk-free): {sortino:.3f}")

    # Interpretation
    if sharpe > 2.0:
        rating = "Excellent"
    elif sharpe > 1.5:
        rating = "Very Good"
    elif sharpe > 1.0:
        rating = "Good"
    elif sharpe > 0.5:
        rating = "Acceptable"
    else:
        rating = "Poor"

    print(f"\n🎯 Strategy Rating: {rating}")
    print(f"   (Based on Sharpe Ratio: {sharpe:.2f})")


def main():
    """Main demonstration function"""

    print("🎯 STRATEGY EVALUATION TOOLKIT DEMONSTRATION")
    print("=" * 60)
    print("This script demonstrates the comprehensive strategy evaluation")
    print("functions for analyzing trading strategy performance.")

    try:
        # Run demonstrations
        demonstrate_basic_metrics()
        compare_strategies()
        demonstrate_comprehensive_analysis()
        demonstrate_real_world_usage()

        print(f"\n\n✅ DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("The strategy evaluation toolkit provides:")
        print("• Sharpe, Sortino, and Calmar ratios")
        print("• Comprehensive risk and return metrics")
        print("• Drawdown analysis")
        print("• VaR and CVaR calculations")
        print("• Trade-level statistics")
        print("• Performance visualization")
        print("\nReady for use with your real trading data!")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        print("This might be due to missing dependencies (matplotlib, pandas)")


if __name__ == "__main__":
    main()
