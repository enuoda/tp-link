#ifndef GILLESPIE_SIMULATOR_H
#define GILLESPIE_SIMULATOR_H

#include <vector>
#include <random>

/**
 * @brief Implementation of the Gillespie algorithm for simulating jump processes
 * 
 * This class implements the exact Gillespie algorithm for simulating stochastic
 * jump processes between discrete states. The algorithm samples jump times from
 * exponential distributions and selects transitions based on their propensities.
 */
class GillespieSimulator {
public:
    /**
     * @brief Structure to represent a point in the trajectory
     */
    struct TrajectoryPoint {
        double time;    ///< Time of the state
        int state;      ///< State index
    };

private:
    std::mt19937 rng;                                    ///< Random number generator
    std::uniform_real_distribution<double> uniform_dist; ///< Uniform distribution [0,1)
    int num_states;                                      ///< Number of discrete states
    std::vector<std::vector<double>> transition_rates;   ///< Transition rate matrix Q_a(X)

public:
    /**
     * @brief Constructor for GillespieSimulator
     * @param n_states Number of discrete states in the system
     * @param seed Random seed for reproducibility (default: 42)
     */
    GillespieSimulator(int n_states, unsigned seed = 42);

    /**
     * @brief Calculate the escape propensity λ(t') from the current state
     * @param current_state Current state index
     * @return Total rate of leaving the current state
     */
    double calculate_escape_propensity(int current_state);

    /**
     * @brief Sample the next jump time from exponential distribution
     * @param lambda Escape propensity (rate parameter)
     * @return Time until next jump
     */
    double sample_jump_time(double lambda);

    /**
     * @brief Select which transition occurs based on propensities
     * @param current_state Current state index
     * @param lambda Total escape propensity
     * @return Index of the next state
     */
    int select_transition(int current_state, double lambda);

    /**
     * @brief Run the main Gillespie algorithm simulation
     * @param initial_state Starting state index
     * @param max_time Maximum simulation time
     * @param max_steps Maximum number of jumps (default: 10000)
     * @return Vector of trajectory points (time, state) pairs
     */
    std::vector<TrajectoryPoint> simulate(int initial_state, double max_time, int max_steps = 10000);

    /**
     * @brief Calculate the log probability of a given trajectory
     * @param trajectory Vector of trajectory points
     * @return Log probability of the trajectory
     */
    double calculate_log_probability(const std::vector<TrajectoryPoint>& trajectory);

    /**
     * @brief Print trajectory to console
     * @param trajectory Vector of trajectory points to print
     */
    void print_trajectory(const std::vector<TrajectoryPoint>& trajectory);

    /**
     * @brief Set a specific transition rate in the rate matrix
     * @param from_state Source state index
     * @param to_state Target state index  
     * @param rate Transition rate value
     */
    void set_transition_rate(int from_state, int to_state, double rate);
};

/**
 * @brief Example function demonstrating usage of the Gillespie algorithm
 */
void run_gillespie_example();

#endif // GILLESPIE_SIMULATOR_H

#!/bin/bash
"""
References
    alpaca.data.historical source code:
        https://github.com/alpacahq/alpaca-py/blob/master/alpaca/data/historical/crypto.py

    supported cryptocurrencies
        https://alpaca.markets/support/what-are-the-supported-coins-pairs
"""

# stdlib
from datetime import datetime, timedelta
import os
from typing import List, Tuple
from zoneinfo import ZoneInfo

# numerics
import numpy as np
import pandas as pd

# visualization
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

# trading
from alpaca.data.live.crypto import CryptoDataStream
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.requests import (
    CryptoBarsRequest,
    CryptoQuoteRequest,
    CryptoTradesRequest,
    CryptoLatestQuoteRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from alpaca.trading.enums import (
    AssetClass,
    AssetStatus,
    OrderSide,
    OrderType,
    TimeInForce,
    QueryOrderStatus,
)
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetAssetsRequest,
    MarketOrderRequest,
    LimitOrderRequest,
    StopLimitOrderRequest,
    GetOrdersRequest,
    ClosePositionRequest,
)
from alpaca.trading.stream import TradingStream


from . import CRYPTO_TICKERS, BTC_PAIRS, USDT_PAIRS, USDC_PAIRS, USD_PAIRS


# ==================================================
# FUNCTIONS FOR INTERACTING WITH THE MARKET
# ==================================================


def get_historical_bars(
    client,
    symbol: str,
    start: str,
    end: str = datetime.now(ZoneInfo("America/New_York")),
    interval: TimeFrameUnit = TimeFrameUnit.Day,
) -> pd.DataFrame:
    req = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(amount=1, unit=interval),
        start=start,
        end=end,
        limit=None,
    )
    return client.get_crypto_bars(req).df


def retrieve_crypto_data(
    symbols: str | List[str],
    start_time,
    end_time,
    frequency: TimeFrameUnit = TimeFrame(1, TimeFrameUnit.Hour),
    limit: int = None,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Retrieve crypto data and store in user-friendly manner

    Returns:
        df : pd.DataFrame
            columns: ['open' 'high' 'low' 'close' 'volume' 'trade_count' 'vwap']
            n_columns := number of columns in this dataframe
        ohlcv : jnp.ndarray, shape (n_symbols, n_columns, n_times)
            symbols are indexed in the order they appear in 'sorted_symbols'
        sorted_symbols : List[str]
            symbols in DataFrame not necessarily ordered in same the
            same way they are ordered in 'symbols' parameter
            'sorted_symbols' provides this ordering, i.e., symbol corresponding
            to sorted_symbols[i] corresponds to data in ohlcv[i]
    """
    client = CryptoHistoricalDataClient()

    req = CryptoBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=frequency,
        start=start_time,
        end=end_time,
        limit=limit,
    )

    data = client.get_crypto_bars(req).df

    # ----- make numpy-friendly -----

    sorted_symbols = data.index.get_level_values("symbol").unique()
    n_timestamps = len(data.loc[symbols[0]])
    n_columns = len(data.columns)
    ohlcv = data.values.reshape(len(symbols), n_timestamps, n_columns).transpose(0, 2, 1)

    # squeeze to remove redundant outer dimension if only one symbol is requested
    return data, np.squeeze(ohlcv), sorted_symbols


# ==================================================
# TRADING INTERFACE
# ==================================================


class CryptoTrader:

    __slots__ = "client"

    def __init__(self) -> None:
        self.client = TradingClient(
            api_key=os.getenv("ALPACA_API_KEY"),
            secret_key=os.getenv("ALPACA_SECRET_KEY"),
            url_override=os.getenv("ALPACA_API_BASE_URL"),
            paper=True,
        )

        # check trading account
        # You can check definition of each field in the following documents
        # ref. https://docs.alpaca.markets/docs/account-plans
        # ref. https://docs.alpaca.markets/reference/getaccount-1
        acct = self.client.get_account()

        # check account configuration
        # ref. https://docs.alpaca.markets/reference/getaccountconfig-1
        acct_config = self.client.get_account_configurations()

    def _list_crypto_assets(self):
        # get list of crypto pairs
        # ref. https://docs.alpaca.markets/reference/get-v2-assets-1
        req = GetAssetsRequest(asset_class=AssetClass.CRYPTO, status=AssetStatus.ACTIVE)
        return self.get_all_assets(req)

    def submit_market_order(self, symbol: str, amt: float) -> None:
        req = MarketOrderRequest(
            symbol=symbol,
            notational=amt,  # specify $(amt) in USD
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC,
        )
        return self.client.submit_order(req)

    def submit_limit_order(self, symbol: str, amt: float, limit_price: float) -> None:
        req = LimitOrderRequest(
            symbol=symbol,
            notational=amt,
            limit_price=limit_price,
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.GTC,
        )
        return self.client.submit_order(req)

    def stop_limit_order(
        self, symbol: str, amt: float, limit_price: float, stop_price: float
    ) -> None:
        req = StopLimitOrderRequest(
            symbol=symbol,
            notational=amt,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
            limit_price=limit_price,
            stop_price=stop_price,
        )
        return self.client.submit_order(req)

    def get_all_orders(self, symbol: str) -> None:
        # get a list of orders including closed (e.g. filled) orders by specifying symbol
        req = GetOrdersRequest(status=QueryOrderStatus.ALL, symbols=[symbol])
        return self.client.get_orders(req)

    def get_open_orders(self, symbol: str) -> None:
        # see all open orders
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        return self.client.get_orders(req)

    def cancel_open_orders(self) -> None:
        self.client.cancel_orders()


# ==================================================
# PLOTTING
# ==================================================


def plot_df(df: pd.DataFrame) -> None:

    styles = {
        "open": {"color": "red", "linestyle": "-", "linewidth": 1.5},
        "high": {"color": "blue", "linestyle": "-", "linewidth": 1.5},
        "low": {"color": "green", "linestyle": "-", "linewidth": 1.5},
        "close": {"color": "magenta", "linestyle": "-", "linewidth": 1.5},
        "vwap": {"color": "black", "linestyle": "--", "linewidth": 2},
    }

    # Assuming your dataframe is called 'df'
    # Get the first symbol's data
    first_symbol = df.index.get_level_values("symbol")[0]
    symbol_data = df.xs(first_symbol, level="symbol")

    # Create figure and GridSpec layout
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 3, figure=fig, height_ratios=[2, 1, 1, 1], width_ratios=[1, 1, 1])

    # Main OHLC + VWAP plot (top row, spanning all columns)
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.plot(
        symbol_data.index, symbol_data["open"], label="Open", **styles["open"], alpha=0.8
    )
    ax_main.plot(
        symbol_data.index, symbol_data["high"], label="High", **styles["high"], alpha=0.8
    )
    ax_main.plot(
        symbol_data.index, symbol_data["low"], label="Low", **styles["low"], alpha=0.8
    )
    ax_main.plot(
        symbol_data.index,
        symbol_data["close"],
        label="Close",
        **styles["close"],
        alpha=0.8,
    )
    ax_main.plot(symbol_data.index, symbol_data["vwap"], label="VWAP", **styles["vwap"])
    ax_main.set_ylabel("Price", fontweight="bold")
    ax_main.legend(loc="upper left", ncol=5)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_title(f"Price Data for {first_symbol}", fontweight="bold", fontsize=14)

    # Volume plot (second row, spanning all columns)
    ax_volume = fig.add_subplot(gs[1, :], sharex=ax_main)
    ax_volume.bar(symbol_data.index, symbol_data["volume"], alpha=0.7, color="steelblue")
    ax_volume.grid(True, alpha=0.3)
    ax_volume.set_title("Volume", fontweight="bold")

    # Individual metric plots (bottom two rows)
    # Row 3: Open, High, Low
    ax_open = fig.add_subplot(gs[2, 0], sharex=ax_main)
    ax_open.plot(symbol_data.index, symbol_data["open"], **styles["open"])
    ax_open.grid(True, alpha=0.3)
    ax_open.set_title("Open Price", fontweight="bold")

    ax_high = fig.add_subplot(gs[2, 1], sharex=ax_main)
    ax_high.plot(symbol_data.index, symbol_data["high"], **styles["high"])
    ax_high.grid(True, alpha=0.3)
    ax_high.set_title("High Price", fontweight="bold")

    ax_low = fig.add_subplot(gs[2, 2], sharex=ax_main)
    ax_low.plot(symbol_data.index, symbol_data["low"], **styles["low"])
    ax_low.grid(True, alpha=0.3)
    ax_low.set_title("Low Price", fontweight="bold")

    # Row 4: Close, Trade Count, and a combined volume/trade_count plot
    ax_close = fig.add_subplot(gs[3, 0], sharex=ax_main)
    ax_close.plot(symbol_data.index, symbol_data["close"], **styles["close"])
    ax_close.grid(True, alpha=0.3)
    ax_close.set_title("Close Price", fontweight="bold")

    ax_trades = fig.add_subplot(gs[3, 1], sharex=ax_main)
    ax_trades.plot(
        symbol_data.index, symbol_data["trade_count"], color="orange", linewidth=1.5
    )
    ax_trades.grid(True, alpha=0.3)
    ax_trades.set_title("Number of Trades", fontweight="bold")

    # Combined volume and trade count (dual y-axis)
    ax_combined = fig.add_subplot(gs[3, 2], sharex=ax_main)
    ax_combined_twin = ax_combined.twinx()

    # Volume on left y-axis
    vol_line = ax_combined.plot(
        symbol_data.index,
        symbol_data["volume"],
        color="steelblue",
        linewidth=1.5,
        label="Volume",
    )
    ax_combined.set_ylabel("Volume", color="steelblue", fontweight="bold")
    ax_combined.tick_params(axis="y", labelcolor="steelblue")

    # Trade count on right y-axis
    trade_line = ax_combined_twin.plot(
        symbol_data.index,
        symbol_data["trade_count"],
        color="orange",
        linewidth=1.5,
        label="Trade Count",
    )
    ax_combined_twin.set_ylabel("Trade Count", color="orange", fontweight="bold")
    ax_combined_twin.tick_params(axis="y", labelcolor="orange")

    ax_combined.grid(True, alpha=0.3)
    ax_combined.set_title("Volume vs Trade Count", fontweight="bold")

    # Format x-axis labels only for bottom row
    for ax in [ax_close, ax_trades, ax_combined]:
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("Time", fontweight="bold")

    # Hide x-axis labels for upper plots to avoid clutter
    plt.setp(ax_main.get_xticklabels(), visible=False)
    plt.setp(ax_volume.get_xticklabels(), visible=False)
    plt.setp(ax_open.get_xticklabels(), visible=False)
    plt.setp(ax_high.get_xticklabels(), visible=False)
    plt.setp(ax_low.get_xticklabels(), visible=False)

    # Add overall title
    # fig.suptitle(
    #     f"Comprehensive Time Series Analysis for {first_symbol}",
    #     fontsize=16,
    #     fontweight="bold",
    #     y=0.98,
    # )

    plt.tight_layout()
    plt.subplots_adjust(top=0.94)  # Make room for the main title

    savename = f"{symbol_data}"
    # plt.savefig(savename, dpi=512)
    plt.show()

    return


def compare_symbols(
    df: pd.DataFrame, symbols: List[str], figsize: tuple = (18, 14)
) -> None:
    """
    Create detailed comparison plots for multiple symbols from the dataframe.

    Parameters:
    -----------
    df : pandas.DataFrame
        Multi-indexed dataframe with ('symbol', 'timestamp') index
    symbols : list
        List of symbols to compare (must exist in the dataframe)
    figsize : tuple
        Figure size (width, height)

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    """

    # Validate symbols exist in dataframe
    available_symbols = df.index.get_level_values("symbol").unique()
    missing_symbols = [s for s in symbols if s not in available_symbols]
    if missing_symbols:
        raise ValueError(f"Symbols not found in dataframe: {missing_symbols}")

    # Extract data for each symbol
    symbol_data = {}
    for symbol in symbols:
        symbol_data[symbol] = df.xs(symbol, level="symbol")

    # Create figure and GridSpec layout
    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = GridSpec(
        4,
        2,
        figure=fig,
        height_ratios=[2, 1, 1, 1],
        width_ratios=[1, 1],
        hspace=0.3,
        wspace=0.25,
    )

    # Define colors for each symbol
    colors = plt.cm.Set1(np.linspace(0, 1, len(symbols)))
    symbol_colors = dict(zip(symbols, colors))

    # 1. Main comparison plot - All OHLC + VWAP (top row, spanning both columns)
    ax_main = fig.add_subplot(gs[0, :])

    for symbol in symbols:
        data = symbol_data[symbol]
        color = symbol_colors[symbol]

        # Plot OHLC with slightly different line styles for distinction
        ax_main.plot(
            data.index,
            data["open"],
            label=f"{symbol} Open",
            alpha=0.7,
            linewidth=1.5,
            color=color,
            linestyle="-",
        )
        ax_main.plot(
            data.index,
            data["high"],
            label=f"{symbol} High",
            alpha=0.7,
            linewidth=1.5,
            color=color,
            linestyle=":",
        )
        ax_main.plot(
            data.index,
            data["low"],
            label=f"{symbol} Low",
            alpha=0.7,
            linewidth=1.5,
            color=color,
            linestyle="--",
        )
        ax_main.plot(
            data.index,
            data["close"],
            label=f"{symbol} Close",
            alpha=0.8,
            linewidth=2,
            color=color,
            linestyle="-",
        )
        ax_main.plot(
            data.index,
            data["vwap"],
            label=f"{symbol} VWAP",
            alpha=0.9,
            linewidth=2,
            color=color,
            linestyle="-.",
        )

    ax_main.set_ylabel("Price", fontweight="bold", fontsize=12)
    ax_main.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=1, fontsize=9)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_title("Price Comparison - All Symbols", fontweight="bold", fontsize=14)

    # Format x-axis with proper tick marks
    ax_main.tick_params(axis="x", rotation=45, which="both")
    if hasattr(list(symbol_data.values())[0].index, "to_pydatetime"):
        ax_main.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        ax_main.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    ax_main.xaxis.set_tick_params(which="major", length=6, width=1.5)
    ax_main.xaxis.set_tick_params(which="minor", length=3, width=1)

    # 2. Individual price component comparisons (remaining rows)
    price_components = [
        ("Open Prices", "open"),
        ("High Prices", "high"),
        ("Low Prices", "low"),
        ("Close Prices", "close"),
        ("VWAP", "vwap"),
    ]

    # Positions for subplots: (1,0), (1,1), (2,0), (2,1), (3,:)
    subplot_positions = [(1, 0), (1, 1), (2, 0), (2, 1), (3, slice(None))]

    for i, ((title, column), pos) in enumerate(zip(price_components, subplot_positions)):
        if pos[1] == slice(None):  # VWAP spans both columns
            ax = fig.add_subplot(gs[pos[0], pos[1]], sharex=ax_main)
        else:
            ax = fig.add_subplot(gs[pos[0], pos[1]], sharex=ax_main)

        # Plot each symbol for this price component
        for symbol in symbols:
            data = symbol_data[symbol]
            color = symbol_colors[symbol]

            ax.plot(
                data.index,
                data[column],
                label=symbol,
                color=color,
                linewidth=2,
                alpha=0.8,
            )

        ax.set_ylabel(title, fontweight="bold")
        ax.legend(loc="upper left" if i < 4 else "upper right")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{title} Comparison", fontweight="bold")

        # Only show x-axis labels on bottom row
        if pos[0] != 3:  # Not bottom row
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.tick_params(axis="x", rotation=45)
            ax.set_xlabel("Time", fontweight="bold")

    # Add overall title
    symbols_str = ", ".join(symbols)
    fig.suptitle(
        f"Multi-Symbol Comparison: {symbols_str}", fontsize=16, fontweight="bold", y=0.96
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, right=0.85)  # Make room for title and legend
    plt.show()

    return


def compare_symbols_normalized(
    df: pd.DataFrame, symbols: List[str], figsize: tuple = (16, 12)
) -> None:
    """
    Create normalized comparison plots showing percentage changes from the first data point.
    This is useful for comparing symbols with very different price ranges.

    Parameters:
    -----------
    df : pandas.DataFrame
        Multi-indexed dataframe with ('symbol', 'timestamp') index
    symbols : list
        List of symbols to compare
    figsize : tuple
        Figure size (width, height)

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    """

    # Extract and normalize data for each symbol
    symbol_data = {}
    for symbol in symbols:
        data = df.xs(symbol, level="symbol")
        # Normalize to percentage change from first value
        normalized_data = data.copy()
        for col in ["open", "high", "low", "close", "vwap"]:
            first_val = data[col].iloc[0]
            normalized_data[col] = ((data[col] - first_val) / first_val) * 100
        symbol_data[symbol] = normalized_data

    # Create figure and GridSpec layout
    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[2, 1, 1],
        width_ratios=[1, 1],
        hspace=0.3,
        wspace=0.25,
    )

    # Define colors for each symbol
    colors = plt.cm.Set1(np.linspace(0, 1, len(symbols)))
    symbol_colors = dict(zip(symbols, colors))

    # Main normalized comparison plot
    ax_main = fig.add_subplot(gs[0, :])

    for symbol in symbols:
        data = symbol_data[symbol]
        color = symbol_colors[symbol]

        ax_main.plot(
            data.index,
            data["close"],
            label=f"{symbol} Close",
            color=color,
            linewidth=2.5,
            alpha=0.8,
        )
        ax_main.plot(
            data.index,
            data["vwap"],
            label=f"{symbol} VWAP",
            color=color,
            linewidth=2,
            alpha=0.7,
            linestyle="--",
        )

    ax_main.axhline(y=0, color="black", linestyle="-", alpha=0.5, linewidth=1)
    ax_main.set_ylabel("Percentage Change (%)", fontweight="bold", fontsize=12)
    ax_main.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax_main.grid(True, alpha=0.3)
    # ax_main.set_title('Normalized Price Comparison (% Change from Start)',
    #  fontweight='bold', fontsize=14)

    # Format x-axis
    ax_main.tick_params(axis="x", rotation=45)
    if hasattr(list(symbol_data.values())[0].index, "to_pydatetime"):
        ax_main.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    # Individual normalized components
    components = [
        ("Close % Change", "close"),
        ("VWAP % Change", "vwap"),
        ("High % Change", "high"),
        ("Low % Change", "low"),
    ]
    positions = [(1, 0), (1, 1), (2, 0), (2, 1)]

    for (title, column), pos in zip(components, positions):
        if column == "vwap":
            ls = "--"
        else:
            ls = "-"

        ax = fig.add_subplot(gs[pos[0], pos[1]], sharex=ax_main)

        for symbol in symbols:
            data = symbol_data[symbol]
            color = symbol_colors[symbol]
            ax.plot(
                data.index,
                data[column],
                label=symbol,
                color=color,
                linestyle=ls,
                linewidth=2,
                alpha=0.8,
            )

        ax.axhline(y=0, color="black", linestyle="-", alpha=0.5, linewidth=0.8)
        ax.set_ylabel("Change (%)", fontweight="bold")
        # ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontweight="bold")

        if pos[0] != 2:  # Not bottom row
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.tick_params(axis="x", rotation=45)
            ax.set_xlabel("Time", fontweight="bold")

    symbols_str = ", ".join(symbols)
    fig.suptitle(
        f"Normalized Multi-Symbol Comparison: {symbols_str}",
        fontsize=16,
        fontweight="bold",
        y=0.96,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, right=0.85)
    plt.show()

    return


# def main():

#     # No API key required for crypto data
#     client = CryptoHistoricalDataClient()
#     start_date = datetime.now(ZoneInfo("America/New_York")) - timedelta(days=100)
#     symbols = USDT_PAIRS

#     # ===== collecting data =====

#     df = get_historical_bars(client, symbols, start=start_date)
#     plot_df(df)

#     # compare_symbols(df, symbols)
#     compare_symbols_normalized(df, symbols)

#     return


if __name__ == "__main__":

    # ----- useful constants -----

    NOW = datetime.now(ZoneInfo("America/New_York"))
    PAST_N_YEARS = dict((n, timedelta(days=n * 365)) for n in range(10))

    TIME_FRAMES = {
        "min": TimeFrame(amount=1, unit=TimeFrameUnit.Minute),
        "hour": TimeFrame(amount=1, unit=TimeFrameUnit.Hour),
        "day": TimeFrame(amount=1, unit=TimeFrameUnit.Day),
        "week": TimeFrame(amount=1, unit=TimeFrameUnit.Week),
        "month": TimeFrame(amount=1, unit=TimeFrameUnit.Month),
    }

    # ----- specify data to be retrieved -----

    # symbols = ["BTC/USD"]
    symbols = USDT_PAIRS

    start_time = NOW - PAST_N_YEARS[2]
    end_time = NOW
    frequency = TIME_FRAMES["day"]
    limit = None  # no limit to number of data points to store

    df, ohlcv, sort_symbols = retrieve_crypto_data(
        symbols=symbols,
        start_time=start_time,
        end_time=end_time,
        frequency=frequency,
        limit=limit,
    )

    # print(ohlcv.shape)
    # print(ohlcv[0, :10, :10])
    # # print(df[["open", "high", "low", "close", "vwap"]])
    # # print(df.columns.values)
    # print(df.loc[sort_symbols[0]])

    plot_df(df)
