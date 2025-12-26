#!/bin/bash
# Primary trading routines
# Sam Dawley
# 08/2025

# stdlib
import datetime
import time

# numerics
import numpy as np

# trading
from alpaca.common.exceptions import APIError
from alpaca.trading.models import Order

# custom libraries
from crypto import CryptoTrader


# ==================================================
# AUXILIARY ROUTINES
# ==================================================


def print_transaction(order: Order) -> str:
    """Print transaction details"""
    result = f"Transaction {order.id} ({order.symbol})\n"
    result += f"\tCreated   @ {order.created_at}\n"
    result += f"\tSubmitted @ {order.submitted_at}\n"
    result += f"\tFilled    @ {order.filled_at}\n"
    result += f"\t{order.side.upper()} ${order.notional} of {order.symbol} @ {order.filled_avg_price} USD/share "
    return result


# ==================================================
# PRIMARY TRADING ROUTINE
# ==================================================


def main():
    """Primary trading routine"""

    print("STARTING TRADING BOT...", flush=True)

    # ----- metadata -----
    minimal_transaction = 10.0  # USD

    lag_secs = 10  # seconds
    initial_hr = datetime.datetime.now()
    elapsed_hr = 0

    # ----- configure trading client -----
    ct = CryptoTrader(paper=True)

    # ----- active trading block -----
    try:
        while True:

            # ----- trading logic -----

            # u = np.random.uniform(0, 1)
            # if u < 0.9:
            i = np.random.randint(len(ct.crypto_universe))
            symbol = ct.crypto_universe[i]
            symbol = "SUSHI/USD"

            buy_signal = np.random.uniform(0, 1) < 0.5
            sell_signal = not buy_signal

            # ----- trading execution -----
            if buy_signal:
                order = ct.buy_market_order(symbol=symbol, notional=minimal_transaction)

                print_transaction(order)

            elif sell_signal:
                order = ct.sell_market_order(symbol=symbol, notional=minimal_transaction)

                print_transaction(order)

            else:
                print("Holding...", flush=True)

            # ----- timing -----
            time.sleep(lag_secs)  # avoid excessive CPU usage
            current_hr = datetime.datetime.now()

            if current_hr.hour != initial_hr.hour:
                print(f"It's {current_hr}...", flush=True)
                initial_hr = current_hr
                elapsed_hr += 1

                ct.print_account_summary()

    except APIError as api_e:
        print(f"\nCaught APIError: {api_e}", flush=True)

    except AssertionError as ae:
        print(f"\nCaught AssertionError: {ae}", flush=True)

    except KeyboardInterrupt:
        print(f"\nExiting by keyboard interrupt...", flush=True)

    except ValueError as ve:
        print(f"\nCaught ValueError: {ve}", flush=True)

    # ----- close trading client -----

    ...

    return


if __name__ == "__main__":
    main()
