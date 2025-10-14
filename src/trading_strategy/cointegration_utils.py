from __future__ import annotations

# stdlib
from typing import List, Tuple

# numerics
import numpy as np

# stats
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm

# project imports
from finance import TIME_FRAMES
from finance.crypto import fetch_crypto_data_for_cointegration


def _normalize_timeframe_key(time_scale: str) -> str:
    """
    Map a human-friendly time scale to a TIME_FRAMES key.
    Supported aliases:
        minutes: ["min", "minute", "minutes", "m"]
        hours:   ["hour", "hours", "h"]
        days:    ["day", "days", "d"]
    """
    key = time_scale.strip().lower()
    if key in {"min", "minute", "minutes", "m"}:
        return "min"
    if key in {"hour", "hours", "h"}:
        return "hour"
    if key in {"day", "days", "d"}:
        return "day"
    # fallback to original if already a valid key
    if key in TIME_FRAMES:
        return key
    raise ValueError(f"Unsupported time_scale: {time_scale}")


def _base_symbol(symbol: str) -> str:
    """Return the base part of a crypto symbol like 'BTC/USD' -> 'BTC'."""
    return symbol.split("/")[0].upper() if "/" in symbol else symbol.upper()


def run_pairwise_cointegration(
    tickers: List[str], time_scale: str, days_back: int, p_threshold: float = 0.05
) -> dict[Tuple[str, str], Tuple[float, float]]:
    """
    Retrieve data and run Engle-Granger cointegration tests on all unordered pairs.

    Returns:
        mapping: dict with keys = (base_ticker_i, base_ticker_j)
                 values = (hedge_ratio, std_dev_of_spread)

    Notes:
        - hedge_ratio is from OLS y ~ a + b*x (b returned)
        - std_dev_of_spread is the standard deviation of (y - b*x) over valid timestamps
    """
    if len(tickers) < 2:
        return [], []

    tf_key = _normalize_timeframe_key(time_scale)

    # Retrieve time-aligned close price arrays and timestamps
    price_arrays, sorted_symbols, _timestamps = fetch_crypto_data_for_cointegration(
        symbols=tickers, days_back=days_back, frequency=TIME_FRAMES[tf_key]
    )

    # Ensure numpy arrays and consistent container types
    arrays = [np.asarray(a, dtype=float) for a in price_arrays]
    names = list(sorted_symbols)

    results: dict[Tuple[str, str], Tuple[float, float]] = {}

    n = len(arrays)
    for i in range(n):
        for j in range(i + 1, n):
            x = arrays[i]
            y = arrays[j]

            # mask to timestamps where both series are valid (non-NaN, finite)
            mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
            x_valid = x[mask]
            y_valid = y[mask]

            # require sufficient points
            if x_valid.size < 10:
                continue

            # Engle-Granger
            try:
                t_stat, p_val, _ = coint(x_valid, y_valid)
            except Exception:
                continue

            if p_val < p_threshold:
                # OLS hedge ratio y ~ a + b*x
                try:
                    X = sm.add_constant(x_valid)
                    model = sm.OLS(y_valid, X).fit()
                    hedge_ratio = float(model.params[1])
                except Exception:
                    # fallback: simple slope via np.polyfit
                    hedge_ratio = float(np.polyfit(x_valid, y_valid, 1)[0])

                # Standard deviation of spread
                spread = y_valid - hedge_ratio * x_valid
                std_spread = float(np.std(spread, ddof=1)) if spread.size > 1 else 0.0

                pair = (_base_symbol(names[i]), _base_symbol(names[j]))
                results[pair] = (hedge_ratio, std_spread)

    return results


def check_higher_order_cointegration(
    tickers: List[str], time_scale: str, days_back: int, p_threshold: float = 0.05
) -> Tuple[bool, np.ndarray | None]:
    """
    Check k>2 series for cointegration using the Johansen procedure and return weights
    for a cointegrating vector if one exists.

    Method: Johansen cointegration test (coint_johansen), trace statistic.
    We declare cointegration if the null of "no cointegration" is rejected at ~5%.

    Returns:
        is_cointegrated: whether at least one cointegrating vector is found
        weights: numpy array of weights (length k) for a cointegrating vector (normalized),
                 or None if not cointegrated.

    Notes:
        - Johansen (1988, 1991) test is the standard multivariate generalization of
          Engle-Granger, commonly used to test for and estimate multiple cointegrating
          relationships. statsmodels exposes it via coint_johansen.
        - Other approaches include Phillips-Ouliaris tests and Bayesian/penalized variants,
          but Johansen is the most commonly used off-the-shelf tool for k>2.
    """
    if len(tickers) < 2:
        return False, None

    tf_key = _normalize_timeframe_key(time_scale)

    # Retrieve time-aligned arrays
    price_arrays, sorted_symbols, _timestamps = fetch_crypto_data_for_cointegration(
        symbols=tickers, days_back=days_back, frequency=TIME_FRAMES[tf_key]
    )

    arrays = [np.asarray(a, dtype=float) for a in price_arrays]
    # Build a composite mask where ALL series are valid
    composite_mask = None
    for a in arrays:
        valid = ~(np.isnan(a) | np.isinf(a))
        composite_mask = valid if composite_mask is None else (composite_mask & valid)

    # Apply mask
    aligned = [a[composite_mask] for a in arrays]
    # Require sufficient length
    if len(aligned) == 0 or aligned[0].size < (len(aligned) * 5):  # heuristic minimal length
        return False, None

    # Stack as T x k for Johansen (observations by columns in statsmodels is rows by cols)
    Y = np.column_stack(aligned)

    try:
        # det_order = -1 lets the routine choose deterministic terms; k_ar_diff small (e.g., 1)
        joh = coint_johansen(Y, det_order=0, k_ar_diff=1)
    except Exception:
        return False, None

    # Use trace statistic to test r=0 (no cointegration) vs r>=1
    # stats in joh.lr1, critical values in joh.cvt (rows for r, cols for [90%,95%,99%])
    trace_stat_r0 = float(joh.lr1[0])
    crit_95_r0 = float(joh.cvt[0, 1])  # 95% column

    if trace_stat_r0 > crit_95_r0:
        # At least one cointegrating vector; take the first eigenvector as weights
        beta = np.asarray(joh.evec[:, 0], dtype=float)
        # Normalize for readability (e.g., last element = 1) if possible
        if np.isfinite(beta[-1]) and abs(beta[-1]) > 1e-12:
            beta = beta / beta[-1]
        return True, beta

    return False, None


