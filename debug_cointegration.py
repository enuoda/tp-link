#!/usr/bin/env python3
"""
Cointegration Debug Script

This script diagnoses issues with the cointegration analysis pipeline:
1. Tests data retrieval from Alpaca API
2. Checks data quality (NaN, missing values)
3. Runs pairwise cointegration tests with detailed output
4. Shows p-values for all pairs regardless of threshold

Usage:
    python debug_cointegration.py
    python debug_cointegration.py --symbols BTC/USD ETH/USD LTC/USD
    python debug_cointegration.py --days 60 --time-scale hour
    python debug_cointegration.py --verbose

Sam Dawley
"""

import argparse
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from statsmodels.tsa.stattools import coint


def setup_logging(verbose: bool = False):
    """Configure logging for debug output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def test_imports(logger):
    """Test that all required modules can be imported."""
    logger.info("=" * 60)
    logger.info("STEP 1: Testing Imports")
    logger.info("=" * 60)
    
    errors = []
    
    try:
        from src.finance import TIME_FRAMES
        logger.info("  ✅ src.finance.TIME_FRAMES")
    except Exception as e:
        logger.error(f"  ❌ src.finance.TIME_FRAMES: {e}")
        errors.append(str(e))
    
    try:
        from src.finance.crypto import fetch_crypto_data_for_cointegration
        logger.info("  ✅ src.finance.crypto.fetch_crypto_data_for_cointegration")
    except Exception as e:
        logger.error(f"  ❌ src.finance.crypto.fetch_crypto_data_for_cointegration: {e}")
        errors.append(str(e))
    
    try:
        from src.trading_strategy.cointegration_utils import run_pairwise_cointegration
        logger.info("  ✅ src.trading_strategy.cointegration_utils.run_pairwise_cointegration")
    except Exception as e:
        logger.error(f"  ❌ src.trading_strategy.cointegration_utils.run_pairwise_cointegration: {e}")
        errors.append(str(e))
    
    try:
        from src.trading_strategy.compute_benchmarks import compute_benchmarks
        logger.info("  ✅ src.trading_strategy.compute_benchmarks.compute_benchmarks")
    except Exception as e:
        logger.error(f"  ❌ src.trading_strategy.compute_benchmarks.compute_benchmarks: {e}")
        errors.append(str(e))
    
    if errors:
        logger.error(f"\n  Import errors detected: {len(errors)}")
        return False
    
    logger.info("  All imports successful!")
    return True


def test_data_retrieval(symbols, days_back, time_scale, logger):
    """Test data retrieval from Alpaca API."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Testing Data Retrieval")
    logger.info("=" * 60)
    logger.info(f"  Symbols: {symbols}")
    logger.info(f"  Days back: {days_back}")
    logger.info(f"  Time scale: {time_scale}")
    
    try:
        from src.finance import TIME_FRAMES
        from src.finance.crypto import fetch_crypto_data_for_cointegration
        
        frequency = TIME_FRAMES.get(time_scale, TIME_FRAMES["hour"])
        logger.info(f"  Frequency: {frequency}")
        
        price_arrays, sorted_symbols, timestamps = fetch_crypto_data_for_cointegration(
            symbols=symbols, days_back=days_back, frequency=frequency
        )
        
        logger.info(f"\n  Retrieved {len(timestamps)} timestamps")
        logger.info(f"  Date range: {timestamps[0]} to {timestamps[-1]}")
        logger.info(f"  Symbols returned: {list(sorted_symbols)}")
        
        # Check data quality for each symbol
        logger.info("\n  Data quality by symbol:")
        all_valid = True
        for i, sym in enumerate(sorted_symbols):
            arr = np.asarray(price_arrays[i], dtype=float)
            total = len(arr)
            valid = np.sum(~np.isnan(arr) & np.isfinite(arr))
            nan_count = np.sum(np.isnan(arr))
            inf_count = np.sum(np.isinf(arr))
            pct_valid = 100 * valid / total if total > 0 else 0
            
            status = "✅" if pct_valid > 90 else ("⚠️" if pct_valid > 50 else "❌")
            logger.info(f"    {sym}: {valid}/{total} valid ({pct_valid:.1f}%) {status}")
            
            if nan_count > 0:
                logger.debug(f"      NaN values: {nan_count}")
            if inf_count > 0:
                logger.debug(f"      Inf values: {inf_count}")
            
            if valid < 10:
                all_valid = False
                logger.warning(f"      ⚠️ Insufficient valid data for cointegration tests!")
        
        return price_arrays, sorted_symbols, timestamps, all_valid
        
    except Exception as e:
        logger.error(f"  ❌ Data retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, False


def test_pairwise_cointegration(price_arrays, sorted_symbols, p_threshold, logger):
    """Run pairwise cointegration tests on all pairs."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Pairwise Cointegration Tests")
    logger.info("=" * 60)
    logger.info(f"  P-value threshold: {p_threshold}")
    
    arrays = [np.asarray(a, dtype=float) for a in price_arrays]
    names = list(sorted_symbols)
    n = len(arrays)
    
    total_pairs = n * (n - 1) // 2
    logger.info(f"  Testing {total_pairs} pairs...")
    
    results = []
    pairs_tested = 0
    pairs_skipped = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            x = arrays[i]
            y = arrays[j]
            pair_name = f"{names[i]} vs {names[j]}"
            
            # Clean data
            mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
            x_valid = x[mask]
            y_valid = y[mask]
            
            if x_valid.size < 10:
                logger.debug(f"    {pair_name}: Skipped (only {x_valid.size} valid points)")
                pairs_skipped += 1
                continue
            
            try:
                t_stat, p_val, crit_vals = coint(x_valid, y_valid)
                pairs_tested += 1
                
                # Determine significance level
                if p_val < 0.01:
                    sig_level = "***"
                elif p_val < 0.05:
                    sig_level = "**"
                elif p_val < 0.10:
                    sig_level = "*"
                else:
                    sig_level = ""
                
                passed = p_val < p_threshold
                status = "✅ PASSED" if passed else "❌"
                
                results.append({
                    'pair': pair_name,
                    't_stat': t_stat,
                    'p_value': p_val,
                    'crit_values': crit_vals,
                    'passed': passed,
                    'sig_level': sig_level,
                    'valid_points': x_valid.size
                })
                
            except Exception as e:
                logger.warning(f"    {pair_name}: Test failed - {e}")
                pairs_skipped += 1
    
    # Sort by p-value
    results.sort(key=lambda x: x['p_value'])
    
    # Display results
    logger.info(f"\n  Results (sorted by p-value, threshold={p_threshold}):")
    logger.info("  " + "-" * 70)
    logger.info(f"  {'Pair':<30} {'t-stat':>10} {'p-value':>10} {'Status':>12} {'Points':>8}")
    logger.info("  " + "-" * 70)
    
    for r in results:
        status = "✅ PASSED" if r['passed'] else ""
        logger.info(
            f"  {r['pair']:<30} {r['t_stat']:>10.3f} {r['p_value']:>10.4f} {r['sig_level']:>3} "
            f"{status:>8} {r['valid_points']:>8}"
        )
    
    logger.info("  " + "-" * 70)
    
    # Summary
    passed_count = sum(1 for r in results if r['passed'])
    logger.info(f"\n  Summary:")
    logger.info(f"    Total pairs: {total_pairs}")
    logger.info(f"    Tested: {pairs_tested}")
    logger.info(f"    Skipped: {pairs_skipped}")
    logger.info(f"    Passed threshold ({p_threshold}): {passed_count}")
    
    # Show critical values for reference
    if results:
        logger.info(f"\n  Critical values (for reference):")
        crit = results[0]['crit_values']
        logger.info(f"    1%:  {crit[0]:.3f}")
        logger.info(f"    5%:  {crit[1]:.3f}")
        logger.info(f"    10%: {crit[2]:.3f}")
    
    return results


def test_full_pipeline(symbols, days_back, time_scale, p_threshold, logger):
    """Test the full compute_benchmarks pipeline."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Full Pipeline Test (compute_benchmarks)")
    logger.info("=" * 60)
    
    try:
        from src.trading_strategy.compute_benchmarks import compute_benchmarks
        
        payload = compute_benchmarks(
            symbols=symbols,
            days_back=days_back,
            time_scale=time_scale,
            max_groups=10,
            p_threshold=p_threshold,
        )
        
        groups = payload.get('cointegration_groups', [])
        logger.info(f"\n  Benchmark computation complete!")
        logger.info(f"  Found {len(groups)} cointegration groups")
        
        if groups:
            logger.info("\n  Groups found:")
            for g in groups:
                assets = g.get('assets', [])
                vectors = g.get('vectors', [{}])
                p_val = vectors[0].get('p_value', 'N/A') if vectors else 'N/A'
                logger.info(f"    {g.get('id')}: {assets} (p={p_val})")
        else:
            logger.warning("  No cointegration groups found!")
            logger.info("  Suggestions:")
            logger.info("    1. Increase p_threshold (e.g., --p-threshold 0.20)")
            logger.info("    2. Increase lookback period (e.g., --days 60)")
            logger.info("    3. Try different time scale (e.g., --time-scale day)")
            logger.info("    4. Use more liquid pairs (BTC, ETH, LTC)")
        
        return payload
        
    except Exception as e:
        logger.error(f"  ❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Debug cointegration analysis issues',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic test with default symbols
    python debug_cointegration.py

    # Test specific symbols
    python debug_cointegration.py --symbols BTC/USD ETH/USD LTC/USD

    # Use different lookback period
    python debug_cointegration.py --days 60

    # Use relaxed p-value threshold
    python debug_cointegration.py --p-threshold 0.20

    # Verbose output (shows all debug messages)
    python debug_cointegration.py --verbose
        """
    )
    
    parser.add_argument(
        '--symbols', 
        nargs='+', 
        default=['BTC/USD', 'ETH/USD', 'LTC/USD', 'LINK/USD', 'UNI/USD'],
        help='Crypto symbols to test (default: BTC/USD ETH/USD LTC/USD LINK/USD UNI/USD)'
    )
    parser.add_argument(
        '--days', 
        type=int, 
        default=30,
        help='Lookback period in days (default: 30)'
    )
    parser.add_argument(
        '--time-scale', 
        type=str, 
        default='hour',
        choices=['min', 'hour', 'day'],
        help='Time scale for data (default: hour)'
    )
    parser.add_argument(
        '--p-threshold', 
        type=float, 
        default=0.10,
        help='P-value threshold for cointegration (default: 0.10)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose debug output'
    )
    
    args = parser.parse_args()
    
    logger = setup_logging(args.verbose)
    
    print("\n" + "=" * 60)
    print("🔍 COINTEGRATION DEBUG TOOL")
    print("=" * 60)
    print(f"Symbols: {args.symbols}")
    print(f"Lookback: {args.days} days")
    print(f"Time scale: {args.time_scale}")
    print(f"P-threshold: {args.p_threshold}")
    print("=" * 60 + "\n")
    
    # Step 1: Test imports
    if not test_imports(logger):
        logger.error("\n❌ Import tests failed. Fix import errors before continuing.")
        return 1
    
    # Step 2: Test data retrieval
    price_arrays, sorted_symbols, timestamps, data_ok = test_data_retrieval(
        args.symbols, args.days, args.time_scale, logger
    )
    
    if price_arrays is None:
        logger.error("\n❌ Data retrieval failed. Check API credentials and network connection.")
        return 1
    
    # Step 3: Run pairwise cointegration tests
    results = test_pairwise_cointegration(
        price_arrays, sorted_symbols, args.p_threshold, logger
    )
    
    # Step 4: Test full pipeline
    payload = test_full_pipeline(
        args.symbols, args.days, args.time_scale, args.p_threshold, logger
    )
    
    # Final summary
    print("\n" + "=" * 60)
    print("📋 FINAL SUMMARY")
    print("=" * 60)
    
    passed_count = sum(1 for r in results if r['passed']) if results else 0
    
    if passed_count > 0:
        print(f"✅ Found {passed_count} cointegrated pairs")
    else:
        print("⚠️  No cointegrated pairs found at current threshold")
        print("\nRecommendations:")
        print(f"  1. Try higher p-threshold: python debug_cointegration.py --p-threshold 0.20")
        print(f"  2. Try longer lookback: python debug_cointegration.py --days 60")
        print(f"  3. Try daily data: python debug_cointegration.py --time-scale day")
    
    print("=" * 60 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

