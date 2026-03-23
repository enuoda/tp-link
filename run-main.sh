#!/usr/bin/env bash
#
# run-main.sh — Launcher for the tp-link cointegration trading bot.
#
# Uncomment the mode you want to run and adjust variables as needed.
# Only one mode should be uncommented at a time.
#

set -euo pipefail # https://gist.github.com/akrasic/380bda362e0420be08709152c91ca1f9

# ==============================================================================
# GLOBAL OPTIONS (used across multiple modes)
# ==============================================================================

LIVE=""                          # set to "--live" for real trading (default: paper/testnet)
SYMBOLS=""                       # e.g., "BTC ETH SOL" (empty = all supported)
EXCLUDE_SYMBOLS=""               # e.g., "DOGE SHIB" (empty = exclude none)

# ==============================================================================
# BENCHMARK COMPUTATION OPTIONS
# ==============================================================================

DAYS=30                          # lookback days for benchmark computation
TIME_SCALE="hour"                # time scale: min | hour | day
MAX_GROUPS=10                    # max cointegration pairs to keep
P_THRESHOLD=0.05                 # p-value threshold (lower = stricter)
BENCHMARK_REFRESH_DAYS=7         # days before benchmarks are considered stale

# ==============================================================================
# TRADING OPTIONS
# ==============================================================================

CYCLE_INTERVAL=30                # seconds between trading cycles
DURATION=10                      # duration in minutes (for timed modes)
HEALTH_INTERVAL=15               # minutes between health status logs
LOOKBACK_BARS=500                # lookback bars for historical data
MAX_STREAM_SYMBOLS=40            # max symbols to stream simultaneously

# ==============================================================================
# Z-SCORE THRESHOLDS
# ==============================================================================

ENTRY_ZSCORE=2.0                 # z-score to enter positions (lower = more trades)
EXIT_ZSCORE=0.5                  # z-score to exit positions (higher = exit sooner)
MAX_ZSCORE=10.0                  # z-score cap: block entries / force exits beyond this

# ==============================================================================
# STALENESS THRESHOLDS (seconds)
# ==============================================================================

ENTRY_STALENESS=30.0             # max price age for entries
EXIT_STALENESS=300.0             # max price age for exits (5 min)
EMERGENCY_STALENESS=900.0        # triggers emergency exit (15 min)

# ==============================================================================
# ROLLING RECALIBRATION
# ==============================================================================

RECALIBRATE_INTERVAL=10          # minutes between recalibrations (0 = disabled)
RECALIBRATE_MIN_OBS=50           # min observations before recalibration

# ==============================================================================
# RISK MANAGEMENT
# ==============================================================================

MAX_LOSS_PER_SPREAD=50.0         # USD stop-loss per spread position
REENTRY_COOLDOWN=300.0           # seconds before re-entering after exit (5 min)
FEE_RATE=0.0005                  # per-side taker fee (0.05%)

# ==============================================================================
# BUILD ARGUMENT STRINGS
# ==============================================================================

# Helper: only add flag if variable is non-empty
GLOBAL_ARGS=""
[[ -n "$LIVE" ]]            && GLOBAL_ARGS+=" $LIVE"
[[ -n "$SYMBOLS" ]]         && GLOBAL_ARGS+=" --symbols $SYMBOLS"
[[ -n "$EXCLUDE_SYMBOLS" ]] && GLOBAL_ARGS+=" --exclude-symbols $EXCLUDE_SYMBOLS"

BENCHMARK_ARGS="
    --days $DAYS
    --time-scale $TIME_SCALE
    --max-groups $MAX_GROUPS
    --p-threshold $P_THRESHOLD
"

TRADING_ARGS="
    --cycle-interval $CYCLE_INTERVAL
    --lookback-bars $LOOKBACK_BARS
    --max-stream-symbols $MAX_STREAM_SYMBOLS
    --entry-zscore $ENTRY_ZSCORE
    --exit-zscore $EXIT_ZSCORE
    --max-zscore $MAX_ZSCORE
    --benchmark-refresh-days $BENCHMARK_REFRESH_DAYS
    --max-loss-per-spread $MAX_LOSS_PER_SPREAD
    --reentry-cooldown $REENTRY_COOLDOWN
    --fee-rate $FEE_RATE
"

STALENESS_ARGS="
    --entry-staleness $ENTRY_STALENESS
    --exit-staleness $EXIT_STALENESS
    --emergency-staleness $EMERGENCY_STALENESS
"

RECALIBRATE_ARGS="
    --recalibrate-interval $RECALIBRATE_INTERVAL
    --recalibrate-min-obs $RECALIBRATE_MIN_OBS
"

# ==============================================================================
# MODE SELECTION — uncomment exactly one
# ==============================================================================

# --- 1. Account info ---
# python main.py --mode account $GLOBAL_ARGS

# --- 2. Compute benchmarks ---
# python main.py --mode compute-benchmarks $GLOBAL_ARGS $BENCHMARK_ARGS

# --- 3. Monitor (stream prices, no trading) ---
# python main.py --mode monitor --duration $DURATION $GLOBAL_ARGS

# --- 4. Trade (timed — runs for DURATION minutes then stops) ---
# python main.py --mode trade --duration $DURATION $GLOBAL_ARGS $BENCHMARK_ARGS $TRADING_ARGS

# --- 5. Trade indefinitely (runs until Ctrl+C) ---
python main.py --mode trade-indefinite --health-interval $HEALTH_INTERVAL $GLOBAL_ARGS $BENCHMARK_ARGS $TRADING_ARGS $STALENESS_ARGS $RECALIBRATE_ARGS
