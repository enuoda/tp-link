#!/usr/bin/env python3

"""
Build a JSON file of pairwise cointegration parameters suitable for a trading agent.

Usage (example):
    python build_cointegration_json.py --tickers BTC/USD ETH/USD LTC/USD \
        --days 7 --time_scale min --out cointegration_params.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from typing import List

from cointegration_utils import run_pairwise_cointegration


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build cointegration parameter JSON")
    p.add_argument("--tickers", nargs="+", required=True, help="List of crypto tickers like BTC/USD ETH/USD ...")
    p.add_argument("--days", type=int, default=7, help="Number of days to look back (default: 7)")
    p.add_argument("--time_scale", type=str, default="min", help="Time scale: min/hour/day (default: min)")
    p.add_argument("--p_threshold", type=float, default=0.05, help="p-value threshold (default: 0.05)")
    p.add_argument("--out", type=str, default="cointegration_params.json", help="Output JSON filename")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Compute pairwise cointegration mapping
    mapping = run_pairwise_cointegration(
        tickers=args.tickers,
        time_scale=args.time_scale,
        days_back=args.days,
        p_threshold=args.p_threshold,
    )

    # Convert to JSON-friendly structure
    # Example format:
    # {
    #   "pairs": [
    #       {"tickers": ["BTC","ETH"], "hedge_ratio": 1.23, "std_spread": 45.6},
    #       ...
    #   ]
    # }
    payload = {
        "pairs": [
            {
                "tickers": [pair[0], pair[1]],
                "hedge_ratio": vals[0],
                "std_spread": vals[1],
            }
            for pair, vals in mapping.items()
        ]
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {len(payload['pairs'])} cointegrated pairs to {out_path}")


if __name__ == "__main__":
    main()




