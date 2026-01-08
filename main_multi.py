#!/usr/bin/env python3
"""
Multi-Worker Trading Launcher

Run multiple trading bots in parallel, each with its own API credentials
and hyperparameters. Uses multiprocessing for isolation between workers.

Features:
    - Load configuration from JSON file
    - Spawn N workers with separate credentials
    - Isolated logging per worker (logs/{worker_name}/)
    - Graceful shutdown handling
    - Dynamic worker count based on config
    - Falls back to single-process mode for 1 worker

Usage:
    # Run multiple workers
    python main_multi.py --config workers_config.json
    
    # Dry run (validate config without starting)
    python main_multi.py --config workers_config.json --dry-run

Sam Dawley
"""

import argparse
import json
import logging
import multiprocessing
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import from main.py
try:
    from main import (
        WorkerConfig,
        run_trading_worker,
        setup_worker_logging,
    )
    from src.finance.credentials import ExchangeCredentials
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nPlease ensure all required dependencies are installed.")
    sys.exit(1)


# Set up module-level logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [LAUNCHER] %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/launcher.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Global tracking for graceful shutdown
active_processes: Dict[str, multiprocessing.Process] = {}
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    logger.info(f"\n🛑 Received signal {signum}. Shutting down all workers...")
    shutdown_requested = True
    
    # Terminate all active processes
    for name, proc in active_processes.items():
        if proc.is_alive():
            logger.info(f"Terminating worker: {name}")
            proc.terminate()


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate the workers configuration from JSON.
    
    Args:
        config_path: Path to the JSON configuration file
        
    Returns:
        Parsed configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r') as f:
        config = json.load(f)
    
    # Validate structure
    if "workers" not in config:
        raise ValueError("Config must contain 'workers' array")
    
    if not isinstance(config["workers"], list):
        raise ValueError("'workers' must be an array")
    
    if len(config["workers"]) == 0:
        raise ValueError("'workers' array cannot be empty")
    
    # Validate each worker
    for i, worker in enumerate(config["workers"]):
        if "api_key" not in worker:
            raise ValueError(f"Worker {i} missing 'api_key'")
        if "secret_key" not in worker:
            raise ValueError(f"Worker {i} missing 'secret_key'")
        if "params" not in worker:
            raise ValueError(f"Worker {i} missing 'params'")
    
    return config


def create_worker_configs(config: Dict[str, Any]) -> List[WorkerConfig]:
    """
    Create WorkerConfig objects from the loaded configuration.
    
    Args:
        config: Parsed JSON configuration
        
    Returns:
        List of WorkerConfig objects
    """
    workers = []
    exchange = config.get("exchange", "kraken")
    
    for i, worker_dict in enumerate(config["workers"]):
        # Set defaults
        name = worker_dict.get("name", f"worker_{i}")
        
        # Create credentials
        creds_dict = {
            "api_key": worker_dict["api_key"],
            "secret_key": worker_dict["secret_key"],
            "demo_mode": worker_dict.get("demo_mode", True),
            "exchange_name": worker_dict.get("exchange_name", exchange),
        }
        
        credentials = ExchangeCredentials.from_dict(creds_dict)
        
        # Create log directory path
        log_dir = f"logs/{name}"
        
        workers.append(WorkerConfig(
            name=name,
            credentials=credentials,
            params=worker_dict.get("params", {}),
            log_dir=log_dir,
        ))
    
    return workers


def worker_process(config: WorkerConfig) -> int:
    """
    Entry point for a worker process.
    
    This function runs in a separate process and handles its own
    logging and error handling.
    
    Args:
        config: WorkerConfig for this worker
        
    Returns:
        Exit code
    """
    try:
        return run_trading_worker(config)
    except Exception as e:
        print(f"❌ Worker {config.name} crashed: {e}", flush=True)
        return 1


def run_single_worker(config: WorkerConfig) -> int:
    """
    Run a single worker without multiprocessing.
    
    Used when only one worker is configured, to avoid the overhead
    of process spawning.
    
    Args:
        config: WorkerConfig for the worker
        
    Returns:
        Exit code
    """
    logger.info(f"Running single worker: {config.name}")
    return run_trading_worker(config)


def run_multi_workers(configs: List[WorkerConfig]) -> int:
    """
    Run multiple workers using multiprocessing.
    
    Each worker runs in its own process with isolated logging.
    If a worker crashes, others continue running.
    
    Args:
        configs: List of WorkerConfig objects
        
    Returns:
        0 if all workers completed successfully, 1 otherwise
    """
    global active_processes, shutdown_requested
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info(f"Starting {len(configs)} workers...")
    
    # Create and start processes
    processes: Dict[str, multiprocessing.Process] = {}
    
    for config in configs:
        logger.info(f"  Spawning worker: {config.name}")
        proc = multiprocessing.Process(
            target=worker_process,
            args=(config,),
            name=config.name,
        )
        proc.start()
        processes[config.name] = proc
        active_processes[config.name] = proc
    
    logger.info(f"✅ All {len(processes)} workers started")
    
    # Monitor processes
    failed_workers = []
    
    try:
        while not shutdown_requested:
            all_done = True
            
            for name, proc in processes.items():
                if proc.is_alive():
                    all_done = False
                elif proc.exitcode is not None and name not in failed_workers:
                    if proc.exitcode != 0:
                        logger.warning(f"⚠️ Worker {name} exited with code {proc.exitcode}")
                        failed_workers.append(name)
                    else:
                        logger.info(f"✅ Worker {name} completed successfully")
            
            if all_done:
                break
            
            time.sleep(1.0)
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
        shutdown_requested = True
    
    # Clean up
    for name, proc in processes.items():
        if proc.is_alive():
            logger.info(f"Terminating worker: {name}")
            proc.terminate()
            proc.join(timeout=5.0)
            if proc.is_alive():
                logger.warning(f"Force killing worker: {name}")
                proc.kill()
    
    # Report results
    logger.info("\n" + "=" * 60)
    logger.info("📋 MULTI-WORKER SESSION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total workers: {len(configs)}")
    logger.info(f"Failed workers: {len(failed_workers)}")
    if failed_workers:
        logger.info(f"  Failed: {', '.join(failed_workers)}")
    logger.info("=" * 60)
    
    return 1 if failed_workers else 0


def main() -> int:
    """Main entry point for multi-worker launcher."""
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    parser = argparse.ArgumentParser(
        description='Multi-Worker Trading Launcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run multiple trading workers
    python main_multi.py --config workers_config.json
    
    # Validate config without running
    python main_multi.py --config workers_config.json --dry-run
    
    # Show parsed config
    python main_multi.py --config workers_config.json --show-config

Configuration File Format (JSON):
    {
        "exchange": "kraken",
        "workers": [
            {
                "name": "worker_0",
                "api_key": "KEY_1",
                "secret_key": "SECRET_1",
                "demo_mode": true,
                "params": {
                    "mode": "trade-indefinite",
                    "cycle_interval": 30,
                    "entry_zscore": 2.0,
                    ...
                }
            },
            ...
        ]
    }
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to workers configuration JSON file'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate config without starting workers'
    )
    parser.add_argument(
        '--show-config',
        action='store_true',
        help='Show parsed configuration and exit'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        logger.info(f"Loading config: {args.config}")
        config = load_config(args.config)
        logger.info(f"✅ Config loaded: {len(config['workers'])} workers defined")
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        return 1
    except (ValueError, json.JSONDecodeError) as e:
        logger.error(f"❌ Invalid config: {e}")
        return 1
    
    # Create worker configs
    try:
        worker_configs = create_worker_configs(config)
    except Exception as e:
        logger.error(f"❌ Failed to create worker configs: {e}")
        return 1
    
    # Show config if requested
    if args.show_config:
        print("\n" + "=" * 60)
        print("PARSED WORKER CONFIGURATIONS")
        print("=" * 60)
        for wc in worker_configs:
            print(f"\nWorker: {wc.name}")
            print(f"  Credentials: {wc.credentials}")
            print(f"  Log dir: {wc.log_dir}")
            print(f"  Params: {json.dumps(wc.params, indent=4)}")
        print("=" * 60)
        return 0
    
    # Dry run - just validate
    if args.dry_run:
        logger.info("✅ Dry run: Config is valid")
        for wc in worker_configs:
            logger.info(f"  Worker: {wc.name} ({wc.credentials.exchange_name})")
        return 0
    
    # Run workers
    if len(worker_configs) == 1:
        # Single worker - run without multiprocessing
        logger.info("Single worker mode (no multiprocessing)")
        return run_single_worker(worker_configs[0])
    else:
        # Multiple workers - use multiprocessing
        logger.info("Multi-worker mode (with multiprocessing)")
        return run_multi_workers(worker_configs)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

