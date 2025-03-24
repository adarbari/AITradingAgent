#!/usr/bin/env python3
"""
Command-line interface for training and managing cached trading models.
"""
import os
import sys
import argparse
from datetime import datetime, timedelta
import pandas as pd
from tabulate import tabulate

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.train.trainer import TrainingManager


def train_model(args):
    """Train a model with the provided arguments."""
    # Create synthetic params if using synthetic data
    synthetic_params = None
    if args.data_source == "synthetic":
        synthetic_params = {
            "initial_price": args.synthetic_initial_price,
            "volatility": args.synthetic_volatility,
            "drift": args.synthetic_drift,
            "volume_min": 1000000,
            "volume_max": 5000000
        }
    
    # Create model params if provided
    model_params = None
    if args.learning_rate or args.n_steps or args.batch_size or args.gamma:
        model_params = {}
        if args.learning_rate:
            model_params["learning_rate"] = args.learning_rate
        if args.n_steps:
            model_params["n_steps"] = args.n_steps
        if args.batch_size:
            model_params["batch_size"] = args.batch_size
        if args.gamma:
            model_params["gamma"] = args.gamma
    
    # Create training manager
    trainer = TrainingManager(
        models_dir=args.models_dir,
        verbose=args.verbose
    )
    
    # Train model for each symbol
    for symbol in args.symbols:
        print(f"\nTraining model for {symbol}...")
        model, model_path = trainer.get_model(
            symbol=symbol,
            train_start=args.train_start,
            train_end=args.train_end,
            feature_count=args.feature_count,
            data_source=args.data_source,
            timesteps=args.timesteps,
            force_train=args.force,
            synthetic_params=synthetic_params,
            model_params=model_params
        )
        
        if model:
            print(f"Model for {symbol} ready at: {model_path}")
        else:
            print(f"Failed to train model for {symbol}")


def list_models(args):
    """List cached models."""
    trainer = TrainingManager(
        models_dir=args.models_dir,
        verbose=args.verbose
    )
    
    # Get cached models
    models = trainer.list_cached_models(symbol=args.symbol)
    
    if not models:
        print(f"No cached models found{' for ' + args.symbol if args.symbol else ''}.")
        return
    
    # Format output
    table_data = []
    for model in models:
        table_data.append([
            model["symbol"],
            model["train_start"],
            model["train_end"],
            model["data_source"],
            model["feature_count"],
            model["created_at"],
            model["hash"][:8],
            model["model_path"]
        ])
    
    # Print as table
    headers = ["Symbol", "Train Start", "Train End", "Data Source", 
               "Features", "Created At", "Hash", "Path"]
    print(tabulate(table_data, headers=headers, tablefmt="pretty"))
    print(f"\nTotal: {len(models)} model(s)")


def clear_models(args):
    """Clear cached models."""
    trainer = TrainingManager(
        models_dir=args.models_dir,
        verbose=args.verbose
    )
    
    # Confirm if not forced
    if not args.force:
        symbol_str = f"symbol {args.symbol}" if args.symbol else "ALL symbols"
        date_str = f" older than {args.older_than}" if args.older_than else ""
        confirm = input(f"This will remove cached models for {symbol_str}{date_str}. Continue? (y/N): ")
        if confirm.lower() != 'y':
            print("Operation cancelled.")
            return
    
    # Clear models
    removed = trainer.clear_cache(symbol=args.symbol, older_than=args.older_than)
    
    if removed > 0:
        print(f"Removed {removed} model(s) from cache.")
    else:
        print("No models were removed.")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Train and manage cached trading models")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--models-dir", type=str, default="models",
                               help="Directory to store models (default: models)")
    common_parser.add_argument("--verbose", type=int, default=1,
                               help="Verbosity level (0: silent, 1: normal, 2: detailed)")
    
    # Train command
    train_parser = subparsers.add_parser("train", parents=[common_parser],
                                        help="Train model(s) for one or more symbols")
    train_parser.add_argument("--symbols", type=str, nargs="+", required=True,
                             help="Stock symbol(s) to train on")
    train_parser.add_argument("--train-start", type=str, default="2020-01-01",
                             help="Start date for training data (default: 2020-01-01)")
    train_parser.add_argument("--train-end", type=str, default="2022-12-31",
                             help="End date for training data (default: 2022-12-31)")
    train_parser.add_argument("--feature-count", type=int, default=21,
                             help="Number of features to use (default: 21)")
    train_parser.add_argument("--data-source", type=str, default="yfinance",
                             choices=["yfinance", "synthetic"],
                             help="Source of data (default: yfinance)")
    train_parser.add_argument("--timesteps", type=int, default=100000,
                             help="Training timesteps (default: 100000)")
    train_parser.add_argument("--force", action="store_true",
                             help="Force retraining even if a cached model exists")
    
    # Synthetic data parameters
    train_parser.add_argument("--synthetic-initial-price", type=float, default=100.0,
                             help="Initial price for synthetic data (default: 100.0)")
    train_parser.add_argument("--synthetic-volatility", type=float, default=0.02,
                             help="Volatility for synthetic data (default: 0.02)")
    train_parser.add_argument("--synthetic-drift", type=float, default=0.001,
                             help="Drift for synthetic data (default: 0.001)")
    
    # Model parameters
    train_parser.add_argument("--learning-rate", type=float,
                             help="Learning rate for the model")
    train_parser.add_argument("--n-steps", type=int,
                             help="Number of steps per update")
    train_parser.add_argument("--batch-size", type=int,
                             help="Batch size for training")
    train_parser.add_argument("--gamma", type=float,
                             help="Discount factor")
    
    # List command
    list_parser = subparsers.add_parser("list", parents=[common_parser],
                                      help="List cached models")
    list_parser.add_argument("--symbol", type=str,
                           help="Filter by symbol")
    
    # Clear command
    clear_parser = subparsers.add_parser("clear", parents=[common_parser],
                                       help="Clear cached models")
    clear_parser.add_argument("--symbol", type=str,
                            help="Symbol to clear cache for (default: all)")
    clear_parser.add_argument("--older-than", type=str,
                            help="Clear models older than this date (YYYY-MM-DD)")
    clear_parser.add_argument("--force", action="store_true",
                            help="Skip confirmation prompt")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run command
    if args.command == "train":
        train_model(args)
    elif args.command == "list":
        list_models(args)
    elif args.command == "clear":
        clear_models(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 