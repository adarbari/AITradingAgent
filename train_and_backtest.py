#!/usr/bin/env python3
"""
Train PPO models using synthetic or real data up to a specified date and backtest
them for a future period, comparing performance against a market index.
"""
import os
import glob
import argparse
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
from src.agent.trading_env import TradingEnvironment
from src.agent import SafeTradingEnvironment
from src.data import BaseDataFetcher, SyntheticDataFetcher, YahooDataFetcher, DataFetcherFactory
from src.backtest import Backtester
from src.models import ModelTrainer
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import warnings
from pathlib import Path
import traceback

# Create results and models directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('data/cache', exist_ok=True)

def main():
    """Main function to run the script"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train and backtest trading models')
    parser.add_argument('--data_source', type=str, default='synthetic', 
                        choices=['synthetic', 'yahoo'],
                        help='Data source to use (synthetic or yahoo)')
    parser.add_argument('--train_start', type=str, default='2020-01-01',
                        help='Start date for training data (YYYY-MM-DD)')
    parser.add_argument('--train_end', type=str, default='2023-12-31',
                        help='End date for training data (YYYY-MM-DD)')
    parser.add_argument('--test_start', type=str, default='2024-01-01',
                        help='Start date for test data (YYYY-MM-DD)')
    parser.add_argument('--test_end', type=str, default='2024-03-31',
                        help='End date for test data (YYYY-MM-DD)')
    parser.add_argument('--symbols', type=str, default='AAPL,AMZN',
                        help='Comma-separated list of stock symbols')
    
    args = parser.parse_args()
    
    # Convert symbols string to list
    symbols = args.symbols.split(',')
    
    print(f"Using {args.data_source} data")
    print(f"Training period: {args.train_start} to {args.train_end}")
    print(f"Testing period: {args.test_start} to {args.test_end}")
    print(f"Symbols: {', '.join(symbols)}")
    
    # Initialize the model trainer
    model_trainer = ModelTrainer(models_dir='models')
    
    # Train models for each symbol
    trained_models = {}
    for symbol in symbols:
        model = model_trainer.train_model(
            symbol=symbol, 
            train_start_date=args.train_start, 
            train_end_date=args.train_end, 
            data_source=args.data_source,
            data_fetcher_factory=DataFetcherFactory,
            trading_env_class=SafeTradingEnvironment
        )
        if model:
            trained_models[symbol] = model
    
    if not trained_models:
        print("No models were successfully trained. Exiting.")
        return
    
    # Initialize the backtester
    backtester = Backtester(results_dir='results')
    
    # Backtest models
    results_dict = {}
    for symbol in trained_models.keys():
        model_name = f"ppo_{symbol}_{args.train_start.split('-')[0]}_{args.train_end.split('-')[0]}"
        results = backtester.backtest_model(
            model_name=model_name,
            test_start_date=args.test_start,
            test_end_date=args.test_end,
            data_source=args.data_source,
            data_fetcher_factory=DataFetcherFactory,
            trading_env_class=SafeTradingEnvironment
        )
        if results:
            results_dict[symbol] = results
    
    if not results_dict:
        print("No models were successfully backtested. Exiting.")
        return
    
    # Get NASDAQ performance for comparison
    nasdaq_performance = backtester.get_market_performance(
        symbol="^IXIC",
        test_start_date=args.test_start,
        test_end_date=args.test_end,
        data_source=args.data_source,
        data_fetcher_factory=DataFetcherFactory
    )
    
    # Plot comparison even with just one model
    backtester.plot_comparison(
        results_dict=results_dict,
        market_performance=nasdaq_performance,
        test_start_date=args.test_start,
        test_end_date=args.test_end,
        market_name="NASDAQ"
    )
    
    print("\nTraining and backtesting completed successfully!")

if __name__ == "__main__":
    main() 