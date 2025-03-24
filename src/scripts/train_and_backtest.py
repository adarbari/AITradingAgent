def train_model(data_fetcher_type, symbol, start_date, end_date,
               trading_env_class=TradingEnvironment, model_path=None):
    """
    Train a trading agent on historical data.
    
    Args:
        data_fetcher_type (str): Type of data fetcher to use (yahoo, synthetic, csv)
        symbol (str): Symbol to fetch data for
        start_date (str): Start date for training data
        end_date (str): End date for training data
        trading_env_class (class): Class of trading environment to use
        model_path (str, optional): Path to save model to
    
    Returns:
        BaseAlgorithm: Trained model
    """
    # Create data fetcher
    data_fetcher = DataFetcherFactory.create_data_fetcher(data_fetcher_type)
    
    # Fetch training data
    training_data = data_fetcher.fetch_data(symbol, start_date, end_date)
    
    # Add technical indicators
    training_data = data_fetcher.add_technical_indicators(training_data)
    
    # Prepare data for agent
    prices, features = data_fetcher.prepare_data_for_agent(training_data)
    
    # Create training environment
    env = trading_env_class(
        prices=prices,
        features=features,
        initial_balance=10000,
        transaction_fee_percent=0.001
    )
    
    # Choose appropriate policy based on observation space type
    if isinstance(env.observation_space, gym.spaces.Dict):
        policy = "MultiInputPolicy"  # For dictionary observation spaces
    else:
        policy = "MlpPolicy"  # For Box observation spaces
    
    # Train agent
    trainer = Trainer()
    model = trainer.train_model(
        env=env,
        total_timesteps=10000,
        model_type="ppo",
        policy=policy
    )
    
    # Save model if path provided
    if model_path:
        trainer.save_model(model, model_path)
    
    return model 