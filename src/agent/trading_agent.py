"""
Trading agent using reinforcement learning.
"""
import os
import random
from collections import deque

import numpy as np
# Import specific keras modules without importing the base package
from keras import Model
from keras import layers
from keras import optimizers

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


class DQNTradingAgent:
    """
    Trading agent using Deep Q-Network (DQN).
    """
    
    def __init__(self, state_size, action_size, model_dir='models', batch_size=64, gamma=0.95, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, learning_rate=0.001, 
                 hidden_units=(128, 64), memory_size=10000, verbose=0):
        """
        Initialize the DQN agent.
        
        Args:
            state_size (tuple): Shape of the state
            action_size (int): Number of possible actions
            model_dir (str, optional): Directory to save models. Default is 'models'.
            batch_size (int, optional): Batch size for training. Default is 64.
            gamma (float, optional): Discount factor. Default is 0.95.
            epsilon (float, optional): Exploration rate. Default is 1.0.
            epsilon_decay (float, optional): Decay rate for exploration. Default is 0.995.
            epsilon_min (float, optional): Minimum exploration rate. Default is 0.01.
            learning_rate (float, optional): Learning rate. Default is 0.001.
            hidden_units (tuple, optional): Sizes of hidden layers. Default is (128, 64).
            memory_size (int, optional): Size of replay memory. Default is 10000.
            verbose (int, optional): Verbosity level for model operations. Default is 0.
        """
        # State and action space
        self.state_size = state_size  # (window_size, num_features)
        self.action_size = action_size  # 3 (hold, buy, sell)
        
        # Agent parameters
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.memory_size = memory_size
        self.verbose = verbose
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Build model
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    
    def _build_model(self):
        """
        Build the neural network model.
        
        Returns:
            keras.Model: Neural network model
        """
        # Feature input (technical indicators)
        feature_input = layers.Input(shape=self.state_size, name='feature_input')
        feature_flatten = layers.Flatten()(feature_input)
        
        # Account info input (balance, shares_held, cost_basis)
        account_input = layers.Input(shape=(3,), name='account_input')
        
        # Combine inputs
        combined = layers.Concatenate()([feature_flatten, account_input])
        
        # Shared layers
        x = combined
        for units in self.hidden_units:
            x = layers.Dense(units, activation='relu')(x)
        
        # Output layer
        output = layers.Dense(self.action_size, activation='linear')(x)
        
        # Create model
        model = Model(inputs=[feature_input, account_input], outputs=output)
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        
        return model
    
    def update_target_model(self):
        """
        Update target model with weights from the main model.
        """
        self.target_model.set_weights(self.model.get_weights())
    
    def act(self, state, training=True):
        """
        Choose an action based on the current state.
        
        Args:
            state (dict): Current state
            training (bool, optional): Whether the agent is training. Default is True.
            
        Returns:
            int: Action to take
        """
        if training and np.random.rand() <= self.epsilon:
            # Exploration: random action
            return random.randrange(self.action_size)
        
        # Exploitation: use model to predict best action
        features = np.expand_dims(state['features'], axis=0)
        account_info = np.expand_dims(state['account_info'], axis=0)
        q_values = self.model.predict([features, account_info], verbose=self.verbose)
        
        return np.argmax(q_values[0])
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory.
        
        Args:
            state (dict): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (dict): Next state
            done (bool): Whether the episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=None):
        """
        Train the agent with experiences from replay memory.
        
        Args:
            batch_size (int, optional): Batch size for training. Default is None.
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        if len(self.memory) < batch_size:
            return
        
        # Sample batch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        # Initialize arrays for features and account info
        feature_shape = (batch_size,) + self.state_size
        states_features = np.zeros(feature_shape)
        states_account = np.zeros((batch_size, 3))
        
        next_states_features = np.zeros(feature_shape)
        next_states_account = np.zeros((batch_size, 3))
        
        actions, rewards, dones = [], [], []
        
        # Fill arrays
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states_features[i] = state['features']
            states_account[i] = state['account_info']
            
            next_states_features[i] = next_state['features']
            next_states_account[i] = next_state['account_info']
            
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
        
        # Predict Q-values for current states
        targets = self.model.predict([states_features, states_account], verbose=self.verbose)
        
        # Predict Q-values for next states using target model
        next_q_values = self.target_model.predict([next_states_features, next_states_account], verbose=self.verbose)
        
        # Update targets for actions taken
        for i, (action, reward, done) in enumerate(zip(actions, rewards, dones)):
            if done:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.gamma * np.amax(next_q_values[i])
        
        # Train the model
        self.model.fit(
            [states_features, states_account], targets, 
            epochs=1, verbose=self.verbose, batch_size=batch_size
        )
        
        # Update exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        """
        Load model from file.
        
        Args:
            name (str): Model name
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        path = os.path.join(self.model_dir, name)
        try:
            print(f"Loading model from {path}")
            self.model.load_weights(path)
            self.update_target_model()
            print("Model loaded successfully")
            return True
        except (OSError, IOError) as e:
            print(f"Error loading model from {path}: File not found or inaccessible")
            return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def save(self, name):
        """
        Save model to file.
        
        Args:
            name (str): Model name
            
        Returns:
            bool: True if model saved successfully, False otherwise
        """
        path = os.path.join(self.model_dir, name)
        try:
            print(f"Saving model to {path}")
            self.model.save_weights(path)
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False


class PPOTradingAgent:
    """
    Trading agent using Proximal Policy Optimization (PPO).
    """
    
    def __init__(self, env, model_dir='models', learning_rate=0.0001, n_steps=2048, batch_size=64, gamma=0.99, verbose=1):
        """
        Initialize the PPO agent.
        
        Args:
            env (gym.Env): Trading environment
            model_dir (str, optional): Directory to save models. Default is 'models'.
            learning_rate (float, optional): Learning rate. Default is 0.0001.
            n_steps (int, optional): Number of steps per update. Default is 2048.
            batch_size (int, optional): Batch size for training. Default is 64.
            gamma (float, optional): Discount factor. Default is 0.99.
            verbose (int, optional): Verbosity level for model operations. Default is 1.
        """
        self.env = env
        self.model_dir = model_dir
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.gamma = gamma
        self.verbose = verbose
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Create a vectorized environment
        self.vec_env = DummyVecEnv([lambda: env])
        
        # Create the PPO agent
        self.model = PPO(
            'MultiInputPolicy', 
            self.vec_env, 
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=gamma,
            verbose=verbose
        )
    
    def train(self, total_timesteps=50000):
        """
        Train the agent.
        
        Args:
            total_timesteps (int, optional): Total timesteps for training. Default is 50000.
        """
        print(f"Training PPO agent for {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps, progress_bar=self.verbose > 0)
    
    def test(self, episodes=10):
        """
        Test the agent.
        
        Args:
            episodes (int, optional): Number of episodes for testing. Default is 10.
            
        Returns:
            float: Mean reward
        """
        print(f"Testing PPO agent for {episodes} episodes...")
        mean_reward, std_reward = evaluate_policy(self.model, self.vec_env, n_eval_episodes=episodes)
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        return mean_reward
    
    def save(self, name):
        """
        Save the model.
        
        Args:
            name (str): Model name
            
        Returns:
            bool: True if model saved successfully, False otherwise
        """
        path = os.path.join(self.model_dir, name)
        try:
            print(f"Saving model to {path}")
            self.model.save(path)
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    def load(self, name):
        """
        Load the model.
        
        Args:
            name (str): Model name
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        path = os.path.join(self.model_dir, name)
        try:
            print(f"Loading model from {path}")
            self.model = PPO.load(path, env=self.vec_env)
            return True
        except (OSError, IOError) as e:
            print(f"Error loading model from {path}: File not found or inaccessible")
            return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def predict(self, observation, deterministic=True):
        """
        Predict action for a given observation.
        
        Args:
            observation (dict): Current observation
            deterministic (bool, optional): Whether to use deterministic action selection. Default is True.
            
        Returns:
            int: Action to take
        """
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action 