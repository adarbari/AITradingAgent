"""
Enhanced Reinforcement Learning Agent for the multi-agent trading system.
Implements advanced RL algorithms with improved learning capabilities.
"""
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import pandas as pd
import os
import random
from collections import deque
import time

# Import specific keras modules
from tensorflow.keras import Model, models, layers, optimizers, callbacks
from tensorflow.keras import backend as K
import tensorflow as tf

# Stable Baselines
from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

# Import base agent class
from src.agent.trading_agent import DQNTradingAgent, PPOTradingAgent


class EnhancedDQNAgent(DQNTradingAgent):
    """
    Enhanced Deep Q-Network Agent with advanced learning capabilities.
    
    This agent extends the base DQNTradingAgent with:
    1. Attention mechanisms for better feature importance weighting
    2. Prioritized experience replay for more efficient learning
    3. Dueling network architecture for better value estimation
    4. Double DQN for reducing overestimation bias
    5. Multi-modal inputs (market data, sentiment, etc.)
    """
    
    def __init__(self, state_size, action_size, model_dir='models', batch_size=64, gamma=0.95,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, learning_rate=0.001,
                 hidden_units=(256, 128), memory_size=100000, use_per=True, use_dueling=True,
                 use_double=True, use_attention=True, n_step=3, verbose=0):
        """
        Initialize the enhanced DQN agent.
        
        Args:
            state_size (tuple): Shape of the state
            action_size (int): Number of possible actions
            model_dir (str): Directory to save models
            batch_size (int): Batch size for training
            gamma (float): Discount factor
            epsilon (float): Exploration rate
            epsilon_decay (float): Decay rate for exploration
            epsilon_min (float): Minimum exploration rate
            learning_rate (float): Learning rate
            hidden_units (tuple): Sizes of hidden layers
            memory_size (int): Size of replay memory
            use_per (bool): Whether to use prioritized experience replay
            use_dueling (bool): Whether to use dueling network architecture
            use_double (bool): Whether to use double DQN
            use_attention (bool): Whether to use attention mechanism
            n_step (int): Number of steps for n-step learning
            verbose (int): Verbosity level for model operations
        """
        # Call the parent constructor first
        super().__init__(
            state_size=state_size,
            action_size=action_size,
            model_dir=model_dir,
            batch_size=batch_size,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            learning_rate=learning_rate,
            hidden_units=hidden_units,
            memory_size=memory_size,
            verbose=verbose
        )
        
        # Enhanced agent parameters
        self.use_per = use_per              # Prioritized Experience Replay
        self.use_dueling = use_dueling      # Dueling Networks
        self.use_double = use_double        # Double DQN
        self.use_attention = use_attention  # Attention Mechanism
        self.n_step = n_step                # N-step learning
        
        # Prioritized experience replay memory
        if self.use_per:
            self.memory = PrioritizedReplayBuffer(memory_size, alpha=0.6)
            self.per_beta = 0.4  # Importance sampling weight
            self.per_beta_increment = 0.001  # Annealing rate for beta
        else:
            # Use standard replay buffer from parent class
            self.memory = deque(maxlen=memory_size)
        
        # N-step learning storage
        self.n_step_buffer = deque(maxlen=n_step)
        
        # Overwrite the model with enhanced architecture
        self.model = self._build_enhanced_model()
        self.target_model = self._build_enhanced_model()
        self.update_target_model()
        
        # Add TensorBoard metrics
        self.loss_history = []
        self.reward_history = []
        self.q_value_history = []

    def _build_enhanced_model(self):
        """
        Build an enhanced neural network model with attention and dueling architecture.
        
        Returns:
            keras.Model: Enhanced neural network model
        """
        # Market data input (technical indicators)
        market_input = layers.Input(shape=self.state_size, name='market_input')
        
        # Process market data with attention if enabled
        if self.use_attention:
            # Reshape for attention layer if needed
            reshaped_market = layers.Reshape((self.state_size[0], self.state_size[1]))(market_input)
            
            # Self-attention layer
            attention = layers.MultiHeadAttention(
                num_heads=4, key_dim=32, dropout=0.1
            )(reshaped_market, reshaped_market)
            
            # Add & Normalize
            norm_attention = layers.LayerNormalization()(attention + reshaped_market)
            
            # Feed-forward network
            ffn = layers.Dense(64, activation='relu')(norm_attention)
            ffn = layers.Dense(self.state_size[1])(ffn)
            
            # Add & Normalize
            norm_output = layers.LayerNormalization()(ffn + norm_attention)
            
            # Pool across sequence dimension
            pooled = layers.GlobalAveragePooling1D()(norm_output)
        else:
            # Simple flattening if attention not used
            pooled = layers.Flatten()(market_input)
        
        # Account info input (balance, shares_held, cost_basis, etc.)
        account_input = layers.Input(shape=(3,), name='account_input')
        
        # Sentiment input (optional)
        sentiment_input = layers.Input(shape=(3,), name='sentiment_input')  # sentiment, volume, volatility
        
        # Combine all inputs
        combined = layers.Concatenate()([pooled, account_input, sentiment_input])
        
        # Shared layers
        x = combined
        for units in self.hidden_units:
            x = layers.Dense(units)(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU(alpha=0.01)(x)
            x = layers.Dropout(0.2)(x)
        
        if self.use_dueling:
            # Dueling network architecture
            # Value stream
            value_stream = layers.Dense(self.hidden_units[-1] // 2, activation='relu')(x)
            value = layers.Dense(1)(value_stream)
            
            # Advantage stream
            advantage_stream = layers.Dense(self.hidden_units[-1] // 2, activation='relu')(x)
            advantage = layers.Dense(self.action_size)(advantage_stream)
            
            # Combine value and advantage
            output = layers.Add()([
                value,
                layers.Subtract()([
                    advantage,
                    layers.Lambda(lambda a: K.mean(a, axis=1, keepdims=True))(advantage)
                ])
            ])
        else:
            # Standard Q-value output
            output = layers.Dense(self.action_size, activation='linear')(x)
        
        # Create model with all inputs
        model = Model(
            inputs=[market_input, account_input, sentiment_input],
            outputs=output
        )
        
        # Compile with Huber loss for stability
        model.compile(
            loss=tf.keras.losses.Huber(),
            optimizer=optimizers.Adam(learning_rate=self.learning_rate)
        )
        
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory with n-step returns.
        
        Args:
            state (dict): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (dict): Next state
            done (bool): Whether the episode is done
        """
        # Store experience in n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # If n-step buffer is not filled yet, return
        if len(self.n_step_buffer) < self.n_step:
            return
        
        # Calculate n-step return
        reward_n, next_state_n, done_n = self._get_n_step_info()
        state_0, action_0, _, _, _ = self.n_step_buffer[0]
        
        if self.use_per:
            # For prioritized replay, we need to calculate initial priority
            # For new experiences, set maximum priority
            self.memory.add(state_0, action_0, reward_n, next_state_n, done_n, max_priority=True)
        else:
            # Standard memory
            self.memory.append((state_0, action_0, reward_n, next_state_n, done_n))
    
    def _get_n_step_info(self):
        """
        Get n-step return information.
        
        Returns:
            Tuple[float, dict, bool]: n-step reward, n-step next state, n-step done
        """
        # Get last experience from n-step buffer
        _, _, _, next_state, done = self.n_step_buffer[-1]
        
        # If last experience is terminal, return its state as n-step next state
        if done:
            return self._get_n_step_return(), next_state, True
            
        # Otherwise, use last state as n-step next state
        return self._get_n_step_return(), next_state, False
        
    def _get_n_step_return(self):
        """
        Calculate n-step return.
        
        Returns:
            float: n-step return
        """
        n_step_return = 0
        
        # Calculate n-step return: R_t + gamma * R_{t+1} + ... + gamma^{n-1} * R_{t+n-1}
        for i in range(len(self.n_step_buffer)):
            n_step_return += self.gamma**i * self.n_step_buffer[i][2]
            
        return n_step_return
    
    def act(self, state, training=True):
        """
        Choose an action based on the current state.
        
        Args:
            state (dict): Current state
            training (bool): Whether the agent is training
            
        Returns:
            int: Action to take
        """
        if training and np.random.rand() <= self.epsilon:
            # Exploration: random action
            return random.randrange(self.action_size)
        
        # Exploitation: use model to predict best action
        market_data = np.expand_dims(state['features'], axis=0)
        account_info = np.expand_dims(state['account_info'], axis=0)
        
        # Add sentiment data if available, otherwise use zeros
        if 'sentiment' in state:
            sentiment_data = np.expand_dims(state['sentiment'], axis=0)
        else:
            sentiment_data = np.zeros((1, 3))
        
        q_values = self.model.predict(
            [market_data, account_info, sentiment_data],
            verbose=0
        )
        
        # Store Q-values for monitoring
        if training:
            self.q_value_history.append(np.max(q_values[0]))
        
        return np.argmax(q_values[0])
    
    def replay(self, batch_size=None):
        """
        Train the agent with experiences from replay memory.
        
        Args:
            batch_size (int): Batch size for training
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        # Check if we have enough samples
        if self.use_per:
            if len(self.memory) < batch_size:
                return
        else:
            if len(self.memory) < batch_size:
                return
        
        # Sample batch from memory
        if self.use_per:
            # Update beta for importance sampling
            self.per_beta = min(1.0, self.per_beta + self.per_beta_increment)
            
            # Get batch with importance sampling weights
            batch, indices, weights = self.memory.sample(batch_size, self.per_beta)
            states, actions, rewards, next_states, dones = zip(*batch)
            weights = np.array(weights)
        else:
            # Random sampling for standard replay buffer
            batch = random.sample(self.memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            weights = np.ones(batch_size)  # Equal weights
        
        # Initialize arrays for features and account info
        market_data = np.zeros((batch_size,) + self.state_size)
        account_info = np.zeros((batch_size, 3))
        sentiment_data = np.zeros((batch_size, 3))
        
        next_market_data = np.zeros((batch_size,) + self.state_size)
        next_account_info = np.zeros((batch_size, 3))
        next_sentiment_data = np.zeros((batch_size, 3))
        
        # Fill arrays
        for i in range(batch_size):
            market_data[i] = states[i]['features']
            account_info[i] = states[i]['account_info']
            
            if 'sentiment' in states[i]:
                sentiment_data[i] = states[i]['sentiment']
            
            next_market_data[i] = next_states[i]['features']
            next_account_info[i] = next_states[i]['account_info']
            
            if 'sentiment' in next_states[i]:
                next_sentiment_data[i] = next_states[i]['sentiment']
        
        # Predict Q-values for current states
        current_q_values = self.model.predict([market_data, account_info, sentiment_data], verbose=0)
        
        if self.use_double:
            # Double DQN: select actions using online network
            next_q_values = self.model.predict([next_market_data, next_account_info, next_sentiment_data], verbose=0)
            next_actions = np.argmax(next_q_values, axis=1)
            
            # Then use target network to estimate Q-values for those actions
            target_q_values = self.target_model.predict([next_market_data, next_account_info, next_sentiment_data], verbose=0)
            target_q = np.array([target_q_values[i, next_actions[i]] for i in range(batch_size)])
        else:
            # Standard DQN
            target_q_values = self.target_model.predict([next_market_data, next_account_info, next_sentiment_data], verbose=0)
            target_q = np.max(target_q_values, axis=1)
        
        # Create targets
        targets = current_q_values.copy()
        for i in range(batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * target_q[i]
        
        # Train the model with importance sampling weights if using PER
        if self.use_per:
            history = self.model.fit(
                [market_data, account_info, sentiment_data],
                targets,
                epochs=1,
                verbose=0,
                batch_size=batch_size,
                sample_weight=weights
            )
            
            # Update priorities in replay buffer
            td_errors = np.abs(current_q_values - targets).sum(axis=1)
            for i, idx in enumerate(indices):
                self.memory.update_priority(idx, td_errors[i])
        else:
            history = self.model.fit(
                [market_data, account_info, sentiment_data],
                targets,
                epochs=1,
                verbose=0,
                batch_size=batch_size
            )
        
        # Store loss for monitoring
        self.loss_history.append(history.history['loss'][0])
        
        # Update exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_model(self):
        """
        Update target model with polyak averaging for stability.
        """
        tau = 0.01  # Small value for soft update
        
        # Get weights from both models
        target_weights = self.target_model.get_weights()
        online_weights = self.model.get_weights()
        
        # Soft update: θ_target = τ*θ_online + (1-τ)*θ_target
        for i in range(len(target_weights)):
            target_weights[i] = tau * online_weights[i] + (1 - tau) * target_weights[i]
        
        # Set the updated weights
        self.target_model.set_weights(target_weights)
        
    def add_reward(self, reward):
        """
        Add reward to history for monitoring.
        
        Args:
            reward (float): Reward received
        """
        self.reward_history.append(reward)
    
    def save_metrics(self, episode):
        """
        Save metrics for TensorBoard visualization.
        
        Args:
            episode (int): Current episode number
        """
        # Create a summary writer for TensorBoard
        log_dir = os.path.join(self.model_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Calculate average metrics
        avg_loss = np.mean(self.loss_history[-100:]) if self.loss_history else 0
        avg_q = np.mean(self.q_value_history[-100:]) if self.q_value_history else 0
        avg_reward = np.mean(self.reward_history[-100:]) if self.reward_history else 0
        
        # Log to file for later visualization
        metrics_log = os.path.join(log_dir, 'metrics.csv')
        metrics_exists = os.path.exists(metrics_log)
        
        with open(metrics_log, 'a') as f:
            if not metrics_exists:
                f.write('episode,loss,q_value,reward,epsilon\n')
            f.write(f'{episode},{avg_loss},{avg_q},{avg_reward},{self.epsilon}\n')
        
        # Clear histories to save memory
        self.loss_history = self.loss_history[-100:]
        self.q_value_history = self.q_value_history[-100:]
        self.reward_history = self.reward_history[-100:]


class EnhancedSACAgent:
    """
    Enhanced Soft Actor-Critic (SAC) agent for trading.
    
    SAC is an off-policy maximum entropy deep reinforcement learning algorithm
    that combines the benefits of DDPG and PPO, suitable for continuous action spaces.
    """
    
    def __init__(self, env, model_dir='models', learning_rate=0.0003, gamma=0.99,
                 tau=0.005, buffer_size=1000000, batch_size=256, train_freq=1, 
                 gradient_steps=1, ent_coef='auto', verbose=1):
        """
        Initialize the SAC agent.
        
        Args:
            env (gym.Env): Trading environment
            model_dir (str): Directory to save models
            learning_rate (float): Learning rate
            gamma (float): Discount factor
            tau (float): Target network update rate
            buffer_size (int): Size of the replay buffer
            batch_size (int): Batch size for training
            train_freq (int): Training frequency
            gradient_steps (int): Number of gradient steps per update
            ent_coef (str or float): Entropy coefficient
            verbose (int): Verbosity level
        """
        self.env = env
        self.model_dir = model_dir
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.ent_coef = ent_coef
        self.verbose = verbose
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Create a vectorized environment
        self.vec_env = DummyVecEnv([lambda: env])
        
        # Create custom policy network with enhanced architecture
        policy_kwargs = {
            "net_arch": {
                "pi": [256, 256],  # Actor network
                "qf": [256, 256]   # Critic network
            },
            "activation_fn": tf.nn.relu,
            "use_sde": True,  # Use state-dependent exploration
            "sde_sample_freq": 4,  # Sample frequency for noise
        }
        
        # Create the SAC agent
        self.model = SAC(
            "MlpPolicy",
            self.vec_env,
            learning_rate=learning_rate,
            gamma=gamma,
            tau=tau,
            buffer_size=buffer_size,
            batch_size=batch_size,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            ent_coef=ent_coef,
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            tensorboard_log=os.path.join(model_dir, 'logs')
        )
        
        # Create a callback for saving the model
        self.callbacks = [SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=os.path.join(model_dir, 'logs'))]
    
    def train(self, total_timesteps=100000, callback=None):
        """
        Train the agent.
        
        Args:
            total_timesteps (int): Total timesteps for training
            callback (Callable): Optional callback function
        """
        print(f"Training SAC agent for {total_timesteps} timesteps...")
        
        callbacks = self.callbacks.copy()
        if callback is not None:
            callbacks.append(callback)
            
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name="sac",
            progress_bar=self.verbose > 0
        )
    
    def predict(self, observation, deterministic=True):
        """
        Predict action for a given observation.
        
        Args:
            observation (dict): Current observation
            deterministic (bool): Whether to use deterministic action selection
            
        Returns:
            int: Action to take
        """
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action
    
    def save(self, filename):
        """
        Save the model.
        
        Args:
            filename (str): Filename for the saved model
        """
        path = os.path.join(self.model_dir, filename)
        self.model.save(path)
        print(f"Model saved to {path}")
        
    def load(self, filename):
        """
        Load the model.
        
        Args:
            filename (str): Filename for the model to load
        """
        path = os.path.join(self.model_dir, filename)
        self.model = SAC.load(path, env=self.vec_env)
        print(f"Model loaded from {path}")
    
    def evaluate(self, n_eval_episodes=10):
        """
        Evaluate the agent.
        
        Args:
            n_eval_episodes (int): Number of episodes for evaluation
            
        Returns:
            Tuple[float, float]: Mean reward, standard deviation
        """
        print(f"Evaluating agent over {n_eval_episodes} episodes...")
        mean_reward, std_reward = evaluate_policy(
            self.model, self.vec_env, n_eval_episodes=n_eval_episodes
        )
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        return mean_reward, std_reward


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving the model when the reward improves.
    """
    
    def __init__(self, check_freq=1000, log_dir=None):
        """
        Initialize the callback.
        
        Args:
            check_freq (int): Frequency of checking for improvements
            log_dir (str): Directory for saving the model
        """
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose=1)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -float('inf')
        
    def _init_callback(self):
        """
        Initialize callback attributes.
        """
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
    
    def _on_step(self):
        """
        Check for saving the model.
        
        Returns:
            bool: Whether to continue training
        """
        if self.n_calls % self.check_freq == 0:
            # Calculate mean episodic reward
            mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
            
            if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Mean reward: {mean_reward:.2f}")
            
            # Save model if improved
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose > 0:
                    print(f"Saving new best model with reward {mean_reward:.2f}")
                self.model.save(self.save_path)
                
        return True


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer for storing experiences.
    Implements the proportional variant from the PER paper.
    """
    
    def __init__(self, max_size, alpha=0.6):
        """
        Initialize the buffer.
        
        Args:
            max_size (int): Maximum size of the buffer
            alpha (float): Alpha parameter for prioritization
        """
        self.max_size = max_size
        self.buffer = []
        self.priorities = np.zeros(max_size, dtype=np.float32)
        self.alpha = alpha
        self.position = 0
        self.size = 0
    
    def __len__(self):
        """
        Get the number of items in the buffer.
        
        Returns:
            int: Number of items in the buffer
        """
        return self.size
    
    def add(self, state, action, reward, next_state, done, max_priority=False):
        """
        Add an experience to the buffer.
        
        Args:
            state (dict): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (dict): Next state
            done (bool): Whether the episode is done
            max_priority (bool): Whether to use maximum priority
        """
        experience = (state, action, reward, next_state, done)
        
        if self.size < self.max_size:
            self.buffer.append(experience)
            self.size += 1
        else:
            self.buffer[self.position] = experience
        
        # Set priority for new experience
        if max_priority:
            self.priorities[self.position] = 1.0 if len(self) <= 1 else max(self.priorities)
        else:
            self.priorities[self.position] = 1.0
        
        # Update position
        self.position = (self.position + 1) % self.max_size
    
    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of experiences from the buffer with prioritization.
        
        Args:
            batch_size (int): Batch size
            beta (float): Beta parameter for importance sampling
            
        Returns:
            Tuple[List, List, List]: Batch of experiences, indices, and weights
        """
        # Calculate sampling probabilities
        if self.size == self.max_size:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.size]
        
        # Convert priorities to probabilities
        probs = priorities ** self.alpha
        probs = probs / np.sum(probs)
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self), batch_size, replace=False, p=probs)
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights = weights / np.max(weights)  # Normalize for stability
        
        # Get experiences
        batch = [self.buffer[idx] for idx in indices]
        
        return batch, indices, weights
    
    def update_priority(self, index, priority):
        """
        Update the priority of an experience.
        
        Args:
            index (int): Index of the experience
            priority (float): New priority
        """
        # Add small constant to ensure non-zero probability
        self.priorities[index] = priority + 1e-5


# Utility for creating custom network architectures
def create_market_state_encoder(input_shape, use_lstm=False, use_attention=True):
    """
    Create a network for encoding market state data.
    
    Args:
        input_shape (tuple): Shape of input data
        use_lstm (bool): Whether to use LSTM layers
        use_attention (bool): Whether to use attention mechanism
        
    Returns:
        Tuple[keras.Input, keras.layers.Layer]: Input tensor and output tensor
    """
    # Input for market data
    market_input = layers.Input(shape=input_shape)
    
    # Initial convolutional layers for feature extraction
    x = layers.Reshape((input_shape[0], input_shape[1]))(market_input)
    x = layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    if use_lstm:
        # LSTM layers for sequential processing
        x = layers.LSTM(128, return_sequences=use_attention)(x)
        x = layers.BatchNormalization()(x)
        
        if not use_attention:
            x = layers.GlobalAveragePooling1D()(x)
    
    if use_attention:
        # Self-attention mechanism
        query_value = layers.Dense(64)(x)
        key_value = layers.Dense(64)(x)
        attention_scores = layers.Dot(axes=(2, 2))([query_value, key_value])
        attention_weights = layers.Softmax()(attention_scores)
        context_vector = layers.Dot(axes=(2, 1))([attention_weights, x])
        
        # Global pooling after attention
        x = layers.GlobalAveragePooling1D()(context_vector)
    
    return market_input, x 