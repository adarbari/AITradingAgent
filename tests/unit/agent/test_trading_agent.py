"""
Tests for the trading agent module.
"""
import os
import pytest
import numpy as np
import tensorflow as tf
import gymnasium
from unittest.mock import patch, MagicMock, PropertyMock
from src.agent.trading_agent import DQNTradingAgent, PPOTradingAgent


class TestDQNTradingAgent:
    """Test cases for the DQNTradingAgent class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Disable GPU for tests
        tf.config.set_visible_devices([], 'GPU')
        
        # Create temp directory for models
        os.makedirs('tests/test_models', exist_ok=True)
        
        # Define state size for tests
        self.state_size = (10, 5)  # (window_size, num_features)
        self.action_size = 3  # hold, buy, sell
        
        # Create agent
        self.agent = DQNTradingAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            model_dir='tests/test_models',
            batch_size=4,  # Small batch size for testing
            epsilon=1.0,
            epsilon_decay=0.99,
            epsilon_min=0.01
        )
        
        # Create sample data for testing
        self.sample_features = np.random.normal(0, 1, self.state_size).astype(np.float32)
        self.sample_account_info = np.array([10000, 0, 10000], dtype=np.float32)
        self.sample_state = {
            'features': self.sample_features,
            'account_info': self.sample_account_info
        }
        
        self.sample_next_state = {
            'features': np.random.normal(0, 1, self.state_size).astype(np.float32),
            'account_info': np.array([9500, 5, 10100], dtype=np.float32)
        }
    
    def teardown_method(self):
        """Clean up after tests"""
        # Delete test model files
        for file in os.listdir('tests/test_models'):
            if file.endswith('.h5') or file.endswith('.keras') or file.endswith('.weights.h5'):
                os.remove(os.path.join('tests/test_models', file))
    
    def test_initialization(self):
        """Test agent initialization"""
        assert self.agent.state_size == self.state_size
        assert self.agent.action_size == self.action_size
        assert self.agent.model_dir == 'tests/test_models'
        assert self.agent.batch_size == 4
        assert self.agent.epsilon == 1.0
        assert len(self.agent.memory) == 0
        assert isinstance(self.agent.model, tf.keras.Model)
        assert isinstance(self.agent.target_model, tf.keras.Model)
    
    def test_build_model(self):
        """Test model architecture"""
        model = self.agent._build_model()
        
        # Check model inputs
        assert len(model.inputs) == 2
        
        # Get the expected shapes from the model inputs
        input_shape_0 = model.inputs[0].shape
        input_shape_1 = model.inputs[1].shape
        
        # Check dimensions match (excluding None/batch dimension)
        assert input_shape_0[1:] == (self.state_size[0], self.state_size[1])
        assert input_shape_1[1:] == (3,)
        
        # Check model outputs
        assert model.outputs[0].shape[1:] == (self.action_size,)
    
    def test_update_target_model(self):
        """Test updating target model weights"""
        # Modify the main model weights
        original_weights = [np.array(w) for w in self.agent.target_model.get_weights()]
        
        # Train the model slightly to change weights
        self.agent.model.fit(
            [np.expand_dims(self.sample_features, 0), 
             np.expand_dims(self.sample_account_info, 0)],
            np.random.random((1, self.action_size)),
            epochs=1,
            verbose=0
        )
        
        # Update target model
        self.agent.update_target_model()
        
        # Check weights are now the same
        for w1, w2 in zip(self.agent.model.get_weights(), self.agent.target_model.get_weights()):
            assert np.array_equal(w1, w2)
    
    def test_act_exploration(self):
        """Test act method in exploration mode"""
        with patch('numpy.random.rand', return_value=0.1):  # Ensure exploration
            action = self.agent.act(self.sample_state, training=True)
            assert 0 <= action < self.action_size
    
    def test_act_exploitation(self):
        """Test act method in exploitation mode"""
        # Test with training=False to force exploitation
        with patch.object(self.agent.model, 'predict', return_value=np.array([[0.1, 0.9, 0.3]])):
            action = self.agent.act(self.sample_state, training=False)
            assert action == 1  # Should choose the action with highest Q-value
    
    def test_remember(self):
        """Test remember method to store experiences"""
        # Add an experience to memory
        self.agent.remember(
            state=self.sample_state,
            action=1,  # buy
            reward=0.1,
            next_state=self.sample_next_state,
            done=False
        )
        
        # Check memory contains the experience
        assert len(self.agent.memory) == 1
        
        # Check content of the memory
        state, action, reward, next_state, done = self.agent.memory[0]
        assert state is self.sample_state
        assert action == 1
        assert reward == 0.1
        assert next_state is self.sample_next_state
        assert done is False
    
    def test_replay_insufficient_memory(self):
        """Test replay method with insufficient memory"""
        # Add less experiences than batch size
        self.agent.remember(
            state=self.sample_state,
            action=1,
            reward=0.1,
            next_state=self.sample_next_state,
            done=False
        )
        
        # Call replay
        self.agent.replay()
        
        # Epsilon should remain unchanged
        assert self.agent.epsilon == 1.0
    
    def test_replay(self):
        """Test replay method for learning"""
        # Add enough experiences to memory
        for _ in range(10):
            self.agent.remember(
                state=self.sample_state,
                action=1,
                reward=0.1,
                next_state=self.sample_next_state,
                done=False
            )
        
        # Initial epsilon
        initial_epsilon = self.agent.epsilon
        
        # Call replay
        self.agent.replay(batch_size=4)
        
        # Epsilon should decrease
        assert self.agent.epsilon < initial_epsilon
    
    def test_save_load(self):
        """Test saving and loading model weights"""
        # Save model
        model_name = 'test_dqn_model.weights.h5'
        success = self.agent.save(model_name)
        
        # Check save was successful and file exists
        assert success
        expected_path = os.path.join('tests/test_models', model_name)
        assert os.path.exists(expected_path)
        
        # Create a new agent
        new_agent = DQNTradingAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            model_dir='tests/test_models'
        )
        
        # Load the model
        success = new_agent.load(model_name)
        assert success
        
        # Check weights are the same
        for w1, w2 in zip(self.agent.model.get_weights(), new_agent.model.get_weights()):
            assert np.array_equal(w1, w2)
    
    def test_act_with_min_epsilon(self):
        """Test act method with minimum epsilon value"""
        # Set epsilon to minimum value to ensure exploitation
        self.agent.epsilon = self.agent.epsilon_min
        
        # Mock the model's predict method to return a known value
        with patch.object(self.agent.model, 'predict', return_value=np.array([[0.1, 0.2, 0.7]])):
            action = self.agent.act(self.sample_state, training=True)
            # Should choose action 2 (highest Q-value)
            assert action == 2
    
    def test_act_with_nan_values(self):
        """Test act method with NaN values in state"""
        # Create a state with NaN values
        invalid_state = {
            'features': np.full(self.state_size, np.nan).astype(np.float32),
            'account_info': np.array([10000, 0, 10000], dtype=np.float32)
        }
        
        # When model returns NaN values, should default to action 0 (hold)
        with patch.object(self.agent.model, 'predict', return_value=np.array([[np.nan, np.nan, np.nan]])):
            action = self.agent.act(invalid_state, training=False)
            # Should default to action 0 when all values are NaN
            assert action == 0
    
    def test_replay_with_custom_batch_size(self):
        """Test replay method with custom batch size"""
        # Add more experiences than default batch size
        for _ in range(20):
            self.agent.remember(
                state=self.sample_state,
                action=1,
                reward=0.1,
                next_state=self.sample_next_state,
                done=False
            )
        
        # Custom batch size
        custom_batch_size = 10
        
        # Verify initial epsilon
        initial_epsilon = self.agent.epsilon
        
        # Call replay with custom batch size
        self.agent.replay(batch_size=custom_batch_size)
        
        # Epsilon should decrease
        assert self.agent.epsilon < initial_epsilon
    
    def test_memory_capacity(self):
        """Test memory capacity (deque maxlen)"""
        # Fill the memory beyond capacity
        capacity = 10000  # Default capacity
        
        # Add capacity + 100 experiences
        for i in range(capacity + 100):
            self.agent.remember(
                state=self.sample_state,
                action=1,
                reward=float(i),  # Unique reward to identify experiences
                next_state=self.sample_next_state,
                done=False
            )
        
        # Memory should only hold 'capacity' items
        assert len(self.agent.memory) == capacity
        
        # Check that the oldest items were removed (rewards should start after 100)
        oldest_reward = self.agent.memory[0][2]  # Get reward from first experience
        assert oldest_reward >= 100.0


class TestPPOTradingAgent:
    """Test cases for the PPOTradingAgent class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Create temp directory for models
        os.makedirs('tests/test_models', exist_ok=True)
        
        # Mock the PPO class from stable_baselines3
        self.mock_ppo = MagicMock()
        # Mock the PPO.load method as a static method
        self.mock_ppo.load = MagicMock(return_value=MagicMock())
        
        # Create a mock for DummyVecEnv
        self.mock_dummy_vec_env = MagicMock()
        self.mock_dummy_vec_env.return_value = MagicMock()
        
        # Create a mock for evaluate_policy
        self.mock_evaluate_policy = MagicMock(return_value=(100.0, 10.0))  # (mean_reward, std_reward)
        
        # Create a mock environment that's also a gymnasium.Env
        self.mock_env = MagicMock()
        # Make the mock_env appear to be a Gymnasium.Env to pass type checking
        self.mock_env.__class__ = gymnasium.Env
        
        # Use patch to replace dependencies
        self.patches = [
            patch('src.agent.trading_agent.PPO', self.mock_ppo),
            patch('src.agent.trading_agent.DummyVecEnv', self.mock_dummy_vec_env),
            patch('src.agent.trading_agent.evaluate_policy', self.mock_evaluate_policy)
        ]
        
        for p in self.patches:
            p.start()
            
        # Create the agent
        self.agent = PPOTradingAgent(
            env=self.mock_env,
            model_dir='tests/test_models',
            learning_rate=0.0001,
            n_steps=64
        )
    
    def teardown_method(self):
        """Clean up after tests"""
        # Stop all patches
        for p in self.patches:
            p.stop()
            
        # Delete test model files
        for file in os.listdir('tests/test_models'):
            if file.endswith('.zip'):
                os.remove(os.path.join('tests/test_models', file))
    
    def test_initialization(self):
        """Test agent initialization"""
        assert self.agent.env is self.mock_env
        assert self.agent.model_dir == 'tests/test_models'
        assert self.agent.learning_rate == 0.0001
        assert self.agent.n_steps == 64
        
        # Check that DummyVecEnv was called
        self.mock_dummy_vec_env.assert_called_once()
        
        # Check that PPO constructor was called with correct parameters
        self.mock_ppo.assert_called_once()
        _, kwargs = self.mock_ppo.call_args
        assert kwargs['learning_rate'] == 0.0001
        assert kwargs['n_steps'] == 64
    
    def test_train(self):
        """Test train method"""
        # Setup
        total_timesteps = 1000
        
        # Call train
        self.agent.train(total_timesteps=total_timesteps)
        
        # Check model.learn was called with correct parameters
        self.agent.model.learn.assert_called_once_with(total_timesteps=total_timesteps, progress_bar=True)
    
    def test_test(self):
        """Test test method"""
        # Setup
        episodes = 10
        
        # Mock the evaluate_policy function to avoid issues with env type checking
        with patch('src.agent.trading_agent.evaluate_policy', return_value=(100.0, 10.0)):
            # Call test
            mean_reward = self.agent.test(episodes=episodes)
            
            # Check return value
            assert mean_reward == 100.0
    
    def test_save(self):
        """Test save method"""
        # Call save with a patch to avoid path issues
        model_path = 'test_ppo_model'  # Use relative path to avoid duplication
        
        # Mock os.path.join to return the exact path we want
        with patch('os.path.join', return_value=model_path):
            self.agent.save(model_path)
        
        # Check model.save was called with correct path
        self.agent.model.save.assert_called_once_with(model_path)
    
    def test_load(self):
        """Test load method"""
        # Mock os.path.join and PPO.load to avoid file system interactions
        model_path = 'test_ppo_model'
        
        with patch('os.path.join', return_value=model_path), \
             patch('stable_baselines3.PPO.load', return_value=MagicMock()) as mock_load:
            # Call load
            self.agent.load(model_path)
        
            # Check PPO.load was called
            self.mock_ppo.load.assert_called_once_with(model_path, env=self.agent.vec_env)
    
    def test_predict(self):
        """Test predict method"""
        # Setup
        state = {'observation': np.array([1, 2, 3])}
        self.agent.model.predict.return_value = (1, None)  # (action, _)
        
        # Call predict
        action = self.agent.predict(state)
        
        # Check model.predict was called with correct parameters
        self.agent.model.predict.assert_called_once_with(state, deterministic=True)
        
        # Check return value
        assert action == 1 