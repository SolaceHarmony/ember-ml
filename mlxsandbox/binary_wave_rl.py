"""
Binary Wave Reinforcement Learning.

This script demonstrates how to use binary wave neural networks
for reinforcement learning tasks.
"""

import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union, Dict, Any
import random
from collections import deque
from ember_ml.nn.tensor.types import TensorLike
from mlx_binary_wave import MLXBinaryWave

class BinaryWaveQLearningAgent:
    """
    Binary Wave Q-Learning Agent.
    
    This agent uses a binary wave neural network to approximate the Q-function
    for reinforcement learning tasks.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 32):
        """
        Initialize the agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of the hidden layer
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Initialize Q-network
        self.q_network = BinaryWaveQNetwork(state_dim, action_dim, hidden_dim)
        
        # Initialize target Q-network
        self.target_q_network = BinaryWaveQNetwork(state_dim, action_dim, hidden_dim)
        self.update_target_network()
        
        # Initialize replay buffer
        self.replay_buffer = deque(maxlen=1000)
        
        # Initialize hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.target_update_freq = 10
        
        # Initialize step counter
        self.steps = 0
    
    def select_action(self, state: TensorLike) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            # Convert state to binary
            state_binary = self._binarize_state(state)
            
            # Get Q-values
            q_values = self.q_network.forward(state_binary)
            
            # Select action with highest Q-value
            return int(mx.argmax(q_values))
    
    def store_transition(self, state: TensorLike, action: int, reward: float, next_state: TensorLike, done: bool):
        """
        Store transition in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train(self):
        """Train the agent using experience replay."""
        # Increment step counter
        self.steps += 1
        
        # Check if replay buffer has enough samples
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*batch)
        from ember_ml.nn import tensor
        # Convert to arrays
        states = tensor.convert_to_tensor(states)
        actions = tensor.convert_to_tensor(actions)
        rewards = tensor.convert_to_tensor(rewards)
        next_states = tensor.convert_to_tensor(next_states)
        dones = tensor.convert_to_tensor(dones)
        
        # Convert states to binary
        states_binary = self._binarize_states(states)
        next_states_binary = self._binarize_states(next_states)
        
        # Get current Q-values
        current_q_values = self.q_network.forward(states_binary)
        
        # Get next Q-values from target network
        next_q_values = self.target_q_network.forward(next_states_binary)
        
        # Compute target Q-values
        max_next_q_values = mx.max(next_q_values, axis=1)
        target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)
        
        # Update Q-values for taken actions
        for i in range(self.batch_size):
            current_q_values[i, actions[i]] = target_q_values[i]
        
        # Update Q-network
        self.q_network.update(states_binary, current_q_values, self.learning_rate)
        
        # Update target network periodically
        if self.steps % self.target_update_freq == 0:
            self.update_target_network()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        """Update target network with current Q-network weights."""
        self.target_q_network.w1 = mx.array(self.q_network.w1)
        self.target_q_network.w2 = mx.array(self.q_network.w2)
        self.target_q_network.b1 = mx.array(self.q_network.b1)
        self.target_q_network.b2 = mx.array(self.q_network.b2)
    
    def _binarize_state(self, state: TensorLike) -> mx.array:
        """
        Convert state to binary representation.
        
        Args:
            state: State to binarize
            
        Returns:
            Binary representation of state
        """
        # Scale state to [0, 1]
        scaled_state = (state - ops.stats.min(state)) / (ops.stats.max(state) - ops.stats.min(state) + 1e-8)
        
        # Convert to binary
        binary_state = mx.array(scaled_state >= 0.5, dtype=mx.int32)
        
        return binary_state
    
    def _binarize_states(self, states: TensorLike) -> mx.array:
        """
        Convert multiple states to binary representation.
        
        Args:
            states: States to binarize
            
        Returns:
            Binary representation of states
        """
        # Scale states to [0, 1]
        scaled_states = (states - ops.stats.min(states)) / (ops.stats.max(states) - ops.stats.min(states) + 1e-8)
        
        # Convert to binary
        binary_states = mx.array(scaled_states >= 0.5, dtype=mx.int32)
        
        return binary_states

class BinaryWaveQNetwork:
    """
    Binary Wave Q-Network.
    
    This network uses binary wave operations to approximate the Q-function.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        """
        Initialize the network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of the hidden layer
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Initialize weights
        self.w1 = mx.array(mx.random.uniform(shape=(state_dim, hidden_dim)) < 0.5, dtype=mx.int32)
        self.w2 = mx.array(mx.random.uniform(shape=(hidden_dim, action_dim)) < 0.5, dtype=mx.int32)
        
        # Initialize biases
        self.b1 = mx.array(mx.random.uniform(shape=(hidden_dim,)) < 0.5, dtype=mx.int32)
        self.b2 = mx.array(mx.random.uniform(shape=(action_dim,)) < 0.5, dtype=mx.int32)
    
    def forward(self, x: mx.array) -> mx.array:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (batch_size, state_dim)
            
        Returns:
            Q-values (batch_size, action_dim)
        """
        # First layer
        h1 = self._binary_layer(x, self.w1, self.b1)
        
        # Second layer
        q_values = self._binary_layer(h1, self.w2, self.b2)
        
        # Convert to float for Q-values
        q_values_float = mx.array(q_values, dtype=mx.float32)
        
        return q_values_float
    
    def _binary_layer(self, x: mx.array, w: mx.array, b: mx.array) -> mx.array:
        """
        Binary layer computation.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            w: Weight tensor (input_dim, output_dim)
            b: Bias tensor (output_dim,)
            
        Returns:
            Output tensor (batch_size, output_dim)
        """
        batch_size = x.shape[0]
        output_dim = w.shape[1]
        
        # Initialize output
        output = mx.zeros((batch_size, output_dim), dtype=mx.int32)
        
        # For each output neuron
        for j in range(output_dim):
            # For each input feature
            for i in range(x.shape[1]):
                # Get input and weight
                x_i = x[:, i]
                w_ij = w[i, j]
                
                # Compute contribution: x_i AND w_ij
                contribution = MLXBinaryWave.bitwise_and(x_i, mx.array([w_ij] * batch_size))
                
                # Add contribution to output
                output = mx.array([
                    *output[:, :j],
                    MLXBinaryWave.bitwise_xor(output[:, j], contribution),
                    *output[:, j+1:]
                ])
            
            # Add bias
            output = mx.array([
                *output[:, :j],
                MLXBinaryWave.bitwise_xor(output[:, j], mx.array([b[j]] * batch_size)),
                *output[:, j+1:]
            ])
        
        return output
    
    def update(self, states: mx.array, target_q_values: mx.array, learning_rate: float):
        """
        Update network weights using binary gradient descent.
        
        Args:
            states: Input states (batch_size, state_dim)
            target_q_values: Target Q-values (batch_size, action_dim)
            learning_rate: Learning rate
        """
        # Forward pass
        h1 = self._binary_layer(states, self.w1, self.b1)
        q_values = self._binary_layer(h1, self.w2, self.b2)
        
        # Convert to float
        q_values_float = mx.array(q_values, dtype=mx.float32)
        
        # Compute error
        error = target_q_values - q_values_float
        
        # Binary gradient descent
        # For simplicity, we'll just flip a random subset of weights
        # proportional to the error magnitude
        
        # Compute error magnitude
        error_magnitude = mx.sum(mx.abs(error))
        
        # Number of weights to flip
        num_flips = int(learning_rate * error_magnitude)
        
        # Randomly flip weights
        for _ in range(num_flips):
            # Randomly select layer (1 or 2)
            layer = np.random.randint(1, 3)
            
            if layer == 1:
                # Flip a weight in layer 1
                i = np.random.randint(0, self.state_dim)
                j = np.random.randint(0, self.hidden_dim)
                self.w1 = mx.array([
                    *self.w1[:i],
                    [*self.w1[i, :j], 1 - self.w1[i, j], *self.w1[i, j+1:]],
                    *self.w1[i+1:]
                ])
            else:
                # Flip a weight in layer 2
                i = np.random.randint(0, self.hidden_dim)
                j = np.random.randint(0, self.action_dim)
                self.w2 = mx.array([
                    *self.w2[:i],
                    [*self.w2[i, :j], 1 - self.w2[i, j], *self.w2[i, j+1:]],
                    *self.w2[i+1:]
                ])

class CartPoleEnv:
    """
    Simple CartPole environment for demonstration.
    
    This is a simplified version of the CartPole environment from OpenAI Gym.
    """
    
    def __init__(self):
        """Initialize the environment."""
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02
        
        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * ops.pi / 360
        self.x_threshold = 2.4
        
        # State: [x, x_dot, theta, theta_dot]
        self.state = None
        
        # Reset the environment
        self.reset()
    
    def reset(self) -> Any:
        """
        Reset the environment.
        
        Returns:
            Initial state
        """
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        from ember_ml.nn import tensor
        return tensor.convert_to_tensor(self.state)
    
    def step(self, action: int) -> Tuple[TensorLike, float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (0 or 1)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        x, x_dot, theta, theta_dot = self.state
        
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = ops.sin(theta)
        
        # Calculate acceleration
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        # Update state
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        from ember_ml.nn import tensor
        self.state = tensor.convert_to_tensor([x, x_dot, theta, theta_dot])
        
        # Check if done
        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        
        # Calculate reward
        reward = 1.0 if not done else 0.0
        
        return tensor.convert_to_tensor(self.state), reward, done, {}

def train_agent(env: CartPoleEnv, agent: BinaryWaveQLearningAgent, num_episodes: int = 100) -> List[float]:
    """
    Train the agent on the environment.
    
    Args:
        env: Environment to train on
        agent: Agent to train
        num_episodes: Number of episodes to train for
        
    Returns:
        List of episode rewards
    """
    episode_rewards = []
    
    for episode in range(num_episodes):
        # Reset environment
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train agent
            agent.train()
            
            # Update state and reward
            state = next_state
            episode_reward += reward
        
        # Store episode reward
        episode_rewards.append(episode_reward)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}, Epsilon: {agent.epsilon:.4f}")
    
    return episode_rewards

def plot_rewards(rewards: List[float]):
    """
    Plot episode rewards.
    
    Args:
        rewards: List of episode rewards
    """
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot rewards
    plt.plot(rewards)
    
    # Add labels and title
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Rewards')
    
    # Add moving average
    window_size = 10
    from ember_ml.nn import tensor
    moving_avg = np.convolve(rewards, tensor.ones(window_size) / window_size, mode='valid')
    plt.plot(range(window_size - 1, len(rewards)), moving_avg, 'r-')
    
    # Add legend
    plt.legend(['Rewards', f'{window_size}-Episode Moving Average'])
    
    # Save figure
    plt.savefig('rl_rewards.png')
    
    print("Episode rewards plot saved to 'rl_rewards.png'")

def main():
    """Run reinforcement learning with binary waves."""
    print("Binary Wave Reinforcement Learning")
    print("=================================\n")
    
    # Create environment
    print("Creating CartPole environment...")
    env = CartPoleEnv()
    
    # Create agent
    print("Creating binary wave Q-learning agent...")
    agent = BinaryWaveQLearningAgent(state_dim=4, action_dim=2, hidden_dim=16)
    
    # Train agent
    print("\nTraining agent...")
    rewards = train_agent(env, agent, num_episodes=100)
    
    # Plot rewards
    print("\nPlotting rewards...")
    plot_rewards(rewards)
    
    print("\nDone!")

if __name__ == "__main__":
    main()