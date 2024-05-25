# AdaptiveBioinkMixer: Real-time control of multi-material bioink mixing using deep reinforcement learning
# Based on the paper "Real-time control of multi-material bioink mixing using deep reinforcement learning" (2024)

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.optimizers import Adam
import gym

### Environment Setup ###

# Define the multi-material bioink mixing environment
class BioinkMixingEnv(gym.Env):
    def __init__(self, num_materials, target_ratios):
        super(BioinkMixingEnv, self).__init__()
        self.num_materials = num_materials
        self.target_ratios = target_ratios
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(num_materials,))
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(num_materials,))
        self.current_ratios = np.zeros(num_materials)
        self.time_step = 0
        
    def reset(self):
        self.current_ratios = np.zeros(self.num_materials)
        self.time_step = 0
        return self.get_state()
    
    def step(self, action):
        # Update mixing ratios based on action
        self.current_ratios += action
        self.current_ratios = np.clip(self.current_ratios, 0, 1)
        self.current_ratios /= np.sum(self.current_ratios)
        
        # Calculate reward based on deviation from target ratios
        reward = -np.sum(np.abs(self.current_ratios - self.target_ratios))
        
        # Increment time step
        self.time_step += 1
        
        # Check if episode is done
        done = (self.time_step >= 100)
        
        return self.get_state(), reward, done, {}
    
    def get_state(self):
        # State includes current mixing ratios and material properties
        state = np.concatenate((self.current_ratios, self.get_material_properties()))
        return state
    
    def get_material_properties(self):
        # Simulate material properties (e.g., viscosity, surface tension)
        # This is a placeholder function and should be replaced with actual material property measurements
        properties = []
        for _ in range(self.num_materials):
            viscosity = np.random.normal(loc=1.0, scale=0.1)
            surface_tension = np.random.normal(loc=0.5, scale=0.05)
            properties.extend([viscosity, surface_tension])
        return np.array(properties)

# Create the multi-material bioink mixing environment
num_materials = 3
target_ratios = [0.6, 0.3, 0.1]
env = BioinkMixingEnv(num_materials, target_ratios)

### PPO Algorithm ###

# Define the policy network
def create_policy_network(state_shape, action_dim):
    state_input = Input(shape=state_shape)
    x = Dense(128, activation='relu')(state_input)
    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(128)(x)
    x = Dense(256, activation='relu')(x)
    output = Dense(action_dim, activation='softmax')(x)
    model = tf.keras.Model(inputs=state_input, outputs=output)
    return model

# Define the value network
def create_value_network(state_shape):
    state_input = Input(shape=state_shape)
    x = Dense(128, activation='relu')(state_input)
    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(128)(x)
    x = Dense(256, activation='relu')(x)
    output = Dense(1)(x)
    model = tf.keras.Model(inputs=state_input, outputs=output)
    return model

# Create the policy and value networks
state_shape = (num_materials * 3,)  # State includes mixing ratios and material properties
action_dim = num_materials
policy_network = create_policy_network(state_shape, action_dim)
value_network = create_value_network(state_shape)

# Define the PPO loss functions
def ppo_loss(advantages, old_log_probs, log_probs, clip_ratio=0.2):
    ratio = tf.exp(log_probs - old_log_probs)
    policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, 
                                             tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages))
    return policy_loss

def value_loss(returns, values):
    return tf.reduce_mean(tf.square(returns - values))

# Define the model optimizer
optimizer = Adam(learning_rate=0.001)

### Training Loop ###

# Hyperparameters
num_episodes = 1000
max_steps_per_episode = 100
gamma = 0.99
lam = 0.95
batch_size = 64
epochs = 10

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    
    states = []
    actions = []
    rewards = []
    values = []
    log_probs = []
    
    for step in range(max_steps_per_episode):
        # Select action based on policy network
        state_input = np.reshape(state, [1, state_shape[0]])
        action_probs = policy_network.predict(state_input)[0]
        action = np.random.choice(action_dim, p=action_probs)
        
        # Take action and observe next state and reward
        next_state, reward, done, _ = env.step(np.eye(action_dim)[action])
        episode_reward += reward
        
        # Store transition
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        values.append(value_network.predict(state_input)[0][0])
        log_probs.append(np.log(action_probs[action]))
        
        # Update state
        state = next_state
        
        if done:
            break
    
    # Calculate advantages and returns
    advantages = []
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * (0 if i == len(rewards) - 1 else values[i + 1]) - values[i]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        returns.insert(0, advantages[0] + values[i])
    
    # Normalize advantages
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
    
    # Train policy and value networks
    for _ in range(epochs):
        # Sample a batch of transitions
        indices = np.random.randint(0, len(states), size=batch_size)
        batch_states = np.array([states[i] for i in indices])
        batch_actions = np.array([actions[i] for i in indices])
        batch_advantages = np.array([advantages[i] for i in indices])
        batch_returns = np.array([returns[i] for i in indices])
        batch_log_probs = np.array([log_probs[i] for i in indices])
        
        # Update policy network
        with tf.GradientTape() as tape:
            new_log_probs = tf.math.log(policy_network(batch_states)[np.arange(batch_size), batch_actions])
            policy_loss_value = ppo_loss(batch_advantages, batch_log_probs, new_log_probs)
        policy_gradients = tape.gradient(policy_loss_value, policy_network.trainable_variables)
        optimizer.apply_gradients(zip(policy_gradients, policy_network.trainable_variables))
        
        # Update value network
        with tf.GradientTape() as tape:
            value_loss_value = value_loss(batch_returns, value_network(batch_states))
        value_gradients = tape.gradient(value_loss_value, value_network.trainable_variables)
        optimizer.apply_gradients(zip(value_gradients, value_network.trainable_variables))
    
    # Print episode summary
    print(f"Episode {episode+1}: Reward = {episode_reward:.2f}")

# Save the trained models
policy_network.save("adaptive_bioink_mixer_policy.h5")
value_network.save("adaptive_bioink_mixer_value.h5")
