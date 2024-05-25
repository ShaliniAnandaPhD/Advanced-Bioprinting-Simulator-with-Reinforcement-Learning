# AdaptiveBioprintRL: Adaptive bioprinting with real-time feedback using deep reinforcement learning
# Based on the paper "Adaptive bioprinting with real-time feedback using deep reinforcement learning" (2024)

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.optimizers import Adam
import random

### Environment Setup ###

# Define the adaptive bioprinting environment
class BioprintingEnv:
    def __init__(self, sensor_data):
        self.sensor_data = sensor_data
        self.current_step = 0
        self.num_steps = len(sensor_data)
        self.state_dim = sensor_data.shape[1]
        self.action_dim = 3  # Adjust temperature, pressure, and print speed
        
    def reset(self):
        self.current_step = 0
        return self.sensor_data[self.current_step]
    
    def step(self, action):
        # Apply the action to the bioprinting process
        temperature_adjustment, pressure_adjustment, speed_adjustment = action
        # Simulate the effect of the action on the print quality
        print_quality = self.simulate_print_quality(temperature_adjustment, pressure_adjustment, speed_adjustment)
        
        # Calculate reward based on print quality
        reward = self.calculate_reward(print_quality)
        
        # Move to the next step
        self.current_step += 1
        
        # Check if the episode is done
        done = (self.current_step == self.num_steps - 1)
        
        # Get the next state
        next_state = self.sensor_data[self.current_step]
        
        return next_state, reward, done
    
    def simulate_print_quality(self, temperature_adjustment, pressure_adjustment, speed_adjustment):
        # Simulate the effect of the action on the print quality
        # This is a placeholder function and should be replaced with a real simulation or prediction model
        print_quality = np.random.random()  # Random value between 0 and 1
        return print_quality
    
    def calculate_reward(self, print_quality):
        # Calculate the reward based on the print quality
        # This is a placeholder function and should be replaced with a appropriate reward function
        reward = print_quality
        return reward

# Load the sensor data
sensor_data = np.loadtxt('sensor_data.csv', delimiter=',')

# Create the bioprinting environment
env = BioprintingEnv(sensor_data)

### DDPG Algorithm ###

# Define the actor network
def create_actor_network(state_dim, action_dim):
    inputs = Input(shape=(state_dim,))
    x = Dense(256, activation='relu')(inputs)
    x = Dense(256, activation='relu')(x)
    x = Dense(action_dim, activation='tanh')(x)
    outputs = x * np.array([10.0, 10.0, 10.0])  # Scale the actions
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Define the critic network
def create_critic_network(state_dim, action_dim):
    state_inputs = Input(shape=(state_dim,))
    state_x = Dense(256, activation='relu')(state_inputs)
    state_x = Dense(256, activation='relu')(state_x)
    
    action_inputs = Input(shape=(action_dim,))
    action_x = Dense(256, activation='relu')(action_inputs)
    
    combined = tf.keras.layers.concatenate([state_x, action_x])
    x = Dense(256, activation='relu')(combined)
    x = Dense(1)(x)
    
    model = Model(inputs=[state_inputs, action_inputs], outputs=x)
    return model

# Create the actor and critic networks
state_dim = env.state_dim
action_dim = env.action_dim
actor_model = create_actor_network(state_dim, action_dim)
critic_model = create_critic_network(state_dim, action_dim)

# Define the target networks
target_actor = create_actor_network(state_dim, action_dim)
target_critic = create_critic_network(state_dim, action_dim)

# Initialize the target networks with the same weights as the original networks
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Define the update function for the target networks
def update_target_networks(target_model, model, tau):
    target_weights = target_model.get_weights()
    original_weights = model.get_weights()
    for i in range(len(target_weights)):
        target_weights[i] = tau * original_weights[i] + (1 - tau) * target_weights[i]
    target_model.set_weights(target_weights)

# Define the replay buffer
class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim):
        self.max_size = max_size
        self.buffer = []
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        return states, actions, rewards, next_states, dones
    
    def size(self):
        return len(self.buffer)

# Create the replay buffer
replay_buffer = ReplayBuffer(max_size=10000, state_dim=state_dim, action_dim=action_dim)

# Define the training function
def train_ddpg(actor_model, critic_model, target_actor, target_critic, replay_buffer, batch_size, gamma, tau):
    if replay_buffer.size() < batch_size:
        return
    
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    
    # Predict the next actions using the target actor network
    next_actions = target_actor.predict(next_states)
    
    # Predict the Q-values using the target critic network
    target_q_values = target_critic.predict([next_states, next_actions])
    
    # Calculate the target Q-values
    targets = rewards + (1 - dones) * gamma * target_q_values
    
    # Train the critic network
    critic_loss = critic_model.train_on_batch([states, actions], targets)
    
    # Train the actor network
    with tf.GradientTape() as tape:
        actions = actor_model(states)
        q_values = critic_model([states, actions])
        actor_loss = -tf.reduce_mean(q_values)
    actor_gradients = tape.gradient(actor_loss, actor_model.trainable_variables)
    actor_optimizer.apply_gradients(zip(actor_gradients, actor_model.trainable_variables))
    
    # Update the target networks
    update_target_networks(target_actor, actor_model, tau)
    update_target_networks(target_critic, critic_model, tau)
    
    return critic_loss, actor_loss

# Define the hyperparameters
batch_size = 64
gamma = 0.99
tau = 0.005
actor_lr = 0.001
critic_lr = 0.002
num_episodes = 1000

# Create the optimizers
actor_optimizer = Adam(learning_rate=actor_lr)
critic_optimizer = Adam(learning_rate=critic_lr)

# Compile the critic model
critic_model.compile(optimizer=critic_optimizer, loss='mse')

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        # Select an action using the actor network
        action = actor_model.predict(np.expand_dims(state, axis=0))[0]
        
        # Add noise to the action for exploration
        action += np.random.normal(0, 0.1, size=action_dim)
        
        # Take a step in the environment
        next_state, reward, done = env.step(action)
        
        # Store the experience in the replay buffer
        replay_buffer.add(state, action, reward, next_state, done)
        
        # Train the DDPG algorithm
        critic_loss, actor_loss = train_ddpg(actor_model, critic_model, target_actor, target_critic, replay_buffer, batch_size, gamma, tau)
        
        state = next_state
        episode_reward += reward
    
    print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Critic Loss = {critic_loss:.4f}, Actor Loss = {actor_loss:.4f}")

# Save the trained models
actor_model.save("adaptive_bioprint_actor.h5")
critic_model.save("adaptive_bioprint_critic.h5")
