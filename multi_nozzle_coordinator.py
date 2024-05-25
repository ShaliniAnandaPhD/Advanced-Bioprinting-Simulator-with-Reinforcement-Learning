# MultiNozzleCoordinator: Multi-nozzle coordination in bioprinting using deep reinforcement learning
# Based on the paper "Multi-nozzle coordination in bioprinting using deep reinforcement learning" (2024)

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import networkx as nx

### Environment Setup ###

# Define the multi-nozzle bioprinting environment
class BioprintingEnv:
    def __init__(self, num_nozzles, grid_size, target_structure):
        self.num_nozzles = num_nozzles
        self.grid_size = grid_size
        self.target_structure = target_structure
        self.nozzle_positions = np.random.randint(0, grid_size, size=(num_nozzles, 2))
        self.printed_structure = np.zeros((grid_size, grid_size), dtype=int)
        
    def reset(self):
        self.nozzle_positions = np.random.randint(0, self.grid_size, size=(self.num_nozzles, 2))
        self.printed_structure = np.zeros((self.grid_size, self.grid_size), dtype=int)
        return self.get_state()
    
    def get_state(self):
        state = []
        for i in range(self.num_nozzles):
            nozzle_state = np.zeros((self.grid_size, self.grid_size, 2))
            nozzle_state[self.nozzle_positions[i][0], self.nozzle_positions[i][1], 0] = 1
            nozzle_state[:, :, 1] = self.printed_structure
            state.append(nozzle_state)
        return np.array(state)
    
    def step(self, actions):
        rewards = np.zeros(self.num_nozzles)
        for i in range(self.num_nozzles):
            action = actions[i]
            if action == 0:  # Move up
                self.nozzle_positions[i][0] = max(0, self.nozzle_positions[i][0] - 1)
            elif action == 1:  # Move down
                self.nozzle_positions[i][0] = min(self.grid_size - 1, self.nozzle_positions[i][0] + 1)
            elif action == 2:  # Move left
                self.nozzle_positions[i][1] = max(0, self.nozzle_positions[i][1] - 1)
            elif action == 3:  # Move right
                self.nozzle_positions[i][1] = min(self.grid_size - 1, self.nozzle_positions[i][1] + 1)
            elif action == 4:  # Print
                self.printed_structure[self.nozzle_positions[i][0], self.nozzle_positions[i][1]] = 1
                rewards[i] = self.calculate_reward(i)
        
        done = self.is_done()
        return self.get_state(), rewards, done
    
    def calculate_reward(self, nozzle_id):
        # Calculate the reward based on the printed structure and target structure
        reward = np.sum(self.printed_structure[self.target_structure == 1]) / np.sum(self.target_structure)
        return reward
    
    def is_done(self):
        # Check if the printed structure matches the target structure
        return np.array_equal(self.printed_structure, self.target_structure)

# Create the multi-nozzle bioprinting environment
num_nozzles = 3
grid_size = 10
target_structure = np.random.randint(0, 2, size=(grid_size, grid_size))
env = BioprintingEnv(num_nozzles, grid_size, target_structure)

### Actor-Critic Models ###

# Define the actor model
def create_actor_model(state_shape, action_dim):
    state_input = Input(shape=state_shape)
    x = Dense(128, activation='relu')(state_input)
    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(128)(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(action_dim, activation='softmax')(x)
    model = Model(inputs=state_input, outputs=output)
    return model

# Define the critic model
def create_critic_model(state_shape):
    state_input = Input(shape=state_shape)
    x = Dense(128, activation='relu')(state_input)
    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(128)(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(1)(x)
    model = Model(inputs=state_input, outputs=output)
    return model

# Create the actor and critic models for each nozzle
state_shape = (grid_size, grid_size, 2)
action_dim = 5
actor_models = [create_actor_model(state_shape, action_dim) for _ in range(num_nozzles)]
critic_models = [create_critic_model(state_shape) for _ in range(num_nozzles)]

# Define the model optimizers
actor_optimizer = Adam(learning_rate=0.001)
critic_optimizer = Adam(learning_rate=0.002)

### Graph Neural Network ###

# Define the graph neural network model
def create_gnn_model(state_shape, num_nozzles):
    state_inputs = [Input(shape=state_shape) for _ in range(num_nozzles)]
    x = Concatenate()(state_inputs)
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(num_nozzles)(x)
    model = Model(inputs=state_inputs, outputs=output)
    return model

# Create the graph neural network model
gnn_model = create_gnn_model(state_shape, num_nozzles)

### Training Loop ###

# Hyperparameters
num_episodes = 1000
max_steps_per_episode = 100
discount_factor = 0.99
actor_update_steps = 10
critic_update_steps = 10
gnn_update_steps = 10

# Training loop
for episode in range(num_episodes):
    states = env.reset()
    episode_rewards = np.zeros(num_nozzles)
    
    for step in range(max_steps_per_episode):
        # Select actions based on actor models
        actions = []
        for i in range(num_nozzles):
            state = np.expand_dims(states[i], axis=0)
            action_probs = actor_models[i].predict(state)[0]
            action = np.random.choice(action_dim, p=action_probs)
            actions.append(action)
        
        # Take actions and observe next states and rewards
        next_states, rewards, done = env.step(actions)
        episode_rewards += rewards
        
        # Update critic models
        for i in range(num_nozzles):
            state = np.expand_dims(states[i], axis=0)
            next_state = np.expand_dims(next_states[i], axis=0)
            target_q_value = rewards[i] + (1 - done) * discount_factor * critic_models[i].predict(next_state)[0]
            with tf.GradientTape() as tape:
                q_value = critic_models[i](state)
                critic_loss = tf.reduce_mean(tf.square(target_q_value - q_value))
            critic_gradients = tape.gradient(critic_loss, critic_models[i].trainable_variables)
            critic_optimizer.apply_gradients(zip(critic_gradients, critic_models[i].trainable_variables))
        
        # Update actor models
        for i in range(num_nozzles):
            state = np.expand_dims(states[i], axis=0)
            with tf.GradientTape() as tape:
                action_probs = actor_models[i](state)
                log_probs = tf.math.log(action_probs)
                expected_q_value = critic_models[i](state)
                actor_loss = -tf.reduce_mean(log_probs * expected_q_value)
            actor_gradients = tape.gradient(actor_loss, actor_models[i].trainable_variables)
            actor_optimizer.apply_gradients(zip(actor_gradients, actor_models[i].trainable_variables))
        
        # Update graph neural network model
        gnn_inputs = [np.expand_dims(state, axis=0) for state in next_states]
        with tf.GradientTape() as tape:
            gnn_outputs = gnn_model(gnn_inputs)
            gnn_targets = np.zeros((1, num_nozzles))
            for i in range(num_nozzles):
                gnn_targets[0, i] = rewards[i] + (1 - done) * discount_factor * critic_models[i].predict(gnn_inputs[i])[0]
            gnn_loss = tf.reduce_mean(tf.square(gnn_targets - gnn_outputs))
        gnn_gradients = tape.gradient(gnn_loss, gnn_model.trainable_variables)
        gnn_optimizer.apply_gradients(zip(gnn_gradients, gnn_model.trainable_variables))
        
        states = next_states
        
        if done:
            break
    
    print(f"Episode: {episode+1}, Total Rewards: {episode_rewards}")

# Save the trained models
for i in range(num_nozzles):
    actor_models[i].save(f"actor_model_{i}.h5")
    critic_models[i].save(f"critic_model_{i}.h5")
gnn_model.save("gnn_model.h5")
