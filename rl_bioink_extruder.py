# AdaptiveExtruder: Adaptive bioink extrusion control using deep reinforcement learning
# Based on the paper "Adaptive bioink extrusion control using deep reinforcement learning" (2024)

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.optimizers import Adam

### Environment Setup ###

# Define the bioink extrusion environment
class BioinkExtrusionEnv:
    def __init__(self, min_rate, max_rate, target_quality):
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.target_quality = target_quality
        self.current_rate = min_rate
        self.time_step = 0
        
    def reset(self):
        self.current_rate = self.min_rate
        self.time_step = 0
        return self.get_state()
    
    def get_state(self):
        # State includes current extrusion rate, print quality, and material properties
        state = [self.current_rate, self.get_print_quality(), self.get_material_properties()]
        return np.array(state)
    
    def step(self, action):
        # Update extrusion rate based on action
        self.current_rate += action
        self.current_rate = np.clip(self.current_rate, self.min_rate, self.max_rate)
        
        # Calculate reward based on print quality and extrusion rate
        print_quality = self.get_print_quality()
        reward = -abs(print_quality - self.target_quality) - 0.1 * abs(action)
        
        # Increment time step
        self.time_step += 1
        
        # Check if episode is done
        done = (self.time_step >= 100)
        
        return self.get_state(), reward, done
    
    def get_print_quality(self):
        # Simulate print quality based on extrusion rate and material properties
        # This is a placeholder function and should be replaced with actual quality measurement
        quality = np.random.normal(loc=self.current_rate/10, scale=0.1)
        return quality
    
    def get_material_properties(self):
        # Simulate material properties (e.g., viscosity, surface tension)
        # This is a placeholder function and should be replaced with actual material property measurements
        viscosity = np.random.normal(loc=1.0, scale=0.1)
        surface_tension = np.random.normal(loc=0.5, scale=0.05)
        return [viscosity, surface_tension]

# Create the bioink extrusion environment
min_rate = 0.1  # Minimum extrusion rate (mL/s)
max_rate = 2.0  # Maximum extrusion rate (mL/s)
target_quality = 0.8  # Target print quality (0-1)
env = BioinkExtrusionEnv(min_rate, max_rate, target_quality)

### Actor-Critic Model ###

# Define the actor model
def create_actor_model(state_shape, action_dim):
    state_input = Input(shape=state_shape)
    x = Dense(128, activation='relu')(state_input)
    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(128)(x)
    x = Dense(256, activation='relu')(x)
    output = Dense(action_dim, activation='tanh')(x)
    model = tf.keras.Model(inputs=state_input, outputs=output)
    return model

# Define the critic model
def create_critic_model(state_shape):
    state_input = Input(shape=state_shape)
    x = Dense(128, activation='relu')(state_input)
    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(128)(x)
    x = Dense(256, activation='relu')(x)
    output = Dense(1)(x)
    model = tf.keras.Model(inputs=state_input, outputs=output)
    return model

# Create the actor and critic models
state_shape = (3,)  # State includes extrusion rate, print quality, and material properties
action_dim = 1  # Action is the change in extrusion rate
actor_model = create_actor_model(state_shape, action_dim)
critic_model = create_critic_model(state_shape)

# Define the model optimizer
actor_optimizer = Adam(learning_rate=0.001)
critic_optimizer = Adam(learning_rate=0.002)

### Training Loop ###

# Hyperparameters
num_episodes = 1000
max_steps_per_episode = 100
discount_factor = 0.99
actor_update_steps = 10
critic_update_steps = 10

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    
    for step in range(max_steps_per_episode):
        # Select action based on actor model
        state_input = np.reshape(state, [1, state_shape[0]])
        action = actor_model.predict(state_input)[0]
        
        # Take action and observe next state and reward
        next_state, reward, done = env.step(action)
        episode_reward += reward
        
        # Store transition in replay buffer
        # (Omitted for brevity)
        
        # Update critic model
        if step % critic_update_steps == 0:
            # Sample a batch of transitions from replay buffer
            # (Omitted for brevity)
            
            # Compute target Q-values
            next_state_inputs = np.reshape(next_states, [batch_size, state_shape[0]])
            target_q_values = critic_model.predict(next_state_inputs)
            target_q_values = rewards + (1 - dones) * discount_factor * target_q_values
            
            # Train critic model on the sampled batch
            state_inputs = np.reshape(states, [batch_size, state_shape[0]])
            critic_loss = critic_model.train_on_batch(state_inputs, target_q_values)
        
        # Update actor model
        if step % actor_update_steps == 0:
            # Compute critic gradients
            with tf.GradientTape() as tape:
                state_inputs = np.reshape(states, [batch_size, state_shape[0]])
                actions = actor_model(state_inputs)
                critic_values = critic_model([state_inputs, actions])
                actor_loss = -tf.reduce_mean(critic_values)
            
            # Compute actor gradients and update actor model
            actor_gradients = tape.gradient(actor_loss, actor_model.trainable_variables)
            actor_optimizer.apply_gradients(zip(actor_gradients, actor_model.trainable_variables))
        
        # Update state
        state = next_state
        
        if done:
            break
    
    # Print episode summary
    print(f"Episode {episode+1}: Reward = {episode_reward:.2f}")

# Save the trained models
actor_model.save("adaptive_extruder_actor.h5")
critic_model.save("adaptive_extruder_critic.h5")
