# MultiRobotBioPrint: Multi-robot coordination in bioprinting using reinforcement learning
# Based on the paper "Multi-robot coordination in bioprinting using reinforcement learning" (2024)

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.optimizers import Adam

### Environment Setup ###

# Define the bioprinting environment
class BioPrintEnv:
    def __init__(self, num_robots, grid_size, target_shape):
        self.num_robots = num_robots
        self.grid_size = grid_size
        self.target_shape = target_shape
        self.robot_positions = np.random.randint(0, grid_size, size=(num_robots, 2))
        self.bioprint_grid = np.zeros((grid_size, grid_size), dtype=int)
        
    def reset(self):
        self.robot_positions = np.random.randint(0, self.grid_size, size=(self.num_robots, 2))
        self.bioprint_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        return self.get_state()
    
    def get_state(self):
        state = []
        for i in range(self.num_robots):
            robot_state = np.zeros((self.grid_size, self.grid_size, 2))
            robot_state[self.robot_positions[i][0], self.robot_positions[i][1], 0] = 1
            robot_state[:, :, 1] = self.bioprint_grid
            state.append(robot_state)
        return np.array(state)
    
    def step(self, actions):
        rewards = np.zeros(self.num_robots)
        for i in range(self.num_robots):
            action = actions[i]
            if action == 0:  # Move up
                self.robot_positions[i][0] = max(0, self.robot_positions[i][0] - 1)
            elif action == 1:  # Move down
                self.robot_positions[i][0] = min(self.grid_size - 1, self.robot_positions[i][0] + 1)
            elif action == 2:  # Move left
                self.robot_positions[i][1] = max(0, self.robot_positions[i][1] - 1)
            elif action == 3:  # Move right
                self.robot_positions[i][1] = min(self.grid_size - 1, self.robot_positions[i][1] + 1)
            elif action == 4:  # Bioprint
                self.bioprint_grid[self.robot_positions[i][0], self.robot_positions[i][1]] = 1
                rewards[i] = self.calculate_reward()
        
        done = self.is_done()
        return self.get_state(), rewards, done
    
    def calculate_reward(self):
        # Calculate the reward based on the similarity between bioprinted grid and target shape
        reward = np.sum(self.bioprint_grid[self.target_shape == 1]) / np.sum(self.target_shape)
        return reward
    
    def is_done(self):
        # Check if the bioprinting task is complete
        return np.array_equal(self.bioprint_grid, self.target_shape)

# Create the bioprinting environment
num_robots = 3
grid_size = 10
target_shape = np.random.randint(0, 2, size=(grid_size, grid_size))
env = BioPrintEnv(num_robots, grid_size, target_shape)

### MADDPG Algorithm ###

# Define the actor and critic networks for MADDPG
def create_actor_network(state_shape, action_dim):
    state_input = Input(shape=state_shape)
    x = Dense(128, activation='relu')(state_input)
    x = Dense(128, activation='relu')(x)
    output = Dense(action_dim, activation='softmax')(x)
    model = Model(inputs=state_input, outputs=output)
    return model

def create_critic_network(state_shape, action_dim):
    state_input = Input(shape=state_shape)
    action_input = Input(shape=(action_dim,))
    x = Dense(128, activation='relu')(state_input)
    x = Concatenate()([x, action_input])
    x = Dense(128, activation='relu')(x)
    output = Dense(1, activation='linear')(x)
    model = Model(inputs=[state_input, action_input], outputs=output)
    return model

# Define the MADDPG agent
class MADDPGAgent:
    def __init__(self, num_agents, state_shape, action_dim, actor_lr, critic_lr, gamma, tau):
        self.num_agents = num_agents
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        
        self.actors = [create_actor_network(state_shape, action_dim) for _ in range(num_agents)]
        self.critics = [create_critic_network(state_shape, action_dim) for _ in range(num_agents)]
        self.target_actors = [create_actor_network(state_shape, action_dim) for _ in range(num_agents)]
        self.target_critics = [create_critic_network(state_shape, action_dim) for _ in range(num_agents)]
        
        self.actor_optimizers = [Adam(learning_rate=actor_lr) for _ in range(num_agents)]
        self.critic_optimizers = [Adam(learning_rate=critic_lr) for _ in range(num_agents)]
        
    def act(self, states):
        actions = []
        for i in range(self.num_agents):
            state = states[i]
            action_probs = self.actors[i].predict(np.expand_dims(state, axis=0))[0]
            action = np.random.choice(self.action_dim, p=action_probs)
            actions.append(action)
        return actions
    
    def train(self, states, actions, rewards, next_states, dones):
        for i in range(self.num_agents):
            with tf.GradientTape(persistent=True) as tape:
                # Compute critic loss
                next_actions = [self.target_actors[j].predict(np.expand_dims(next_states[j], axis=0))[0] for j in range(self.num_agents)]
                next_critic_inputs = [np.expand_dims(next_states[j], axis=0) for j in range(self.num_agents)] + next_actions
                next_q_values = self.target_critics[i].predict(next_critic_inputs)
                target_q_values = rewards[i] + self.gamma * next_q_values * (1 - dones)
                critic_inputs = [np.expand_dims(states[j], axis=0) for j in range(self.num_agents)] + [np.expand_dims(actions[j], axis=0) for j in range(self.num_agents)]
                q_values = self.critics[i].predict(critic_inputs)
                critic_loss = tf.reduce_mean(tf.square(target_q_values - q_values))
                
                # Compute actor loss
                actor_inputs = [np.expand_dims(states[j], axis=0) for j in range(self.num_agents)]
                actor_outputs = self.actors[i].predict(actor_inputs[i])
                critic_inputs = actor_inputs + [actor_outputs]
                actor_loss = -tf.reduce_mean(self.critics[i].predict(critic_inputs))
                
            # Update critic
            critic_grad = tape.gradient(critic_loss, self.critics[i].trainable_variables)
            self.critic_optimizers[i].apply_gradients(zip(critic_grad, self.critics[i].trainable_variables))
            
            # Update actor
            actor_grad = tape.gradient(actor_loss, self.actors[i].trainable_variables)
            self.actor_optimizers[i].apply_gradients(zip(actor_grad, self.actors[i].trainable_variables))
            
    def update_target_networks(self):
        for i in range(self.num_agents):
            actor_weights = self.actors[i].get_weights()
            target_actor_weights = self.target_actors[i].get_weights()
            for j in range(len(actor_weights)):
                target_actor_weights[j] = self.tau * actor_weights[j] + (1 - self.tau) * target_actor_weights[j]
            self.target_actors[i].set_weights(target_actor_weights)
            
            critic_weights = self.critics[i].get_weights()
            target_critic_weights = self.target_critics[i].get_weights()
            for j in range(len(critic_weights)):
                target_critic_weights[j] = self.tau * critic_weights[j] + (1 - self.tau) * target_critic_weights[j]
            self.target_critics[i].set_weights(target_critic_weights)
            
    def save_models(self):
        for i in range(self.num_agents):
            self.actors[i].save(f'actor_{i}.h5')
            self.critics[i].save(f'critic_{i}.h5')
            
    def load_models(self):
        for i in range(self.num_agents):
            self.actors[i] = tf.keras.models.load_model(f'actor_{i}.h5')
            self.critics[i] = tf.keras.models.load_model(f'critic_{i}.h5')

### Training ###

# Hyperparameters
num_episodes = 1000
batch_size = 32
actor_lr = 0.001
critic_lr = 0.002
gamma = 0.99
tau = 0.005

# Create MADDPG agent
state_shape = (grid_size, grid_size, 2)
action_dim = 5
agent = MADDPGAgent(num_robots, state_shape, action_dim, actor_lr, critic_lr, gamma, tau)

# Training loop
for episode in range(num_episodes):
    states = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        actions = agent.act(states)
        next_states, rewards, done = env.step(actions)
        episode_reward += np.mean(rewards)
        
        agent.train(states, actions, rewards, next_states, done)
        agent.update_target_networks()
        
        states = next_states
        
    print(f"Episode: {episode+1}, Reward: {episode_reward}")
    
# Save trained models
agent.save_models()
