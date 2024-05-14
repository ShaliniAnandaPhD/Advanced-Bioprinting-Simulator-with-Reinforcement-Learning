import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO

class BioprintingAgent(nn.Module):
    def __init__(self, observation_shape, action_space):
        super(BioprintingAgent, self).__init__()
        self.observation_shape = observation_shape
        self.action_space = action_space
        
        # Define the neural network architecture
        self.features_extractor = nn.Sequential(
            nn.Conv3d(observation_shape[3], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.policy_net = nn.Sequential(
            nn.Linear(self._get_feature_size(), 256),
            nn.ReLU(),
            nn.Linear(256, self.action_space.n)
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(self._get_feature_size(), 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def _get_feature_size(self):
        # Calculate the size of the output from the features_extractor
        dummy_input = torch.zeros(1, *self.observation_shape)
        features = self.features_extractor(dummy_input)
        return features.view(-1).size(0)
    
    def forward(self, observation):
        # Extract features from the observation
        features = self.features_extractor(observation)
        
        # Compute action probabilities and state value
        action_logits = self.policy_net(features)
        state_value = self.value_net(features)
        
        return action_logits, state_value

def train_agent(env, hyperparameters):
    # Create the reinforcement learning agent
    observation_shape = env.observation_space.shape
    action_space = env.action_space
    agent = BioprintingAgent(observation_shape, action_space)
    
    # Define the optimizer and learning rate scheduler
    optimizer = optim.Adam(agent.parameters(), lr=hyperparameters["learning_rate"])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=hyperparameters["lr_decay_steps"],
                                             gamma=hyperparameters["lr_decay_rate"])
    
    # Create the PPO model using Stable Baselines3
    model = PPO(
        policy=agent,
        env=env,
        learning_rate=hyperparameters["learning_rate"],
        n_steps=hyperparameters["n_steps"],
        batch_size=hyperparameters["batch_size"],
        n_epochs=hyperparameters["n_epochs"],
        gamma=hyperparameters["gamma"],
        gae_lambda=hyperparameters["gae_lambda"],
        clip_range=hyperparameters["clip_range"],
        clip_range_vf=hyperparameters["clip_range_vf"],
        max_grad_norm=hyperparameters["max_grad_norm"],
        optimizer=optimizer,
        use_sde=hyperparameters["use_sde"],
        sde_sample_freq=hyperparameters["sde_sample_freq"],
        tensorboard_log=hyperparameters["tensorboard_log"]
    )
    
    # Train the agent
    total_timesteps = hyperparameters["total_timesteps"]
    model.learn(total_timesteps=total_timesteps, callback=lr_scheduler)
    
    return model

if __name__ == "__main__":
    # Load the bioprinting environment
    from organ_bioprint_env import OrganBioprintEnv
    env = OrganBioprintEnv()
    
    # Define hyperparameters for training
    hyperparameters = {
        "learning_rate": 3e-4,
        "lr_decay_steps": 1000,
        "lr_decay_rate": 0.9,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "clip_range_vf": None,
        "max_grad_norm": 0.5,
        "use_sde": False,
        "sde_sample_freq": -1,
        "tensorboard_log": "./tensorboard_logs/",
        "total_timesteps": 1000000
    }
    
    # Train the agent
    model = train_agent(env, hyperparameters)
    
    # Save the trained model
    model.save("trained_bioprinting_agent")